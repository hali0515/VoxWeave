"""Split preview, confirm and undo workflow of the speaker audition server."""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from voxweave import artifacts, fsio, sidecars, turnembed, vocals, voiceembed
from voxweave.core.overlay import spans_in
from voxweave.speakerserve_mapping import (
    _check_names,
    _mapping_speakers,
    _strict_json_loads,
)
from voxweave.voicebase import (
    MAX_PROVENANCE_STRING_BYTES,
    VOICEPRINTS_MAX_BYTES,
    Phase2DataError,
    canonical_turns_digest,
    encode_json_bytes,
    media_fingerprint,
    require_mapping,
    require_string,
    strict_json_object_loads,
    strict_turn_projection,
    validate_voiceprint_conjunction,
)
from voxweave.voiceepisode import episode_lock
from voxweave.voicematch import (
    build_compatibility_fingerprint,
    delete_suggest,
    require_known_compatibility,
)
from voxweave.vocalscache import SeparatorIdentity, validate_separator_identity

if TYPE_CHECKING:
    from voxweave.speakerserve import SpeakerHTTPServer

MAX_UNDO_BYTES = 64 * 1024 * 1024


class SplitConflict(RuntimeError):
    """The staged split no longer matches authoritative episode inputs."""


@dataclass(frozen=True, slots=True)
class _SplitProposal:
    speaker_id: str
    sibling_bytes: bytes
    voiceprints_path: Path
    voiceprints_bytes: bytes
    media_fingerprint: str
    assignment: tuple[tuple[int, str], ...]
    embeddings: tuple[tuple[int, tuple[float, ...]], ...]
    # Decoupled lane only: the centroid-v1 voiceprints of groups "A" / "B"
    # (a group without a usable segment is absent). None = legacy lane, whose
    # confirm averages the per-turn embeddings above.
    recipe_centroids: tuple[tuple[str, tuple[float, ...]], ...] | None = None


@dataclass(frozen=True, slots=True)
class _StagedSplit:
    sibling_bytes: bytes
    voiceprints_path: Path
    voiceprints_bytes: bytes
    media_fingerprint: str
    turns: tuple[tuple[float, float, str], ...]
    selected_indices: tuple[int, ...]
    embedding_dim: int
    embedding_model: str
    embedding_checkpoint: str
    pyannote_version: str
    audio_separated: bool
    audio_normalized: bool
    audio_separator: SeparatorIdentity | None
    embedding_lane: str = turnembed.LANE_LEGACY
    # The sibling's voiced and sung spans, for picking clean preview clips.
    vad_speech: tuple[tuple[float, float], ...] = ()
    sing_spans: tuple[tuple[float, float], ...] = ()

    def embedding_identity(self) -> turnembed.EmbeddingIdentity:
        """The provider identity the bound voiceprints were captured with."""
        if self.embedding_lane == turnembed.LANE_DECOUPLED:
            return turnembed.EmbeddingIdentity.decoupled(
                self.embedding_model, self.embedding_checkpoint
            )
        return turnembed.EmbeddingIdentity(
            model=self.embedding_model,
            checkpoint_sha256=self.embedding_checkpoint,
            pyannote_version=self.pyannote_version,
        )


def _absolute(path: Path) -> Path:
    return Path(os.path.realpath(os.path.abspath(os.fspath(Path(path)))))


def _validated_speaker_request(value: object, speaker_ids: Sequence[str]) -> str:
    if not isinstance(value, dict) or set(value) != {"speaker_id"}:
        raise ValueError("split request must contain only speaker_id")
    speaker_id = value["speaker_id"]
    if not isinstance(speaker_id, str) or speaker_id not in set(speaker_ids):
        raise ValueError("split request contains an unknown speaker id")
    return speaker_id


def _prepare_split_wav(media_path: Path, staged: _StagedSplit) -> Path:
    """Reproduce the audio transform recorded by the bound voiceprints."""
    from voxweave.chunking import decode_to_wav
    from voxweave.vocalscache import (
        cache_lock,
        load_cache_companion,
        validate_cache_pair,
    )

    audio_filter = vocals.ASR_LOUDNORM if staged.audio_normalized else None
    if not staged.audio_separated:
        return decode_to_wav(
            media_path,
            sample_rate=turnembed.SAMPLE_RATE,
            mono=True,
            audio_filter=audio_filter,
        )

    if staged.audio_separator is None:
        raise SplitConflict("separated voiceprints lack a resolved separator identity")
    cache_path = vocals.cache_vocals_path(media_path)
    with cache_lock(cache_path) as handle:
        try:
            companion, _validated = load_cache_companion(handle.companion_path)
            validate_cache_pair(
                companion,
                handle.cache_path,
                media_fingerprint=staged.media_fingerprint,
                separator=staged.audio_separator,
            )
        except (OSError, Phase2DataError) as exc:
            raise SplitConflict(
                "the separated vocals cache does not match this voiceprint capture"
            ) from exc
        return decode_to_wav(
            handle.cache_path,
            sample_rate=turnembed.SAMPLE_RATE,
            mono=True,
            audio_filter=audio_filter,
        )


@dataclass(frozen=True, slots=True)
class _StagedProvenance:
    embedding_lane: str
    embedding_model: str
    embedding_checkpoint: str
    pyannote_version: str
    audio_separated: bool
    audio_normalized: bool
    audio_separator: SeparatorIdentity | None


def _staged_provenance(sidecar: Mapping[str, object]) -> _StagedProvenance:
    provenance = require_mapping(sidecar.get("provenance"), "provenance")
    require_known_compatibility(build_compatibility_fingerprint(provenance))
    decoupled = provenance.get("embedding_lane") == turnembed.LANE_DECOUPLED
    embedding_model = require_string(
        provenance.get("embedding_model"),
        "provenance.embedding_model",
        max_bytes=MAX_PROVENANCE_STRING_BYTES,
    )
    embedding_checkpoint = require_string(
        provenance.get("embedding_checkpoint"),
        "provenance.embedding_checkpoint",
        max_bytes=MAX_PROVENANCE_STRING_BYTES,
    )
    if decoupled:
        # The recipe is what split-confirm must reproduce; one this version
        # does not implement cannot be recomputed faithfully.
        recipe = provenance.get("embedding_recipe")
        if recipe != voiceembed.CENTROID_RECIPE:
            raise SplitConflict(
                f"voiceprints use centroid recipe {recipe!r}, which this voxweave "
                f"version cannot reproduce (it implements "
                f"{voiceembed.CENTROID_RECIPE!r})"
            )
        # Informational only on this lane: pyannote never runs the embedder.
        raw_version = provenance.get("pyannote_version")
        pyannote_version = raw_version if isinstance(raw_version, str) else ""
    else:
        pyannote_version = require_string(
            provenance.get("pyannote_version"),
            "provenance.pyannote_version",
            max_bytes=MAX_PROVENANCE_STRING_BYTES,
        )
    audio = require_mapping(provenance.get("audio"), "provenance.audio")
    separated = audio.get("separated")
    normalized = audio.get("normalized")
    sample_rate = audio.get("sample_rate")
    if type(separated) is not bool or type(normalized) is not bool:
        raise Phase2DataError("provenance.audio separated/normalized must be booleans")
    if type(sample_rate) is not int or sample_rate != turnembed.SAMPLE_RATE:
        raise Phase2DataError(
            f"provenance.audio.sample_rate must be {turnembed.SAMPLE_RATE}"
        )
    separator = (
        validate_separator_identity(audio.get("separator")) if separated else None
    )
    return _StagedProvenance(
        embedding_lane=(
            turnembed.LANE_DECOUPLED if decoupled else turnembed.LANE_LEGACY
        ),
        embedding_model=embedding_model,
        embedding_checkpoint=embedding_checkpoint,
        pyannote_version=pyannote_version,
        audio_separated=separated,
        audio_normalized=normalized,
        audio_separator=separator,
    )


def _stage_split_inputs(
    server: SpeakerHTTPServer,
    speaker_id: str,
) -> _StagedSplit:
    with episode_lock(server.media_path):
        sibling_bytes = server.sibling_path.read_bytes()
        sibling = strict_json_object_loads(
            sibling_bytes,
            max_bytes=max(1, len(sibling_bytes)),
            source=server.sibling_path.name,
        )
        turns = strict_turn_projection(sibling.get("speaker_turns"))
        selected = tuple(
            index
            for index, (_start, _end, label) in enumerate(turns)
            if label == speaker_id
        )
        if len(selected) < 2:
            raise turnembed.UnsplittableSpeakerError(
                "a speaker needs at least two turns to split"
            )
        sidecar_path = sidecars.voiceprints_path(server.media_path)
        try:
            sidecar_bytes = sidecar_path.read_bytes()
        except OSError as exc:
            raise SplitConflict(
                "speaker splitting requires bound voiceprints; rerun with "
                "--diarize --voiceprints"
            ) from exc
        sidecar = strict_json_object_loads(
            sidecar_bytes,
            max_bytes=VOICEPRINTS_MAX_BYTES,
            source=sidecar_path.name,
        )
        fingerprint = media_fingerprint(server.media_path)
        validated = validate_voiceprint_conjunction(
            sidecar,
            sibling,
            fingerprint,
        )
        staged = _staged_provenance(sidecar)
        return _StagedSplit(
            sibling_bytes=sibling_bytes,
            voiceprints_path=sidecar_path,
            voiceprints_bytes=sidecar_bytes,
            media_fingerprint=fingerprint,
            turns=turns,
            selected_indices=selected,
            embedding_dim=validated.embedding_dim,
            embedding_model=staged.embedding_model,
            embedding_checkpoint=staged.embedding_checkpoint,
            pyannote_version=staged.pyannote_version,
            audio_separated=staged.audio_separated,
            audio_normalized=staged.audio_normalized,
            audio_separator=staged.audio_separator,
            embedding_lane=staged.embedding_lane,
            vad_speech=tuple(spans_in(sibling.get("vad_speech")) or ()),
            sing_spans=tuple(spans_in(sibling.get("sing_spans")) or ()),
        )


# Stand-in speaker labels of the two proposed groups while preview clips are
# picked (never a real diarizer id).
_SPLIT_GROUP_LABEL = "\x00split-group-{}"


def _proposal_groups(
    wav_path: Path,
    staged: _StagedSplit,
    assignment: Mapping[int, str],
) -> list[dict[str, object]]:
    from voxweave import speakers

    turns = staged.turns
    # The page's clip rules: outside every other speaker's turn (the other
    # group's included), inside VAD speech, outside singing.
    clean = speakers.select_snippets(
        [
            (
                start,
                end,
                _SPLIT_GROUP_LABEL.format(assignment[index])
                if index in assignment
                else label,
            )
            for index, (start, end, label) in enumerate(turns)
        ],
        staged.vad_speech,
        staged.sing_spans,
    )
    groups: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="voxweave_split_clips_") as temp_dir:
        root = Path(temp_dir)
        for label in ("A", "B"):
            indices = sorted(
                index for index, group in assignment.items() if group == label
            )
            spans = [(turns[index][0], turns[index][1]) for index in indices]
            # A group without a clean stretch still gets clips of its turns.
            picks = clean.get(
                _SPLIT_GROUP_LABEL.format(label)
            ) or speakers._pick_spread(spans, speakers.MAX_SNIPPETS_PER_SPEAKER)
            samples: list[dict[str, object]] = []
            for clip_index, (start, end) in enumerate(picks):
                clip_path = root / f"{label}-{clip_index}.mp3"
                speakers.extract_clip(wav_path, start, end, clip_path)
                encoded = base64.b64encode(clip_path.read_bytes()).decode("ascii")
                samples.append(
                    {
                        "start": start,
                        "end": end,
                        "src": f"data:audio/mpeg;base64,{encoded}",
                    }
                )
            groups.append(
                {
                    "label": label,
                    "turn_indices": indices,
                    "turn_count": len(indices),
                    "total_duration": math.fsum(
                        turns[index][1] - turns[index][0] for index in indices
                    ),
                    "samples": samples,
                }
            )
    return groups


def _recheck_split_inputs(server: SpeakerHTTPServer, staged: _StagedSplit) -> None:
    with episode_lock(server.media_path):
        if server.sibling_path.read_bytes() != staged.sibling_bytes:
            raise SplitConflict("speaker turns changed during split preview; retry")
        current_sidecar_path = sidecars.voiceprints_path(server.media_path)
        if current_sidecar_path != staged.voiceprints_path:
            raise SplitConflict(
                "voiceprints storage changed during split preview; retry"
            )
        if current_sidecar_path.read_bytes() != staged.voiceprints_bytes:
            raise SplitConflict("voiceprints changed during split preview; retry")
        if media_fingerprint(server.media_path) != staged.media_fingerprint:
            raise SplitConflict("media changed during split preview; retry")


def _require_embedding_identity(
    staged: _StagedSplit,
    embeddings: object,
) -> None:
    if not isinstance(embeddings, turnembed.AttestedTurnEmbeddings):
        raise turnembed.TurnEmbeddingError(
            "turn embedding provider did not attest its loaded identity"
        )
    identity = embeddings.identity
    if (
        identity.lane != staged.embedding_lane
        or identity.model != staged.embedding_model
        or identity.checkpoint_sha256 != staged.embedding_checkpoint
        # pyannote's version is part of the identity only on the legacy lane,
        # where pyannote itself runs the embedding checkpoint.
        or (
            staged.embedding_lane == turnembed.LANE_LEGACY
            and identity.pyannote_version != staged.pyannote_version
        )
    ):
        raise SplitConflict(
            "the turn embedding provider does not match the voiceprint capture"
        )


# Stand-in label for group B while its centroid-v1 segments are computed: the
# real SPEAKER_NN id is only minted at confirm time, and the recipe needs no
# more than "a speaker other than A and everyone else".
_SPLIT_B_PLACEHOLDER = "\x00split-group-b"


def _split_recipe_centroids(
    wav_path: Path,
    staged: _StagedSplit,
    speaker_id: str,
    assignment: Mapping[int, str],
    identity: turnembed.EmbeddingIdentity,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """centroid-v1 voiceprints of both split groups, as capture would compute them."""
    relabeled = [
        (start, end, _SPLIT_B_PLACEHOLDER if assignment.get(index) == "B" else label)
        for index, (start, end, label) in enumerate(staged.turns)
    ]
    try:
        centroids = turnembed.recipe_centroids(
            wav_path,
            relabeled,
            (speaker_id, _SPLIT_B_PLACEHOLDER),
            identity,
        )
    except turnembed.EmbeddingIdentityMismatch as exc:
        raise _capture_mismatch(exc) from exc
    groups: list[tuple[str, tuple[float, ...]]] = []
    for group, label in (("A", speaker_id), ("B", _SPLIT_B_PLACEHOLDER)):
        vector = centroids.get(label)
        if vector is None:
            continue
        if len(vector) != staged.embedding_dim:
            raise turnembed.TurnEmbeddingError(
                "recomputed voiceprints do not match the bound voiceprint dimension"
            )
        groups.append((group, tuple(float(value) for value in vector)))
    return tuple(groups)


def _capture_mismatch(exc: turnembed.EmbeddingIdentityMismatch) -> SplitConflict:
    """The 409 for an embedder this installation cannot reproduce.

    Same answer as :func:`_require_embedding_identity`: the capture is intact,
    the local embedder differs from the one it recorded.
    """
    return SplitConflict(
        f"the turn embedding provider does not match the voiceprint capture: {exc}"
    )


def _build_split_proposal(
    server: SpeakerHTTPServer,
    speaker_id: str,
) -> tuple[_SplitProposal, dict[str, object]]:
    staged = _stage_split_inputs(server, speaker_id)
    selected_turns = [staged.turns[index] for index in staged.selected_indices]
    identity = staged.embedding_identity()
    embedding_request = turnembed.AttestedTurnRequest(
        selected_turns,
        identity=identity,
    )
    wav_path = _prepare_split_wav(server.media_path, staged)
    try:
        try:
            provider_embeddings = turnembed.turn_embeddings(wav_path, embedding_request)
        except turnembed.EmbeddingIdentityMismatch as exc:
            raise _capture_mismatch(exc) from exc
        _require_embedding_identity(staged, provider_embeddings)
        expected = set(range(len(selected_turns)))
        if set(provider_embeddings) != expected:
            raise turnembed.TurnEmbeddingError(
                "turn embedding provider did not return every requested turn"
            )
        local_embeddings = {
            index: turnembed.normalized_centroid([provider_embeddings[index]])
            for index in sorted(expected)
        }
        if any(
            len(local_embeddings[index]) != staged.embedding_dim for index in expected
        ):
            raise turnembed.TurnEmbeddingError(
                "turn embeddings do not match the bound voiceprint dimension"
            )
        local_assignment = turnembed.bisect_embeddings(local_embeddings)
        assignment = {
            staged.selected_indices[local_index]: group
            for local_index, group in local_assignment.items()
        }
        embeddings = {
            staged.selected_indices[local_index]: tuple(
                float(value) for value in local_embeddings[local_index]
            )
            for local_index in sorted(local_embeddings)
        }
        recipe = (
            _split_recipe_centroids(wav_path, staged, speaker_id, assignment, identity)
            if staged.embedding_lane == turnembed.LANE_DECOUPLED
            else None
        )
        groups = _proposal_groups(wav_path, staged, assignment)
    finally:
        wav_path.unlink(missing_ok=True)
        # Confirming needs no model; do not hold VRAM while the page is open.
        turnembed.release()
    _recheck_split_inputs(server, staged)
    ordered_assignment = tuple(sorted(assignment.items()))
    proposal = _SplitProposal(
        speaker_id=speaker_id,
        sibling_bytes=staged.sibling_bytes,
        voiceprints_path=staged.voiceprints_path,
        voiceprints_bytes=staged.voiceprints_bytes,
        media_fingerprint=staged.media_fingerprint,
        assignment=ordered_assignment,
        embeddings=tuple(sorted(embeddings.items())),
        recipe_centroids=recipe,
    )
    response: dict[str, object] = {
        "speaker_id": speaker_id,
        "assignment": {str(index): group for index, group in ordered_assignment},
        "groups": groups,
    }
    return proposal, response


def _validate_confirmation(value: object, proposal: _SplitProposal) -> None:
    if not isinstance(value, dict) or set(value) != {"speaker_id", "assignment"}:
        raise ValueError("confirmation must contain speaker_id and assignment")
    if value["speaker_id"] != proposal.speaker_id:
        raise ValueError("confirmation speaker id does not match the proposal")
    raw_assignment = value["assignment"]
    if not isinstance(raw_assignment, dict):
        raise ValueError("confirmation assignment must be an object")
    expected = {str(index): group for index, group in proposal.assignment}
    if raw_assignment != expected:
        raise ValueError("confirmation assignment does not match the proposal")


def _mapping_document(
    raw: bytes, *, source: str
) -> tuple[dict[str, object], dict[str, str]]:
    """Parse the on-disk mapping a split rewrites (or an undo restores).

    The same rules as ``/serve-info`` (:func:`_mapping_speakers`), except that
    every id is kept: a split adds an id and must not drop anyone's name.
    """
    try:
        value = _strict_json_loads(raw)
        speakers = _mapping_speakers(value)
        _check_names(speakers)
    except (UnicodeError, ValueError) as exc:
        raise SplitConflict(f"{source} is not a valid speaker mapping: {exc}") from exc
    return value, speakers


def _next_speaker_id(used: set[str]) -> str:
    index = 0
    while True:
        candidate = f"SPEAKER_{index:02d}"
        if candidate not in used:
            return candidate
        index += 1


def _json_bytes(value: object, *, newline: bool) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _undo_record(path: Path, before: bytes, after: bytes) -> dict[str, object]:
    return {
        "path": os.fspath(_absolute(path)),
        "before": base64.b64encode(before).decode("ascii"),
        "after_size": len(after),
        "after_sha256": _sha256(after),
    }


def _undo_bytes(
    *,
    fingerprint: str,
    sibling_path: Path,
    sibling_before: bytes,
    sibling_after: bytes,
    voiceprints_path: Path,
    voiceprints_before: bytes,
    voiceprints_after: bytes,
    mapping_path: Path,
    mapping_before: bytes,
    mapping_after: bytes,
) -> bytes:
    value = {
        "version": 1,
        "media_fingerprint": fingerprint,
        "files": {
            "sibling": _undo_record(sibling_path, sibling_before, sibling_after),
            "voiceprints": _undo_record(
                voiceprints_path,
                voiceprints_before,
                voiceprints_after,
            ),
            "mapping": _undo_record(mapping_path, mapping_before, mapping_after),
        },
    }
    return encode_json_bytes(value, max_bytes=MAX_UNDO_BYTES)


def _write_bytes(path: Path, raw: bytes, *, private: bool = True) -> None:
    """Replace ``path``; ``private=False`` only for the sibling JSON, a user
    deliverable (it keeps its mode), everything else here stays 0600."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SplitConflict(f"{path.name} is not valid UTF-8") from exc
    fsio.atomic_write_text(path, text, private=private)


def _restore_files(files: Iterable[tuple[Path, bytes, bool]]) -> None:
    for path, raw, private in files:
        _write_bytes(path, raw, private=private)


def _confirm_split(server: SpeakerHTTPServer, proposal: _SplitProposal) -> str:
    with episode_lock(server.media_path):
        sibling_bytes = server.sibling_path.read_bytes()
        if sibling_bytes != proposal.sibling_bytes:
            raise SplitConflict("speaker turns changed after the split preview; retry")
        sidecar_path = sidecars.voiceprints_path(server.media_path)
        if sidecar_path != proposal.voiceprints_path:
            raise SplitConflict("voiceprints storage changed after the split preview")
        sidecar_bytes = sidecar_path.read_bytes()
        if sidecar_bytes != proposal.voiceprints_bytes:
            raise SplitConflict("voiceprints changed after the split preview; retry")
        fingerprint = media_fingerprint(server.media_path)
        if fingerprint != proposal.media_fingerprint:
            raise SplitConflict("media changed after the split preview; retry")

        sibling = strict_json_object_loads(
            sibling_bytes,
            max_bytes=max(1, len(sibling_bytes)),
            source=server.sibling_path.name,
        )
        turns = strict_turn_projection(sibling.get("speaker_turns"))
        sidecar = strict_json_object_loads(
            sidecar_bytes,
            max_bytes=VOICEPRINTS_MAX_BYTES,
            source=sidecar_path.name,
        )
        validated = validate_voiceprint_conjunction(sidecar, sibling, fingerprint)

        mapping_path = sidecars.speakers_mapping_path(server.media_path)
        mapping_bytes = mapping_path.read_bytes()
        mapping, mapping_entries = _mapping_document(
            mapping_bytes,
            source=mapping_path.name,
        )
        sidecar_speakers = sidecar.get("speakers")
        if not isinstance(sidecar_speakers, dict):
            raise Phase2DataError("voiceprints speakers must be an object")
        used_ids = {
            *(label for _start, _end, label in turns),
            *sidecar_speakers.keys(),
            *mapping_entries.keys(),
        }
        new_id = _next_speaker_id(used_ids)

        assignments = dict(proposal.assignment)
        vectors = dict(proposal.embeddings)
        if set(assignments) != set(vectors):
            raise SplitConflict("split proposal embeddings are incomplete")
        for index in assignments:
            if index >= len(turns) or turns[index][2] != proposal.speaker_id:
                raise SplitConflict("split proposal no longer addresses the same turns")
        group_a = [
            vectors[index] for index, group in assignments.items() if group == "A"
        ]
        group_b = [
            vectors[index] for index, group in assignments.items() if group == "B"
        ]
        if any(
            len(vector) != validated.embedding_dim for vector in (*group_a, *group_b)
        ):
            raise SplitConflict("split proposal embedding dimension changed")
        if proposal.recipe_centroids is None:
            # Legacy lane: the mean of the whole-turn pyannote embeddings.
            group_centroids: dict[str, list[float] | None] = {
                "A": turnembed.normalized_centroid(group_a),
                "B": turnembed.normalized_centroid(group_b),
            }
        else:
            # Decoupled lane: the centroid-v1 voiceprints the proposal computed
            # with the capture's embedder, so the split sidecar matches what a
            # fresh capture of the relabelled turns would write.
            recipe = dict(proposal.recipe_centroids)
            group_centroids = {
                group: (list(recipe[group]) if group in recipe else None)
                for group in ("A", "B")
            }
            if any(
                vector is not None and len(vector) != validated.embedding_dim
                for vector in group_centroids.values()
            ):
                raise SplitConflict("split proposal embedding dimension changed")

        updated_sibling = copy.deepcopy(sibling)
        raw_turns = updated_sibling.get("speaker_turns")
        if not isinstance(raw_turns, list):
            raise Phase2DataError("speaker_turns must be an array")
        for index, group in assignments.items():
            raw_turn = raw_turns[index]
            if not isinstance(raw_turn, list) or len(raw_turn) != 3:
                raise Phase2DataError(f"speaker_turns[{index}] must be an array")
            if group == "B":
                raw_turn[2] = new_id
        updated_turns = strict_turn_projection(updated_sibling.get("speaker_turns"))

        updated_sidecar = copy.deepcopy(sidecar)
        updated_speakers = updated_sidecar.get("speakers")
        updated_binding = updated_sidecar.get("binding")
        if not isinstance(updated_speakers, dict) or not isinstance(
            updated_binding, dict
        ):
            raise Phase2DataError("voiceprints speakers and binding must be objects")
        for group, label in (("A", proposal.speaker_id), ("B", new_id)):
            centroid = group_centroids[group]
            if centroid is None:
                # Same rule as capture: no usable segment, no voiceprint.
                updated_speakers.pop(label, None)
            else:
                updated_speakers[label] = centroid
        updated_binding["turns_digest"] = canonical_turns_digest(updated_turns)
        validate_voiceprint_conjunction(
            updated_sidecar,
            updated_sibling,
            fingerprint,
        )

        updated_mapping_entries = dict(mapping_entries)
        updated_mapping_entries[new_id] = ""
        mapping["speakers"] = updated_mapping_entries
        sibling_after = _json_bytes(updated_sibling, newline=False)
        sidecar_after = encode_json_bytes(
            updated_sidecar,
            max_bytes=VOICEPRINTS_MAX_BYTES,
        )
        mapping_after = _json_bytes(mapping, newline=True)

        undo_path = artifacts.speaker_split_undo_path(server.media_path)
        previous_undo = undo_path.read_bytes() if undo_path.exists() else None
        snapshot = _undo_bytes(
            fingerprint=fingerprint,
            sibling_path=server.sibling_path,
            sibling_before=sibling_bytes,
            sibling_after=sibling_after,
            voiceprints_path=sidecar_path,
            voiceprints_before=sidecar_bytes,
            voiceprints_after=sidecar_after,
            mapping_path=mapping_path,
            mapping_before=mapping_bytes,
            mapping_after=mapping_after,
        )
        suggest_path = sidecars.speakers_suggest_path(server.media_path)
        _write_bytes(undo_path, snapshot)
        written: list[tuple[Path, bytes, bool]] = []
        try:
            for path, before, after, private in (
                (sidecar_path, sidecar_bytes, sidecar_after, True),
                (server.sibling_path, sibling_bytes, sibling_after, False),
                (mapping_path, mapping_bytes, mapping_after, True),
            ):
                written.append((path, before, private))
                _write_bytes(path, after, private=private)
            delete_suggest(suggest_path)
        except BaseException:
            _restore_files(reversed(written))
            if previous_undo is None:
                undo_path.unlink(missing_ok=True)
            else:
                _write_bytes(undo_path, previous_undo)
            raise

    server.speaker_ids = (*server.speaker_ids, new_id)
    server.pristine_mapping_path = None
    server.pristine_mapping_generation = None
    server.split_proposal = None
    server.session_terminal = True
    server.report(
        f"Split {proposal.speaker_id} into {proposal.speaker_id} and {new_id}"
    )
    server.report("Restart `voxweave speakers serve` to re-audition")
    return new_id


def _decoded_undo_record(
    value: object,
    *,
    field: str,
) -> tuple[Path, bytes, int, str]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "before",
        "after_size",
        "after_sha256",
    }:
        raise ValueError(f"undo {field} record has an invalid schema")
    raw_path = value["path"]
    encoded = value["before"]
    after_size = value["after_size"]
    after_hash = value["after_sha256"]
    if not isinstance(raw_path, str) or not isinstance(encoded, str):
        raise ValueError(f"undo {field} path and bytes must be strings")
    if type(after_size) is not int or after_size < 0:
        raise ValueError(f"undo {field} size must be a non-negative integer")
    if (
        not isinstance(after_hash, str)
        or len(after_hash) != 64
        or any(character not in "0123456789abcdef" for character in after_hash)
    ):
        raise ValueError(f"undo {field} digest is invalid")
    try:
        before = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"undo {field} bytes are invalid base64") from exc
    path = Path(raw_path)
    if not path.is_absolute() or path != _absolute(path):
        raise ValueError(f"undo {field} path is not normalized")
    return path, before, after_size, after_hash


def _load_undo(path: Path) -> tuple[str, dict[str, tuple[Path, bytes, int, str]]]:
    if not path.is_file():
        raise SplitConflict(
            "there is no speaker split to undo: it was already undone or purged"
        )
    raw = path.read_bytes()
    value = strict_json_object_loads(
        raw,
        max_bytes=MAX_UNDO_BYTES,
        source=path.name,
    )
    if set(value) != {"version", "media_fingerprint", "files"}:
        raise ValueError("speaker split undo snapshot has an invalid schema")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("speaker split undo snapshot has an invalid version")
    fingerprint = value["media_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("speaker split undo media fingerprint is invalid")
    files = value["files"]
    if not isinstance(files, dict) or set(files) != {
        "sibling",
        "voiceprints",
        "mapping",
    }:
        raise ValueError("speaker split undo files have an invalid schema")
    return fingerprint, {
        field: _decoded_undo_record(files[field], field=field)
        for field in ("sibling", "voiceprints", "mapping")
    }


# How an undo refusal names each file it checks; the page shows the message as is.
_UNDO_FILE_LABELS = {
    "sibling": "the transcript JSON",
    "voiceprints": "the voiceprints",
    "mapping": "the speaker mapping",
}


def _undo_split(server: SpeakerHTTPServer) -> None:
    undo_path = artifacts.speaker_split_undo_path(server.media_path)
    with episode_lock(server.media_path):
        fingerprint, records = _load_undo(undo_path)
        try:
            current_paths = {
                "sibling": _absolute(server.sibling_path),
                "voiceprints": _absolute(sidecars.voiceprints_path(server.media_path)),
                "mapping": _absolute(sidecars.speakers_mapping_path(server.media_path)),
            }
        except OSError as exc:
            raise SplitConflict(
                "undo refused: the episode's artifact storage changed since the split"
            ) from exc
        for field, expected_path in current_paths.items():
            recorded_path, _before, _size, _digest = records[field]
            if recorded_path != expected_path:
                raise SplitConflict(
                    f"undo refused: the storage of {_UNDO_FILE_LABELS[field]} "
                    f"({expected_path.name}) changed since the split"
                )
        try:
            current_fingerprint = media_fingerprint(server.media_path)
        except OSError as exc:
            raise SplitConflict(
                "undo refused: the media file changed since the split"
            ) from exc
        if current_fingerprint != fingerprint:
            raise SplitConflict("undo refused: the media file changed since the split")

        current: dict[str, bytes] = {}
        for field, path in current_paths.items():
            changed = SplitConflict(
                f"undo refused: {_UNDO_FILE_LABELS[field]} ({path.name}) "
                "changed since the split"
            )
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise changed from exc
            _recorded_path, before, expected_size, expected_hash = records[field]
            matches_after = len(raw) == expected_size and _sha256(raw) == expected_hash
            if raw != before and not matches_after:
                raise changed
            current[field] = raw

        sibling_before = records["sibling"][1]
        sidecar_before = records["voiceprints"][1]
        mapping_before = records["mapping"][1]
        sibling = strict_json_object_loads(
            sibling_before,
            max_bytes=max(1, len(sibling_before)),
            source=current_paths["sibling"].name,
        )
        sidecar = strict_json_object_loads(
            sidecar_before,
            max_bytes=VOICEPRINTS_MAX_BYTES,
            source=current_paths["voiceprints"].name,
        )
        validate_voiceprint_conjunction(sidecar, sibling, fingerprint)
        _mapping_document(mapping_before, source=current_paths["mapping"].name)

        restored: list[tuple[Path, bytes, bool]] = []
        try:
            for field, before in (
                ("voiceprints", sidecar_before),
                ("sibling", sibling_before),
                ("mapping", mapping_before),
            ):
                path = current_paths[field]
                private = field != "sibling"
                restored.append((path, current[field], private))
                _write_bytes(path, before, private=private)
            undo_path.unlink()
        except BaseException:
            _restore_files(reversed(restored))
            raise

    restored_turns = strict_turn_projection(sibling.get("speaker_turns"))
    server.speaker_ids = tuple(
        dict.fromkeys(label for _start, _end, label in restored_turns)
    )
    server.pristine_mapping_path = None
    server.pristine_mapping_generation = None
    server.split_proposal = None
    server.session_terminal = True
    server.report("Restored the previous speaker split generation")
    server.report("Restart `voxweave speakers serve` to re-audition")
