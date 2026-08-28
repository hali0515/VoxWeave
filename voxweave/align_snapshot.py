"""Exact align input snapshots and injective immutable JSON values.

The selected legacy reader and the strict P6 reader intentionally coexist:
default ``json.loads`` supplies tolerant last-key-wins semantics, while a
separate lexical tree retains duplicate occurrences, order, scalar type, exact
binary64 bits, and nonfinite class.  Neither projection is reconstructed from
the other.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from voxweave import realign
from voxweave.subformats import decode_subtitle_bytes, sniff_format


class FrozenJSONDomainError(TypeError):
    """A selected value cannot be represented in the closed JSON sum type."""

    def __init__(self, detail_code: str, message: str):
        super().__init__(message)
        self.detail_code = detail_code


@dataclass(frozen=True)
class FrozenAbsent:
    pass


@dataclass(frozen=True)
class FrozenNull:
    pass


@dataclass(frozen=True)
class FrozenBool:
    value: bool


@dataclass(frozen=True)
class FrozenInt:
    value: int


FloatSourceClass = Literal["finite", "nan", "positive-infinity", "negative-infinity"]


@dataclass(frozen=True)
class FrozenFloat:
    binary64_hex: str
    source_class: FloatSourceClass


@dataclass(frozen=True)
class FrozenString:
    value: str


@dataclass(frozen=True)
class FrozenArray:
    items: tuple[Any, ...]


@dataclass(frozen=True)
class FrozenObject:
    items: tuple[tuple[str, Any], ...]


FrozenJSON = (
    FrozenAbsent
    | FrozenNull
    | FrozenBool
    | FrozenInt
    | FrozenFloat
    | FrozenString
    | FrozenArray
    | FrozenObject
)

FROZEN_ABSENT = FrozenAbsent()
FROZEN_NULL = FrozenNull()


def _float_source_class(value: float) -> FloatSourceClass:
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "positive-infinity"
    if value == -math.inf:
        return "negative-infinity"
    return "finite"


def _freeze_float(value: float) -> FrozenFloat:
    return FrozenFloat(
        binary64_hex=struct.pack(">d", value).hex(),
        source_class=_float_source_class(value),
    )


def freeze_json(value: Any) -> FrozenJSON:
    """Freeze one JSON-domain value without coercion or key sorting."""

    def visit(node: Any, active: set[int]) -> FrozenJSON:
        if isinstance(
            node,
            (
                FrozenAbsent,
                FrozenNull,
                FrozenBool,
                FrozenInt,
                FrozenFloat,
                FrozenString,
                FrozenArray,
                FrozenObject,
            ),
        ):
            return node
        if node is FROZEN_ABSENT:
            return FROZEN_ABSENT
        if node is None:
            return FROZEN_NULL
        if type(node) is bool:
            return FrozenBool(node)
        if type(node) is int:
            return FrozenInt(node)
        if type(node) is float:
            return _freeze_float(node)
        if type(node) is str:
            return FrozenString(node)
        if isinstance(node, Mapping):
            identity = id(node)
            if identity in active:
                raise FrozenJSONDomainError(
                    "selected-json-cycle", "JSON object contains a cycle"
                )
            active.add(identity)
            try:
                items: list[tuple[str, FrozenJSON]] = []
                for key, member in node.items():
                    if type(key) is not str:
                        raise FrozenJSONDomainError(
                            "non-string-selected-object-key",
                            "JSON object key is not an exact string",
                        )
                    items.append((key, visit(member, active)))
                return FrozenObject(tuple(items))
            finally:
                active.remove(identity)
        if type(node) is list:
            identity = id(node)
            if identity in active:
                raise FrozenJSONDomainError(
                    "selected-json-cycle", "JSON array contains a cycle"
                )
            active.add(identity)
            try:
                return FrozenArray(tuple(visit(member, active) for member in node))
            finally:
                active.remove(identity)
        raise FrozenJSONDomainError(
            "unsupported-selected-json-node",
            f"unsupported JSON node type {type(node).__name__}",
        )

    return visit(value, set())


def _frame(payload: bytes) -> bytes:
    return len(payload).to_bytes(8, "big") + payload


_FLOAT_CLASS_TAG = {
    "finite": b"f",
    "nan": b"n",
    "positive-infinity": b"p",
    "negative-infinity": b"m",
}


def encode_frozen_json(value: FrozenJSON) -> bytes:
    """Prefix-frame the tagged tree injectively for seals and stable digests."""
    if isinstance(value, FrozenAbsent):
        return b"X"
    if isinstance(value, FrozenNull):
        return b"N"
    if isinstance(value, FrozenBool):
        return b"B" + (b"\x01" if value.value else b"\x00")
    if isinstance(value, FrozenInt):
        return b"I" + _frame(str(value.value).encode("ascii"))
    if isinstance(value, FrozenFloat):
        raw = bytes.fromhex(value.binary64_hex)
        if len(raw) != 8:
            raise FrozenJSONDomainError(
                "unsupported-selected-json-node", "float seal is not binary64"
            )
        return b"F" + _FLOAT_CLASS_TAG[value.source_class] + raw
    if isinstance(value, FrozenString):
        return b"S" + _frame(value.value.encode("utf-8", errors="surrogatepass"))
    if isinstance(value, FrozenArray):
        payload = len(value.items).to_bytes(8, "big") + b"".join(
            _frame(encode_frozen_json(member)) for member in value.items
        )
        return b"A" + payload
    if isinstance(value, FrozenObject):
        members: list[bytes] = [len(value.items).to_bytes(8, "big")]
        for key, member in value.items:
            members.append(_frame(key.encode("utf-8", errors="surrogatepass")))
            members.append(_frame(encode_frozen_json(member)))
        return b"O" + b"".join(members)
    raise FrozenJSONDomainError(
        "unsupported-selected-json-node", "value is not a FrozenJSON member"
    )


def frozen_json_digest(value: FrozenJSON) -> str:
    return hashlib.sha256(encode_frozen_json(value)).hexdigest()


def thaw_json(value: FrozenJSON) -> Any:
    """Return a fresh legacy-compatible mutable projection.

    ``FrozenObject`` may carry duplicate names.  A normal Python mapping cannot,
    so thaw follows the legacy parser's last-occurrence semantics while
    retaining the full lexical object in the immutable source value.
    """
    if isinstance(value, FrozenAbsent):
        raise FrozenJSONDomainError(
            "unsupported-selected-json-node", "absent has no JSON value"
        )
    if isinstance(value, FrozenNull):
        return None
    if isinstance(value, FrozenBool):
        return value.value
    if isinstance(value, FrozenInt):
        return value.value
    if isinstance(value, FrozenFloat):
        return struct.unpack(">d", bytes.fromhex(value.binary64_hex))[0]
    if isinstance(value, FrozenString):
        return value.value
    if isinstance(value, FrozenArray):
        return [thaw_json(member) for member in value.items]
    if isinstance(value, FrozenObject):
        out: dict[str, Any] = {}
        for key, member in value.items:
            out[key] = thaw_json(member)
        return out
    raise FrozenJSONDomainError(
        "unsupported-selected-json-node", "value is not a FrozenJSON member"
    )


def encode_frozen_json_document(
    value: FrozenJSON,
    *,
    allow_nan: bool,
) -> bytes:
    """Encode a frozen JSON tree with the selected-writer two-space layout.

    Unlike ``json.dumps(thaw_json(value))``, this projection retains every
    ``FrozenObject`` pair, including nested duplicate names and their order.
    Scalar rendering delegates to the standard encoder so finite binary64
    values, signed zero, and the legacy nonfinite spellings keep the historical
    selected-writer representation.
    """

    def scalar(node: FrozenJSON) -> str:
        if isinstance(node, FrozenAbsent):
            raise FrozenJSONDomainError(
                "unsupported-selected-json-node",
                "absent has no JSON document representation",
            )
        if isinstance(node, FrozenNull):
            projected: Any = None
        elif isinstance(node, FrozenBool):
            projected = node.value
        elif isinstance(node, FrozenInt):
            projected = node.value
        elif isinstance(node, FrozenFloat):
            projected = struct.unpack(">d", bytes.fromhex(node.binary64_hex))[0]
        elif isinstance(node, FrozenString):
            projected = node.value
        else:
            raise FrozenJSONDomainError(
                "unsupported-selected-json-node",
                "compound JSON value reached scalar projection",
            )
        return json.dumps(
            projected,
            ensure_ascii=False,
            allow_nan=allow_nan,
        )

    def render(node: FrozenJSON, depth: int) -> str:
        if isinstance(node, FrozenArray):
            if not node.items:
                return "[]"
            member_indent = " " * (2 * (depth + 1))
            close_indent = " " * (2 * depth)
            members = [
                member_indent + render(member, depth + 1) for member in node.items
            ]
            return "[\n" + ",\n".join(members) + "\n" + close_indent + "]"
        if isinstance(node, FrozenObject):
            if not node.items:
                return "{}"
            member_indent = " " * (2 * (depth + 1))
            close_indent = " " * (2 * depth)
            members = [
                member_indent
                + json.dumps(key, ensure_ascii=False)
                + ": "
                + render(member, depth + 1)
                for key, member in node.items
            ]
            return "{\n" + ",\n".join(members) + "\n" + close_indent + "}"
        return scalar(node)

    return render(value, 0).encode("utf-8")


@dataclass(frozen=True)
class _LexInt:
    token: str


@dataclass(frozen=True)
class _LexFloat:
    token: str


@dataclass(frozen=True)
class _LexConstant:
    token: str


@dataclass(frozen=True)
class _LexObject:
    items: tuple[tuple[str, Any], ...]


def _freeze_lexical(node: Any) -> FrozenJSON:
    if isinstance(node, _LexInt):
        return FrozenInt(int(node.token, 10))
    if isinstance(node, (_LexFloat, _LexConstant)):
        return _freeze_float(float(node.token))
    if isinstance(node, _LexObject):
        return FrozenObject(
            tuple((key, _freeze_lexical(value)) for key, value in node.items)
        )
    if type(node) is list:
        return FrozenArray(tuple(_freeze_lexical(value) for value in node))
    return freeze_json(node)


def _lexical_json_loads(text: str) -> FrozenJSON:
    value = json.loads(
        text,
        object_pairs_hook=lambda pairs: _LexObject(tuple(pairs)),
        parse_int=_LexInt,
        parse_float=_LexFloat,
        parse_constant=_LexConstant,
    )
    return _freeze_lexical(value)


def _has_duplicate_name(value: FrozenJSON) -> bool:
    if isinstance(value, FrozenObject):
        seen: set[str] = set()
        for key, member in value.items:
            if key in seen or _has_duplicate_name(member):
                return True
            seen.add(key)
    elif isinstance(value, FrozenArray):
        return any(_has_duplicate_name(member) for member in value.items)
    return False


def _has_nonfinite(value: FrozenJSON) -> bool:
    if isinstance(value, FrozenFloat):
        return value.source_class != "finite"
    if isinstance(value, FrozenObject):
        return any(_has_nonfinite(member) for _key, member in value.items)
    if isinstance(value, FrozenArray):
        return any(_has_nonfinite(member) for member in value.items)
    return False


@dataclass(frozen=True)
class StrictInputStatus:
    kind: Literal["valid", "invalid"]
    detail_code: Literal["sibling-json-duplicate-key", "sibling-json-nonfinite"] | None


@dataclass(frozen=True)
class RawJSONCarrier:
    present: bool
    value: FrozenJSON | None


_CARRIER_KEYS = (
    "vad_speech",
    "shot_changes",
    "sing_spans",
    "speaker_turns",
    "voiceprint_capture",
    "voiceprint_media",
    "segmentation",
)


def _last_occurrence(root: FrozenJSON, key: str) -> RawJSONCarrier:
    if not isinstance(root, FrozenObject):
        return RawJSONCarrier(False, None)
    for name, value in reversed(root.items):
        if name == key:
            return RawJSONCarrier(True, value)
    return RawJSONCarrier(False, None)


@dataclass(frozen=True)
class SiblingJSONSnapshot:
    name: str
    present: bool
    size: int | None
    sha256: str | None
    legacy_semantic: FrozenObject
    lexical: FrozenJSON
    strict_input_status: StrictInputStatus
    carriers: tuple[tuple[str, RawJSONCarrier], ...]
    digest: str

    def thaw_legacy(self) -> dict[str, Any]:
        value = thaw_json(self.legacy_semantic)
        assert isinstance(value, dict)
        return value

    def carrier(self, name: str) -> RawJSONCarrier:
        for key, carrier in self.carriers:
            if key == name:
                return carrier
        return RawJSONCarrier(False, None)


def _sibling_digest(
    *,
    present: bool,
    size: int | None,
    sha256: str | None,
    semantic: FrozenObject,
    lexical: FrozenJSON,
) -> str:
    return frozen_json_digest(
        FrozenArray(
            (
                FrozenString("sibling-json-snapshot"),
                FrozenBool(present),
                FROZEN_NULL if size is None else FrozenInt(size),
                FROZEN_NULL if sha256 is None else FrozenString(sha256),
                semantic,
                lexical,
            )
        )
    )


def decode_sibling_json_snapshot(name: str, raw: bytes | None) -> SiblingJSONSnapshot:
    """Build tolerant semantic and strict lexical projections from exact J0."""
    if raw is None:
        semantic = FrozenObject(())
        lexical: FrozenJSON = FROZEN_ABSENT
        return SiblingJSONSnapshot(
            name=name,
            present=False,
            size=None,
            sha256=None,
            legacy_semantic=semantic,
            lexical=lexical,
            strict_input_status=StrictInputStatus("valid", None),
            carriers=tuple((key, RawJSONCarrier(False, None)) for key in _CARRIER_KEYS),
            digest=_sibling_digest(
                present=False,
                size=None,
                sha256=None,
                semantic=semantic,
                lexical=lexical,
            ),
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"{name} is corrupt JSON (invalid UTF-8);"
            " re-run transcribe/process to regenerate it"
        ) from exc
    try:
        semantic_value = json.loads(text)
        lexical = _lexical_json_loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{name} is corrupt JSON ({exc.msg} at line {exc.lineno});"
            " re-run transcribe/process to regenerate it"
        ) from exc
    if not isinstance(semantic_value, dict) or not isinstance(lexical, FrozenObject):
        raise RuntimeError(
            f"{name}: expected a JSON object, got {type(semantic_value).__name__};"
            " re-run transcribe/process to regenerate it"
        )
    semantic_frozen = freeze_json(semantic_value)
    assert isinstance(semantic_frozen, FrozenObject)
    if _has_duplicate_name(lexical):
        strict_status = StrictInputStatus("invalid", "sibling-json-duplicate-key")
    elif _has_nonfinite(lexical):
        strict_status = StrictInputStatus("invalid", "sibling-json-nonfinite")
    else:
        strict_status = StrictInputStatus("valid", None)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    return SiblingJSONSnapshot(
        name=name,
        present=True,
        size=len(raw),
        sha256=raw_sha256,
        legacy_semantic=semantic_frozen,
        lexical=lexical,
        strict_input_status=strict_status,
        carriers=tuple((key, _last_occurrence(lexical, key)) for key in _CARRIER_KEYS),
        digest=_sibling_digest(
            present=True,
            size=len(raw),
            sha256=raw_sha256,
            semantic=semantic_frozen,
            lexical=lexical,
        ),
    )


@dataclass(frozen=True)
class ParsedVTTBlock:
    text: str
    start: float | None
    end: float | None
    lyric: bool
    speaker: str | None
    speakers: tuple[tuple[str | None, str], ...] | None


@dataclass(frozen=True)
class SubtitleSnapshot:
    name: str
    size: int
    sha256: str
    decoded_text: str
    blocks: tuple[ParsedVTTBlock, ...]


def decode_subtitle_snapshot(name: str, raw: bytes) -> SubtitleSnapshot:
    """Decode exact V0 and parse VTT directly without timestamp sorting."""
    text = decode_subtitle_bytes(raw, Path(name).name)
    sniffed = sniff_format(text)
    if sniffed == "ass":
        suffix = Path(name).suffix.lower().lstrip(".") or "no extension"
        raise RuntimeError(
            f"{Path(name).name}: content is ASS/SSA but the extension says {suffix};"
            " rename the file to its real format"
        )
    raw_blocks = realign.parse_vtt_blocks(text)
    if not raw_blocks:
        raise RuntimeError(f"no cues in {Path(name).name}")
    blocks: list[ParsedVTTBlock] = []
    for block in raw_blocks:
        raw_speakers = block.get("speakers")
        speakers = (
            tuple((name, line) for name, line in raw_speakers)
            if isinstance(raw_speakers, list)
            else None
        )
        speaker = block.get("speaker")
        blocks.append(
            ParsedVTTBlock(
                text=block["text"],
                start=block.get("start"),
                end=block.get("end"),
                lyric=bool(block.get("lyric")),
                speaker=speaker if isinstance(speaker, str) else None,
                speakers=speakers,
            )
        )
    return SubtitleSnapshot(
        name=Path(name).name,
        size=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        decoded_text=text,
        blocks=tuple(blocks),
    )


@dataclass(frozen=True)
class AlignBlockContent:
    source_index: int
    text: str
    lyric: bool
    speaker: str | None
    speakers: tuple[tuple[str | None, str], ...] | None
    alignment_text: str
    content_sha256: str


@dataclass(frozen=True)
class RouteBound:
    source_index: int
    start: float | None
    end: float | None


def _optional_string(value: str | None) -> FrozenJSON:
    return FROZEN_NULL if value is None else FrozenString(value)


def _speaker_lines(
    value: tuple[tuple[str | None, str], ...] | None,
) -> FrozenJSON:
    if value is None:
        return FROZEN_NULL
    return FrozenArray(
        tuple(
            FrozenArray((_optional_string(name), FrozenString(line)))
            for name, line in value
        )
    )


def _block_frozen(
    *,
    source_index: int,
    text: str,
    lyric: bool,
    speaker: str | None,
    speakers: tuple[tuple[str | None, str], ...] | None,
    alignment_text: str,
) -> FrozenArray:
    return FrozenArray(
        (
            FrozenInt(source_index),
            FrozenString(text),
            FrozenBool(lyric),
            _optional_string(speaker),
            _speaker_lines(speakers),
            FrozenString(alignment_text),
        )
    )


@dataclass(frozen=True)
class AlignSnapshot:
    subtitle: SubtitleSnapshot
    sibling: SiblingJSONSnapshot
    blocks: tuple[AlignBlockContent, ...]
    route_bounds: tuple[RouteBound, ...]
    qwen_delivery_order: tuple[int, ...]
    block_content_sha256: str


def decode_align_snapshot(
    vtt_name: str,
    vtt_bytes: bytes,
    sibling_json_bytes: bytes | None,
    *,
    effective_iso: str,
    sibling_snapshot: SiblingJSONSnapshot | None = None,
) -> AlignSnapshot:
    """Build immutable content and disjoint route views from exact V0/J0."""
    subtitle = decode_subtitle_snapshot(vtt_name, vtt_bytes)
    sibling_name = f"{Path(vtt_name).stem}.json"
    if sibling_snapshot is None:
        sibling = decode_sibling_json_snapshot(sibling_name, sibling_json_bytes)
    else:
        expected_sha256 = (
            None
            if sibling_json_bytes is None
            else hashlib.sha256(sibling_json_bytes).hexdigest()
        )
        if (
            sibling_snapshot.name != sibling_name
            or sibling_snapshot.present != (sibling_json_bytes is not None)
            or sibling_snapshot.size
            != (None if sibling_json_bytes is None else len(sibling_json_bytes))
            or sibling_snapshot.sha256 != expected_sha256
        ):
            raise ValueError("predecoded sibling snapshot does not match exact J0")
        sibling = sibling_snapshot
    separator = "" if effective_iso in realign.NO_SPACE_LANGS else " "
    contents: list[AlignBlockContent] = []
    block_values: list[FrozenJSON] = []
    bounds: list[RouteBound] = []
    for source_index, block in enumerate(subtitle.blocks):
        alignment_text = separator.join(block.text.split("\n")).strip()
        frozen = _block_frozen(
            source_index=source_index,
            text=block.text,
            lyric=block.lyric,
            speaker=block.speaker,
            speakers=block.speakers,
            alignment_text=alignment_text,
        )
        block_values.append(frozen)
        contents.append(
            AlignBlockContent(
                source_index=source_index,
                text=block.text,
                lyric=block.lyric,
                speaker=block.speaker,
                speakers=block.speakers,
                alignment_text=alignment_text,
                content_sha256=frozen_json_digest(frozen),
            )
        )
        start, end = block.start, block.end
        if start is not None and end is not None and start > end:
            start, end = end, start
        bounds.append(RouteBound(source_index, start, end))

    if bounds and all(bound.start is not None for bound in bounds):
        qwen_order = tuple(
            bound.source_index
            for bound in sorted(
                bounds,
                key=lambda bound: (
                    bound.start,
                    bound.end if bound.end is not None else bound.start,
                ),
            )
        )
    else:
        qwen_order = tuple(range(len(bounds)))
    return AlignSnapshot(
        subtitle=subtitle,
        sibling=sibling,
        blocks=tuple(contents),
        route_bounds=tuple(bounds),
        qwen_delivery_order=qwen_order,
        block_content_sha256=frozen_json_digest(FrozenArray(tuple(block_values))),
    )
