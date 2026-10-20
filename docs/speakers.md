# Speakers

## Speaker models

Speaker diarization support ships by default on both variants but remains opt-in at runtime.
The default model is `pyannote/speaker-diarization-community-1`, which separates and counts
multiple speakers noticeably better than 3.1. It is gated separately on Hugging Face: before
using `--diarize`, accept the conditions for the model you select, then set `VOXWEAVE_HF_TOKEN` /
`HF_TOKEN` or authenticate once with `hf auth login` (a `uv tool` install does not put `hf` on
PATH; run it as `uvx --from huggingface_hub hf auth login`). Users who have accepted only the 3.1
gate can keep it with `--diarize-model 3.1`.

Measured with pyannote.audio 4.0.7 on the same Japanese 16 kHz test waveform and an RTX PRO
4000 24 GB. Peak allocated CUDA memory was identical in that run:

| Short name              | Resolved pipeline                          | Model license | Peak allocated VRAM |
| ----------------------- | ------------------------------------------ | ------------- | ------------------- |
| `community-1` (default) | `pyannote/speaker-diarization-community-1` | CC-BY-4.0     | 2,723.963 MiB       |
| `3.1`                   | `pyannote/speaker-diarization-3.1`         | MIT           | 2,723.963 MiB       |

Community-1 generally improves speaker counting and separation; 3.1 stays selectable per run
for users who have accepted only the existing 3.1 gate.

**Voiceprint embedders.** With `--voiceprints`, pyannote still finds the speaker turns, but the
per-speaker voiceprints (cross-episode matching and **Split this speaker**) come from a
dedicated speaker-embedding model chosen per language. Each checkpoint is downloaded once into
`~/.cache/voxweave/audio/`, pinned by size and SHA-256, and verified at the start of a
`--voiceprints` run, before any audio work (with `auto` and no `--language`, both checkpoints). If
that fails, the run warns and continues without voiceprints:

| `--voiceprint-model`                                  | Checkpoint                                                                                                                                                          | Languages     | Dim | Weights licence / caveats                                                                                                                            |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | --- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `redimnet2` (default for every language but Japanese) | ReDimNet2-B6 `b6-vb2+vox2+cnc2_v0-lm.pt` ([PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2) release v1.0.0)                                             | any           | 192 | Code MIT; the weights are trained on VoxBlink2, so treat them as **CC BY-NC-SA 4.0 (non-commercial)**.                                               |
| `anime-va` (default for Japanese)                     | [`litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm`](https://huggingface.co/litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm) (pinned revision) | Japanese only | 192 | Model card says MIT; the training data (a visual-novel voice corpus) has **unclear provenance**. Trained to tell voice actors apart, not characters. |
| `pyannote` (legacy)                                   | the diarization pipeline's own embedding head                                                                                                                       | any           | 256 | Keeps matching voice stores built before the dedicated embedders.                                                                                    |

`auto` (the default) picks `anime-va` for Japanese episodes and `redimnet2` otherwise. Voice
stores are per embedding space: a store built by one embedder never matches another. Because the
dedicated embedders are independent of the diarizer, switching `--diarize-model` no longer
orphans a store built by `redimnet2` or `anime-va`.

**Voiceprint speaker clustering (opt-in).** By default, `--diarize` keeps pyannote's own answer
to _who is who_. `--speaker-clustering voiceprint` (or `VOXWEAVE_DIARIZE_CLUSTERING=voiceprint`,
or `[diarize].clustering = "voiceprint"` in the config) regroups pyannote's turns by voice
instead. It uses the `redimnet2` checkpoint from the table above whatever the language, so the
same **CC BY-NC-SA 4.0 (non-commercial)** weights licence applies.

How it works:

- pyannote still finds the turns, including overlaps. Each turn keeps only the speech nobody
  else talks over; turns with at least 1 s of that speech are clustered by voice. A cluster needs
  6 s of it to count as a speaker.
- Every other turn goes to the closest voice nearby. If no voice is close enough (background
  voices, noise, fragments too short to tell), the turn is dropped from `speaker_turns` and the
  subtitle words it covered stay with the surrounding speaker.

Measured against pyannote's clustering:

- It confused speakers less on three of four public test sets (AliMeeting, VoxConverse,
  JVS-conv) and slightly more on AMI.
- It attributed 37 of 40 short backchannels (under 1 s) in an online meeting correctly, against 32.
- It can split one person into more speakers than pyannote does, and a speaker who talks very
  little may be merged into others or dropped.

Speaker-count bounds and fallbacks:

- `--min-speakers` and `--max-speakers` apply to this stage too. A `--min-speakers` above the
  number of voices it finds is met by dividing voices, because someone with a few seconds of
  speech cannot form a speaker of their own. pyannote does the same and measured worse: given
  the true speaker count on the public recordings where the stage finds too few, the stage's DER
  rose from 19% to 23% and pyannote's from 19% to 30%. Set it only when a missing speaker
  matters more than a split one.
- If the stage fails (a download, out of memory), cannot meet the bounds, or finds no turn long
  enough to anchor a voiceprint, the run warns and keeps pyannote's speakers. With
  `--voiceprint-model pyannote` the run also keeps pyannote's clustering, because those legacy
  voiceprints are keyed by pyannote's labels.

If you change the setting (or `--diarize-model`) on an episode whose speakers you already named,
the speakers are renumbered while the saved names stay keyed by speaker id. A transcription run
warns when a named id's turns changed. Review the names with `voxweave speakers <media>` before
`voxweave speakers enroll` stores a voice under a wrong name. The setting never changes which
voice stores an episode's voiceprints match.

## Label speakers

After a diarized transcription, `voxweave speakers serve <media>` prepares an audition page in
memory, serves it on `127.0.0.1` by default, and opens it in your browser. The page embeds up to three
clean, non-overlapping speech clips per diarizer id. Listen, enter names, then select **Save**
to write them directly to the episode's speaker mapping. Each Save then re-renders the
episode's VTT from `<stem>.json` so it shows the new names (`--no-render` turns this off).
The VTT is replaced only when it differs from that render in speaker names alone: if it holds
cue text or timing the JSON does not, such as hand edits not yet aligned, it is left unchanged
and the page says why. The server runs until Ctrl+C; use
`--no-open` to print the access link without opening a browser, or `--port N` to choose its HTTP
port. No audition HTML is written to disk. `voxweave speakers <media>` is the supported
shorthand for `serve`; `--manual` disables voice matching for that session.

To access the audition from another device, use
`voxweave speakers serve episode.mkv --host 0.0.0.0 --port 8765 --no-open`, then open the
printed `http://0.0.0.0:8765/?k=...` link with `0.0.0.0` replaced by the server's IP
address. This binds all IPv4 interfaces.

For an ngrok tunnel on the same machine, run
`voxweave speakers episode.mkv --port 9999 --no-open --ngrok` and, in another terminal,
`ngrok http http://127.0.0.1:9999`. Open the tunnel's URL with the printed `/?k=...`
appended. No public domain needs to be supplied: VoxWeave reads the local agent API at
`127.0.0.1:4040` and refreshes the accepted URLs as tunnels change.
Only tunnels forwarding to this local port are accepted, including HTTPS access and saves.

> [!WARNING]
> Access is by link. The URL `speakers serve` prints and opens carries a random access key
> (`?k=...`, new on every start); opening it sets a session cookie, and every request without
> that cookie is refused, on `127.0.0.1` too. Anyone who has the link can play the episode audio,
> read and change speaker names and run splits, so share it only with people you trust and stop
> the server when you are done. With `--host 0.0.0.0` the connection is plain HTTP, so the link
> and the audio are not encrypted on your network.

```bash
voxweave episode.mkv --diarize
voxweave speakers serve episode.mkv
# Save names in the browser; each Save re-renders episode.vtt with them
```

If pyannote merged two people under one diarizer id, select **Split this speaker** on that
card. VoxWeave clusters the id's individual turns and lets you audition both proposed groups
before applying the split. This action requires the episode to have been captured with
`--voiceprints`; the turns are embedded with the same embedder (and checkpoint) the voiceprints
were captured with, and the proposal is refused if that embedder or the audio provenance cannot
be reproduced (for example, when a bound separated-vocals cache is missing or stale). A
confirmed split rewrites `speaker_turns` and the bound voiceprint centroids (recomputed with the
capture's centroid recipe) and asks you to restart `voxweave speakers serve` to audition and name
the new id. It also saves the files it rewrote as a one-level snapshot in the episode cache
(`speaker-split.undo.json`). Until you restart, the card offers **Undo this split**, which
restores the transcript JSON, the voiceprints and the speaker mapping from that snapshot. The
undo is refused, and the page says which file is the reason, when one of them changed after the
split (or the media did), or when there is no snapshot left; a refused undo restores nothing.
Either way, restart `voxweave speakers serve` to audition again. The page offers the undo only
in the session that applied the split; after a restart, take a split back by transcribing the
episode again with `--diarize --voiceprints`, which diarizes it from scratch.
`voxweave speakers purge` deletes the snapshot.

Voice matching across episodes is a separate, opt-in layer. Capture centroids with
`--diarize --voiceprints`, review the ordinary empty mapping, then enroll only those
human-entered names into your voice library:

```bash
voxweave episode.mkv --diarize --voiceprints
voxweave speakers serve episode.mkv
# Save reviewed names in the browser first
voxweave speakers enroll episode.mkv

# Later episodes, in this folder or any other: suggestions appear in the served page and a
# regenerable cache record.
voxweave speakers serve ../season-2/episode-01.mkv
```

### Voice library

`speakers enroll` saves voices into one library shared by every media folder, so a voice
enrolled in one season, show or folder of recordings can be suggested in another. The library
directory is chosen by, first match wins:

1. `--voices-dir DIR` (on `speakers serve`, `speakers enroll` and every `voices` command);
2. `VOXWEAVE_VOICES_DIR`;
3. `dir` under `[voices]` in `~/.config/voxweave.conf` (relative to that file's directory);
4. `$XDG_DATA_HOME/voxweave/voices`, else `~/.local/share/voxweave/voices` (every platform).

`voxweave voices where` prints the directory and which of these chose it. The library is plain
JSON, one file per concern, and never holds audio:

```
identities.json              names, aliases and scopes of every identity; space files; forgotten ids
spaces/<model>-<fp12>.json   the voice vectors of one embedding space
history.jsonl                log of changes (ids, scopes, episode labels, counts; never vectors or display names)
.library.lock                one lock for the whole library
```

Each identity has a random id; names are attributes, so two identities may both be called
"Alex". Names are shared by every embedding space (a rename applies everywhere), while vectors
are kept per space: an episode captured with `redimnet2` is only compared with `redimnet2`
vectors, and switching embedders starts a new space in the same library. A voice sample keeps a
pointer to its source (media path, media fingerprint, capture id, speaker label, episode) so a
future re-embedding can find the media again.

**Scopes and tiers.** Every enrollment is tagged with a scope: `--show NAME`, else the name of
the media's folder. When you review an episode, identities of its own scope are suggested as
before (tier 1). Identities of every other scope (the same voice actor in another work, the same
person in another folder of recordings) are suggested only above a stricter similarity bar
(tier 2: `VOXWEAVE_VOICES_GLOBAL_SUGGEST`), are never prefilled, and are labelled with the scopes
they come from. The bar is stricter because false suggestions grow with the library: if one
unrelated voice clears it with probability _p_, a library of _N_ unrelated voices produces at
least one false suggestion with probability 1 - (1 - _p_)^_N_. A generic folder name, one that
many shows share (a season, disc, part or year number such as `Season 1`, `S02`, `Disc 1` or
`2023`, or `Specials`), is qualified with its parent folder: episodes in `Frieren/Season 1` get
the scope `Frieren / Season 1`. Other folder names are used as they are, so two unrelated
folders with the same name (say `Recordings`) share a scope; pass `--show` to keep them apart.

A name you enter links to an existing identity only when you used that identity's suggestion
button, or when an identity of the same scope carries that name. Otherwise enrollment creates a
new identity, even if another scope has someone of the same name; if a name is ambiguous within
a scope, enrollment stops and asks you to rename one (`voxweave voices rename ID NAME`).

**Sharing on a NAS.** Several machines can point `[voices].dir` at the same directory on a NAS,
as the same user account. Writes take an exclusive `flock` on `.library.lock`; Linux NFS
clients emulate it with byte-range locks, which works on NFSv4 and on NFSv3 with the lock
manager, but a `nolock` mount keeps each lock local to its machine, so use a mount with working
locks. Every write also checks, just
before replacing each file, that it still holds what the write read, and otherwise stops
("re-run the command"). That check narrows the window but is not atomic: without working locks,
two simultaneous writers can both pass it and one update can be lost. Voice samples such a race
leaves behind for an identity that `identities.json` does not list are ignored by every reader
and kept on disk, as are those left when an older `identities.json` is restored; only the
samples of identities removed with `voices forget` are deleted by the next write.
`identities.json` lists every space file, so `voices forget` finds all of them
even when an NFS client's cached directory listing is a minute old, and stops (asking you to
re-run) if a listed file is not visible yet. Files are replaced by an atomic rename within the
directory. File-sync services are not a lock and can produce conflicting copies; use a real
network mount. A library belongs to one user account: every file in it (the lock too) is
created `0600`, and a directory VoxWeave creates is `0700`; a directory you created beforehand
keeps its own permissions. Sharing one library between different user accounts is not
supported: machines that share it must reach it as the same user (the same uid on NFS). Only
the built-in location gets missing parent directories created: for `--voices-dir`,
`VOXWEAVE_VOICES_DIR` or `[voices].dir` the parent must already exist, so an unmounted share (an
empty mount point) makes `speakers enroll` fail instead of starting a second library on the
local disk, and `speakers serve` warns that the library is missing.

**Privacy.** The library holds voice biometrics of the people you name; the first write into a
new library prints a notice saying so. `voxweave voices list` and `voxweave voices show ID|NAME`
inspect it, and deleting the directory removes everyone. `voxweave voices forget ID` removes one
person from every space and keeps only the id as a tombstone, so that person is never suggested,
imported or enrolled again, not even from a review page made before the forget or from a
per-folder store of an earlier version. It does not edit those per-folder stores: it lists the
ones that still hold the person (the stores you imported, and the ones next to their media), and
their vectors stay on disk until you delete those files. The history keeps the forgotten id's
entries without their scopes and episode labels. It never holds voice data or display names, but
other entries keep scopes (folder or show names), episode labels (media file names) and, for
imports, the imported file's path, all of which can name people too.

**Per-show stores from earlier versions.** A `voxweave.voices.json` next to the media is still
read for suggestions, never written, with a one-time hint to merge it with
`voxweave voices import PATH`. By default the imported voices keep the scopes under which the
store already served suggestions: its folder's scope and the store's show (a store with another
file name, only ever used with `--voices`, keeps its show). `--scope NAME` imports them under
that scope instead, and importing again with another `--scope` adds that scope. Import keeps the
store's ids, so importing it again adds no voice samples, skips samples older than the five an
identity already keeps, and leaves the file unchanged. `--voices FILE`
still selects a per-show store explicitly, with its previous behavior (a missing one is created
only with both `--voices` and `--show`); it cannot be combined with `--voices-dir`.

A space belongs to one embedding model (see the voiceprint embedders under [Speaker models](#speaker-models));
a per-show store captured with another embedder is skipped with a warning that names both
spaces, and enrollment into such a store is refused. A store built before the dedicated
embedders keeps working with `--voiceprint-model pyannote` (see [MIGRATING.md](https://github.com/hali0515/VoxWeave/blob/main/MIGRATING.md)).

Use `speakers enroll --replace` to replace this episode's prior contribution (the same episode
label within the same scope, or the same media). Enrolling an already enrolled capture again
after its folder was renamed, or with another `--show`, moves its voice samples to the new scope
without `--replace`; a new capture of the same media in the new scope needs `--replace`, and so
does enrolling again after a confirmed speaker split, which changes the voices of a capture that
keeps its id.
The shipped matching policy is **suggest-only**: `VOXWEAVE_VOICES_ACCEPT` defaults to `off`,
so stored names appear as review buttons and never become authoritative mapping values.
Even when an operator configures a finite accept threshold, a machine prefill exists only in
the served input; the v1 mapping is still created with empty names and must be reviewed and
saved by the operator.

The versioned mapping is intentionally small:

```json
{ "version": 1, "speakers": { "SPEAKER_00": "Aoi" } }
```

`render` renders mapped names as WebVTT voice tags while keeping transcript text and the
sibling JSON clean. Empty or missing names remain unlabeled. An existing mapping is the normal
edit flow: the server re-reads it for the page and replaces it only when you select **Save**.
When burning a mapped SRT, recovered speaker identity is retained in the temporary ASS `Name`
field while the visible dialogue remains prefix-free.

`voxweave speakers purge episode.mkv` removes that episode's voiceprint
artifact and suggestion record from both supported legacy and cache locations. It also removes
the cache-only one-level speaker-split undo snapshot and any legacy audition HTML while holding
the episode transaction lock. It works after the media has been removed and deliberately keeps
the human-edited mapping.

`voxweave speakers list episode.mkv` inspects that episode's speaker turns, reviewed
mapping, and voiceprint state without changing them. Add `--json` for machine-readable
output. It does not read the voice library; use `voxweave voices list` for that.
