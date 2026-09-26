# calibration

Tracked data contracts and golden inputs for the two voxweave quality rulers. The unit
suite answers "did behaviour change unintentionally"; this directory answers "did
acoustic boundaries get more accurate" and "did subtitle segmentation get better".

```text
calibration/
  schemas/       JSON Schema (draft 2020-12) contracts, tracked and stable
  alignment/     manifest example and synthetic reference fixtures (no tracked baseline)
  segmentation/  corpus registry, golden cases, recorded baseline
  align-shadow/  align-shadow corpus, manifest and baseline for scripts/calib_align_shadow.py
  p6-oracle/     detached oracle corpus for scripts/p6_oracle.py (see its README)
```

Shared helpers live in `scripts/calib_common.py`: schema validation, the single type-7
percentile definition, the canonical JSON digest, language-tag canonicalization, micro
aggregation and the exit codes. It stays importable in a bare environment whose only
third-party package is `jsonschema` — no torch, no model code.

## Exit codes

Every calibration CLI uses the same contract:

| code | meaning |
|---:|---|
| `0` | data valid, all enabled gates passed |
| `1` | data valid, a quality gate failed |
| `2` | manifest / schema / coverage / reference / tooling invalid — this run has no standing to judge quality |

A broken corpus must never report as a quality regression, so exit 2 is never downgraded
to exit 1.

## The two truth lanes

The alignment ruler never pools its ground-truth sources. Each lane is keyed by
`(source_kind, language)` and answers a different question; inside a lane every item keeps
its `reference_id` and its own metrics, so the pooled lane numbers stay traceable.

| lane | ground truth | answers | primary metrics |
|---|---|---|---|
| `mfa_words` / `manual_words` | same-language word boundaries from MFA 3.0 or human annotation | is the acoustic alignment accurate | word start/end MAE, median, p90, threshold hit rates |
| `commercial_cues` / `manual_cues` | same-language release or human cue boundaries | how close is the final subtitle to a shipped track | cue start/end median, p90, `%<=0.25s`, `%<=1s` |

Rules that are not negotiable:

- MFA is not a segmentation-style reference, and a release subtitle is not word-level
  acoustic truth. Mixing their samples into one percentile is meaningless.
- No cross-language pairing. An English release track paired against a Japanese lane is
  `reference_language_mismatch` — exit 2 for that item, not a degraded mode.
- For `ja`, `mfa_words` is a first-class truth source, not a fallback, because a
  same-language commercial track often does not exist.
- Matching hypothesis to reference is text-driven. Timestamps are what is under test and
  must never be used to pair units.

`scripts/calib_alignment.py` has four subcommands (`--help` lists their flags):

```bash
uv run python scripts/calib_alignment.py inspect-tracks MEDIA --lang ja [--json]
uv run python scripts/calib_alignment.py report --manifest M [--json-out P]
uv run python scripts/calib_alignment.py check --manifest M --baseline B
uv run python scripts/calib_alignment.py record-baseline --manifest M --report R --output O
```

`report` writes `build/calibration/alignment-report.json` unless `--json-out` says
otherwise. `--source` / `--item` narrow a `report` for exploration; such a report records
its filters, and `check` and `record-baseline` refuse it, because a baseline gates the
whole manifest. `calibration/alignment/manifest.example.json` is the manifest shape to
copy; no alignment baseline is tracked yet, so `--output` names wherever you record one.

The segmentation ruler is a separate, zero-GPU lane. It stores no expected subtitle text:
each case in `segmentation/cases/` is a real captured `word_segments` stream plus the
production inputs (`vad_speech`, `shot_changes`, `sing_spans`, `speaker_turns`), replayed
through the same production entry point the pipeline uses, then reduced to four gated
metrics — `len_break_mid_phrase_rate`, `over_7s_rate`, `cps_p90`, `forbidden_end_rate`.
All percentages are micro-aggregated: numerators and denominators are summed across
cases and kept in the report, never averaged per case, so a 60 s clip cannot outweigh a
150 s one.

Case data is tracked in Git, so it should be redistributable (self-recorded, CC,
public-domain, or consented). `third-party` is the explicit maintainer escape hatch:
the case is honestly marked `redistributable: false`, and tracking it is the repo
owner's decision, never the tool's default. The current `third-party` cases (all of
`zh-*` and `ja-*`) are tracked under that explicit owner decision, recorded 2026-08.
Timestamps are rebased to 0, speakers are `S0/S1/...`, and
no audio, video, source filename or real speaker name is stored. Private extension
corpora are supplied through `VOXWEAVE_CALIB_ROOT` and are reported separately — they
never change the denominator of the public PR gate.

## Baselines

`segmentation/baseline.json` (and any alignment baseline recorded with
`calib_alignment.py record-baseline`) is a recorded reference point, not a target invented
by hand. Gates are one-sided in the direction that means "worse" (errors may not rise, hit
rates and coverage may not fall), so an improvement can never fail. For an error metric:

```python
allowed = baseline_value + max(absolute_tolerance, baseline_value * relative_tolerance)
passed = current <= allowed and (absolute_max is None or current <= absolute_max)
```

Updating a baseline is a reviewed, human action:

```bash
uv run python scripts/calib_segmentation.py record-baseline \
  --corpus calibration/segmentation/corpus.json \
  --report build/calibration/segmentation-report.json \
  --output calibration/segmentation/baseline.json
```

- `record-baseline` is never run by CI and is not part of any default `make` target.
- It refuses to run unless the report is valid and the corpus digest matches, so a
  regression cannot be laundered into a new baseline by rerunning the harness.
- A mismatch in corpus digest, `metric_definition_version` or recorded dependency
  versions is exit 2: re-record deliberately and review the diff, do not paper over it.
- Never grandfather a currently bad value into the absolute target. If head misses the
  absolute goal, block relative regressions first, keep the absolute gate at `warning`,
  and reach the goal in its own PR.

Ordinary GitHub Actions runs only the zero-GPU segmentation replay and the synthetic
harness tests. Real media and MFA models belong to a self-hosted or manually dispatched
quality workflow: a public runner should never download private media or multi-GB models.
