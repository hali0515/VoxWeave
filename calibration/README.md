# calibration

Tracked data contracts and golden inputs for the two voxweave quality rulers. The unit
suite answers "did behaviour change unintentionally"; this directory answers "did
acoustic boundaries get more accurate" and "did subtitle segmentation get better".

```text
calibration/
  schemas/       JSON Schema (draft 2020-12) contracts, tracked and stable
  alignment/     manifest example, reference fixtures, recorded baseline
  segmentation/  corpus registry, golden cases, recorded baseline
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

The alignment ruler never pools its ground-truth sources. Each lane is kept separate by
`(source_kind, language, reference_id)` and answers a different question.

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
owner's decision, never the tool's default. Timestamps are rebased to 0, speakers are `S0/S1/...`, and
no audio, video, source filename or real speaker name is stored. Private extension
corpora are supplied through `VOXWEAVE_CALIB_ROOT` and are reported separately — they
never change the denominator of the public PR gate.

## Baselines

`alignment/baseline.json` and `segmentation/baseline.json` are recorded reference points,
not targets invented by hand. Gates are one-sided (lower is better), so an improvement can
never fail:

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
