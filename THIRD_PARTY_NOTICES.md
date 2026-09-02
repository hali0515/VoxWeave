# Third-Party Notices

voxweave is licensed under the MIT License (see `LICENSE`). It incorporates and
builds upon the third-party work listed below. Each retains its own license;
the relevant notices are reproduced or referenced here.

## Vendored source code (included in this repository)

### Mel-Band RoFormer — `voxweave/vendor/`

`voxweave/vendor/mel_band_roformer.py` and `voxweave/vendor/attend.py` are a frozen
copy of the Mel-Band RoFormer implementation from
[lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)
(by way of the `audio-separator` / `uvr_lib_v5` copy). Licensed under MIT:

```
MIT License

Copyright (c) 2023 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction... (full MIT terms; see the upstream
repository for the complete text).
```

### Subtitle splitting — `voxweave/core/smart_split.py`

The subtitle-splitting pipeline (`split_at_sentence_end`,
`split_long_cues_with_word_timings`, etc.) is adapted from
[dashed/whisperx-subtitles-replicate](https://github.com/dashed/whisperx-subtitles-replicate)
(`predict.py`), which is MIT-licensed. Modified to add CJK / no-space language
awareness and the voxweave-specific cue heuristics.

## Models downloaded at runtime (NOT bundled in this repository)

voxweave orchestrates the following models; users download the weights themselves.
Each is governed by its own license — verify before commercial use.

| Model                                                         | Used for                 | License (verify upstream)                                       |
| ------------------------------------------------------------- | ------------------------ | --------------------------------------------------------------- |
| Kim Mel-Band RoFormer vocals (`KimberleyJSN/melbandroformer`) | vocal separation         | MIT (author granted on the HF repo / GitHub issue #18, 2026-04) |
| Qwen3-ASR (`Qwen/Qwen3-ASR-*`)                                | ASR                      | Qwen license / Apache-2.0 — read the model card                 |
| wav2vec2-large-xlsr-53-japanese (`jonatasgrosman/...`)        | JA CTC alignment         | Apache-2.0 (verify)                                             |
| torchaudio WAV2VEC2_ASR_LARGE_LV60K_960H                      | EN CTC alignment         | as distributed by torchaudio                                    |
| PANNs Cnn14                                                   | song/music detection     | Apache/MIT (verify)                                             |
| silero-vad                                                    | voice activity detection | MIT                                                             |
| pyannote speaker-diarization-community-1 (`pyannote/speaker-diarization-community-1`) | speaker diarization (`--diarize`, default pipeline) | CC-BY-4.0 - attribution required (gated; accept the model-card conditions on Hugging Face) |
| pyannote speaker-diarization-3.1 (`pyannote/speaker-diarization-3.1`)                 | speaker diarization (`--diarize --diarize-model 3.1`, opt-in legacy) | MIT (gated separately; also requires accepting `pyannote/segmentation-3.0`) |

Downstream model licenses are the responsibility of the deployer. For
commercial deployment, confirm each model card's terms (some carry usage
conditions, e.g. large-MAU clauses).
