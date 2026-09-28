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
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### ReDimNet2 — `voxweave/vendor/redimnet2/`

The speaker-embedding network behind the default voiceprint embedder is a copy of
the model definition from [PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2)
at commit `c5bbe0b76e37df698c403f8844e41304ceab6307` (imports made
package-relative; nothing else changed). The upstream MIT license text is shipped
as `voxweave/vendor/redimnet2/LICENSE`. Most `layers/` files carry an MIT notice
of ID R&D, Inc.; `layers/poolings.py` is Apache License 2.0 (Shuai Wang,
wespeaker), keeps its upstream notice, and is not covered by the MIT license: its
license text is shipped as `voxweave/vendor/redimnet2/LICENSE.Apache-2.0`.

### SpeechBrain ECAPA-TDNN + Fbank — `voxweave/vendor/speechbrain_ecapa/`

The ECAPA-TDNN network and Fbank front end used by the Japanese voice-actor
embedder are copied from [speechbrain/speechbrain](https://github.com/speechbrain/speechbrain)
tag `v1.0.3` (commit `31c1e329048c0380dc7f2acbe680c44a036b6286`), Apache
License 2.0; the license text is shipped as
`voxweave/vendor/speechbrain_ecapa/LICENSE` and each module header lists the
upstream files and the modifications.

## Models downloaded at runtime (NOT bundled in this repository)

voxweave orchestrates the following models; users download the weights themselves.
Each is governed by its own license — verify before commercial use.

| Model                                                         | Used for                 | License (verify upstream)                                       |
| ------------------------------------------------------------- | ------------------------ | --------------------------------------------------------------- |
| Kim Mel-Band RoFormer vocals (`KimberleyJSN/melbandroformer`) | vocal separation         | MIT (author granted on the HF repo / GitHub issue #18, 2026-04) |
| Qwen3-ASR (`Qwen/Qwen3-ASR-*`)                                | ASR                      | Apache-2.0 (model card)                                         |
| Qwen3-ForcedAligner (`Qwen/Qwen3-ForcedAligner-0.6B`)         | zh/yue alignment         | Apache-2.0 (model card)                                         |
| MMS-300m forced aligner ONNX (`deskpai/ctc_forced_aligner`)   | ja alignment (default)   | CC-BY-NC-4.0 (model card) - non-commercial                      |
| wav2vec2 LV60K-self (`facebook/wav2vec2-large-960h-lv60-self`) | en CTC alignment         | Apache-2.0 (model card)                                         |
| PANNs Cnn14 (`thelou1s/panns-inference`)                      | song/music detection     | Apache-2.0 (model card)                                         |
| silero-vad                                                    | voice activity detection | MIT                                                             |
| pyannote speaker-diarization-community-1 (`pyannote/speaker-diarization-community-1`) | speaker diarization (`--diarize`, default pipeline) | CC-BY-4.0 - attribution required (gated; accept the model-card conditions on Hugging Face) |
| pyannote speaker-diarization-3.1 (`pyannote/speaker-diarization-3.1`)                 | speaker diarization (`--diarize --diarize-model 3.1`, opt-in legacy) | MIT (gated separately; also requires accepting `pyannote/segmentation-3.0`) |
| ReDimNet2-B6 `b6-vb2+vox2+cnc2_v0-lm.pt` (PalabraAI/redimnet2 release v1.0.0)         | voiceprints (`--voiceprints`, default embedder for every language except Japanese) | code MIT; weights trained on VoxBlink2 -> treat as CC BY-NC-SA 4.0 (non-commercial) |
| Anime voice-actor ECAPA-TDNN (`litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm`) | voiceprints for Japanese (`auto` routing) | model card says MIT; training-data provenance unclear (visual-novel corpus) |

Downstream model licenses are the responsibility of the deployer. For
commercial deployment, confirm each model card's terms (some carry usage
conditions, e.g. large-MAU clauses).
