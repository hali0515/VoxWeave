# How it works

## The edit-and-resync workflow

```
voxweave episode.mkv          # 1. transcribe  -> episode.vtt + episode.json
  └─ (optional) correct       # 2. LLM ASR fix -> episode.asrfix.vtt (review it; or --apply to rewrite + re-align)
edit episode.vtt by hand      # 3. fix wording / line breaks
voxweave align episode.vtt    # 4. re-derive timestamps from audio (overwrites VTT + JSON)
voxweave translate episode.vtt --target zh   # 5. context-aware translation
voxweave pack episode.zh.vtt             # 6. soft-mux into the media (or burn for hardsubs)
```

Timestamps are **always** derived from the audio by the forced aligner — you never hand-edit
them. Edit the text freely; `align` puts the timing back.

## How it works

| Stage           | What runs                                                                                                                                                                                                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Separation**  | Mel-Band Roformer (full-band 44.1k stereo, vendored pure-torch) isolates vocals; downsampled to 16k afterwards.                                                                                                                                                           |
| **Song-skip**   | PANNs (route ii) flags singing/music on the separated vocals before ASR; songs are excised mid-segment with cuts snapped into real silences, and PANNs clean-speech evidence rescues dialogue the waveform VAD under-scores.                                              |
| **Chunking**    | Silero VAD splits speech into ≤120s chunks (longer risks ASR repetition-loop collapse).                                                                                                                                                                                   |
| **ASR + align** | Qwen3-ASR (default, text + units in one pass) / Whisper hybrid (faster-whisper on cuda, mlx-whisper on mps) / dual-ASR fusion — the pipeline is engine-agnostic.                                                                                                          |
| **Alignment**   | `ja` → MMS-300m + uroman, `en` → wav2vec2-LV60K CTC (both full-file single pass, WhisperX-gold); `zh`·`yue` → Qwen. During transcribe the pass is cropped to the transcribed envelope and excised songs are muted, so untranscribed music can never host stretched words. |
| **Layout**      | gap-aware `smart_split`: word-level gaps + BudouX phrase atoms + line-length, on a shared timeline forked per language.                                                                                                                                                   |
