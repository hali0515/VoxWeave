# 代码审计报告（2026-09-26）

分支 `claude/code-audit-quality-hmei1j`，基线 `94b6ff5`（feat(burn): cap the video bitrate at the source's）。

## 范围与方法

- **范围**：`voxweave/`（约 7.8 万行，不含 `vendor/`）、`scripts/` 与 `experiments/`（约 1.8 万行）、README / MIGRATING / calibration 文档、构建与 CI 配置、测试套件卫生。
- **维度**（按要求）：结构、质量、问题（bug）、健壮/稳定性、引导不合理（CLI 帮助、报错提示、文档与代码不符）、死代码、无效注释。
- **做法**：代码按子系统切成 24 个切片，每片由一个审计 agent 通读并给出带 `文件:行号` 与原文引用的发现；再由独立的复核 agent 以"先尝试驳倒"的方式逐条复核，高危 bug 额外做一次复现式复核。修复阶段，每个修复 agent 在动手前都会重新核对（能复现的先复现），核对不成立的一律跳过。
- **约束**：本仓库的 P6 oracle（`scripts/p6_oracle.py`）锁定了 `align_delta_registry.py`、`align_failures.py`、`engine_registry.py`、oracle schema 与 `uv.lock` 的 sha256，并检查若干源文件中的字面 token、执行清单里引用的测试；分段质量标尺（`calib_segmentation.py`）以已记录的 baseline 做门禁，仓库约定"重新记录 baseline 必须由人审核"。因此：
  - 上述被锁定的文件一个字节都没动；oracle 引用的测试没有删改；
  - 会改变分段输出（从而需要重新记录 baseline）的修复一律**没有做**，列入下文"需要决策"；
  - `scripts/p6_oracle.py` 本身（门禁工具，禁止出现 `write_text` 等 token）只报告、不修改。

## 结果概览

- 24 个切片共 **509 条发现**：高 23、中 147、低 339。
- 独立复核覆盖 397 条：确认 213、部分确认（细节/严重度修正）150、驳回 34；其余 112 条在截止时间前未排到复核（附录中标"未复核"）。**所有已落地的修复**都由修复 agent 在改动前对照代码重新核实过（能复现的先复现），不依赖复核队列。
- 修复落在本分支的 20 余个提交里（按子系统分组）；每组都跑了 ruff、pyright 和相关测试，涉及 P6/分段的组另外跑了 oracle 或标尺前后对比。

## 已修复（按用户影响排序）

### 数据丢失 / 输出错误（高）

| 问题 | 位置 | 提交 |
|---|---|---|
| `identities.json` 缺失或被旧版本覆盖时，下一次 enroll/import（任意空间）会静默删除声纹库里的声音样本；现在缺失时拒绝读取，回滚时孤儿样本保留在磁盘上直到恢复，只有被 forget 的身份才会删除 | `voicelibrary.py` | `90cdde1`、`d3259e8` |
| `align` 会删掉以 Note/Noted/Style/Region/WebVTT 开头的台词（如 "Noted, sir."），纯文本草稿尤甚 | `realign.py` | `1229596` |
| zh/yue（Qwen 路线）对齐：已对齐字幕之后出现一条无法路由的插入字幕，就会以晦涩的 evidence 错误整体失败 | `align_evidence.py` | `f8b6ca1` |
| 俄语/希腊语/带重音拉丁字母被按双宽计算，行宽预算减半、字幕数量翻倍 | `core/layout.py` | `8959143` |
| 人声缓存命中时悄悄改用分离人声做 VAD 时间参考，重跑结果与首次不同 | `pipeline.py` | `31eef6e` |
| burn：路径含 `:` 或 `'`（如 "Star Wars: A New Hope.ass"）时 ffmpeg 滤镜解析失败 | `mux.py` | `bc302d8` |
| pack/burn：GBK/Big5 等非 UTF-8 字幕原样交给 ffmpeg，中日文字幕丢失或失败 | `mux.py` | `bc302d8` |
| 重新分离说话人（如 `--max-speakers 2`）后，试听页因旧映射里残留的说话人 id 返回 500，保存/拆分永远失败 | `speakerserve.py` | `5656b72` |
| `--hybrid` 融合把数字里的小数点/千分位去掉（14.2 → 142），并在中日文里重复标点（。。） | `realign.py` | `1229596` |
| `correct --apply` 先覆盖 VTT 再自动对齐，媒体缺失时差异丢失；现在显式 `--media` 不存在会在调用 LLM 前拒绝，自动对齐失败会记录已应用的修改和该执行的命令 | `pipeline.py` | `31eef6e` |
| `correct --apply` 的自动重新对齐忽略 `[defaults]` 的 separate/normalize/vad_mask | `cli.py` | `0e23f9f` |
| `asrfix`：同一条字幕的多个修正互相覆盖；修改双说话人字幕会丢掉说话人标签；输出过长时整份稿子重试 4 次才拆分 | `asrfix.py` | `79b2225` |
| `--language ja` 等 ISO 代码（文档声明可用）在默认 torch Qwen 引擎上让每个块都失败 | `backend.py` | `79b2225` |
| 超过约 30 分钟的媒体：DP 分块规划器只按字幕跨度计预算、实际裁剪更宽，被校验器拒绝，导致 `align` 失败、`transcribe` 退回逐块对齐 | `chunking.py` | `79b2225` |

### 健壮性 / 稳定性

- PANNs（歌曲检测）把检查点路径打印到 stdout，破坏"stdout 只输出结果路径"的约定；不遵守 `VOXWEAVE_DEVICE`/MPS，并在所有 GPU 上 DataParallel；滑窗一次性物化（约 2 倍音频的额外内存）；标签 CSV 下载无超时、不校验、非原子写入；`--sdh` 会触发 panns 自带的 `wget` 引导（`6ca29d1`）。
- 多处导入期 `float()/int()` 解析环境变量：任何一个 `VOXWEAVE_*` 写错都会让所有命令（包括 `--help`）直接崩溃。现在 `config._env_int/_env_float` 会警告并回退，pipeline/songdet/shotdet/translate/chunking 等改用它们（`31eef6e`、`6ca29d1`、`bc302d8` 等）。
- 分离过程中 Ctrl-C 泄漏 GB 级临时 WAV；模型释放失败会掩盖原始错误并跳过临时文件清理（`31eef6e`）。
- 声纹库：只读挂载/他人拥有的锁文件导致共享读锁失败（`90cdde1`）。
- 试听服务：`/save` 的磁盘/权限错误直接断开连接；连接无超时（慢速连接可占满线程）（`5656b72`）。
- 翻译：OpenAI SDK 自带的 2 次重试 + 600 s 超时叠加在我们自己的重试下，挂起的端点要约 90 分钟才失败（`bc302d8`）。
- 诊断（diarize）：新增 `diarize.preflight()`，在不加载 torch 的前提下做与正式运行相同的门禁/令牌检查和说话人数检查；正式运行也会在加载 pipeline 前校验人数；pyannote 遵守 `VOXWEAVE_DEVICE`；离线/坏 revision 时给出"无法获取 config.yaml"而不是内部错误；被拒的子模型（如 segmentation-3.0）会被点名（`a752674`）。注：把 preflight 接入 `process()` 会让依赖"无 token 也能 mock 诊断"的现有测试失败，因此暂未接入，留作后续。
- 配置：`VOXWEAVE_CONFIG=~/...` 不展开 `~`，会在当前目录建出 `./~/` 并忽略真实配置；表内键名拼错（如 `[llm] base-url`，会把字幕发给 api.openai.com）无任何提示；`ctc_max_dp_frames` 接受 0/负数（`9062097`）。

### 引导不合理（CLI、报错、文档）

- 提示用户使用不存在或已隐藏的选项/命令：`--model`（应为 `--asr-model`）、`--lang`（应为 `--language`）、`voxweave[songdet]`/`voxweave[translate]` 这类不存在的 extra、`process` 命令、`--replace-episode`、`--no-match`、`--to`、speakers 子命令不接受的 `--diarize-model`。
- `correct` 的工作流提示（CLI 与 README）让用户在 `--apply` 后再跑一次 `align`（其实已自动对齐），且 `--apply` 并不会采用已审阅的 sidecar。
- 顶层帮助 "correct -> edit -> align -> render"：`edit` 不是命令，且 align 之后 render 会丢弃手工修改。
- `--min-speakers` 帮助建议"已知人数就传"，与项目自己测得的"会把 DER 从 19% 拉到 30%"相矛盾。
- 所有 `FileNotFoundError` 都提示"或 ffmpeg 不在 PATH"；错误面板显示内部类名（如 `Phase2DataError`）；缺依赖提示只给 `make install`（PyPI 用户无法执行）；`hf auth login` 在 `uv tool install` 后并不在 PATH 上。
- 试听服务 403 页总是建议 `--ngrok`（与真实原因无关）；purge 后的报错让用户重新采集刚删除的声纹。
- README/MIGRATING：`VOXWEAVE_MIN_CUE_SEC` 默认值写成 0.8（实际 0 = 关闭）；称非 MMS 的 CTC 对齐器按字幕裁剪（实际全文件一次）；`load_strategy=sum` 描述为"逐块并发"；PyPI 页面的相对链接失效；PyPI 安装命令缺少 onnxruntime override（CPU 版会顶掉 GPU 版）；未写明 `uv tool install --torch-backend` 需要 uv ≥ 0.9.19；以及多个未文档化的环境变量/配置键（`b1a3e44` 以外的各 docs 提交）。
- `--help` 会写入/迁移 `~/.config/voxweave.conf` 并把日志混进帮助输出（`0e23f9f`）。

### 死代码与无效注释（节选）

- 删除：`experiments/song_detect.py`（无引用且已与生产逻辑分叉）、`config.DEFAULT_ASR_MODEL` 的重复定义（改为 backend 引用它）、`shotdet.detect_shot_changes`（移入测试）、`timing.VISIBLE_GAP_MIN_S`、`schema.Atom`、`Seal.to_dict`、`segmentation_orchestration._swap_ext`、`align_acquisition._default_digest/_fresh_evidence_inputs`、`align_context` 的三个未用 API、`SCOPE_ORDER`、`SpeakerMatch.top_identity_id/top_similarity`、`speakercluster.Embed`、pyannote 3.x 的不可达调用路径、若干不可达分支。
- 修正与代码矛盾的注释/文档字符串：`core/__init__` 的包说明、smart_split 阶段描述、SourceUnit.provenance、langsets、timing_preview、align_seed（RAT-1 早已批准）、realign 路由/裁剪说明、diarize（split→render、TF32）、voiceembed、backend/backend_mlx 的对齐路径、Makefile/CI/pyproject 中过时的说明，以及引用仓库外设计文档（"§5.3""design 3.x""Phase-0"等）的注释。

### 构建 / CI / 打包

- `.gitignore` 的 `*.lock` 会匹配被 oracle 锁定哈希的 `uv.lock`（新增 `!uv.lock`）。
- CI：删除"语料不存在就跳过门禁"这一步（语料早已入库，保留它只会在语料意外丢失时静默放行）；为 workflow 加最小权限与超时。
- 打包：sdist/wheel 带上 `THIRD_PARTY_NOTICES.md`；sdist 不再带无法运行的 tests/（缺 conftest、scripts、calibration）。已用 `uv build` 验证。
- 测试：`tests/test_cli_voices.py` 中依赖目录只读权限的测试在 root 下必然失败，已按 root 跳过；songdet 的 fake torch 不再污染进程级设备缓存。

## 需要决策 / 未修复的重要问题

以下问题已确认存在，但修复要么会改变受门禁保护的输出（需要人工重新记录 baseline），要么涉及设计取舍、P6 法定区域或安全模型，不适合在本次审计中擅自改动。按严重度排序。

### 高

1. **音色试听服务无鉴权**（`speakerserve.py`）。`--host 0.0.0.0` 或 `--ngrok` 时，任何能访问端口/隧道的人都能播放剧集音频、读写说话人名称、触发 split（会重写字幕和声纹），而 `/serve-info` 会把保存用的 token 直接发给任何人。本次只做了缓解：启动时打印"无密码"警告、`--help` 与 README 标明风险、连接加 30 s 超时。真正的修复需要设计访问密钥（URL 中带一次性 secret，首个 GET 换成 HttpOnly cookie），或在未配置 ngrok 鉴权时拒绝 `--ngrok`。
2. **所有交付文件都是 0600**（`fsio.py`、`episode_transaction.py`）。VTT/SRT/ASS/JSON、pack/burn 的输出都经 `mkstemp` + `os.replace` 写出，因此永远是 0600，覆盖已有文件时也会把原权限降为 0600；同机其它用户、Plex/Jellyfin 等以其它账号运行的媒体服务读不到字幕。修复需要区分"公开交付物"（按 umask 创建、保留原权限）与"私有数据"（声纹库、缓存，保持 0600），并改到 P6 事务写入器，属于需要设计的改动。
3. **pack/burn 默认输出名会静默覆盖**（`mux.py:235`）。默认输出文件名不含字幕语言，也不检查已存在的文件，第二次对另一语言执行 pack/burn 会覆盖上一次的成品。
4. **>30 分钟媒体的对齐分块**（`align_dp_safety.py`，P6 法定区域）。DP 预算校验拒绝重叠/嵌套的字幕（voxweave 自己的 `rescue_tiny_cues` 就会写出这种重叠），导致"对齐 → 编辑 → 再对齐"在长片上失败；规划器与校验器度量的量也不一致（规划器侧已修复，见上文；校验器侧属 P6 源码门禁范围）。建议由 P6 负责人决定：对重叠字幕先投影成单调包络再校验，并在报错中给出字幕序号。
5. **P6 权威分配器是二次复杂度**（`align_distribution.py:697`）。每个区间都扫描到调用末尾，在真实整集全文件对齐上总是耗尽生产预算。属于算法设计问题。
6. **分段质量标尺的 `UnitLocator.locate_last` 取错单元**（`scripts/calib_segmentation.py:908`）。多个同一时间戳的零时长单元时取了第一个，导致带标点的边界被误分类，ja 的 forbidden_end 门禁被错误地提升为阻断。修复会改变标尺指标，需重新记录 baseline。
7. **shadow-v2 schema-2 准入对 refined 文档总是失败**（`core/shadow_v2.py:1627`）。仅影响可选的 shadow 通道（`VOXWEAVE_SEG_V2_SHADOW=1`），不影响用户输出。
8. **诊断的提前拒绝尚未接入主流程**。`diarize.preflight()` 已实现（门禁 token、说话人数），CLI 也已在参数层面校验人数；但把它接入 `pipeline.process()` 会让一批"无 token 也 mock 诊断"的现有测试失败，本次未接入。接入时需同时给这些测试提供假 token。

### 中（会改变分段输出，需重新记录 baseline 后再修）

- `core/kinsoku.py:43`：`_BIND_END_MED = frozenset("とまでより")` 被拆成单字 で/よ/り/ま/と，导致以「で」「よ」「り」结尾的行被当作粘连结尾惩罚（本意是 と/まで/より）。
- `core/timing.py:355`：延长字幕时直接夹到下一条开始，产生零间隔字幕，违反模块声明的 2 帧最小间隔。
- `core/smart_split.py:1398`：无时间戳的超长 atom 借用父字幕时间，产生重叠字幕并被当作语音锚点。
- `core/layout.py:375`：中日文两行折行按单字而非词评分，「目的」被当作「的」惩罚。
- 同样的宽度规则在 `core/align_compare.py:83`（`_visual_width`）还有一份旧副本，属于 P6 比较器，未改。

### 中（设计/结构）

- `pipeline.py` 是 4.6k 行的上帝模块：`transcribe`（约 550 行）与 `align`（约 660 行）混合了互不相关的阶段，人声缓存协议在两处各写一份且已漂移；约 25 处延迟 import 反向依赖它，只为拿路径小工具。建议拆出 `paths.py`、`vocals_cache.acquire_16k()`，并把 translate/correct 移出。
- `pipeline._write_siblings` 等约 180 行旧 sibling 写入链只被测试使用，测试因此在验证一个生产不走的写入器（`test_manifest`、`test_speech_anchors`、`test_pipeline_vad`）。
- `backend.transcribe_align` 及其单块辅助函数（约 110 行）只被测试使用；这些测试被 P6 oracle 清单引用，按约束未改。
- `voxweave/p6_ratifications.py` 只被测试 import，其 `*_ENABLED` 常量不控制任何代码路径；相关测试被 oracle 引用，未删。
- ASR 模型加载在逐块容错里进行：模型名写错或无网络时，会对每个块（整部电影约 80 块）各重试一次，`--hybrid` 下缺一个引擎还会静默降级。
- 分离器全局设置 TF32 精度且不恢复，导致后续 fp32 阶段的数值随"本次是否跑了分离"而变（缓存命中 vs 未命中不可复现）。
- 绑定声纹的 `align` 在人声缓存未命中时重写缓存却不写 companion，之后每次 align 都会重新分离。
- `speakercluster` 中约 200 行 spectral/refine 聚类只被测试可达；`turnembed` 旧通道对短 turn 零填充、对长 turn 不分窗。
- 发布流程（`release.yml`）打 tag 即发布 PyPI，不跑测试、不要求 CI 绿。
- CI 中 `ruff format` 未锁定版本，ruff 的年度风格更新会让未改动的代码直接变红。
- `debug/` 在 `.gitignore` 中却被强制跟踪了十余份历史报告，其中部分内容已过时（默认模型、缓存布局等）；建议统一策略（移到 `docs/history/` 或取消跟踪）。
- 声纹库多用户共享 NAS 目录时，锁文件和新文件都会被收窄成 0600（与文档承诺"不收窄预先建好的共享目录"矛盾），需先决定是否支持多账号共享。
- `scripts/p6_oracle.py`：依赖门禁漏掉了 `from voxweave import X` 与相对 import 两种写法；工具性失败退出码为 1（应为 2）；环境不匹配时报错不写出期望值与实际值。按约束只报告。

## 验证

| 门禁 | 结果 |
|---|---|
| `ruff check .` / `ruff format --check .` | 通过 |
| `pyright`（basic，生产代码） | 0 errors, 0 warnings |
| P6 oracle `validate` / `compare --check` / `source-gates --check` | 均退出 0（在最终 HEAD 上重跑） |
| 分段质量标尺 `calib_segmentation.py evaluate --check` | `status=pass cases=20 failures=0 warnings=1`，与修改前逐项一致 |
| `pytest tests/`（全量） | 4746 passed, 10 skipped；另有 2 个 oracle 测试（调用 `make quality-p6-oracle`）在无 `.venv` 的 git worktree 中因解释器不匹配失败，在带 3.13.12 venv 的主检出中重跑均通过。全量运行基于 `d3259e8`；最后的 backend 批次（`79b2225`）另跑了相关测试 413 passed，并重跑 oracle compare/source-gates 均为 0 |
| `uv build`（sdist + wheel） | 通过；sdist 不含 tests/，两者都含 THIRD_PARTY_NOTICES.md |

说明：

- 修改前的基线全量测试为 4492 passed / 17 failed / 12 errors，失败全部来自环境：克隆是浅克隆，缺少 oracle 引用的历史提交（本次已 `git fetch --unshallow` 补齐），另有一个依赖只读目录权限、在 root 下必然失败的测试（本次已改为 root 下跳过）。
- 测试集中约 12 个 diarize 测试会真实访问 huggingface.co（离线时每个约 24 s），另有若干 oracle 测试会调用 `uv run` / `make`；这些问题已列入附录（tests-hygiene 切片），本次未改。

## 过程说明

- 审计分 24 个切片并行进行，修复由 14 个修复 agent 按互不重叠的文件集合并行完成，每批由我审阅 diff 后单独提交。
- 两处提交内容与说明不完全对应（并发暂存所致，内容本身正确）：`experiments/song_detect.py` 的删除落在 `4db7b39`（README 提交）里；`REPORT-split-cluster.md` 移入 `debug/` 落在 `1229596`（realign 提交）里。
- 未修改：被 oracle 锁定哈希的文件、oracle 引用的测试、`scripts/p6_oracle.py`、`uv.lock`。

## 附录：全部发现（按切片）

复核列："确认/部分确认/驳回"来自独立复核 agent；"未复核"表示截止前未排到复核。部分确认通常指严重度或修复方案被修正，而问题本身成立。已修复与否以正文为准。

### align-core

This slice is the forced-alignment core. align_ctc (wav2vec2, windowed emissions, star wildcards, one global DP) and align_mms (MMS-300m ONNX with uroman) are the two full-pass backends. align_common holds the shared DP-chunking driver (_prepare_dp_calls / _execute_dp_calls), audio loading, song muting and VAD emission masking. align_dp_safety is a fail-closed validator for over-budget route hints and plans. realign holds VTT parsing, char-level routing, punctuation fusion and reinjection, VAD positioning and cue-duration finalisation. align_inputs and align_runtime are P6 input projections and an opt-in trace recorder. The alignment maths itself is sound: frame-to-time mapping, token/unit counts in _distribute_units, and monotonicity inside one DP all check out. The serious problems are at the edges. The over-budget (>30 min) path is mostly unusable, because the planner budgets cue spans while the validator budgets the wider physical crops. On 50 random 60-minute layouts, 66% were refused, and nested cues are refused too, including ones voxweave's own rescue_tiny_cues writes. parse_vtt_blocks silently drops plain-text cues that start with "Note"/"Region"/"Style". Hybrid fusion strips decimal points from numbers. The rest is lower severity: stale per-cue-crop docstrings, P6 plumbing and helpers that are dead or used only by tests, two drifted copies of the preparation code, and fragile env-knob parsing.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | high | bug | `voxweave/align_dp_safety.py:184` | Over-budget DP plans are refused for most real long media: planner budgets cue spans, validator budgets wider physical crops | 确认 |
| 1 | high | bug | `voxweave/align_dp_safety.py:106` | Over-budget hint validation rejects overlapping/nested cues, including the ones voxweave's own align output writes | 确认 |
| 2 | high | bug | `voxweave/realign.py:77` | parse_vtt_blocks silently drops cues whose text starts with Note/Region/Style/WebVTT (plain-text drafts lose dialogue) | 确认 |
| 3 | medium | bug | `voxweave/realign.py:266` | Hybrid fusion strips decimal points and thousands separators inside numbers (14.2 -> 142) | 确认 |
| 4 | medium | guidance | `voxweave/align_dp_safety.py:113` | Plain-text draft -> align fails on >30-min en/ja media with a jargon-only error and no remedy | 确认 |
| 5 | medium | guidance | `voxweave/align_mms.py:311` | --vad-mask is advertised for all alignment but is a silent no-op for ja (MMS) and zh/yue (Qwen), while the manifest records it as on | 部分确认 |
| 6 | low | robustness | `voxweave/align_ctc.py:48` | Env knobs parsed with bare float() at import: a typo crashes every command, WINDOW_S<=0 loops forever | 确认 |
| 7 | low | bug | `voxweave/realign.py:264` | No-space fusion duplicates punctuation where Whisper already punctuated (。。 / 、、) | 驳回 |
| 8 | low | stale-comment | `voxweave/realign.py:1060` | realign docstrings still describe per-cue routing, cropping and back-fill that the default en/ja full-pass align no longer uses | 部分确认 |
| 9 | low | stale-comment | `voxweave/realign.py:35` | realign.MIN_CUE_SEC = 0.8 and its 'align must enforce this itself' comment contradict production (floor disabled, default 0) | 驳回 |
| 10 | low | dead-code | `voxweave/align_ctc.py:252` | interp_missing in the CTC path can never change a unit; its observer plumbing and docstrings describe a safety net that does not exist | 确认 |
| 11 | low | dead-code | `voxweave/realign.py:1183` | realign.render_vtt is used only by tests; the production align renderer is a separate copy | 部分确认 |
| 12 | low | dead-code | `voxweave/align_common.py:374` | Legacy three-argument pass_fn contract in _execute_dp_calls/_dp_chunked_pass exists only for tests | 未复核 |
| 13 | low | dead-code | `voxweave/align_ctc.py:326` | Unused `iso` parameter in _ctc_flat_pass and a CtcAligner.sep_id field read only by tests | 未复核 |
| 14 | low | stale-comment | `voxweave/align_mms.py:310` | Comment implies MMS does not use offset_s, but the deferred projection and raw observer depend on it | 未复核 |
| 15 | low | stale-comment | `voxweave/realign.py:640` | ZeroDurationDiagnostics claims repairs are never inferred from float diffs, but exact-zero repairs are | 未复核 |
| 16 | low | structure | `voxweave/align_ctc.py:505` | align_blocks_full_ctc/_mms each keep two copies of the preparation sequence, and the copies have drifted | 未复核 |
| 17 | low | quality | `voxweave/realign.py:478` | route_blocks discards all VTT timestamps when any single block lacks one; the pipeline error then claims the VTT has none | 未复核 |
| 18 | low | quality | `voxweave/align_inputs.py:158` | Profile key validation has a misleading tuple-order clause and an unreachable length check | 确认 |

### asr-backend

The asr-backend slice covers the local model backend (backend.py: separator load/demix, Qwen/faster-whisper/fusion ASR passes with per-chunk failure containment and optional batching, full-file vs per-chunk alignment dispatch, singleton lifecycle), the Apple-Silicon adapters (backend_mlx.py), ffmpeg decode/VAD/chunk planning (chunking.py), language normalization/reconciliation (lang.py), the LLM correction sidecar (asrfix.py) and a one-function timestamps.py. Main entry points are backend.transcribe_chunks / separate_vocals / align_text / release, chunking.decode_to_wav / vad_speech_segments / pack_speech_segments / plan_dp_chunks / slice_wav, and asrfix.correct_cues / apply_fixes / render_vtt. The code is generally careful, but two high-impact defects stand out. The torch Qwen engine passes `--language` through unnormalized, so ISO codes, which the CLI help documents as valid, fail every chunk. plan_dp_chunks budgets cue spans rather than the crop windows it emits, so the DP-safety validator rejects most long ja/en media, and `align` then fails outright. Robustness gaps cluster around error paths. Deterministic model-load failures are retried once per chunk. A missing ffmpeg surfaces as a raw FileNotFoundError and leaks a temp file. The friendly separator-download hint is unreachable. On the LLM-correction side, a truncated response is retried four times before the cue set is split, duplicate fixes silently overwrite each other, and per-line speaker tags are dropped on corrected lines. Vulture's backend_mlx 'unused params' (return_time_stamps/hotwords/vad_filter) are deliberate API-compat shims passed by backend._asr_only/_qwen_asr_kwargs and are not dead. transcribe_align and its helpers are production code used only by tests. Note: the working tree has uncommitted edits by another process; one stale comment reported below (fusion whisper default) is already corrected there, and all findings are against HEAD.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | high | bug | `voxweave/backend.py:691` | Torch Qwen engine passes --language raw; ISO codes (documented as valid) fail every chunk | 确认 |
| 1 | high | bug | `voxweave/chunking.py:163` | plan_dp_chunks emits crop windows larger than max_sec; DP validator refuses most long media | 确认 |
| 2 | medium | robustness | `voxweave/backend.py:1012` | Deterministic model-load failures retried once per chunk; fusion silently degrades | 部分确认 |
| 3 | medium | guidance | `voxweave/backend.py:214` | Friendly separator-download hint (--no-separate, manual path) is unreachable | 部分确认（改判 low） |
| 4 | medium | robustness | `voxweave/chunking.py:240` | decode_to_wav: missing ffmpeg raises raw FileNotFoundError and leaks the temp wav | 部分确认（改判 low） |
| 6 | medium | bug | `voxweave/asrfix.py:216` | apply_fixes: duplicate fixes for one cue silently overwrite each other; audit reports both applied | 确认 |
| 7 | medium | bug | `voxweave/asrfix.py:227` | Correcting a line in a per-line-speaker (dash) cue drops that line's voice tag | 确认 |
| 8 | medium | robustness | `voxweave/asrfix.py:340` | Length-truncated correction retried 4x on the full transcript before splitting; warning blames vLLM | 驳回 |
| 9 | medium | dead-code | `voxweave/backend.py:1358` | transcribe_align and its single-chunk helpers are production code used only by tests | 确认 |
| 5 | low | guidance | `voxweave/chunking.py:245` | ffmpeg error message dumps full stderr (banner) and never says it timed out | 确认 |
| 10 | low | stale-comment | `voxweave/backend.py:91` | Comment says fusion whisper defaults to large-v3-turbo; built-in default is large-v3 | 部分确认 |
| 11 | low | stale-comment | `voxweave/backend.py:67` | Whisper-hybrid and aligner-singleton comments describe Qwen timestamps that no longer apply | 部分确认 |
| 12 | low | stale-comment | `voxweave/backend_mlx.py:5` | MLX module/align docstrings claim ALL alignment goes through the MLX aligner | 部分确认 |
| 13 | low | robustness | `voxweave/chunking.py:17` | Malformed numeric env vars crash CLI import (even --help) with a raw traceback | 确认 |
| 14 | low | bug | `voxweave/chunking.py:47` | pack_speech_segments emits an arbitrarily short hard-cut remainder as its own chunk | 确认 |
| 15 | low | quality | `voxweave/backend.py:1262` | Local `release` bool shadows the module-level release() inside transcribe_chunks | 确认 |
| 16 | low | quality | `voxweave/backend.py:349` | TF32 matmul precision set process-wide on separator load and never restored | 确认 |
| 17 | low | robustness | `voxweave/backend.py:505` | separate_vocals leaks its temp FLAC when writing fails | 确认 |
| 18 | low | dead-code | `voxweave/backend.py:16` | backend re-exports interp_missing and _load_yaml exist only for tests | 确认 |
| 19 | low | dead-code | `voxweave/lang.py:40` | _ISO1_TO_ISO3 entries for ar/hi/nl/tr/vi/th/id/uk/pl are unreachable | 部分确认 |
| 20 | low | structure | `voxweave/backend.py:114` | Separate undocumented MODEL_DIR root ignores VOXWEAVE_CACHE_ROOT; import-time directory rename | 部分确认 |
| 21 | low | guidance | `voxweave/backend_mlx.py:100` | MLX backend silently swaps custom/explicit ASR and distil model ids for stock repos | 部分确认 |
| 22 | low | structure | `voxweave/chunking.py:336` | Generic slice_wav carries P6 failure-taxonomy plumbing via private kwargs | 确认 |
| 23 | low | robustness | `voxweave/chunking.py:295` | vad_speech_segments validates sample rate with assert | 部分确认 |
| 24 | low | stale-comment | `voxweave/backend.py:757` | 'whisper has no Cantonese code' is false for large-v3/turbo, the default whisper models | 部分确认 |

### cli-ux

The cli-ux slice is split into clear parts. cli.py holds the root DefaultGroup and the transcribe, render, align, translate, export, pack, burn and correct commands, with tri-state flag resolution for CLI > env > conf > built-in. cli_compat.py handles routing a bare media path to transcribe, plus the hidden renamed options and the deprecated aliases, all documented in MIGRATING 0.16.0. cli_speakers.py and cli_voices.py build the two sub-groups around a shared `_run` error wrapper. ui.py has the rich progress reporter, the result panels and the error hints; progress.py is the renderer-free Reporter base class; runtime.py covers device, dtype, dependency hints and bridging Hugging Face download progress into the reporter. Overall the code is careful: stdout/stderr are kept apart, the aliases are deliberate, and non-tty output is handled. The main problems are:
- `correct --apply` re-aligns without the [defaults] separate/normalize/vad_mask settings.
- User guidance has drifted from the code: the correct/apply workflow advice, the top-level "edit -> align -> render" chain, the VOXWEAVE_HF_TOKEN hint, internal exception names shown to users, and hints that repeat the error message.
- Several failures skip the error panel or crash with a raw traceback: glossary loading, a very long argument token, and rich markup in the correct summary.
- There is no early validation of speaker counts, so a bad value only fails after the full ASR run.
Note: the working tree was being edited by another process during this audit (ui.py:351 OOM hint and config.py template now say --asr-model), so that already-fixed item is left out.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | medium | bug | `voxweave/cli.py:1040` | `correct --apply` auto re-align ignores conf [defaults].separate / normalize / vad_mask | 确认 |
| 1 | medium | guidance | `voxweave/ui.py:503` | Correct-workflow hint (and README) says `correct --apply` applies the reviewed sidecar and then needs `align`; neither is true | 确认 |
| 2 | medium | robustness | `voxweave/ui.py:477` | Correct summary feeds filenames and LLM/cue text into rich markup: names get mangled, and a `[/tag]` crashes after the files are written | 确认 |
| 3 | medium | robustness | `voxweave/cli.py:753` | Glossary loading runs outside `_run`: a bad --glossary gives a raw Python traceback instead of the error panel | 确认（改判 low） |
| 4 | medium | robustness | `voxweave/cli.py:372` | --min-speakers/--max-speakers are unvalidated: bad bounds fail only after the full ASR run, and 0 silently means 'unbounded' | 部分确认（改判 low） |
| 5 | medium | guidance | `voxweave/runtime.py:117` | HF download failure hint tells users to set VOXWEAVE_HF_TOKEN, which these download helpers never read | 确认（改判 low） |
| 6 | medium | guidance | `voxweave/cli.py:239` | Top-level help workflow 'correct -> edit -> align -> render' names a non-command and steers users to re-layout after align | 确认 |
| 7 | low | guidance | `voxweave/ui.py:486` | --apply summary points to an audit JSON that isn't written, and says 'text changed' when nothing changed | 确认 |
| 8 | low | guidance | `voxweave/cli.py:69` | Flag-source labels are wrong: '--no-diarize' is reported as 'CLI --diarize', and 'off (--no-separate)' appears when config disabled it | 确认 |
| 9 | low | robustness | `voxweave/cli.py:70` | A blank VOXWEAVE_VOICEPRINTS makes every transcribe fail with a usage error | 确认 |
| 10 | low | robustness | `voxweave/cli.py:245` | `voxweave <cmd> --help` writes (or migrates) ~/.config/voxweave.conf and logs it into the help output | 确认 |
| 11 | low | guidance | `voxweave/ui.py:331` | Every FileNotFoundError gets the 'or ffmpeg is not on PATH' hint, including missing sibling JSON/media in `speakers` commands | 确认 |
| 12 | low | guidance | `voxweave/ui.py:391` | Error panel shows internal exception class names such as 'Phase2DataError' to users | 部分确认 |
| 13 | low | quality | `voxweave/ui.py:355` | _hint_for matches on message substrings, repeats guidance the message already gives, and keeps a marker nothing produces | 部分确认 |
| 14 | low | guidance | `voxweave/runtime.py:95` | Missing-dependency hints only suggest `make install`, which needs a source checkout; the README's primary install is PyPI | 确认 |
| 15 | low | robustness | `voxweave/cli_compat.py:104` | Default-command routing crashes with a raw OSError on an over-long token | 确认 |
| 16 | low | guidance | `voxweave/cli_compat.py:172` | On click >= 8.4, typo suggestions include hidden commands (e.g. the deprecated 'split') | 确认 |
| 17 | low | structure | `voxweave/cli_speakers.py:371` | Serve has a special-case call branch that exists only to fit single-argument test doubles | 部分确认 |
| 18 | low | guidance | `voxweave/cli_voices.py:15` | --voices-dir help leaves out $XDG_DATA_HOME, and the text is copied three times | 确认 |
| 19 | low | guidance | `voxweave/cli.py:406` | --timestamps help promises word-level timestamps in the VTT; the output only has cue timing lines | 确认 |
| 20 | low | guidance | `voxweave/cli.py:648` | Command docstrings show raw Markdown/RST in --help and describe out-of-date behaviour | 确认 |
| 21 | low | guidance | `voxweave/cli.py:305` | --keep-lyrics says 'detection still runs', but with separation off it silently does nothing | 确认 |
| 22 | low | robustness | `voxweave/cli.py:531` | Debug-dir display calls the state-changing artifacts.claim_paths outside `_run` | 部分确认 |
| 23 | low | guidance | `voxweave/export.py:233` | Export error names the hidden legacy flag `--to` instead of `-f/--format` | 确认 |
| 24 | low | guidance | `voxweave/cli.py:270` | --asr-model help omits its env var and config-file default, unlike every sibling option | 部分确认 |
| 25 | low | dead-code | `voxweave/runtime.py:135` | runtime._load_yaml is only used by tests | 确认 |

### config

voxweave/config.py is a pure-stdlib accessor module. It holds the built-in defaults, the cache layout, the first-run TOML template (_TEMPLATE plus ensure_default_config with qsub.conf migration), and about 25 conf_*/resolve_* accessors that each re-read the TOML through _load(). It also holds the env-only segmentation thresholds (gap_thresholds). Precedence (CLI > env > file > default) is implemented correctly for the keys it covers (asr_model, llm.*, diarize.*, voices.dir, batch, autocast). No config key is dead: every top-level key and section key is read by some consumer. The weak spots are uneven validation and stale text. Some accessors warn on bad values (autocast, [llm] ints, [align]); others silently drop typos and invalid values (load_strategy, [batch], [fusion], [defaults] keys, section-level typos, env ints). ctc_max_dp_frames has no lower bound. VOXWEAVE_CONFIG is not tilde-expanded. Every accessor re-parses the file and repeats the same warnings. The template and the README Configuration section have drifted from the code in several places: the fusion whisper default, the claim that non-MMS CTC aligners run per-cue, the description of load_strategy "sum", the VOXWEAVE_MIN_CUE_SEC default, and the claim that the template is fully commented out. The README also leaves several real knobs undocumented, VOXWEAVE_CONFIG among them.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | medium | guidance | `voxweave/config.py:143` | Fusion whisper default is large-v3 in code, but the template and backend comment both say large-v3-turbo | 确认 |
| 1 | medium | guidance | `voxweave/config.py:234` | Template, comments and README say non-MMS CTC aligners run per-cue crop and that xlsr full-file OOMs; both are false | 部分确认（改判 low） |
| 2 | medium | guidance | `README.md:881` | README gives VOXWEAVE_MIN_CUE_SEC a default of 0.8; the code default is 0 (disabled) | 部分确认（改判 low） |
| 3 | medium | robustness | `voxweave/config.py:703` | ctc_max_dp_frames accepts 0/negative values and silently ignores bad env values; alignment then fails with an unrelated "route hints" error | 部分确认（改判 low） |
| 4 | medium | robustness | `voxweave/config.py:685` | Several accessors silently discard invalid values or non-table sections, inconsistent with the ones that warn | 部分确认（改判 low） |
| 5 | medium | robustness | `voxweave/config.py:250` | Typos in keys inside a section are never flagged; only top-level keys are checked | 确认 |
| 6 | medium | bug | `voxweave/config.py:247` | VOXWEAVE_CONFIG is not tilde-expanded: the real config is ignored and a literal ./~/ directory is created in the cwd | 驳回 |
| 7 | low | quality | `voxweave/config.py:270` | _load() re-parses the file on every accessor call and repeats the same warnings; the 'warned about once' claims are false | 部分确认 |
| 8 | low | guidance | `README.md:915` | README says the auto-written template has everything commented out; its [align] entries are active | 驳回 |
| 9 | low | guidance | `voxweave/config.py:112` | load_strategy 'sum' is described as per-chunk ASR+align in one pass, but the pass structure is identical to 'peak' | 部分确认 |
| 10 | low | dead-code | `voxweave/config.py:19` | DEFAULT_ASR_MODEL is unused anywhere, and its 'mirrors backend.ASR_MODEL / FUSION_*' comment names constants that don't exist | 驳回 |
| 11 | low | guidance | `voxweave/config.py:98` | Template points to the hidden legacy --model flag, and conf_asr_model's docstring gets env/config precedence backwards | 驳回 |
| 12 | low | guidance | `README.md:805` | README Configuration omits VOXWEAVE_CONFIG and several real config knobs | 部分确认 |
| 13 | low | structure | `voxweave/config.py:80` | 'All weights go under VOXWEAVE_CACHE_ROOT', but backend keeps a second root (MODEL_DIR) that ignores it | 确认 |
| 14 | low | quality | `voxweave/config.py:322` | _nonempty_str checks the stripped value but returns the unstripped one, so padded values reach model loaders | 驳回 |

### dead-code-sweep

This slice is a dead-code sweep of the whole `voxweave` package, vendor code excluded (~72k LOC). I ran vulture twice, once on the whole repo and once on production code only. I added an AST scan for unused parameters and a transitive reachability pass over production symbols, which catches helpers only called by other dead helpers. The package is mostly well connected: click commands, HTTP handlers, the documented pipeline/backend re-exports and the documented `noqa: ARG002` duck-typing parameters in backend_mlx are deliberate and were not reported. Most real findings fall into three groups. (1) Large legacy entry points that production replaced but that are still kept alive, and still tested, only through tests: `pipeline._write_siblings` and its JSON-writer chain (now a different shape from the projector writer production actually uses), and `backend.transcribe_align` with its three single-chunk helpers. (2) Test-only fault-injection and qualification seams built into P6 production modules: the authority-limit test profiles in align_distribution, the simulated boundary-row qualification in segmentation_candidates, and a `_verifier_cut_mutator` parameter passed through acquisition that nobody ever sets. (3) Small unused or test-only helpers, closed-vocabulary constants and always-constant predicates left over from P6/P7, including the whole `p6_ratifications` module, whose `*_ENABLED` flags are never read. Several of these carry docstrings that now contradict the code: `shadow_artifact` "the one call the Wave B hook makes", `_dump_sibling_json` "shared by process and align", and a finalizer.DELTA_IDS list that names FD-2, a delta this module never fires. Note: another process was editing backend.py, config.py, pipeline.py, songdet.py and ui.py while I worked. Line numbers for those files were re-checked at the end, and config.DEFAULT_ASR_MODEL was dropped because the working tree now uses it.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | medium | dead-code | `voxweave/pipeline.py:1805` | _write_siblings / _dump_sibling_json / _sibling_json_data / _persistable_cue chain is only used by tests and has drifted from the real sibling writer | 确认 |
| 1 | medium | dead-code | `voxweave/p6_ratifications.py:78` | p6_ratifications module is never imported by production; its *_ENABLED 'flags' are read by nothing | 部分确认（改判 low） |
| 2 | medium | dead-code | `voxweave/backend.py:1358` | backend.transcribe_align and its single-chunk helpers (_transcribe_fusion/_transcribe_whisper_align/_transcribe_qwen_align) are reachable only from tests | 确认 |
| 3 | medium | structure | `voxweave/align_distribution.py:205` | Test-only authority-limit qualification machinery lives in the production allocator and is consulted on every align run | 部分确认（改判 low） |
| 4 | medium | structure | `voxweave/segmentation_candidates.py:235` | Simulated boundary-row qualification (test-only) is exported from production and adds an unreachable acceptance branch to verification | 部分确认（改判 low） |
| 5 | low | dead-code | `voxweave/align_acquisition.py:1446` | _verifier_cut_mutator fault-injection parameter is threaded through acquisition but never set by any caller | 确认 |
| 6 | low | dead-code | `voxweave/align_distribution.py:1292` | build_authority_distribution public entry point is only used by tests (production uses _build_context_authority_distribution) | 部分确认 |
| 7 | low | dead-code | `voxweave/align_context.py:390` | retire_context_role, verify_context_content and ContextRole are unused anywhere; role_events is only used by tests | 确认 |
| 8 | low | dead-code | `voxweave/align_acquisition.py:2200` | _fresh_evidence_inputs and _default_digest are unused anywhere; _fresh_core_inputs is only used by one test | 确认 |
| 9 | low | dead-code | `voxweave/align_failures.py:357` | is_detail_dormant always returns False (RATIFICATION_DORMANT_DETAILS is empty) and is only called by tests | 确认 |
| 10 | low | dead-code | `voxweave/core/align_compare.py:49` | semantic_comparison_available() is a constant-True gate that only tests call | 确认 |
| 11 | low | stale-comment | `voxweave/core/finalizer.py:164` | finalizer.DELTA_IDS ('rows this module can fire') is unused and lists FD-2, which this module never fires | 确认 |
| 12 | low | stale-comment | `voxweave/core/boundary_v2.py:2128` | shadow_artifact docstring 'The one call the Wave B hook makes' is false; the function is only used by tests | 驳回 |
| 13 | low | stale-comment | `voxweave/pipeline.py:1740` | Sibling-writer docstrings describe _write_siblings/_dump_sibling_json as the shared production writer | 部分确认 |
| 14 | low | dead-code | `voxweave/core/smart_split.py:1874` | smart_split_segments accepts min_duration/desired_wps and forwards them only to parameters documented as unused | 确认 |
| 15 | low | dead-code | `voxweave/align_ctc.py:321` | _ctc_flat_pass accepts iso but never uses it | 确认 |
| 16 | low | dead-code | `voxweave/translate.py:346` | render_translated_vtt and TRANSLATE_MODEL are only used by tests | 确认 |
| 17 | low | dead-code | `voxweave/shotdet.py:234` | shotdet.detect_shot_changes blocking wrapper is only used by tests | 驳回 |
| 18 | low | dead-code | `voxweave/voicebase.py:531` | Several exported voice/speaker helpers are only used by tests (write_voiceprints, voiceprint_conjunction_valid, script_json, cache_pair_valid, suggest_bytes, episode_lock_path, load_speaker_mapping, delta_registry_bytes, embedding_spans) | 未复核 |
| 19 | low | dead-code | `voxweave/core/timing.py:28` | Constants never read anywhere: VISIBLE_GAP_MIN_S, LAYOUT_SOURCES, PROFILE_VIOLATION_REASONS, SCOPE_ORDER, _r_sha helper | 未复核 |
| 20 | low | dead-code | `voxweave/core/schema.py:38` | schema.Atom TypedDict is never used as a type | 未复核 |
| 21 | low | dead-code | `voxweave/debug.py:53` | DebugSink.enabled and DebugSink.root are never read by production | 未复核 |
| 22 | low | robustness | `voxweave/segmentation_candidates.py:151` | P6 id()-keyed authority registries are never pruned and keep every run's VTT/JSON bytes alive for the life of the process | 未复核 |

### diarize

The slice has three parts. diarize.py (1.9k lines) holds the pyannote 4.x loader with its provenance (config and embedding pinning, the 3.1 PLDA shim, the HF gate errors), the diarization run with its TF32 handling and the opt-in voiceprint regrouping stage, and the pure speaker-aware cue formatter that `render` replay and the core shadow lane also use. speakercluster.py is the numpy-only `voiceprint-v1` recipe. Its only production caller is diarize (a deferred `from voxweave import speakercluster` inside `_voiceprint_clustering`), which always builds `ClusteringParams()` (method `ahc`). turnembed.py is the split-speaker embedding provider. It is imported at module level by speakerserve.py, and scripts/calibrate_voiceprints.py calls its `_load_inference`. The code is defensive: most numeric and identity edge cases are guarded, and the risky stage falls back to pyannote's result. The problems are these:
- User-facing guidance is weak around HF gating and failures to reach the Hub. The gate error names the wrong repo, one hint is stale, and an offline 3.1 load gives a cryptic error.
- Diarization preconditions are only checked after the full ASR run.
- The pyannote pipeline ignores VOXWEAVE_DEVICE.
- The legacy turn-embedding lane zero-pads short turns and runs long turns unwindowed, which the project's own voiceembed module rejects.
- About 200 lines of spectral/refine clustering are reachable only from tests.
- Plus small dead branches, stale docstrings and one layering shortcut.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | high | robustness | `voxweave/diarize.py:1171` | Diarization preconditions (HF token for gated models, min/max speaker sanity) are only checked after the whole separation+ASR+alignment run | 部分确认 |
| 1 | medium | guidance | `voxweave/diarize.py:866` | Gate error names the configured pipeline's model card, not the repo that actually refused (e.g. segmentation-3.0 for 3.1) | 部分确认（改判 low） |
| 2 | medium | guidance | `voxweave/diarize.py:749` | Offline / unreachable Hub or a bad revision for `--diarize-model 3.1` surfaces as an internal 'verified AgglomerativeClustering pipeline config' error | 确认 |
| 3 | medium | robustness | `voxweave/diarize.py:870` | pyannote pipeline placement ignores VOXWEAVE_DEVICE (and MPS); every other model uses runtime.get_device() | 确认 |
| 4 | medium | guidance | `voxweave/diarize.py:836` | Escape-hatch hint says 'voiceprint stores are per-model', contradicting README/config for the default (decoupled) voiceprint lane | 确认（改判 low） |
| 5 | medium | dead-code | `voxweave/speakercluster.py:285` | `spectral` and `refine` clustering methods (~200 lines) are reachable only from tests | 确认 |
| 6 | medium | quality | `voxweave/turnembed.py:582` | Legacy-lane split embeddings zero-pad every sub-2 s turn to 2 s and pool without a mask, the approach voiceembed documents as corrupting voice statistics | 部分确认 |
| 7 | low | robustness | `voxweave/turnembed.py:583` | Legacy-lane turn embedding runs each whole turn in one forward pass (no windowing) | 确认 |
| 8 | low | robustness | `voxweave/turnembed.py:145` | turnembed has no release(): the legacy embedding model stays resident on the GPU for the whole `speakers serve` lifetime | 部分确认 |
| 9 | low | robustness | `voxweave/diarize.py:1358` | _span_speaker silently mis-attributes when turns are not sorted; the replay path never sorts persisted turns | 确认 |
| 10 | low | robustness | `voxweave/speakercluster.py:643` | Anchor AHC is O(n^3) time / O(n^2) memory with no cap on the anchor count | 确认 |
| 11 | low | dead-code | `voxweave/diarize.py:1217` | pyannote 3.x call path is unreachable (dependency pinned to pyannote-audio>=4,<5); only tests reach it by faking the version | 部分确认 |
| 12 | low | dead-code | `voxweave/diarize.py:415` | _expand_model_references `parent_subfolder` is never non-None (unreachable branch) | 确认 |
| 13 | low | dead-code | `voxweave/diarize.py:1428` | Unreachable guards in the speaker-run passes | 确认 |
| 14 | low | dead-code | `voxweave/speakercluster.py:92` | `Embed` alias unused anywhere; `embedding_spans` used only by tests | 部分确认 |
| 15 | low | dead-code | `voxweave/turnembed.py:545` | Unattested turn_embeddings path (plain turn list -> default WeSpeaker model) is only used by tests and contradicts the module contract | 部分确认 |
| 16 | low | stale-comment | `voxweave/turnembed.py:540` | turn_embeddings docstring describes only the legacy padding; the decoupled lane repeats to the embedder minimum and windows long turns | 确认 |
| 17 | low | stale-comment | `voxweave/diarize.py:10` | Module docstring refers to the `split` command, which was renamed to `render` | 部分确认 |
| 18 | low | stale-comment | `voxweave/diarize.py:990` | _clustering_embedding_numerics docstring says pyannote's span leaves TF32 off at this point, but that span is restored before clustering runs | 确认 |
| 19 | low | structure | `voxweave/diarize.py:42` | diarize.py mixes the model loader with the pure cue formatter, imports a private backend helper, and turnembed re-imports diarize privates | 部分确认 |

### docs-guidance

The docs slice has five files. README.md is the user reference for install, every subcommand, configuration, the data contract and performance knobs. MIGRATING.md holds per-release upgrade notes. calibration/README.md and calibration/p6-oracle/README.md are maintainer docs for the quality rulers and the P6 oracle corpus. REPORT-split-cluster.md is a branch hand-off report sitting in the repo root. I ran `--help` for every subcommand and cross-checked the docs against cli.py, config.py and the pipeline code. Most flags, defaults, env-var defaults, file layouts and voice-library behaviour match. The real problems are guidance errors that steer users into failure or wasted work: the documented `--lang` flag does not exist, `correct --apply` is documented as needing a manual `align` that it already runs, the MIGRATING advice to start a new `--voices` store fails without `--show`, the `--min-speakers` help contradicts the README's own measurements, and `hf auth login` is not on PATH after the documented `uv tool install`. I also found one real code bug: the auto re-align inside `correct --apply` ignores the `[defaults]` section. Commits and edits from other agents landed while I worked (4db7b39, 9062097). They already fixed the PyPI onnxruntime override, the `VOXWEAVE_MIN_CUE_SEC` default, the per-cue-crop comments, the load-strategy comment in the config block, the `--lang` code comments and the `--replace-episode` hints, so those are left out. All line numbers refer to the current working tree.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | medium | guidance | `README.md:183` | README and MIGRATING tell users about a `--lang` option that does not exist (it is `--language`) | 驳回 |
| 1 | medium | guidance | `README.md:603` | README says to run `align` after `correct --apply`, but `--apply` already re-aligns by default; the options table leaves out `--align/--no-align` and `--media` | 驳回 |
| 2 | medium | bug | `voxweave/cli.py:1040` | The auto re-align in `correct --apply` ignores `[defaults].separate/normalize/vad_mask`, unlike `voxweave align` | 驳回 |
| 3 | medium | guidance | `MIGRATING.md:101` | MIGRATING advises enrolling into "a new `--voices` file", which fails without `--show` and contradicts the new voice library in the same release | 驳回 |
| 4 | medium | guidance | `voxweave/cli.py:377` | The `--min-speakers` help says to pass it whenever the count is known, which contradicts the README's measured advice | 部分确认 |
| 5 | medium | guidance | `README.md:165` | Docs and error hints say `hf auth login`, but the documented `uv tool install` does not put `hf` on PATH | 部分确认（改判 low） |
| 6 | medium | guidance | `calibration/README.md:65` | The corpus licensing record says only zh-*/ja-* cases are third-party, but every en-* case is too | 部分确认（改判 low） |
| 7 | low | guidance | `README.md:153` | The Apple Silicon alignment stack is described three different ways (README says CoreML/CPU, pyproject says MLX-only / torch fallback, code runs ONNX MMS on CPU) | 部分确认 |
| 8 | low | stale-comment | `README.md:37` | The Hardware note still calls `load_strategy = "sum"` "concurrent" | 确认 |
| 9 | low | guidance | `README.md:225` | Docs say "`process` warns"; there is no `process` command, and the follow-up command lacks its argument | 确认 |
| 10 | low | guidance | `voxweave/cli.py:750` | The translate help shows the output name as `<stem>.<to>.<ext>`, using the hidden legacy option name | 确认 |
| 11 | low | guidance | `voxweave/speakers.py:1538` | Speaker-serve refusal messages recommend the deprecated `--no-match` instead of `--manual` | 确认 |
| 12 | low | stale-comment | `README.md:595` | The render section says "the CLI rename", release-note wording that means nothing in the reference | 确认 |
| 13 | low | guidance | `README.md:168` | Internal milestone jargon "Phase-0 measurements" in the user-facing Setup section | 确认 |
| 14 | low | quality | `README.md:1133` | The align/VTT-forms paragraph sits inside the "Sensitive and derived speaker data" section | 确认 |
| 15 | low | stale-comment | `calibration/README.md:74` | calibration README names a nonexistent `alignment/baseline.json` and a directory map that omits align-shadow/ and p6-oracle/ | 确认 |
| 16 | low | guidance | `calibration/README.md:86` | The baseline-recording command skips the Makefile target and uses a bare `uv run`, which CI warns against | 确认 |
| 17 | low | structure | `REPORT-split-cluster.md:5` | A stale branch hand-off report is committed at the repo root | 确认 |
| 18 | low | guidance | `README.md:355` | The README configuration reference leaves out keys and env vars that CLI help and error messages point users to | 部分确认 |

### media-output

The media-output slice turns finished subtitles into deliverables: mux.py builds and runs the ffmpeg soft-mux (pack) and hard-sub (burn) commands, including the new bitrate cap; export.py and subformats.py convert between VTT, SRT and ASS; translate.py sends cues to an OpenAI-compatible LLM in windows and saves resumable progress; sdh.py places PANNs sound-event tags. Underneath, fsio.py does the atomic writes, artifacts.py holds the per-media cache claims, vocalscache.py handles the separated-vocals cache companions and locks, and mediasnapshot.py makes private media snapshots. The command builders separate probing from execution cleanly and are well tested. But the ffmpeg edge is weak. Filtergraph escaping is single-level, so ASS paths containing `'` or `:` break burn. Non-UTF-8 subtitles that the loader accepts are handed to ffmpeg raw. Existing default subtitle dispositions, mov_text tracks, cover-art streams and the `-o` file extension are not accounted for. The default output name ignores the subtitle language and other existing files, so it can silently overwrite them. fsio creates every deliverable (VTT/SRT/ASS, packed or burned media) with mode 0600, whatever the umask. Translation retries at two layers (ours and the SDK's) with a 600 s SDK timeout. Structurally, the slice's modules reach into the 4.7k-line pipeline.py through deferred imports for small path helpers, and the media lookup has drifted between mux and pipeline. The rest is small test-only leftovers and a few stale or garbled comments.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | high | bug | `voxweave/mux.py:657` | _filter_escape escapes only one level, so an ASS path containing ':' or "'" breaks burn | 未复核 |
| 1 | high | bug | `voxweave/mux.py:320` | pack hands non-UTF-8 subtitles (which the loader accepts) to ffmpeg raw, so their cues are dropped or the run fails | 未复核 |
| 2 | high | robustness | `voxweave/fsio.py:32` | Every deliverable (VTT/SRT/ASS, packed or burned media) is created 0600, and replacing a file downgrades it to 0600 | 未复核 |
| 3 | high | bug | `voxweave/mux.py:235` | The default pack/burn output name ignores the subtitle language and other existing files, so it silently overwrites them | 未复核 |
| 4 | medium | bug | `voxweave/mux.py:204` | resolve_media has drifted from pipeline's subtitle-media lookup: pack/burn cannot find the media for X.sdh.vtt or X.asrfix.vtt | 未复核 |
| 5 | medium | bug | `voxweave/mux.py:700` | burn picks the real (non-cover) video stream for sizing and the bitrate cap, but ffmpeg is told to encode 0:v:0 | 未复核 |
| 6 | medium | bug | `voxweave/mux.py:323` | pack into mkv stream-copies existing mov_text subtitles, which Matroska cannot store (despite "mkv holds everything") | 未复核 |
| 7 | medium | bug | `voxweave/mux.py:369` | pack flags the new track default but leaves an existing default subtitle flagged, so players may still pick the old track | 未复核 |
| 9 | medium | robustness | `voxweave/mux.py:396` | pack and burn take the container from --container/source and ignore the -o extension, which actually picks ffmpeg's muxer | 未复核 |
| 10 | medium | robustness | `voxweave/mux.py:119` | ffmpeg/libass warnings are thrown away when the encode succeeds (burn progress path and pack) | 未复核 |
| 11 | medium | robustness | `voxweave/translate.py:471` | LLM client keeps the SDK's 2 retries and 600 s timeout under our own 3-attempt retry, so a hung endpoint takes up to ~90 min to fail | 未复核 |
| 12 | medium | robustness | `voxweave/subformats.py:243` | Loading any SRT fails if an unrelated artifact-cache marker is unreadable; only the mapping read itself is guarded | 未复核 |
| 13 | medium | robustness | `voxweave/fsio.py:116` | The hard-link fallback in atomic_write_text_new misses ENOTSUP (macOS) and ENOSYS (FUSE mounts) | 未复核 |
| 14 | medium | bug | `voxweave/subformats.py:72` | ASS vector-drawing events (\p1) come out as text cues like 'm 0 0 l 1920 0 …' | 未复核 |
| 15 | medium | bug | `voxweave/export.py:154` | The common SRT position tag {\an8} becomes a literal '(\an8)' in ASS export and burned video; the docstring claims 'raised positioning' support | 未复核 |
| 16 | medium | bug | `voxweave/export.py:104` | export and burn silently drop untimed cues from a partly timed VTT; translate rejects the same input | 未复核 |
| 17 | medium | bug | `voxweave/translate.py:1011` | VOXWEAVE_TRANSLATE_CONTEXT_TAIL=0 sends the whole previous window as context, because win[-0:] is the whole list | 未复核 |
| 19 | medium | structure | `voxweave/artifacts.py:118` | Leaf output modules reach into the 4.7k-line pipeline.py through deferred imports for small path helpers and private functions | 未复核 |
| 8 | low | robustness | `voxweave/mux.py:40` | The pack precheck skips mp4 audio and webm cover art, so both fail as cryptic muxer errors | 未复核 |
| 18 | low | robustness | `voxweave/translate.py:37` | Translate env knobs are parsed with int() at import time, so a bad value crashes every command, even --help | 未复核 |
| 20 | low | dead-code | `voxweave/translate.py:346` | render_translated_vtt is used only by tests | 未复核 |
| 21 | low | dead-code | `voxweave/translate.py:30` | TRANSLATE_MODEL is never read by production code; only a test and a comment refer to it | 未复核 |
| 22 | low | dead-code | `voxweave/vocalscache.py:365` | cache_pair_valid is exported but used only by tests | 未复核 |
| 23 | low | dead-code | `voxweave/artifacts.py:293` | _claim_directory's `except OSError` branch can never run | 未复核 |
| 24 | low | dead-code | `voxweave/mediasnapshot.py:318` | MediaSnapshot.copy_method is set in production but read only by tests | 未复核 |
| 25 | low | stale-comment | `voxweave/sdh.py:61` | The SDH comments say short events are "padded", but the code drops them; the fit_events_to_gaps docstring is garbled | 未复核 |
| 26 | low | stale-comment | `voxweave/mux.py:8` | The mux module docstring says pack and burn drop nothing from the source, which is not true | 未复核 |
| 27 | low | stale-comment | `voxweave/translate.py:880` | The _collapse_units docstring is garbled | 未复核 |
| 28 | low | guidance | `voxweave/translate.py:461` | The missing-openai hint tells users to install an extra that does not exist: voxweave[translate] | 未复核 |
| 29 | low | guidance | `voxweave/translate.py:93` | Untranslated cues are reported by 0-based index, while SRT/VTT cue numbers start at 1 | 未复核 |
| 30 | low | structure | `voxweave/translate.py:646` | Production translate code is shaped around test fakes | 未复核 |
| 31 | low | robustness | `voxweave/mediasnapshot.py:287` | The stale-snapshot sweep can raise a raw OSError, which skips the callers' non-fatal SnapshotUnavailable handling | 未复核 |
| 32 | low | quality | `voxweave/vocalscache.py:166` | The fallback for a missing autocast value is tied to the configurable default, not to the historical "off" the comment describes | 未复核 |
| 33 | low | robustness | `voxweave/fsio.py:40` | Hidden .part temp files (possibly multi-GB burn outputs) survive SIGTERM/SIGKILL and are never swept next to the media | 未复核 |

### p6-distribution

This slice is the P6 "authority" core. align_snapshot freezes the VTT and sibling-JSON inputs into immutable, hashable trees, and its frozen-JSON primitives are also the digest layer for about 19 modules. align_distribution holds the legacy count-slice lane plus a budgeted allocator/verifier that splits every captured unit among the cues; align_distribution_reference is a deliberately duplicated replay of that allocator. align_context issues single-use orchestration contexts and roles; align_failures is the closed failure vocabulary; engine_registry maps language to engine family (every language is legacy-v1); p6_ratifications is a record module. The code is internally consistent and heavily validated, but it has one serious flaw. The allocator's inner loop scans every interval end up to the end of the call, so on a normal-length English (ctc-full) or Japanese (mms-full) episode it burns about 8-13 s of CPU per lane and then always hits the production budget. The mandatory reference replay pays the same cost again. As a result the fresh-alignment evidence lane only ever records 'allocation-budget'. Most of the rest is dead code and test-only scaffolding living in the package: unused context APIs, test-qualification hooks, a fault-injection mutator threaded through production, and a p6_ratifications module that only tests import. There are also checks that can never fail, fields that are computed but never read, failure-registry pairs that no code produces, and a sibling-JSON error hint that names a command that does not exist and would overwrite the user's edited VTT.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | high | bug | `voxweave/align_distribution.py:697` | Authority allocator scans every interval end to the end of the call: quadratic cost always exhausts the production budget on real full-pass episodes | 确认 |
| 1 | medium | guidance | `voxweave/align_snapshot.py:503` | Corrupt-sibling-JSON hint names a non-existent `process` command and, during align, steers the user to overwrite the edited VTT | 确认 |
| 2 | medium | bug | `voxweave/pipeline.py:375` | VTT decode classification: 'vtt-encoding' is unreachable and BOM decode errors escape unclassified | 确认 |
| 3 | medium | dead-code | `voxweave/p6_ratifications.py:78` | p6_ratifications is a test-only module; its *_ENABLED flags gate nothing | 部分确认（改判 low） |
| 4 | low | structure | `voxweave/align_distribution.py:205` | Test-qualification machinery and a verifier fault-injection hook live in the production allocator path | 部分确认 |
| 5 | low | dead-code | `voxweave/align_distribution.py:51` | SCOPE_ORDER unused anywhere; build_authority_distribution used only by tests | 确认 |
| 6 | low | dead-code | `voxweave/align_context.py:390` | Unused context APIs: retire_context_role, verify_context_content (unused anywhere), role_events (tests only), ContextRole alias (unused) | 确认 |
| 7 | low | dead-code | `voxweave/align_context.py:451` | verify_context_roles_terminal can never fail in production: every caller retires all live roles immediately before it | 部分确认 |
| 8 | low | dead-code | `voxweave/align_context.py:462` | verify_context_expected_vtt_generation compares a hash with itself in production | 部分确认 |
| 9 | low | robustness | `voxweave/align_snapshot.py:506` | Sibling snapshot decode costs ~60x a plain json.loads and holds ~160 MB for a 10 MB sidecar | 确认 |
| 10 | low | dead-code | `voxweave/align_snapshot.py:561` | Snapshot fields computed on every align but never read (digest only by tests) | 确认 |
| 11 | low | dead-code | `voxweave/align_failures.py:357` | is_detail_dormant always returns False; used only by a test | 部分确认 |
| 12 | low | quality | `voxweave/align_distribution.py:90` | Registered failure pairs are never produced: AuthorityLimitProfileError and FrozenJSONDomainError carry detail_code but no CanonicalFailure | 部分确认 |
| 13 | low | bug | `voxweave/align_distribution_reference.py:194` | Reference replay rejects a correct zero-call receipt under a test-only profile | 部分确认 |
| 14 | low | robustness | `voxweave/align_context.py:132` | Issued-context registry is never pruned; each record pins the full stable input | 确认 |
| 15 | low | structure | `voxweave/align_snapshot.py:21` | Leaf frozen-JSON digest primitives drag in realign/speakers/voicelibrary via align_snapshot imports | 部分确认 |
| 16 | low | quality | `voxweave/align_distribution.py:1093` | Quadratic bookkeeping: base rows built eagerly then rebuilt, each scanning all claims; route overlap check is O(n^2) | 确认 |
| 17 | low | structure | `voxweave/align_distribution.py:1419` | Legacy-parity lane uses a different no-space language set than the legacy code it mirrors | 部分确认 |
| 18 | low | guidance | `voxweave/align_snapshot.py:572` | ASS-content-in-.vtt error tells align users to rename the file, which align then rejects | 确认 |
| 19 | low | dead-code | `voxweave/align_snapshot.py:127` | Unreachable FROZEN_ABSENT branch in freeze_json | 确认 |

### p6-evidence-A

These three modules hold the P6 fresh-alignment evidence machinery. align_acquisition.py owns the issuer/session that observes every physical backend call, captures and transforms units, seals digests, and keeps module-level registries. align_evidence_core.py builds a "producer" EvidenceCore and a separately written "reference" EvidenceCore and compares them (the ALD-6 gate). align_evidence.py binds, validates and encodes the durable `.align-evidence.json` sidecar, and also has a path verifier that only tests and scripts/p6_oracle_public.py use. All of this runs on the default `voxweave align` path (pipeline.align -> begin/seal_fresh_alignment -> align_orchestration.build_align_selection -> build/project_evidence_core -> bind/encode_align_evidence). It is fail-closed: any disagreement between producer, reference and durable validator aborts the user's re-align, even though no production code ever reads the sidecar. That design turns real validator bugs into crashes. I reproduced two in a scratch harness: a Qwen-route cue that gets skipped after an aligned cue, and a reversed unit from the aligner. The bookkeeping is also costly: with a fake aligner, P6 alone spent about 24 s on 3,000 units and about 56 s on 6,000. The rest is dead or test-only surface and comments that point to an external spec (§5.3/§9/RAT-2/W1/HEAD) that is not in the repo.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | high | bug | `voxweave/align_evidence.py:1319` | Durable validator expects a skip claim's owner_index to equal the delivery index; producer writes the skip ordinal, so Qwen-route align crashes when a skipped cue follows an aligned one | 确认 |
| 1 | medium | bug | `voxweave/align_evidence_core.py:1223` | Reference projector checks the geometry failure before the capture status, the reverse of the producer, so ALD-6 cross-link fails when both are invalid | 部分确认（改判 low） |
| 2 | medium | robustness | `voxweave/align_evidence.py:999` | Durable schema hard-rejects a reversed legacy unit, so one aligner quirk aborts align instead of marking evidence invalid | 部分确认（改判 low） |
| 3 | medium | robustness | `voxweave/align_acquisition.py:1917` | Every _fresh_record access recomputes all seven seal digests over the whole acquisition, costing tens of seconds per align | 部分确认 |
| 4 | medium | guidance | `voxweave/align_evidence.py:370` | Evidence/seal failures reach the user as raw internal codes with no hint | 确认 |
| 5 | medium | robustness | `voxweave/align_evidence.py:529` | Mandatory, fail-closed evidence binding blocks the user's align output although no production code reads the sidecar | 部分确认 |
| 6 | low | quality | `voxweave/align_acquisition.py:1650` | Swallowed exception gives legacy_absolute_digest=None, which every downstream consumer rejects | 部分确认 |
| 7 | low | dead-code | `voxweave/align_acquisition.py:801` | _default_digest and _fresh_evidence_inputs unused anywhere; _fresh_core_inputs used only by a test | 确认 |
| 8 | low | dead-code | `voxweave/align_evidence_core.py:754` | _r_sha is never called, so the reference projector skips the digest-shape checks the producer does | 部分确认 |
| 9 | low | dead-code | `voxweave/align_acquisition.py:1446` | Unused plumbing: begin_fresh_alignment(_verifier_cut_mutator=...) and the per-call digest overrides on _observe_physical_call | 确认 |
| 10 | low | structure | `voxweave/align_acquisition.py:1197` | Test-only instrumentation in production: AcquisitionAdmissionLedger and the default source-facts branch | 部分确认 |
| 11 | low | structure | `voxweave/align_evidence.py:2130` | verify_align_evidence and its helpers (~170 lines) are used only by tests and an oracle script, and lazily import private pipeline helpers | 部分确认 |
| 12 | low | stale-comment | `voxweave/align_evidence.py:3` | Module docstring says the binder receives the independently projected core; it receives the producer core | 驳回 |
| 13 | low | stale-comment | `voxweave/align_acquisition.py:460` | Comments cite an external spec and a moment in git history that are not in the repo (§5.3, §9, section 9.3, W1, HEAD's arithmetic) | 部分确认 |
| 14 | low | robustness | `voxweave/align_acquisition.py:786` | Module-global id()-keyed registries are never pruned and keep full deep copies of every alignment (and the live issuer after a failed align) | 部分确认 |
| 15 | low | dead-code | `voxweave/align_evidence.py:335` | SelectedOutputs.main_json_sha256 is unused; EvidenceCore.core_digest is referenced only as a dead getattr default | 部分确认 |

### p6-orchestration

This slice holds the P6 align orchestration and publication layer. align_orchestration.build_align_selection runs AO-14 to AO-21: the adapter, the mandatory ALD-6 evidence core, the optional v2 comparison, candidate encoding and evidence binding. align_adapter issues context-bound adapter and evaluated results, and runs the W1 finalizer under the shadow flag. align_projector renders VTT and JSON bytes. align_shadow and align_shadow_minimal build the rich and fallback shadow observations. align_delta_registry defines ALD-0 to ALD-6. episode_transaction stages bytes, takes the episode lock, runs the compare-and-swap checks and publishes in order. pipeline.py imports every module lazily inside process, split, align and correct. The exception is the rich shadow builders: only the private `_shadow_observer` argument of `pipeline.align` reaches them, and only scripts/calib_align_shadow.py and tests pass it, so the CLI never does. Shadow isolation for align is careful: W1 and comparator failures turn into invalid v2 statuses when the engine is legacy-v1, and every language maps to legacy-v1. The main weaknesses are in publication and lifecycle. Published outputs always end up mode 0600. A failed cleanup unlink stops the steps that follow, even though the primary files have already landed. The landed and leftover facts are computed for every error but never shown to the user. Long filenames cannot be staged. Registries keyed by id() are never pruned. There is also some test-only or dead scaffolding: the AO phase tuple, the registry metadata fields and a few unreachable branches.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | medium | robustness | `voxweave/episode_transaction.py:390` | Published VTT/JSON/SDH files are always mode 0600, and an existing file's mode is reset | 未复核 |
| 1 | medium | robustness | `voxweave/align_adapter.py:155` | _ADAPTERS / _EVALUATED registries keyed by id() are never pruned (unbounded growth per align) | 未复核 |
| 2 | medium | dead-code | `voxweave/align_shadow.py:524` | Rich/minimal align shadow modules are unreachable from the CLI; with VOXWEAVE_SEG_V2_SHADOW=1 align computes v2 results and throws them away | 未复核 |
| 3 | medium | robustness | `voxweave/episode_transaction.py:742` | A failed cleanup unlink stops the voiceprint sidecar and evidence steps after the primaries have landed | 未复核 |
| 4 | medium | guidance | `voxweave/episode_transaction.py:195` | Partial publication is never shown to the user: the landed and leftover paths exist only as exception attributes that no production code reads | 未复核 |
| 5 | medium | robustness | `voxweave/episode_transaction.py:391` | Staging fails with ENAMETOOLONG for valid output names longer than 240 bytes (about 80 CJK characters) | 未复核 |
| 6 | low | structure | `voxweave/episode_transaction.py:30` | The module docstring says there is no model or renderer dependency, but importing voxweave.speakers pulls in numpy, the renderers and the voice stack | 未复核 |
| 7 | low | dead-code | `voxweave/align_delta_registry.py:18` | AlignDeltaDefinition.primitive_fields and .relation are never read, have drifted from the comparator, and are outside the registry digest | 未复核 |
| 8 | low | quality | `voxweave/align_shadow.py:356` | The lazy-semantic delta set is taken by position (ALIGN_DELTA_IDS[:-1]) instead of by registry phase | 未复核 |
| 9 | low | dead-code | `voxweave/align_orchestration.py:63` | ALIGN_AO_PHASE_ORDER is test/oracle-only data in a production module | 未复核 |
| 10 | low | dead-code | `voxweave/align_orchestration.py:557` | retire_align_selection's AlignSelection branch is never used | 未复核 |
| 11 | low | dead-code | `voxweave/align_adapter.py:744` | Unreachable fallbacks in align_adapter: the getattr default for receipt_digest and a repeated seed_blocks None check | 未复核 |
| 12 | low | robustness | `voxweave/episode_transaction.py:241` | _mapping_stat catches BaseException, which turns Ctrl-C/SystemExit into a mapping observation | 未复核 |
| 13 | low | guidance | `voxweave/episode_transaction.py:229` | CAS rechecks report read errors as 'input/media changed; re-run' | 未复核 |
| 14 | low | robustness | `voxweave/episode_transaction.py:388` | Staged temp files survive a crash with no sweep, and os.replace is never followed by a directory fsync | 未复核 |
| 15 | low | quality | `voxweave/align_orchestration.py:349` | Pointless deferred imports: align_inputs is already a top-level import, and secrets is imported inline | 未复核 |
| 16 | low | guidance | `voxweave/align_orchestration.py:512` | Legacy-candidate and ALD-6 failures abort align with bare, jargon-laden RuntimeErrors that drop their cause | 未复核 |
| 17 | low | guidance | `voxweave/episode_transaction.py:175` | ArtifactCleanupError claims 'JSON/VTT outputs landed' for `correct`, which only writes the VTT | 未复核 |
| 18 | low | structure | `voxweave/episode_transaction.py:632` | commit_primary_outputs(context=None) is an unauthorized commit path used only by tests | 未复核 |

### pipeline

voxweave/pipeline.py (~4.6k lines) is the orchestration god-module. It holds path helpers (swap_ext, MEDIA_EXTS, sibling-media lookup, artifact owner), the vocals cache, P6 failure-annotation helpers, transcribe (~550 lines), segment_document and the manifest, a legacy sibling writer, process/split, align (~660 lines), and the unrelated LLM commands translate/correct. Lower layers (artifacts, mux, export, ui, sdh, subformats, speakers, core/shadow_v2) import back into it through about 25 deferred imports. The worst defects: a vocals-cache hit silently swaps the original-audio 0.25-threshold VAD timing reference for separated-vocals VAD, so re-runs change vad_speech and snapping. And `correct --apply` overwrites the VTT before its automatic align can fail on missing media, losing the diff; the suggested --media re-run is a no-op. There is also sizeable test-only code: the _write_siblings/_dump_sibling_json writer chain and the exception-annotation plumbing on process/split/correct. Two issues seen at HEAD are already fixed in the uncommitted working tree and are not reported: the `voxweave[songdet]` extra hint and the README `VOXWEAVE_MIN_CUE_SEC` default of 0.8. Line numbers are for the current working tree, which another process was editing during the audit.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | high | bug | `voxweave/pipeline.py:1385` | Vocals-cache hit silently replaces the original-audio sensitive VAD timing reference with separated-vocals VAD | 确认 |
| 1 | high | bug | `voxweave/pipeline.py:4579` | correct --apply commits the rewritten VTT before the auto-align, so a missing-media failure loses the diff, and the suggested fix is a no-op | 确认（改判 medium） |
| 2 | medium | robustness | `voxweave/pipeline.py:3287` | Bound (voiceprint) align cache miss rewrites the vocals cache without a companion, so every later bound align re-separates | 确认 |
| 3 | medium | robustness | `voxweave/pipeline.py:819` | _separate_to_16k_32k self-cleanup catches Exception only, so Ctrl-C during separation leaks the full-band temp WAV | 确认 |
| 4 | medium | robustness | `voxweave/pipeline.py:132` | Env knobs parsed with bare float()/int() at import time: one malformed value crashes every CLI command, including --help | 部分确认（改判 low） |
| 5 | medium | dead-code | `voxweave/pipeline.py:1805` | The _write_siblings / _dump_sibling_json / _sibling_json_data / _persistable_cue / _UNPERSISTED_CUE_KEYS writer chain is used only by tests, and its docstrings claim it is the production writer | 确认 |
| 6 | medium | dead-code | `voxweave/pipeline.py:393` | Exception-annotation plumbing on process/split/correct (failure, landed, auxiliary_landed, panns_release_exception, secondary_exceptions) has no production reader | 部分确认（改判 low） |
| 7 | medium | structure | `voxweave/pipeline.py:678` | mux.resolve_media re-implements subtitle-to-media lookup and has drifted from _find_subtitle_media: pack/burn cannot find media for .sdh.vtt / .asrfix.vtt | 确认 |
| 8 | medium | structure | `voxweave/pipeline.py:621` | Leaf and core modules import back into the orchestration god-module through about 25 deferred imports, only to reach path and parsing helpers | 部分确认 |
| 9 | medium | structure | `voxweave/pipeline.py:992` | God module and functions: transcribe (~550 lines) and align (~660 lines) mix unrelated stages, and the vocals-cache logic is duplicated and has drifted | 确认 |
| 10 | medium | bug | `voxweave/pipeline.py:2231` | _reconcile_word_segment_language reads only unit['text'], so `word`-keyed units silently skip the language repair | 确认 |
| 11 | low | bug | `voxweave/pipeline.py:3531` | Per-cue (zh/yue Qwen) alignment progress bar stays frozen until every cue is aligned, then jumps to 100% | 确认 |
| 12 | low | robustness | `voxweave/pipeline.py:3092` | render/split on a sibling with non-object word_segments entries crashes with a bare AttributeError | 确认 |
| 13 | low | robustness | `voxweave/pipeline.py:588` | Missing ffprobe or a probe timeout is reported as an unreadable vocals cache and forces re-separation on every run | 确认 |
| 14 | low | robustness | `voxweave/pipeline.py:1530` | Unguarded backend.release() in transcribe/align finally blocks can mask the original error and skip temp-file cleanup | 部分确认 |
| 15 | low | dead-code | `voxweave/pipeline.py:1931` | SegmentationResult.thresholds_used is never read by production code (only tests) | 部分确认 |
| 16 | low | quality | `voxweave/pipeline.py:3316` | _retain_qwen_owner_slice is an identity function kept as a test monkeypatch seam, and the `invoker is None` branches in _align_blocks are test-only | 部分确认 |
| 17 | low | stale-comment | `voxweave/pipeline.py:1657` | mark_lyric_cues docstring names the dead `_write_siblings` as the VTT display layer | 确认 |
| 18 | low | stale-comment | `voxweave/pipeline.py:906` | _load_cues docstring claims it is shared by align/translate/correct; only translate uses it | 确认 |
| 19 | low | stale-comment | `voxweave/pipeline.py:3613` | align() docstring and comments describe only the per-cue Qwen routing path and cite a non-existent 'memory' document | 确认 |
| 20 | low | stale-comment | `voxweave/pipeline.py:2000` | segment_document 'exactly what production runs' pass list omits the stranded-tail repair step | 确认 |
| 21 | low | guidance | `voxweave/pipeline.py:842` | Corrupt-sibling errors tell users to 're-run transcribe/process', but there is no `process` command | 部分确认 |
| 22 | low | guidance | `voxweave/pipeline.py:1366` | User-visible warning uses internal jargon ('aligner set', 'smart_split') | 部分确认 |
| 23 | low | structure | `voxweave/pipeline.py:152` | Align-stage cue-duration constants are defined twice, and realign's defaults have drifted from pipeline's | 驳回 |
| 24 | low | structure | `voxweave/pipeline.py:3075` | split() decodes the same sibling bytes with two different decoders; _load_sibling_json_bytes duplicates align_snapshot's decoder | 部分确认 |
| 25 | low | structure | `voxweave/pipeline.py:1672` | lyric_display_text is the canonical lyric wrap, but the same f-string is re-implemented in 8 other places | 部分确认 |

### repo-hygiene

The repo-hygiene slice covers packaging (pyproject.toml, MANIFEST.in, overrides.txt, uv.lock), the developer Makefile, GitHub CI, release and dependabot workflows, .gitignore, the tracked debug/ reports, tests/conftest.py and the vendor package docstring. The build works: sdist and wheel build cleanly from a git archive, `uv lock --check` passes, and every action tag referenced exists (checkout@v7, upload-artifact@v7, download-artifact@v8, setup-uv@v7). Most problems sit between these files and how the project is really used, not inside the code. The documented PyPI install skips the onnxruntime override that the Makefile applies. The release publishes to PyPI with no tests. The sdist leaves out license notices and ships tests without their conftest. The test suite fails outright (instead of skipping) on any host that differs from the P6 oracle's recorded environment, and conftest lets documented VOXWEAVE_* env knobs leak into tests. Many comments in the Makefile, pyproject, CI and conftest describe behaviour that has since changed: MLX alignment routing, the ruff config, the whisper engine on mps, the cache layout and gate sample counts. .gitignore ignores `*.lock` and `debug/`, yet uv.lock (whose digest the oracle pins) and the debug/ reports are tracked.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | medium | guidance | `pyproject.toml:133` | PyPI install path never gets the onnxruntime override, so CPU onnxruntime can shadow onnxruntime-gpu for [cuda] users | 部分确认 |
| 1 | medium | robustness | `Makefile:70` | `make dev`/`make test` claim to match CI, but the suite hard-fails outside the oracle's exact interpreter, platform and full git history | 确认 |
| 2 | medium | robustness | `.github/workflows/release.yml:8` | Tag push publishes to PyPI without running tests or requiring green CI | 确认 |
| 3 | medium | quality | `MANIFEST.in:1` | Vendored MIT code ships without its license notice; THIRD_PARTY_NOTICES.md is neither packaged nor complete | 部分确认（改判 low） |
| 4 | low | robustness | `tests/conftest.py:23` | conftest isolates only a few VOXWEAVE_* variables; documented user knobs leak into the suite and break it | 确认 |
| 5 | low | stale-comment | `tests/conftest.py:10` | _isolate_voxweave_cache docstring claims episode artifacts follow VOXWEAVE_CACHE_ROOT; they now live beside the media | 确认 |
| 6 | low | robustness | `.gitignore:7` | `*.lock` ignores the tracked, digest-pinned uv.lock, and no workflow uses --locked | 部分确认 |
| 7 | low | quality | `.gitignore:10` | `debug/` is ignored, yet ten reports under debug/ are force-tracked and present superseded behaviour as current | 驳回 |
| 8 | low | stale-comment | `pyproject.toml:102` | [mps]/[cuda]/core dependency comments contradict the actual alignment routing, cache location and CLI flag names | 部分确认 |
| 9 | low | stale-comment | `Makefile:10` | Makefile header contradicts its own auto-detection and the [mps] whisper engine | 驳回 |
| 10 | low | stale-comment | `Makefile:81` | lint target comment says the repo has no ruff config | 驳回 |
| 11 | low | guidance | `Makefile:57` | install/reinstall report a clean git state when changes are staged or untracked, and fail when the tool bin dir is not on PATH | 确认 |
| 12 | low | guidance | `Makefile:56` | Install commands need `uv tool install --torch-backend` (uv >= 0.9.19), but no minimum uv version is stated or checked | 部分确认 |
| 13 | low | robustness | `MANIFEST.in:1` | sdist ships tests/test_*.py but not conftest.py, scenarios, scripts/ or calibration/, so a test run from the sdist is broken and not isolated | 驳回 |
| 14 | low | stale-comment | `.github/workflows/ci.yml:62` | Segmentation gate comment has a stale sample count and misstates per-language promotion | 驳回 |
| 15 | low | dead-code | `.github/workflows/ci.yml:83` | 'Check for a captured corpus' step and its `if:` guards can no longer be false | 驳回 |
| 16 | low | robustness | `.github/workflows/ci.yml:45` | Unpinned ruff in `ruff format --check` lets a ruff style release break CI on untouched code; jobs have no timeout or token scope | 部分确认 |
| 17 | low | stale-comment | `voxweave/vendor/__init__.py:1` | vendor package docstring and ruff comment describe vendor/ as only the Mel-Band RoFormer | 驳回 |
| 18 | low | guidance | `pyproject.toml:8` | PyPI long description uses relative links, so the logo and MIGRATING.md migration guide are broken on PyPI | 驳回 |

### scripts-alignment

This slice holds three calibration CLIs. `calib_alignment.py` (2.8k lines) is the alignment accuracy harness. It pairs segments using text only, via n-gram anchors and a monotonic DP, discovers subtitle tracks with ffprobe, builds per-lane metrics and gates reports against a baseline (inspect-tracks/report/check/record-baseline). `calib_align_shadow.py` runs the P6 align-shadow corpus: each case variant runs as a monkeypatched subprocess, and the result is compared to a byte-exact baseline. `mfa_to_word_segments.py` converts MFA TextGrids into alignment-reference JSON, with its own Praat parser, shard placement, OOV classification and provenance. Overall the code is careful and well commented, but the edges are weak in four ways. Several paths break the shared 0/1/2 exit-code contract: uncaught exceptions exit 1, and a baseline-header mismatch in the shadow harness also exits 1. The baseline/filter logic lets a filtered report quietly remove gates from other lanes. The MFA converter drifts from the loader checks it claims to mirror. And a few hints and messages point users to flags that do not exist or leave out what actually failed. Dead code is limited to unused manifest fields and a DP retry that can never run. The fault-injection machinery in the shadow harness is reachable only from tests.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | medium | bug | `scripts/mfa_to_word_segments.py:850` | OOV phone check pools phone tiers of all speakers, so following the tool's own `--tier` advice turns spn-realized OOV words into truth samples | 确认 |
| 1 | medium | bug | `scripts/calib_alignment.py:2716` | A report filtered with --item/--source keeps the full-manifest digest, so it can be recorded as a baseline that silently ungates every other lane | 确认 |
| 2 | medium | robustness | `scripts/calib_alignment.py:2417` | Malformed baseline JSON crashes with KeyError, and the traceback exits 1, which is the 'gate regressed' code | 部分确认（改判 low） |
| 3 | medium | guidance | `scripts/mfa_to_word_segments.py:777` | Missing-provenance hint tells users to pass `--tool-version`, which does not exist (the flag is `--mfa-version`) | 确认 |
| 4 | medium | robustness | `scripts/mfa_to_word_segments.py:790` | Schema-valid or slightly wrong provenance values crash with an uncaught TypeError/ValueError (exit 1), although the converter 'never exits 1' | 确认 |
| 5 | medium | bug | `scripts/mfa_to_word_segments.py:943` | Converter writes references with a negative offset_s that the alignment loader then rejects: the 'loader's own checks' are only partly mirrored | 确认 |
| 6 | medium | guidance | `scripts/calib_align_shadow.py:274` | Environment pin failure says only 'runtime environment does not match the manifest' and never names which field differs | 部分确认（改判 low） |
| 7 | medium | guidance | `scripts/calib_align_shadow.py:1020` | `check` exits 1 silently, and a baseline header mismatch (changed manifest/schema digest) is reported as 1 instead of 2 | 部分确认（改判 low） |
| 8 | low | structure | `scripts/calib_alignment.py:2602` | inspect-tracks re-implements track selection and has drifted: it skips the Japanese script-ratio floor and the ambiguity check | 确认 |
| 9 | low | quality | `scripts/calib_alignment.py:161` | Banded DP is claimed to 'never degrade in correctness', but it returns suboptimal pairings, and the unbanded retry meant to guarantee this can never run | 确认 |
| 10 | low | guidance | `scripts/calib_alignment.py:2675` | `check --report R` never checks R against --manifest and silently ignores --source/--item/--pairs/--pairs-limit | 部分确认 |
| 11 | low | dead-code | `scripts/calib_alignment.py:1055` | Manifest fields parsed or accepted but never consumed (ItemSpec.tags/media, ItemOutcome.excluded_reference_segments, hypothesis.*_units/alignment_health) | 部分确认 |
| 12 | low | robustness | `scripts/calib_alignment.py:440` | Rows with null start/end are dropped silently and never counted, which shrinks the coverage denominator | 确认 |
| 13 | low | guidance | `scripts/calib_alignment.py:1313` | Cue-lane hypothesis rejects voxweave's own sibling JSON (cues are stored under `segments`), and the error does not point to the subtitle file | 确认（改判 medium） |
| 14 | low | guidance | `calibration/README.md:35` | README and report schema describe lanes keyed by reference_id, an `evaluate` subcommand, and an alignment/baseline.json, none of which exist | 确认 |
| 15 | low | stale-comment | `scripts/calib_alignment.py:115` | Comments cite 'design 3.x' sections of a design document that is not in the repository | 确认 |
| 16 | low | robustness | `scripts/mfa_to_word_segments.py:336` | TextGrid count 'inf'/'nan' crashes `_Values.integer` with an uncaught OverflowError/ValueError (exit 1) | 确认 |
| 17 | low | quality | `scripts/mfa_to_word_segments.py:803` | provenance.command silently records the converter's own argv when --mfa-command is omitted, while the flag defines it as the `mfa align` command | 确认 |
| 18 | low | guidance | `scripts/calib_align_shadow.py:1028` | Harness error messages drop the path and reason (only the exception type name, or a generic 'JSON input is unavailable') | 确认 |
| 19 | low | guidance | `scripts/calib_align_shadow.py:989` | --help shows the hidden worker as '_worker ==SUPPRESS==', leaks 'P6' jargon, and documents no option | 确认 |
| 20 | low | structure | `scripts/calib_align_shadow.py:1004` | Fault-injection machinery in the CLI is reachable only from tests via a private `main(_injections=...)` keyword | 驳回 |
| 21 | low | robustness | `scripts/calib_align_shadow.py:128` | Report JSON is written non-atomically, and the harness duplicates calib_common I/O instead of reusing it | 确认 |

### scripts-oracle

This slice holds three kinds of dev tooling. The first is the detached P6 oracle: p6_oracle.py (about 3.1k lines, which validates, compares and runs source gates), its isolated subprocess worker p6_oracle_public.py, the shared execution-pin helpers in p6_oracle_environment.py, and the release-refresh writer. The second is the voiceprint threshold calibrator (calibrate_voiceprints.py). The third is the song-skip / segmentation fixture capturer (capture_scenario.py). The oracle is carefully fail-closed and its monkeypatch seam targets all still exist in production. Its weak spots are a dependency gate that misses the codebase's dominant `from voxweave import X` / relative import forms, an exit-code contract (1 = mismatch, 2 = invalid) that tooling failures break, environment-mismatch errors that name neither the expected nor the observed value, and a release-refresh path that refuses lock-only changes, which pushes maintainers to hand-edit digests. It also has a few dead parameters and a mandatory `--check` flag that does nothing. The two calibration/capture scripts work on the happy path but leak large temp WAVs, silently overwrite hand-annotated golden fixtures, and crash with tracebacks (losing results) after expensive work. Their segment and PANNs logic are copies of production code that have drifted.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | medium | bug | `scripts/p6_oracle.py:2885` | Dependency gate (_imports) misses `from voxweave import X` and relative imports, so layering violations pass silently | 部分确认（改判 low） |
| 1 | medium | robustness | `scripts/p6_oracle.py:3057` | Tooling failures exit 1 ('mismatch') instead of the documented 2 ('invalid') | 部分确认（改判 low） |
| 2 | medium | guidance | `scripts/p6_oracle.py:540` | Environment-pin errors name neither the recorded nor the observed value, and `make quality-p6-oracle` does not select the pinned interpreter | 部分确认（改判 low） |
| 3 | medium | guidance | `scripts/p6_oracle_release_refresh.py:191` | The 'one sanctioned' refresh refuses dependency-only lock changes, so maintainers hand-edit digests | 部分确认（改判 low） |
| 4 | medium | robustness | `scripts/capture_scenario.py:629` | capture_songdet never deletes the temp WAV/FLAC files that decode_to_wav and separate_vocals hand to the caller | 部分确认（改判 low） |
| 5 | medium | robustness | `scripts/capture_scenario.py:704` | Song-skip fixture silently overwrites an existing tests/scenarios/<name>.json and discards its hand-filled speech_present_at anchors | 部分确认（改判 low） |
| 6 | medium | robustness | `scripts/calibrate_voiceprints.py:479` | Failures after argument parsing crash with a traceback, contradicting 'Exit codes: 0 / 2', and all scored models are lost | 部分确认 |
| 7 | medium | structure | `scripts/calibrate_voiceprints.py:96` | calibration_segments is a drifted copy of voiceembed.centroid_segments (no short-turn fallback) but the report claims the production recipe | 部分确认（改判 low） |
| 8 | medium | guidance | `scripts/calibrate_voiceprints.py:234` | --models errors advertise choices the script rejects, and --help lists no valid model names | 部分确认（改判 low） |
| 9 | low | bug | `scripts/calibrate_voiceprints.py:456` | --max-segments accepts 0 and negative values and silently changes the populations | 确认 |
| 10 | low | robustness | `scripts/calibrate_voiceprints.py:282` | pyannote embed() re-implements voiceembed._embed_span without its out-of-audio check and fails with a cryptic error | 确认 |
| 11 | low | robustness | `scripts/calibrate_voiceprints.py:353` | Vocals cache is used without the duration-freshness check production applies | 确认 |
| 12 | low | robustness | `scripts/calibrate_voiceprints.py:479` | All episodes are decoded up front and held in memory for the whole run | 部分确认 |
| 13 | low | structure | `scripts/capture_scenario.py:648` | PANNs window scoring is copied from songdet instead of calling songdet.window_probs, and crashes on sub-2 s audio | 确认 |
| 14 | low | robustness | `scripts/capture_scenario.py:636` | Scenario capture reuses the vocals cache without the freshness check and creates cache claim state as a side effect | 确认 |
| 15 | low | guidance | `scripts/p6_oracle.py:3030` | `--check` is a required store_true flag that nothing reads, and subcommands have no help | 部分确认 |
| 16 | low | stale-comment | `scripts/p6_oracle.py:1858` | _public_artifact_path keeps a dead cache_root parameter with a justification that no longer holds | 确认 |
| 17 | low | dead-code | `scripts/p6_oracle.py:2974` | _source_gates' optional manifest_path default and its fallback are never used | 确认 |
| 18 | low | quality | `scripts/p6_oracle.py:2272` | Scenario phase table duplicated outside EXPECTED_RUNTIME_SCENARIOS; KeyError makes the 'no validator' else-branch unreachable | 部分确认 |
| 19 | low | structure | `scripts/p6_oracle.py:2028` | _execute_runtime_scenario duplicates _execute_public_case's sandbox/subprocess harness and borrows cases[0]'s environment | 部分确认 |
| 20 | low | quality | `scripts/p6_oracle.py:1026` | _load_delivery silently takes the first basename match, while its error and sibling _case_input require exactly one | 部分确认 |
| 21 | low | structure | `scripts/p6_oracle.py:263` | Three copies of the file-hash / canonical-digest helpers feed the same pinned digests | 部分确认 |

### scripts-segmentation

scripts/calib_common.py (611 lines) holds the shared calibration primitives: the 0/1/2 exit-code contract and run_cli, canonical JSON and digests, atomic write_json, JSON-schema helpers, language-tag canonicalization, the type-7 percentile, metric blocks and micro-aggregation. scripts/calib_segmentation.py (6.6k lines) is one god-module with two jobs. The first is the v1 quality ruler (validate-corpus, evaluate, record-baseline and the legacy compare-video-dir): it loads the corpus, replays each case through pipeline.segment_document, maps cue boundaries back onto source units, computes the four micro-aggregated metrics and runs one-sided baseline gates. The second is a large P5 shadow harness (lane/row matrix, N1/N3/N6/N14 gates, coarse family, OAT ablation, perturbation probes). Both files are generally careful: numerators and denominators are kept, exits are explicit, and many invariants are checked. The most serious defect is in the boundary mapper. UnitLocator.locate_last resolves a cue ending in several zero-duration units that share one timestamp (e.g. '啊' + reinjected '，') to the first of them. That hides the punctuation, so 60 punctuated boundaries on the tracked corpus land in the forbidden_end_rate denominator. They are 55 of ja's 145 samples, and without them ja drops below min_samples (100), so they are what promotes the ja gate to blocking. Other issues: the baseline workflow resets the tracked gate policy to all-warning, and its hints send users into a loop. `evaluate` without --check prints status=pass beside failures, and gates are vacuous without a baseline. Several paths exit 1 (reads as a regression) on malformed input. There is also some dead or test-only helper code and a few stale comments.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | high | bug | `scripts/calib_segmentation.py:908` | UnitLocator.locate_last picks the FIRST of several same-timestamp zero-duration units, so punctuated boundaries are misclassified and the ja forbidden_end gate is wrongly promoted | 确认 |
| 1 | medium | bug | `scripts/calib_segmentation.py:1951` | record-baseline copies the report's gate table, which is DEFAULT_GATES (all 'warning') on the only path that can produce a recordable report after a corpus/metric change | 部分确认（改判 low） |
| 2 | medium | guidance | `scripts/calib_segmentation.py:1917` | Baseline-mismatch hint 'make quality-record-segmentation' cannot work: it needs a fresh report that `make quality-segmentation` refuses to produce | 确认 |
| 3 | medium | guidance | `scripts/calib_segmentation.py:2334` | `evaluate` without --check prints `status=pass` next to a non-zero failure count (shadow explicitly avoids this) | 部分确认（改判 low） |
| 4 | medium | guidance | `scripts/calib_segmentation.py:6490` | `--baseline` help says 'omitted = absolute gates only', but without a baseline no gate is enforced at all, and the Makefile silently drops it when the file is missing | 部分确认（改判 low） |
| 5 | medium | robustness | `scripts/calib_common.py:129` | run_cli only maps CalibrationError to exit 2; malformed-but-parseable input crashes with a traceback and exit 1 ("quality gate failed") | 部分确认（改判 low） |
| 6 | low | robustness | `scripts/calib_common.py:175` | read_json does not catch UnicodeDecodeError despite promising CalibrationError 'on any read/parse problem' | 确认 |
| 7 | low | guidance | `scripts/calib_common.py:261` | schema_errors sorts least-relevant first and then truncates, so the most relevant errors are the ones suppressed; the suppression notice is also added when nothing was suppressed | 确认 |
| 8 | low | guidance | `scripts/calib_segmentation.py:58` | Module docstring (also the --help text) says the last stdout line is 'always' a machine summary, but every exit-2 path via CalibrationError prints none | 确认 |
| 9 | low | dead-code | `scripts/calib_segmentation.py:316` | metric_definition_digest() is unused anywhere; the four call sites inline the same digest | 部分确认 |
| 10 | low | dead-code | `scripts/calib_common.py:395` | Exported helpers that production never calls: is_calibration_language (unused anywhere); exit_code, merge_ratios, MicroAggregator.groups/metrics (tests only) | 确认 |
| 11 | low | dead-code | `scripts/calib_segmentation.py:3862` | load_coarse_manifest's `except OSError` is unreachable, so its 'coarse corpus not found' message never appears | 确认 |
| 12 | low | structure | `scripts/calib_segmentation.py:4501` | Coarse-gate p90 uses a private nearest-rank percentile, contradicting the pinned 'one percentile definition' contract | 部分确认 |
| 13 | low | stale-comment | `scripts/calib_segmentation.py:1878` | load_baseline docstring claims it rejects segmenter-version mismatches; it does not | 确认 |
| 14 | low | stale-comment | `scripts/calib_segmentation.py:999` | _ZERO_SHAPE_WARN comment says it 'mirrors' ZERO_DURATION_MAX_RUN and that runs this long survived the repair; the repair still handles runs of exactly 8, and its runs are counted differently | 确认 |
| 15 | low | guidance | `scripts/calib_segmentation.py:2097` | `evaluate --private` silently does nothing when VOXWEAVE_CALIB_ROOT is unset or its segmentation/corpus.json is missing | 确认 |
| 16 | low | guidance | `scripts/calib_segmentation.py:6593` | --perturb-max-probes help misstates semantics: a cap does not always mark the run non-exhaustive, it applies per case, negatives are accepted, and perturbation options are silently ignored without --perturb | 确认 |
| 17 | low | guidance | `scripts/calib_segmentation.py:5984` | Shadow summary mislabels N3a rows as 'v2 vs v1 baseline' and prints the P1/P2/P3 counts in inconsistent orders | 确认 |
| 18 | low | guidance | `scripts/calib_common.py:222` | Missing-jsonschema hint tells users to install the CUDA inference stack (wrong on Apple Silicon), contradicting the module's 'bare environment' promise | 部分确认 |
| 19 | low | quality | `scripts/calib_segmentation.py:1404` | len_break_mid_phrase_rate gate for spaced languages can never fail but is shown as an 'ok' blocking gate | 确认 |
| 20 | low | quality | `scripts/calib_segmentation.py:3536` | N3b expressed-rate gate threshold is an unexplained magic literal | 确认 |
| 21 | low | guidance | `scripts/calib_segmentation.py:6561` | CLI help and error messages lean on design-doc identifiers (P5, AD-2, AD3-2, N7, N19, W1, C13) whose defining documents are not in the repo | 部分确认 |

### seg-adapters

This slice connects production segmentation (process/split) to the P6 candidate/authority machinery. segmentation_orchestration.build_segmentation_selection is reachable from production: pipeline.py:2860/3117 import it lazily on every `process` and `split`, and it drives segmentation_adapter, segmentation_candidates, segmentation_projector and reference_projector. The same files also carry the align-side candidate_encoder and reference projection, plus the core helpers (canonical_text, subunit refiner, authority ledger, align_seed, align_compare, providers, policy_delta). Most of the logic is careful and heavily self-checking, and that checking causes the main problems:
- The default path deep-copies, re-digests and deep-compares the full delivery several times. I measured about 13–17 s of mostly bookkeeping for a 20k-word document.
- Selection errors lose their root cause, so the user sees only an internal-jargon error.
- The segmentation adapter has drifted from the align adapter. It never builds v2 unless the shadow env var is set, so the documented one-edit engine-registry cutover would break process/split.
- The subunit conservation check is quadratic.
- align_compare crashes on the very shape mismatch it tries to report.
The rest is test-only hooks living in production (simulated-row qualification, semantic_comparison_available, LINEAGE_FIELDS), an unreachable validation branch, an unused helper, registries that are never pruned, duplicated encoder skeletons, and a few stale docstrings (RAT-1 pending, CANONICAL_PASS_FACTOR arithmetic, BudouX coverage).

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | medium | bug | `voxweave/segmentation_adapter.py:511` | Segmentation adapter builds v2 only when the shadow env is on, so the documented engine-registry cutover makes process/split fail | 确认 |
| 1 | medium | robustness | `voxweave/candidate_encoder.py:298` | Selection throws away the candidate failure and the original exception; users get only 'selected-render-invalid/renderer/selected-candidate-missing' | 部分确认 |
| 2 | medium | robustness | `voxweave/segmentation_adapter.py:256` | Default process/split path spends ~10 s per 20k-word file on deepcopy snapshots and repeated delivery digests | 部分确认（改判 low） |
| 3 | medium | robustness | `voxweave/core/subunit.py:607` | assert_refinement_conserved is O(parents × units) and runs 2–5 times per document on the shadow path | 部分确认（改判 low） |
| 4 | medium | bug | `voxweave/core/align_compare.py:786` | compare_semantic_deltas crashes with ValueError on the shape mismatch it is meant to report | 部分确认（改判 low） |
| 5 | low | dead-code | `voxweave/segmentation_adapter.py:243` | Unreachable v2 range validation; its comment describes a check that never runs | 确认 |
| 6 | low | dead-code | `voxweave/segmentation_candidates.py:89` | Simulated-row qualification machinery is test-only code in production that bypasses the registry family gate | 部分确认 |
| 7 | low | dead-code | `voxweave/segmentation_orchestration.py:50` | _swap_ext is never called | 确认 |
| 8 | low | stale-comment | `voxweave/core/canonical_text.py:73` | CANONICAL_PASS_FACTOR claims an 'exact worst case' of 6 passes; canonical_text charges 7, and the constant is only used by a test | 部分确认 |
| 9 | low | stale-comment | `voxweave/core/align_seed.py:3` | Module docstring still says RAT-1 is pending and there are two authority kinds | 确认 |
| 10 | low | stale-comment | `voxweave/core/providers.py:165` | _atoms_slot docstring says th/lo/my have BudouX; the loader returns None for them | 确认 |
| 11 | low | robustness | `voxweave/segmentation_adapter.py:121` | Issuance registries are never pruned and pin full documents and deliveries for the life of the process | 部分确认 |
| 12 | low | structure | `voxweave/segmentation_candidates.py:124` | Segmentation encoder duplicates candidate_encoder's skeleton, imports its privates, and has already drifted | 部分确认 |
| 13 | low | dead-code | `voxweave/core/align_compare.py:49` | semantic_comparison_available() is a constant True used only by a test | 部分确认 |
| 14 | low | dead-code | `voxweave/core/authority.py:64` | LINEAGE_FIELDS claims to keep the probe and tests in sync but the probe doesn't use it; Seal.to_dict is unused | 确认 |
| 15 | low | dead-code | `voxweave/segmentation_adapter.py:258` | IssuedLegacySegmentation.provider_ledger is built, deep-copied and tamper-checked, but never consumed | 确认 |
| 16 | low | quality | `voxweave/segmentation_adapter.py:387` | v1-adoption refusal is mislabeled as a 'delivery-unit-range' failure | 部分确认 |

### seg-core

seg-core is the shipped v1 subtitle engine: layout.py (width/wrap primitives), smart_split.py (sentence/clause planning, atom packing, bound-particle repair and the smart_split_segments orchestrator), timing.py (micro-merge, glue, cleanup, shot snap), the leaf tables (kinsoku, breakpoints, gap_split, conjunctions, langsets, unit_repair), and the P3-P5 records around it (schema, segdoc, timing_preview and the 2000-line finalizer). The core is carefully written, but I confirmed three output bugs on the default path. `_vis_width` counts every non-ASCII character as double width, so Russian and accented Latin text gets half its line budget. `_cleanup_cues` can extend a cue right up to the next cue's start and leave no gap. The ja particle table was built from a string, so it penalizes the single characters で/よ/り/ま. Production always passes thresholds and word timings, so the legacy thresholds=None path, the split_sentence_heuristically/conjunctions chain and the min_duration/desired_wps parameters are only reached by tests. The main structural costs are the cleanup and shot-snap rules restated in four places (timing, timing_preview, finalizer, trace_validator), and the finalizer reaching up into private helpers in align_acquisition and boundary_v2. Several docstrings also no longer match the code: core/__init__, langsets, segdoc.SourceUnit, the smart_split module docstring and the timing_preview seam.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | high | bug | `voxweave/core/layout.py:246` | _vis_width charges every non-ASCII char as double width, halving line budgets for Russian and accented Latin languages | 确认 |
| 1 | medium | bug | `voxweave/core/timing.py:355` | Cleanup extension clamps to next_start, leaving back-to-back cues with no gap that chaining cannot restore | 部分确认 |
| 2 | medium | bug | `voxweave/core/kinsoku.py:43` | _BIND_END_MED is built from a string, so it penalizes the single characters で/よ/り/ま, not まで/より | 确认 |
| 3 | medium | guidance | `voxweave/core/smart_split.py:666` | Every Portuguese, Korean or Cantonese run logs a WARNING the user can do nothing about | 确认 |
| 4 | medium | dead-code | `voxweave/core/smart_split.py:1924` | Legacy thresholds=None mode and the whole split_sentence_heuristically/conjunctions chain are only reached by tests | 确认 |
| 5 | medium | bug | `voxweave/core/smart_split.py:1398` | An untimed oversized atom borrows the parent cue span, producing overlapping cues and fabricated speech anchors | 确认 |
| 6 | low | dead-code | `voxweave/core/smart_split.py:1752` | min_duration/desired_wps parameters and DEFAULT_MIN_DURATION are never read | 确认 |
| 7 | low | dead-code | `voxweave/core/timing.py:28` | VISIBLE_GAP_MIN_S is unused, and the 'visible gaps are left untouched' claim is false | 确认 |
| 8 | low | dead-code | `voxweave/core/schema.py:38` | schema.Atom TypedDict is unused anywhere and is missing keys the engine writes | 确认 |
| 9 | low | stale-comment | `voxweave/core/__init__.py:1` | Package docstring lists 7 of 29 modules and calls core 'pure logic, no models' while it imports up into pipeline, diarize and align_acquisition | 确认 |
| 10 | low | stale-comment | `voxweave/core/langsets.py:3` | langsets rationale is stale: no circular import exists and a second, different no-space set is used by the pipeline joiner | 确认 |
| 11 | low | stale-comment | `voxweave/core/segdoc.py:67` | SourceUnit docstring says provenance/confidence are never read, but several modules read provenance | 确认 |
| 12 | low | stale-comment | `voxweave/core/smart_split.py:9` | smart_split docstrings describe a pipeline that no longer matches what ships | 确认 |
| 13 | low | stale-comment | `voxweave/core/timing_preview.py:31` | 'The cost model does not change a line' is contradicted by boundary_cost type-switching on LegacyCleanupPreview | 部分确认 |
| 14 | low | structure | `voxweave/core/timing_preview.py:172` | The per-cue cleanup and shot-snap rules are restated four times (timing, timing_preview, finalizer, trace_validator) | 驳回 |
| 15 | low | structure | `voxweave/core/finalizer.py:1859` | finalizer reaches into private helpers of a higher layer, and several of its deferred imports protect no import cycle | 部分确认 |
| 16 | low | quality | `voxweave/core/finalizer.py:1258` | finalize() takes a profile separate from the sealed stream.profile and never checks they match | 确认 |
| 17 | low | quality | `voxweave/core/unit_repair.py:317` | unit_repair reads unit surfaces with a different accessor than the engine | 部分确认 |
| 18 | low | quality | `voxweave/core/layout.py:375` | Two-line wrap passes single CJK characters to a scorer built for whole words, so 目的 is penalized like the particle 的 | 部分确认（改判 medium） |
| 19 | low | stale-comment | `voxweave/core/smart_split.py:1104` | _attach_end_penalties claims mid-phrase atoms are never break candidates, but the emergency path makes every edge a candidate | 确认 |

### seg-v2

This slice is the v2 boundary optimizer, which runs as a shadow. boundary_lattice.py enumerates hard-legal cues. It builds the atom layer, barriers, intervals and coalescing, and handles cap, relief and granularity. It admits edges through either IncrementalPacker or canonical FinalText. boundary_cost.py prices each cut and each edge (pause ramp, layout, reading). boundary_v2.py runs an exact per-interval DP, applies the v1-margin selection policy, computes pinned-neighbour margins, falls back to v1 cues, and assembles the artifact. None of this reaches users by default. Every language maps to legacy-v1 in engine_registry, so optimize_document only runs when VOXWEAVE_SEG_V2_SHADOW=1, called from shadow_v2 (three solves per document) and segmentation_adapter. Every production caller passes speakers and a speaker_weight, so the policy-1, packer and LegacyCleanupPreview paths are exercised only by tests and scripts. The core DP is correct and deterministic: no infinite costs, a stable tuple-sort tie-break, and empty documents are handled. The real problems are in the code around it. An adopted v1 fallback can overlap its neighbour. Cap nodes exposed after the relief rescan are dropped. granularity_check is not the lower bound it claims to be. The v1 reference is scored under different legality and pricing rules than v2. Aggregate breakdowns silently lose every feature. Two artifact-only passes take quadratic time. And a whole predecessor-state DP subsystem, plus the runner-up machinery, can never run in production.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | medium | bug | `voxweave/core/boundary_v2.py:1319` | Adopted-v1 fallback that expands backwards overlaps the preceding optimized interval, duplicating units and creating a false exit-driving v2 violation | 确认 |
| 1 | medium | bug | `voxweave/core/boundary_lattice.py:2075` | Cap nodes exposed by the post-relief rescan are ignored, and the granularity-widened node set is discarded, producing false relief-insufficient fallbacks | 确认 |
| 2 | medium | bug | `voxweave/core/boundary_lattice.py:1512` | granularity_check is not the 'strict lower bound' it claims: it charges a separator at line breaks, so legal multi-line cues are typed coarse-granularity | 部分确认（改判 low） |
| 3 | medium | bug | `voxweave/core/boundary_v2.py:1135` | score_v1_global judges and prices v1 under different rules than the v2 lattice and tables in the production (speaker/P5) mode | 部分确认（改判 low） |
| 4 | medium | bug | `voxweave/core/boundary_cost.py:385` | sum_breakdowns drops every feature missing from any part, so aggregates of edges plus cuts (every PathResult.breakdown and V1Reference.global_cost) have empty features | 部分确认（改判 low） |
| 5 | medium | bug | `voxweave/core/boundary_cost.py:273` | pause_evidence labels zero-length or overlapping gaps as vad_state 'silence' even when VAD shows speech there | 部分确认 |
| 6 | medium | robustness | `voxweave/core/boundary_v2.py:946` | _pinned_neighbour_margins is quadratic per interval and dominates optimizer runtime | 确认 |
| 9 | medium | dead-code | `voxweave/core/boundary_v2.py:769` | The whole predecessor-state DP subsystem is unreachable: no edge can ever be evidence_deferred | 确认 |
| 7 | low | robustness | `voxweave/core/boundary_v2.py:1466` | _resolve_selected_path rebuilds edges_from with an O(nodes x edges) nested scan | 确认 |
| 8 | low | robustness | `voxweave/core/boundary_v2.py:325` | sentence_cross_count scans every document sentence node for every edge (O(edges x sentences) across the whole document) | 确认 |
| 10 | low | dead-code | `voxweave/core/boundary_v2.py:182` | CostTables.speaker_pricing_refused is write-only | 部分确认 |
| 11 | low | dead-code | `voxweave/core/boundary_v2.py:746` | Runner-up path is computed for every interval and discarded; only tests read it, and the module docstring still advertises it | 确认 |
| 12 | low | dead-code | `voxweave/core/boundary_lattice.py:995` | IncrementalPacker's no-space branches are unreachable and untested; _lang is never read | 确认 |
| 13 | low | stale-comment | `voxweave/core/boundary_cost.py:509` | LAYOUT_SOURCES is unused anywhere and omits the 'preview-final-text' source that edge_cost actually emits | 部分确认 |
| 14 | low | dead-code | `voxweave/core/boundary_lattice.py:128` | Reason-vocabulary constants are test-only or unused | 驳回 |
| 15 | low | dead-code | `voxweave/core/boundary_lattice.py:1620` | Unused to_dict serializers (SpanViolation, HardBarrier, HardInterval, IntervalLattice, DPResult) | 确认 |
| 16 | low | stale-comment | `voxweave/core/boundary_v2.py:2137` | shadow_artifact claims to be 'The one call the Wave B hook makes' but no production code calls it; the default policy-1 path has no live caller | 驳回 |
| 17 | low | stale-comment | `voxweave/core/boundary_v2.py:1957` | Comment cites a module docstring statement that does not exist | 驳回 |
| 18 | low | stale-comment | `voxweave/core/boundary_lattice.py:1599` | cap_relief_nodes comment says relief_injections is the layout (C16) rescue, but relief_injections also counts duration splits | 未复核 |
| 19 | low | robustness | `voxweave/core/boundary_lattice.py:265` | preflight_profile lets NaN or infinite knobs through: NaN clause_ms crashes, NaN max_cue_s silently disables the cap | 未复核 |
| 20 | low | bug | `voxweave/core/boundary_lattice.py:1843` | Packer-mode edge scan breaks on over-cap at a non-candidate node before the duration ladder runs, so a held multi-word unit gets no-path instead of a held-chain waiver | 未复核 |
| 21 | low | structure | `voxweave/core/boundary_lattice.py:2148` | Evidence-span resolution is duplicated between the lattice and the solver and has drifted; speaker policy lives in the 'hard-legal' module via a deferred import | 未复核 |
| 22 | low | quality | `voxweave/core/boundary_cost.py:320` | pause_cut_cost ignores the evidence's own uncertainty_ms and relies on keyword defaults bound at definition time | 未复核 |
| 23 | low | quality | `voxweave/core/boundary_lattice.py:2094` | An interval that fails only because a run of words is untimed is typed as a generic no-path or relief-insufficient | 未复核 |
| 24 | low | quality | `voxweave/core/boundary_v2.py:1503` | When no v1 reference is supplied, an infeasible interval's missing coverage is attributed to 'v1' and made non-exit-driving | 未复核 |
| 25 | low | bug | `voxweave/core/boundary_v2.py:1122` | score_v1_global silently drops v1 cuts that land inside the last atom and misaligns v1.cues after rounding | 未复核 |

### shadow-v2

The slice is the opt-in BoundaryOptimizer v2 measurement lane. pipeline.segment_document calls _maybe_shadow_v2, which checks VOXWEAVE_SEG_V2_SHADOW == "1" before running shadow_v2.run_shadow. shadow_v2._shadow_v2_artifact is one ~900-line function: it runs three optimizations, three or four finalizations, speaker measurement and a refiner counterfactual, then must pass the closed schema-2 contract in shadow_schema.validate_shadow_v2_payload. That validator re-derives rows with partition_check.check_partition and trace_validator.replay_trace/stability_check. The lane is isolated from production: it gets a nested quiet degradation capture, works on deep copies, and run_shadow turns any exception into an error block. Isolation held in every probe I ran. The main problems are elsewhere. (1) The refiner-off row is written in parent-unit coordinates but checked against refined units, so any refined document whose v1 stream projects always fails admission. I reproduced this. (2) replay_trace never checks that a leg's target is the boundary its rule acts on. I built a forged fixed-point trace that it accepts. (3) Any failed admission throws away the whole assembled artifact and leaves only a generic error, so the typed invalid-row fields can never be persisted. Everything else is lower-risk: a very large function with duplicated branches, lane names defined in three places, vocabulary constants and a parameter used only by tests, stale exit-driving docstrings, and error text that hides the actionable cause or grows without bound.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | high | bug | `voxweave/core/shadow_v2.py:1627` | refiner-off finalizer row is serialized in parent-unit space but validated against refined units, so schema-2 admission always fails for refined documents | 未复核 |
| 1 | medium | bug | `voxweave/core/trace_validator.py:614` | replay_trace never checks that a leg's target/slot is the boundary its rule acts on, so a forged leg can move the wrong boundary and pass | 未复核 |
| 2 | medium | robustness | `voxweave/core/shadow_v2.py:1748` | Any failed schema-2 admission throws away the whole assembled artifact; typed invalid-row states (budget exhaustion, measurement refusal) can never be persisted | 未复核 |
| 3 | medium | guidance | `voxweave/core/shadow_v2.py:1096` | Invalid display profile is reported as 'optimizer profile preflight failed' with the offending key hidden; the harness's dedicated AD3-2 message is unreachable | 未复核 |
| 6 | medium | structure | `voxweave/core/shadow_v2.py:844` | _shadow_v2_artifact is a ~900-line function with a nested class and an incomplete branch that duplicates the complete tail | 未复核 |
| 4 | low | robustness | `voxweave/core/shadow_v2.py:1789` | Shadow failure is logged without a traceback and with a message that grows without bound | 未复核 |
| 5 | low | quality | `voxweave/core/shadow_v2.py:1366` | Incomplete-artifact reasons exist only as free text that the harness matches verbatim | 未复核 |
| 7 | low | structure | `voxweave/core/shadow_v2.py:1116` | Raw-stage validator computed twice with different attribution rules; the shadow copy overwrites boundary_v2's | 未复核 |
| 8 | low | structure | `voxweave/core/shadow_schema.py:22` | Lane-name vocabulary is defined three times and shadow_schema's copy is pinned by no test | 未复核 |
| 9 | low | structure | `voxweave/core/shadow_v2.py:321` | Core-layer module depends on private helpers of the top-level pipeline, a cycle kept alive by deferred imports | 未复核 |
| 10 | low | quality | `voxweave/core/shadow_v2.py:1784` | run_shadow re-validates an artifact that was just validated, and gates it on a hard-coded literal 2 | 未复核 |
| 11 | low | quality | `voxweave/core/shadow_v2.py:491` | Finalizer v1 row is labelled 'solver-partition' although its partition comes from surface projection | 未复核 |
| 12 | low | stale-comment | `voxweave/core/partition_check.py:177` | Docstrings say only raw/core violations are exit-driving; finalizer is exit-driving too | 未复核 |
| 13 | low | dead-code | `voxweave/core/partition_check.py:55` | ORIGINS, STAGES and VIOLATION_KINDS are used only by tests; the VIOLATION_KINDS comment claims an effect it does not have | 未复核 |
| 14 | low | dead-code | `voxweave/core/partition_check.py:340` | check_partition's expect_no_overlap parameter is used only by a test | 未复核 |
| 15 | low | dead-code | `voxweave/core/shadow_v2.py:609` | _shadow_diff_classification's optional stream/seed_cues and its `fact is None` branches cannot be reached | 未复核 |
| 16 | low | dead-code | `voxweave/core/shadow_v2.py:58` | SHADOW_LANE_DELIVERY compatibility alias is read only by tests | 未复核 |
| 17 | low | robustness | `voxweave/core/trace_validator.py:513` | Malformed cycle evidence crashes the validator with IndexError instead of producing an error | 未复核 |
| 18 | low | quality | `voxweave/core/shadow_v2.py:106` | v1 projection failure mode blames origin translation, which cannot fail | 未复核 |
| 19 | low | quality | `voxweave/core/shadow_v2.py:1783` | Incomplete envelopes keep empty placeholder ledgers inside `diagnostic` while the real ones go only to the top level | 未复核 |

### songdet-shot-debug

This slice has five parts. voxweave/songdet.py loads the PANNs Cnn14 model, scores sliding windows and holds the pure song-skip interval logic (merge, expand, excise, group, rescue); the pipeline's plan_song_skip, sdh.py and speakers.py all import it. voxweave/shotdet.py runs the background ffmpeg scene-score job used by process(). voxweave/debug.py holds the no-op and file-writing debug sinks. scripts/song_scores.py is a diagnostic CLI, and experiments/song_detect.py is an old spike. The pure interval algorithms are well tested and read correctly, and the shotdet job lifecycle is careful. The real problems are all in the PANNs integration. The library prints to stdout, which breaks the documented stdout-only-result-paths rule. It ignores VOXWEAVE_DEVICE and MPS. It holds every overlapping window in memory at once. The label CSV bootstrap is fragile (no timeout, an existence-only check, and --sdh can bypass it). A malformed tuning env var crashes every CLI command at import. There is also misleading guidance (a non-existent voxweave[songdet] extra, wrong gate formulas in song_scores.py), dead or test-only code (experiments/, detect_shot_changes, parts of the DebugSink API, a raw-output dump that just copies the text dump), and a few stale docstrings.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | medium | bug | `voxweave/songdet.py:118` | PANNs AudioTagging prints checkpoint path / GPU count to stdout, breaking the 'stdout = result paths only' contract | 确认 |
| 1 | medium | bug | `voxweave/songdet.py:115` | PANNs device ignores VOXWEAVE_DEVICE and never uses MPS (README says it does); DataParallel spreads it over every GPU | 确认 |
| 2 | medium | robustness | `voxweave/songdet.py:562` | window_probs materialises every overlapping 2 s window at once (about 2x the decoded audio size in extra RAM) | 部分确认（改判 low） |
| 3 | medium | robustness | `voxweave/songdet.py:102` | PANNs label bootstrap: urlopen has no timeout, any existing file is trusted forever, write is non-atomic, and errors carry no context | 部分确认 |
| 4 | medium | bug | `voxweave/sdh.py:148` | --sdh imports panns_inference before songdet pre-places the labels CSV, so it hits panns' wget bootstrap that songdet exists to avoid | 部分确认 |
| 5 | medium | robustness | `voxweave/songdet.py:57` | Malformed VOXWEAVE_SONG_CORE_MERGE_SEC / VOXWEAVE_SPEECH_RESCUE_MIN_S crashes every CLI command (even --help) at import | 部分确认（改判 low） |
| 6 | medium | guidance | `voxweave/pipeline.py:1204` | Song-detection warning tells users to install a non-existent `voxweave[songdet]` extra | 部分确认（改判 low） |
| 7 | medium | guidance | `scripts/song_scores.py:76` | song_scores diagnostic prints a gate formula that is not what song_flags computes, so its miss labels point at the wrong gate | 确认（改判 low） |
| 8 | medium | dead-code | `experiments/song_detect.py:1` | experiments/song_detect.py is unreferenced, superseded, has drifted from production and does not run on a fresh host | 部分确认（改判 low） |
| 9 | low | stale-comment | `experiments/song_detect.py:6` | Docstring advertises a `separate` mode that does not exist | 确认 |
| 10 | low | stale-comment | `voxweave/debug.py:102` | Debug sink claims to save raw ASR output with markers, but the only caller passes raw=text, so .raw.txt just duplicates .text.txt | 确认 |
| 11 | low | dead-code | `voxweave/debug.py:53` | DebugSink.enabled / base-class root and FileDebugSink's stem+base constructor mode are only exercised by tests | 部分确认 |
| 12 | low | robustness | `voxweave/debug.py:131` | The fixed debug directory is reused across runs without clearing, so stale chunk and health files mix with the new run | 确认 |
| 13 | low | dead-code | `voxweave/shotdet.py:234` | detect_shot_changes is test-only, yet the module docstring presents it as the entry point | 确认 |
| 14 | low | stale-comment | `voxweave/songdet.py:388` | expand_spans_to_voiced_blocks docstring defines the song core by block_gap, but the code clusters by SONG_CORE_MERGE_SEC | 确认 |
| 15 | low | stale-comment | `voxweave/songdet.py:584` | detect_song_spans docstring says song spans are 'used to drop VAD segments', but segments are now excised, not dropped | 确认 |
| 16 | low | robustness | `scripts/song_scores.py:57` | Read-only diagnostic claims (creates) the media's artifact cache dir and can crash instead of falling back to the raw mix | 确认 |
| 17 | low | robustness | `scripts/song_scores.py:27` | _slice_32k leaks the mkstemp fd, leaks the temp file when ffmpeg fails, and hides ffmpeg's error | 确认 |
| 18 | low | quality | `scripts/song_scores.py:65` | Diagnostic scores a window grid that is offset from the pipeline's, so it may not reproduce the miss it is diagnosing | 确认 |
| 19 | low | guidance | `voxweave/shotdet.py:39` | Undocumented VOXWEAVE_SHOT_SCENE knob silently ignores invalid or out-of-range values | 确认 |
| 20 | low | quality | `voxweave/songdet.py:42` | Comments rely on undefined 'pit-2 protection' jargon | 确认 |
| 21 | low | structure | `voxweave/songdet.py:320` | Generic interval algebra lives in the PANNs module and the union loop is copy-pasted four times | 部分确认 |
| 22 | low | structure | `voxweave/songdet.py:597` | detect_song_spans' span derivation is not factored out, so scripts/capture_scenario.py re-implements it and window_probs by hand | 部分确认 |
| 23 | low | robustness | `voxweave/songdet.py:556` | Sample-rate validation of PANNs input relies on assert | 部分确认 |

### speakers

This slice covers the speaker-labelling workflow. speakers.py (2.3k lines) builds the in-memory audition page, selects clips, reads and writes the speaker-name mapping, renders voice tags, enrolls voices into the store or library, and purges voiceprints. speakerserve.py serves the page over ThreadingHTTPServer with /save, /split, /split-confirm and /split-undo. ngrok.py discovers tunnel origins from the local agent. core/speaker_evidence.py is the deterministic speaker attribution, pricing and lineage module used by the P5 boundary optimizer. The loopback-default server is carefully hardened: Host/Origin checks against DNS rebinding, a CSRF token, body caps, strict JSON, a lock around every action, atomic writes, and a guarded one-level undo with rollback. Split, confirm and undo also check that the transcript, voiceprints and media did not change in the meantime. The main problems:
- `--host 0.0.0.0` and `--ngrok` expose every route, including the one that hands out the token (`/serve-info`), with no authentication.
- There are three mapping parsers whose rules have drifted apart. One of them makes the page unusable after a re-diarization that drops a speaker id.
- The undo feature the README advertises cannot be reached from the page or the CLI.
- Several error messages point users to the wrong fix: `--ngrok` for a hostname problem, the deprecated `--no-match`, recapturing voiceprints after a purge.
- Smaller items: dead implicit-store branches from before the voice library, dead write-only state, an unhandled OSError in /save, and no socket timeouts.
speaker_evidence.py is clean apart from write-only fields and a no-op filter.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | high | robustness | `voxweave/speakerserve.py:238` | Security: --host 0.0.0.0 / --ngrok give anyone who can reach the port unauthenticated read/write access (the CSRF token is handed out by /serve-info) | 部分确认（改判 medium） |
| 1 | high | bug | `voxweave/speakerserve.py:1346` | Audition page cannot save or split when the saved mapping still has an id that is gone after a re-diarization (serve-info returns 500) | 确认 |
| 2 | medium | guidance | `README.md:405` | README advertises a one-level split undo, but neither the page nor any CLI command can trigger /split-undo | 确认 |
| 3 | medium | guidance | `voxweave/speakers.py:1531` | After `voxweave speakers purge`, a plain `speakers serve` refuses and tells the user to re-capture the biometrics they just deleted (or use a deprecated flag) | 确认 |
| 4 | medium | guidance | `voxweave/speakerserve.py:223` | 403 for an unknown Host always suggests --ngrok, even for localhost or a LAN hostname | 确认 |
| 5 | medium | structure | `voxweave/speakerserve.py:886` | Three mapping parsers with drifted validation rules (speakers._mapping_entries_bytes, speakerserve._validated_speakers, speakerserve._mapping_document) | 部分确认（改判 low） |
| 6 | low | robustness | `voxweave/speakerserve.py:342` | /save does not handle OSError: client gets a dropped connection and the terminal gets a raw traceback | 确认 |
| 7 | low | robustness | `voxweave/speakerserve.py:151` | No socket timeout on request handling: every idle or slow connection pins a thread forever | 部分确认 |
| 8 | low | guidance | `voxweave/speakers.py:1341` | Page labels a failed session load as 'Save failed' and enables a Save button that can only 403 | 确认 |
| 9 | low | quality | `voxweave/speakerserve.py:679` | Split preview clips skip the 'clean snippet' filtering the main page applies (overlapping speakers, non-speech, singing) | 部分确认 |
| 10 | low | guidance | `voxweave/speakers.py:1538` | Error messages recommend the deprecated hidden `--no-match` instead of `--manual` | 确认 |
| 11 | low | guidance | `voxweave/speakerserve.py:354` | Suggested follow-up commands print unquoted paths (and a bare file name without its directory) | 确认 |
| 12 | low | dead-code | `voxweave/speakers.py:143` | Implicit ('discovered') voices-store branches are unreachable: both callers always pass an explicit `voices` path | 确认 |
| 13 | low | dead-code | `voxweave/speakerserve.py:350` | `SpeakerHTTPServer.mapping_path` is written after save/split/undo but never read | 部分确认 |
| 14 | low | dead-code | `voxweave/speakers.py:648` | `load_speaker_mapping` (path variant) is only used by tests | 确认 |
| 15 | low | dead-code | `voxweave/core/speaker_evidence.py:347` | `RawSpeakerEvent.left_label` / `right_label` are write-only fields | 确认 |
| 16 | low | dead-code | `voxweave/core/speaker_evidence.py:1425` | measure_speaker_events: `raw_by_id` membership filter is always true, and `_lineage`'s third return value is discarded | 确认 |
| 17 | low | structure | `voxweave/speakers.py:1559` | speakers <-> pipeline import cycle worked around with 14 deferred imports and private cross-module helpers | 部分确认 |

### tests-hygiene

tests/ (about 105k lines, 161 test modules plus conftest and scenario fixtures). The suite is broad and mostly well written: it relies on conftest isolation fixtures, fake pipelines and injected seams, and it has many source-law and contract tests for the P6 subsystem. The biggest health problems are in environment isolation, not assertions. (1) About 12 diarize tests make real HuggingFace HTTP requests through diarize._build_provenance. Offline, each one burns about 24s in hub retry backoff, which accounts for most of the 6+ minute runtime. (2) Two P6 oracle tests shell out to `uv run --extra cuda` and `make ... VARIANT=cuda`. That re-syncs the developer's venv and can rewrite uv.lock. (3) The oracle's closed-env public-command worker runs the real CLI with HOME and VOXWEAVE_CONFIG unset. It therefore creates, reads, or even migrates (renames) the developer's real ~/.config/voxweave.conf; this is confirmed by a byte-identical template that appeared in /root/.config during this session. Several sizeable production code paths now exist only to be tested: the old sibling writer cluster in pipeline.py, the single-chunk backend.transcribe_align path, the whole p6_ratifications module, and a handful of P6 helpers. Smaller issues: stale comments, guards that silently turn assertions into no-ops, a reap test that cannot fail (verified by mutation), importorskip on hard dependencies, isolation fixtures duplicated across files, and multiprocessing and subprocess waits with no timeout. Line numbers refer to the audit base 94b6ff5, which matches the current HEAD for every file cited except where noted.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | high | robustness | `tests/test_warning_hygiene.py:123` | ~12 diarize tests hit huggingface.co for real (fake pipelines lack _OUTER_CONFIG_ATTR); ~24s each offline | 未复核 |
| 1 | high | robustness | `tests/test_p6_fix_oracle_rat4.py:253` | Tests shell out to `uv run --extra cuda` / `make ... VARIANT=cuda`, re-syncing the dev venv and possibly uv.lock | 未复核 |
| 2 | high | robustness | `tests/test_p6_oracle_public_commands.py:57` | Oracle public-command worker runs the real CLI with HOME/VOXWEAVE_CONFIG unset; tests create, read or migrate the developer's ~/.config | 未复核 |
| 3 | medium | dead-code | `tests/test_pipeline_vad.py:13` | Old sibling writer cluster (pipeline._write_siblings & co., ~185 lines) is production-dead and kept alive only by tests | 未复核 |
| 4 | medium | dead-code | `tests/test_backend.py:53` | backend.transcribe_align and its single-chunk engines are unreachable from production; ~20 tests exercise only them | 未复核 |
| 5 | medium | dead-code | `tests/test_p6_ratification_defaults.py:34` | voxweave/p6_ratifications.py is imported only by tests; its six *_ENABLED 'switches' gate nothing | 未复核 |
| 6 | low | dead-code | `tests/test_p6_context.py:56` | Several production helpers exist only to be called by tests | 未复核 |
| 7 | low | structure | `tests/test_p6_distribution.py:87` | Test-only qualification hooks live in production modules and are consulted on every align | 未复核 |
| 8 | low | stale-comment | `tests/test_backend.py:1054` | Comment says the default routes to fusion; the test passes asr_model='fusion' and the default is Qwen | 未复核 |
| 9 | low | stale-comment | `tests/test_subunit.py:823` | Docstring/comment reference a probe_prop_error.py script that does not exist | 未复核 |
| 10 | low | stale-comment | `tests/test_shadow_hook.py:182` | Docstring cites a `--no-<flag>` CLI option for the shadow env var that does not exist | 未复核 |
| 11 | low | quality | `tests/test_shot_snap.py:407` | Reap test polls the child itself, so it passes even if the drain thread's reap is deleted | 未复核 |
| 12 | low | quality | `tests/test_p6_fix_authority.py:243` | Existence guards turn assertions and resets into silent no-ops if a name is renamed | 未复核 |
| 13 | low | quality | `tests/test_p6_fix_authority.py:88` | Negative source-substring assertions check names that exist nowhere and cannot fail meaningfully | 未复核 |
| 14 | low | structure | `tests/test_cli.py:11` | Conftest isolation re-implemented in ~10 per-file fixtures; script loaders copy-pasted 9x | 未复核 |
| 15 | low | structure | `tests/test_p5_close.py:15` | Test modules import helpers from other test modules; importorskip cascades silently | 未复核 |
| 16 | low | robustness | `tests/test_voicestore.py:279` | Multiprocessing and subprocess waits have no timeout; a child failure hangs the whole suite | 未复核 |
| 17 | low | quality | `tests/test_realign.py:542` | importorskip used for hard dependencies and first-party code turns breakage into silent skips | 未复核 |
| 18 | low | structure | `tests/test_scenarios.py:36` | Scenario replay re-implements detect_song_spans' span derivation; golden check vacuous if the key is missing | 未复核 |
| 19 | low | quality | `tests/test_cli.py:131` | CLI and removal tests drive the hidden deprecated `split` alias instead of the canonical `render` command | 未复核 |

### voice-library

The voice-library slice covers three layers. voicebase holds strict JSON, vector and ID primitives plus atomic writes. voicestore holds the legacy per-show store, its flock and the shared indexed-enrollment relation. voicelibrary is the global library: identities.json, per-space vector files, history.jsonl, a library-wide flock, compare-and-swap commits, orphan healing, import and forget. voicematch covers compatibility fingerprints, thresholds, two-tier matching and suggestion records. voiceembed and voiceembed_models handle pinned checkpoint download and verification, the resident embedder and the centroid recipe. voiceepisode provides the per-episode lock. Both voicelibrary and voiceembed_models are reachable in production: speakers.py imports voicelibrary at module level and cli_voices imports it lazily; voiceembed's registered loaders import voiceembed_models lazily. The code is careful overall: strict validation, atomic fsync+rename writes, CAS before each rename, SHA-256 and size pins checked before torch.load. The main problems are four. First, the orphan auto-healing turns a missing or rolled-back identities.json into silent mass deletion of every stored voice vector. Second, lock and permission handling has drifted between the per-show store and the library: read-only stores are handled in one place but not the other, and a pre-created shared directory is narrowed to 0600 anyway. Third, the shared enrollment relation gives user-facing advice to use a deprecated or nonexistent flag (--replace-episode) and shows internal scoped keys. Fourth, there is a layering inversion (voicestore lazily imports the top-level speakers module) plus a cluster of test-only APIs and dead result fields, including the evicted_exemplar_id fields raised in the focus notes.

| # | 严重度 | 类别 | 位置 | 发现 | 复核 |
|---|---|---|---|---|---|
| 0 | high | robustness | `voxweave/voicelibrary.py:750` | Missing/rolled-back identities.json makes the next write delete every stored voice vector (orphan auto-heal) | 确认 |
| 1 | medium | guidance | `voxweave/voicestore.py:479` | Enrollment and import refusals tell users to 'use --replace-episode' (deprecated hidden flag; absent on `voices import`) and show internal scoped keys | 确认 |
| 2 | medium | robustness | `voxweave/voicestore.py:322` | Per-show store lock needs write access and ownership even for a shared (read) lock; the library's copy of the same logic tolerates both | 确认 |
| 3 | medium | robustness | `voxweave/voicelibrary.py:811` | A pre-created shared library directory is narrowed anyway: lock fchmod'ed to 0600 on every use, all files and spaces/ created private | 部分确认（改判 low） |
| 4 | medium | guidance | `voxweave/voicematch.py:179` | VOXWEAVE_VOICES_ACCEPT below the per-embedder suggest default silently disables all matching for that embedder; errors don't name the vars; empty values handled inconsistently | 确认 |
| 5 | low | guidance | `voxweave/voicelibrary.py:2193` | A legacy store whose samples `voices import` refuses stays flagged 'unimported' forever, so serve keeps advising an import that cannot succeed | 确认 |
| 6 | low | guidance | `voxweave/voicematch.py:510` | LEGACY_DIARIZE_HINT tells users to run `speakers` with --diarize-model, which no speakers subcommand accepts | 驳回 |
| 7 | low | structure | `voxweave/voicestore.py:130` | Low-level store validator lazily imports the top-level speakers module (layering inversion); the justifying comment is stale | 确认 |
| 8 | low | structure | `voxweave/voiceepisode.py:96` | _episode_owner duplicates pipeline._artifact_owner (and its MEDIA_EXTS / {asrfix,sdh} lists) with a different precedence | 确认 |
| 9 | low | structure | `voxweave/voicelibrary.py:1134` | ID minting/time helpers copied verbatim between voicestore and voicelibrary; matching limits hardcoded instead of shared constants | 部分确认 |
| 10 | low | dead-code | `voxweave/voicestore.py:84` | evicted_exemplar_id (and the other outcome fields) are never read by production; evictions are never reported to the user | 确认 |
| 11 | low | dead-code | `voxweave/voicebase.py:531` | Production functions used only by tests: write_voiceprints, voiceprint_conjunction_valid, script_json (voicebase), suggest_bytes (voicematch) | 确认 |
| 12 | low | dead-code | `voxweave/voiceepisode.py:46` | episode_lock_path / _lock_path_for_owner are test-only; the yielded EpisodeLockHandle is never consumed | 部分确认 |
| 13 | low | dead-code | `voxweave/voicematch.py:140` | SpeakerMatch.top_identity_id / top_similarity are computed but never read or serialized | 驳回 |
| 14 | low | dead-code | `voxweave/voicelibrary.py:351` | Unreachable unknown-lane branch in space_model_slug; 'split' history action is never written | 驳回 |
| 15 | low | quality | `voxweave/voicebase.py:255` | No-op try/except in load_json_object | 驳回 |
| 16 | low | stale-comment | `voxweave/voiceembed.py:3` | voiceembed docstrings describe the pre-library/pre-cache layout and a nonexistent --lang flag | 驳回 |
