# P6 detached oracle corpus

This directory is immutable comparison data for `scripts/p6_oracle.py`. The
`inputs`, `media`, and `backend-receipts` trees describe reproducible inputs,
while `expected` contains the approved reference sets. The detached runner
projects the sole candidate authority in memory from those inputs and receipts. It also
executes each recorded public command in an isolated clean source root and compares the
bytes read from that command directly with the standalone projection. CI comparison is
read-only and has no rerecording path or checked-in candidate-output tree.

The selected-v2 reference sets contain complete VTT and main-JSON primaries. Align
cases also contain the complete closed RAT-2 evidence sidecar, including physical-call
geometry and selected-primary hash links. The combined case is a separate full golden,
not a textual delta mask. Public output has no parallel golden authority: every artifact
must equal its case's independent standalone projection byte for byte. Every declared
matrix vector names executable pytest evidence; `source-gates --check` executes the
deduplicated set. Acoustic quality calibration remains owned by `scripts/calib_alignment.py`.

Run the complete oracle gate program through the locale-pinned entry point:

```text
make quality-p6-oracle VARIANT=cuda
```

That target runs validation, byte comparison, and source coverage with the exact
recorded environment and writes the comparison report outside this immutable corpus.
The runner itself deliberately retains its strict environment check, so invoking it
directly under an arbitrary shell locale may return the documented invalid-environment
exit code.

For a package release, first update `pyproject.toml` and `uv.lock`, then use the one
sanctioned execution-record refresh procedure:

```text
env -i PATH="$PATH" LANG=zh_CN.UTF-8 LC_ALL=C.UTF-8 uv run --extra cuda --locked python scripts/p6_oracle_release_refresh.py
```

It requires the project, editable lock row, and installed distribution versions to agree,
then updates only `execution.package_version`, `execution.dependency_lock_sha256`, and
`execution.container_digest`. It refuses a dirty expected tree, any pre-existing manifest
edit, any non-execution write plan, or any projected/public byte change beyond the recorded
version. The selected-v2 segmentation golden stores a fixed version sentinel; the same
standalone projector emits both that immutable template and the execution-version bytes,
so a release never rerecords expected artifacts.
