"""``voxweave.core.shadow_v2`` holds the v2 shadow lane; ``segmentation`` keeps the hook.

The move is a seam, and a seam only holds if both sides agree on it:

* the flag and lane names ``pipeline`` re-exports are the very objects
  ``shadow_v2`` defines, so a reader of either module sees one truth;
* ``shadow_v2`` never imports ``pipeline`` or ``segmentation`` -- both import
  it eagerly for the constants, so the reverse edge would be a cycle; the
  overlays it replays come from the ``core.overlay`` leaf;
* the hook resolves the lane from ``shadow_v2``'s globals at call time, so a
  monkeypatch on ``shadow_v2`` is what a run actually executes, and a flag-off
  run never reaches the lane at all.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

from tests.test_shadow_hook import _case_plain, _segment
from voxweave import pipeline
from voxweave.core import shadow_v2
from voxweave.core.segdoc import THRESHOLD_KEYS

RE_EXPORTS = (
    "SEG_V2_SHADOW_ENV",
    "SHADOW_LANE_CORE",
    "SHADOW_LANE_DELIVERY_LEGACY",
    "SHADOW_LANE_FINALIZER",
    "SHADOW_LANE_LEGACY_DISPLAY",
)


def test_pipeline_re_exports_the_lane_constants_by_identity() -> None:
    for name in RE_EXPORTS:
        assert getattr(pipeline, name) is getattr(shadow_v2, name), name
    # The P4 compatibility alias is retired: one name per lane.
    assert not hasattr(shadow_v2, "SHADOW_LANE_DELIVERY")
    assert not hasattr(pipeline, "SHADOW_LANE_DELIVERY")


def _module_scope_imports(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    """Every import that runs (or is type-checked) at module scope, ``if`` bodies included."""
    found: list[ast.Import | ast.ImportFrom] = []
    pending: list[ast.stmt] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append(node)
        elif isinstance(node, (ast.If, ast.Try)):
            pending.extend(ast.iter_child_nodes(node))  # type: ignore[arg-type]
    return found


def test_shadow_v2_never_imports_pipeline_at_module_scope() -> None:
    tree = ast.parse(inspect.getsource(shadow_v2))
    for node in _module_scope_imports(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            names = [node.module or ""]
        assert not any(
            name == "voxweave.pipeline" or name.startswith("voxweave.pipeline.")
            for name in names
        ), ast.dump(node)
        if isinstance(node, ast.ImportFrom) and node.module == "voxweave":
            assert "pipeline" not in {alias.name for alias in node.names}


def test_shadow_v2_imports_neither_pipeline_nor_segmentation_anywhere() -> None:
    upward = {"voxweave.pipeline", "voxweave.segmentation"}
    for node in ast.walk(ast.parse(inspect.getsource(shadow_v2))):
        if isinstance(node, ast.Import):
            names = {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {node.module or ""}
            if node.module == "voxweave":
                names |= {f"voxweave.{alias.name}" for alias in node.names}
        else:
            continue
        assert not names & upward, ast.dump(node)


def test_hook_resolves_the_lane_from_shadow_v2_at_call_time(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_shadow(document, cues, *, thresholds):
        calls.append(
            {"document": document, "cues": list(cues), "thresholds": dict(thresholds)}
        )
        return {"kind": "sentinel"}

    monkeypatch.setattr(shadow_v2, "run_shadow", fake_run_shadow)

    monkeypatch.delenv(pipeline.SEG_V2_SHADOW_ENV, raising=False)
    off = _segment(_case_plain())
    assert off.shadow is None
    assert calls == []

    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    on = _segment(_case_plain())
    assert on.shadow is not None
    assert on.shadow["kind"] == "sentinel"
    assert len(calls) == 1
    assert calls[0]["document"] is on.document
    # the lane measures with the thresholds v1 ran, which the manifest quotes
    assert calls[0]["thresholds"] == {
        key: on.manifest["profile"][key] for key in THRESHOLD_KEYS
    }
    # ``segment_document`` still stamps its two origin-typed blocks onto
    # whatever the lane returned; nothing else in the hook touches the artifact.
    assert set(on.shadow) == {"kind", "production_degraded", "providers"}
    assert on.cues == off.cues
    assert on.units == off.units
