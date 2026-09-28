"""Only the command layer imports ``voxweave.pipeline``.

The pipeline orchestrates and imports downward. A lower module reaching back
into it (even lazily, inside a function) is how the old import cycles grew; the
shared helpers live in leaf modules (``paths``, ``sidecars``, ``vocals``,
``segmentation``, ``core.overlay``) instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import voxweave

PACKAGE = Path(voxweave.__file__).parent
# The CLI drives the pipeline, and ``llm_commands.correct --apply`` re-runs
# ``pipeline.align``; both sit above it.
ALLOWED = {"cli.py", "llm_commands.py"}


def _imports_pipeline(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "voxweave.pipeline" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "voxweave.pipeline":
                return True
            if node.module == "voxweave" and any(
                alias.name == "pipeline" for alias in node.names
            ):
                return True
    return False


def test_only_the_command_layer_imports_pipeline() -> None:
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "vendor" in path.parts or path.name == "pipeline.py":
            continue
        if path.parent == PACKAGE and (
            path.name in ALLOWED or path.name.startswith("cli_")
        ):
            continue
        if _imports_pipeline(ast.parse(path.read_text(encoding="utf-8"))):
            offenders.append(str(path.relative_to(PACKAGE)))
    assert offenders == []
