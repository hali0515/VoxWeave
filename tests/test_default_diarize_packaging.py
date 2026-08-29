from __future__ import annotations

import re
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYANNOTE_REQUIREMENT = "pyannote-audio>=3.4,<4"


def _project() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]


def _locked_voxweave() -> dict[str, object]:
    packages = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))[
        "package"
    ]
    matches = [
        package
        for package in packages
        if package["name"] == "voxweave" and package["source"] == {"editable": "."}
    ]
    assert len(matches) == 1
    return matches[0]


def test_diarization_dependency_is_core_with_an_empty_compatibility_alias() -> None:
    project = _project()
    dependencies = project["dependencies"]
    optional = project["optional-dependencies"]

    assert dependencies.count(PYANNOTE_REQUIREMENT) == 1
    assert optional["diarize"] == []
    assert not any(
        requirement.startswith("pyannote-audio")
        for requirements in optional.values()
        for requirement in requirements
    )


def test_cuda_mps_conflict_contract_is_unchanged() -> None:
    document = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert document["tool"]["uv"]["conflicts"] == [
        [{"extra": "cuda"}, {"extra": "mps"}]
    ]


def test_lock_records_pyannote_as_core_and_preserves_the_empty_alias() -> None:
    package = _locked_voxweave()
    dependencies = package["dependencies"]
    optional = package["optional-dependencies"]
    metadata = package["metadata"]

    assert {"name": "pyannote-audio"} in dependencies
    assert optional["diarize"] == []
    assert {
        "name": "pyannote-audio",
        "specifier": ">=3.4,<4",
    } in metadata["requires-dist"]
    assert "diarize" in metadata["provides-extras"]


def test_make_install_has_no_diarize_extra_detection() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "TOOL_SITE" not in makefile
    assert "EXTRAS" not in makefile
    assert "INSTALL_SPEC = .[$(VARIANT)]" in makefile
    assert "pyannote" not in makefile.lower()


def test_readme_describes_default_install_and_opt_in_runtime() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()

    assert "voxweave[diarize]" not in lowered
    assert "`[diarize]`" not in lowered
    assert re.search(r"semantic[-_ ]split", readme, flags=re.IGNORECASE) is None
    assert "pyannote-audio" in lowered
    assert "ships by default" in lowered
    assert "--diarize" in readme
    assert "pyannote/speaker-diarization-3.1" in readme
    assert "hf auth login" in readme
    assert "VOXWEAVE_HF_TOKEN" in readme
    assert "HF_TOKEN" in readme
    assert re.search(r"^diarize\s*=\s*false", readme, flags=re.MULTILINE)
