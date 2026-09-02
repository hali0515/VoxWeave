from __future__ import annotations

import os
import re
import subprocess
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYANNOTE_REQUIREMENT = "pyannote-audio>=4,<5"
STALE_SEMANTIC_SPLIT = re.compile(
    r"\bsemantic\b[^\r\n]{0,96}?\bsplit\w*\b",
    flags=re.IGNORECASE,
)


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
    assert "diarize" not in optional  # uv omits empty optional-dependency tables
    assert {
        "name": "pyannote-audio",
        "specifier": ">=4,<5",
    } in metadata["requires-dist"]
    assert "diarize" in metadata["provides-extras"]
    locked_pyannote = next(
        package
        for package in tomllib.loads(
            (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
        )["package"]
        if package["name"] == "pyannote-audio"
    )
    assert locked_pyannote["version"].split(".", 1)[0] == "4"


def test_make_install_has_no_diarize_extra_detection(tmp_path: Path) -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "TOOL_SITE" not in makefile
    assert "pyannote" not in makefile.lower()
    environment = {
        "HOME": str(tmp_path),
        "PATH": os.environ.get("PATH", os.defpath),
    }
    for variant in ("cuda", "mps"):
        result = subprocess.run(
            [
                "make",
                "-n",
                "install",
                f"VARIANT={variant}",
                "TORCH_BACKEND=cpu",
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "extras=none" in result.stdout
        assert f'".[{variant}]"' in result.stdout

    compatibility = subprocess.run(
        [
            "make",
            "-n",
            "install",
            "VARIANT=cuda",
            "TORCH_BACKEND=cpu",
            "EXTRAS=diarize",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    assert compatibility.returncode == 0, compatibility.stdout + compatibility.stderr
    assert '".[cuda,diarize]"' in compatibility.stdout


def test_readme_describes_default_install_and_opt_in_runtime() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()

    assert "voxweave[diarize]" not in lowered
    assert "`[diarize]`" not in lowered
    assert "pyannote-audio" in lowered
    assert "ships by default" in lowered
    assert "--diarize" in readme
    assert "--diarize-model" in readme
    assert "VOXWEAVE_DIARIZE_MODEL" in readme
    assert "pyannote/speaker-diarization-3.1" in readme
    assert "pyannote/speaker-diarization-community-1" in readme
    assert "CC-BY-4.0" in readme
    assert "2,723.963 MiB" in readme
    assert "hf auth login" in readme
    assert "VOXWEAVE_HF_TOKEN" in readme
    assert "HF_TOKEN" in readme
    assert re.search(r"^diarize\s*=\s*false", readme, flags=re.MULTILINE)

    for source_path in (REPO_ROOT / "voxweave").rglob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        assert "voxweave[diarize]" not in source.lower(), source_path


def _extract_click_option_block(source: str, flag: str) -> str:
    """Return the raw source of a ``@click.option(flag, ...)`` call body.

    Matches from the flag's own string literal up to the following line that
    is exactly a lone closing paren (the end of the click.option(...) call),
    so a literal ")" inside the option's own help text (e.g. "(default)")
    cannot truncate the match early.
    """
    match = re.search(rf'"{re.escape(flag)}",\n(.*?)\n\)\n', source, flags=re.DOTALL)
    assert match, f"could not find click.option block for {flag!r} in cli.py"
    return match.group(1)


def _option_help_text(option_block: str) -> str:
    """Concatenate the string-literal pieces of a ``help=(...)`` value."""
    remainder = option_block.split("help=(", 1)[1]
    return "".join(re.findall(r'"([^"]*)"', remainder))


def test_diarize_model_help_names_community1_as_default_not_31() -> None:
    cli_source = (REPO_ROOT / "voxweave" / "cli.py").read_text(encoding="utf-8")
    help_text = _option_help_text(
        _extract_click_option_block(cli_source, "--diarize-model")
    )

    assert "community-1 (default)" in help_text
    assert "3.1 (default)" not in help_text
    assert "(the default)" not in help_text


def test_readme_diarize_docs_name_community1_as_default_everywhere() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    stale_31_default_phrases = [
        "3.1 (default)",
        "3.1` (the default)",
        "3.1` (default)",
        "default `pyannote/speaker-diarization-3.1`",
        '"3.1" (built-in default)',
    ]
    for phrase in stale_31_default_phrases:
        assert phrase not in readme, f"stale 3.1-is-default wording found: {phrase!r}"

    # The CLI options table, the env-var reference, and the sample conf file
    # must each independently say community-1 is the default.
    assert "`community-1` (the default)" in readme
    assert "default `pyannote/speaker-diarization-community-1`" in readme
    assert '"community-1" (built-in default)' in readme


def test_tracked_documentation_has_no_removed_semantic_split_language() -> None:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md", "*.rst", "*.txt"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    documentation = [
        REPO_ROOT / raw_path.decode("utf-8")
        for raw_path in result.stdout.split(b"\0")
        if raw_path
    ]
    assert documentation

    stale = {
        str(path.relative_to(REPO_ROOT)): match.group(0)
        for path in documentation
        if (match := STALE_SEMANTIC_SPLIT.search(path.read_text(encoding="utf-8")))
    }
    assert stale == {}
