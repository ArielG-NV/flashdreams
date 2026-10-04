# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enforce the repository's Zensical Markdown documentation layout."""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOCS_ROOT = PurePosixPath("docs/src/content/docs")
_SKILLS_ROOT = PurePosixPath("skills")
_FLASHDREAMS_README = PurePosixPath("flashdreams/README.md")
_INTEGRATION_LINKS_RE = re.compile(
    r"(?m)^## Integration links\n\n"
    r"[-*] \*\*Applications:\*\* .+\n"
    r"[-*] \*\*Configuration:\*\* .+$"
)


def _is_below(path: PurePosixPath, root: PurePosixPath) -> bool:
    return path != root and root in path.parents


def _allows_multiline_markdown(path: PurePosixPath) -> bool:
    return (
        _is_below(path, _DOCS_ROOT)
        or path == _FLASHDREAMS_README
        or (path.name == "SKILL.md" and _is_below(path, _SKILLS_ROOT))
    )


def _has_flashdreams_intro(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(
        lines
        and lines[0].startswith("# ")
        and any(not line.startswith(("#", "- ", "* ", "[", "<")) for line in lines[1:])
    )


def find_violations(repo_root: Path, paths: Iterable[Path]) -> list[str]:
    """Return documentation-layout violations for repository-relative paths."""
    violations: list[str] = []
    for path in sorted(paths):
        relative = PurePosixPath(path.as_posix())
        suffix = relative.suffix.lower()
        if relative.name == "SKILL.md" and not _is_below(relative, _SKILLS_ROOT):
            violations.append(f"{relative}: SKILL.md files must live under skills/**")
        elif suffix == ".rst":
            violations.append(f"{relative}: use Markdown instead of reStructuredText")
        elif suffix == ".md" and not _allows_multiline_markdown(relative):
            line_count = len(
                (repo_root / path).read_text(encoding="utf-8").splitlines()
            )
            if line_count > 1:
                violations.append(
                    f"{relative}: Markdown must be at most one line unless it is "
                    "docs/src/content/docs/**, skills/**/SKILL.md or "
                    f"flashdreams/README.md (found {line_count})"
                )
    return violations


def find_required_doc_violations(repo_root: Path) -> list[str]:
    """Require the package introduction."""
    readme = repo_root / Path(_FLASHDREAMS_README.as_posix())
    if not readme.is_file():
        return ["flashdreams/README.md: required package introduction is missing"]
    if not _has_flashdreams_intro(readme.read_text(encoding="utf-8")):
        return [
            "flashdreams/README.md: expected a heading and its own introductory "
            "prose, not only links"
        ]
    return []


def find_site_integration_violations(repo_root: Path) -> list[str]:
    """Require Zensical and repository pages in the same documentation site."""
    violations: list[str] = []
    config = repo_root / "docs/zensical.toml"
    if not config.is_file() or 'docs_dir = "src/content/docs"' not in config.read_text(
        encoding="utf-8"
    ):
        violations.append("docs/zensical.toml: expected Zensical docs directory")
    if not (repo_root / "docs/src/content/docs/index.md").is_file():
        violations.append("docs/src/content/docs/index.md: missing site landing page")
    if not (repo_root / "docs/src/content/docs/repository/CONTRIBUTING.md").is_file():
        violations.append(
            "docs/src/content/docs/repository/CONTRIBUTING.md: missing repository pages"
        )
    return violations


def find_integration_link_violations(repo_root: Path) -> list[str]:
    """Require uniform application and configuration links in integration docs."""
    root = repo_root / "docs/src/content/docs/repository/integrations_v2"
    readmes = sorted(root.rglob("*.md"))
    violations = []
    for readme in readmes:
        if not _INTEGRATION_LINKS_RE.search(readme.read_text(encoding="utf-8")):
            relative = readme.relative_to(repo_root).as_posix()
            violations.append(
                f"{relative}: expected a standard Integration links section"
            )
    return violations


def _repository_paths(repo_root: Path) -> list[Path]:
    output = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ]
    )
    paths = [Path(raw.decode()) for raw in output.split(b"\0") if raw]
    return [path for path in paths if (repo_root / path).is_file()]


def main() -> int:
    violations = find_violations(_REPO_ROOT, _repository_paths(_REPO_ROOT))
    violations += find_required_doc_violations(_REPO_ROOT)
    violations += find_site_integration_violations(_REPO_ROOT)
    violations += find_integration_link_violations(_REPO_ROOT)
    if violations:
        print("Documentation layout check failed:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
