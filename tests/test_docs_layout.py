# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Zensical Markdown repository documentation layout."""

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "flashdreams_docs_layout_check", _ROOT / "tools" / "check_docs_layout.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
find_integration_link_violations = _MODULE.find_integration_link_violations
find_required_doc_violations = _MODULE.find_required_doc_violations
find_violations = _MODULE.find_violations

pytestmark = pytest.mark.ci_cpu


def test_docs_layout_rules(tmp_path: Path) -> None:
    files = {
        "docs/src/content/docs/guide.md": "---\ntitle: Guide\n---\n\nBody\n",
        "docs/source/legacy.rst": "Legacy\n======\n",
        "README.md": "[Guide](docs/src/content/docs/guide.md#guide)\n",
        "skills/example/SKILL.md": "---\nname: example\n---\n\n# Example\n",
        "flashdreams/README.md": "# FlashDreams package\n\nPackage source.\n",
        "bad.md": "first\nsecond\n",
        "docs/src/content/docs/bad/SKILL.md": "---\nname: bad\n---\n",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    assert find_violations(tmp_path, map(Path, files)) == [
        "bad.md: Markdown must be at most one line unless it is "
        "docs/src/content/docs/**, skills/**/SKILL.md or "
        "flashdreams/README.md (found 2)",
        "docs/source/legacy.rst: use Markdown instead of reStructuredText",
        "docs/src/content/docs/bad/SKILL.md: SKILL.md files must live under skills/**",
    ]


def test_flashdreams_readme_requires_introductory_prose(tmp_path: Path) -> None:
    readme = tmp_path / "flashdreams/README.md"
    readme.parent.mkdir()
    readme.write_text(
        "[Package docs](../docs/src/content/docs/index.md)\n", encoding="utf-8"
    )
    assert find_required_doc_violations(tmp_path) == [
        "flashdreams/README.md: expected a heading and its own introductory prose, "
        "not only links"
    ]
    readme.unlink()
    assert find_required_doc_violations(tmp_path) == [
        "flashdreams/README.md: required package introduction is missing"
    ]


def test_integration_links_use_standard_section(tmp_path: Path) -> None:
    root = tmp_path / "docs/src/content/docs/repository/integrations_v2"
    valid = root / "model/README.md"
    invalid = root / "model/tests/validation.md"
    valid.parent.mkdir(parents=True)
    invalid.parent.mkdir(parents=True)
    valid.write_text(
        "## Integration links\n\n"
        "- **Applications:** [T2V](apps/t2v/README/)\n"
        "- **Configuration:** [config.py](https://example.com/config.py)\n",
        encoding="utf-8",
    )
    invalid.write_text("# T2V\n", encoding="utf-8")
    assert find_integration_link_violations(tmp_path) == [
        "docs/src/content/docs/repository/integrations_v2/model/tests/validation.md: "
        "expected a standard Integration links section"
    ]
