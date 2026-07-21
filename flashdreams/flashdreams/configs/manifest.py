# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema loaders for global and per-demo test-suite manifests."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BackendEntry:
    """One model backend declared by a global manifest."""

    id: str
    adapter: str


@dataclass(frozen=True)
class DecoderEntry:
    """One decoder declared by a global manifest."""

    id: str
    adapter: str


@dataclass(frozen=True)
class NamedVariant:
    """A named backend configuration that demos can opt into."""

    id: str
    backend: str
    settings: dict[str, Any]


@dataclass(frozen=True)
class DemoEntry:
    """One demo declared by a global manifest."""

    id: str
    adapter: str
    config_path: str
    supported_backends: tuple[str, ...]


@dataclass(frozen=True)
class GlobalManifest:
    """Parsed top-level backend/demo manifest."""

    path: Path
    schema_version: int
    backends: tuple[BackendEntry, ...]
    decoders: tuple[DecoderEntry, ...]
    named_variants: tuple[NamedVariant, ...]
    demos: tuple[DemoEntry, ...]
    suites: dict[str, tuple[str, ...]]

    @property
    def backend_ids(self) -> frozenset[str]:
        """Return declared backend ids."""
        return frozenset(entry.id for entry in self.backends)

    @property
    def decoder_ids(self) -> frozenset[str]:
        """Return declared decoder ids."""
        return frozenset(entry.id for entry in self.decoders)

    @property
    def variant_ids(self) -> frozenset[str]:
        """Return declared named variant ids."""
        return frozenset(entry.id for entry in self.named_variants)

    @property
    def demo_ids(self) -> frozenset[str]:
        """Return declared demo ids."""
        return frozenset(entry.id for entry in self.demos)

    def demo(self, demo_id: str) -> DemoEntry:
        """Return one declared demo by id."""
        for entry in self.demos:
            if entry.id == demo_id:
                return entry
        raise ValueError(
            f"Unknown demo {demo_id!r}; known demos: {sorted(self.demo_ids)}"
        )

    def validate_demo_manifest(
        self, demo_manifest: DemoManifest, *, demo_id: str | None = None
    ) -> None:
        """Validate a per-demo manifest against this global manifest."""
        demo: DemoEntry | None = None
        if demo_id is not None:
            demo = self.demo(demo_id)

        variants_by_id = {entry.id: entry for entry in self.named_variants}
        unknown_variants = sorted(
            {case.variant for case in demo_manifest.cases} - self.variant_ids
        )
        if unknown_variants:
            raise ValueError(
                "demo manifest cases reference variants not in global "
                f"named_variants: {unknown_variants}"
            )

        if demo is not None:
            unsupported_cases = sorted(
                case.id
                for case in demo_manifest.cases
                if variants_by_id[case.variant].backend not in demo.supported_backends
            )
            if unsupported_cases:
                raise ValueError(
                    f"demo {demo.id!r} supports backends "
                    f"{list(demo.supported_backends)!r}, but cases use "
                    f"unsupported backends: {unsupported_cases}"
                )

        case_ids = demo_manifest.case_ids
        for suite_name, suite_cases in demo_manifest.suites.items():
            unknown_cases = sorted(set(suite_cases) - case_ids)
            if unknown_cases:
                raise ValueError(
                    f"demo suite {suite_name!r} references unknown cases: "
                    f"{unknown_cases}"
                )


@dataclass(frozen=True)
class DemoCase:
    """One test case in a per-demo manifest."""

    id: str
    fixture: str
    variant: str


@dataclass(frozen=True)
class DemoManifest:
    """Parsed per-demo test-suite manifest."""

    path: Path
    schema_version: int
    fixtures: dict[str, dict[str, Any]]
    cases: tuple[DemoCase, ...]
    suites: dict[str, tuple[str, ...]]

    @property
    def fixture_ids(self) -> frozenset[str]:
        """Return declared fixture ids."""
        return frozenset(self.fixtures)

    @property
    def case_ids(self) -> frozenset[str]:
        """Return declared case ids."""
        return frozenset(case.id for case in self.cases)


def load_global_manifest(path: str | Path) -> GlobalManifest:
    """Read and validate a global manifest."""
    manifest_path = Path(path)
    data = _load_mapping(manifest_path)
    schema_version = _schema_version(data, context="global manifest")

    backends = tuple(
        _parse_backend(item, index=i)
        for i, item in enumerate(_required_list(data, "backends"))
    )
    decoders = tuple(
        _parse_decoder(item, index=i)
        for i, item in enumerate(_required_list(data, "decoders"))
    )
    named_variants = tuple(
        _parse_named_variant(item, index=i)
        for i, item in enumerate(_required_list(data, "named_variants"))
    )
    demos = tuple(
        _parse_demo(item, index=i)
        for i, item in enumerate(_required_list(data, "demos"))
    )
    suites = _parse_suites(data, context="global manifest")

    _ensure_unique("global manifest.backends", (entry.id for entry in backends))
    _ensure_unique("global manifest.decoders", (entry.id for entry in decoders))
    _ensure_unique(
        "global manifest.named_variants", (entry.id for entry in named_variants)
    )
    _ensure_unique("global manifest.demos", (entry.id for entry in demos))

    backend_ids = {entry.id for entry in backends}
    for variant in named_variants:
        if variant.backend not in backend_ids:
            raise ValueError(
                f"named_variant {variant.id!r} references unknown backend "
                f"{variant.backend!r}"
            )

    decoder_ids = {entry.id for entry in decoders}
    for variant in named_variants:
        for decoder_id in _setting_values(variant.settings, "decoder", "decoders"):
            if decoder_id not in decoder_ids:
                raise ValueError(
                    f"named_variant {variant.id!r} references unknown decoder "
                    f"{decoder_id!r}"
                )

    demo_ids = {entry.id for entry in demos}
    for suite_name, suite_demos in suites.items():
        unknown_demos = sorted(set(suite_demos) - demo_ids)
        if unknown_demos:
            raise ValueError(
                f"global suite {suite_name!r} references unknown demos: "
                f"{unknown_demos}"
            )

    for demo in demos:
        unknown_backends = sorted(set(demo.supported_backends) - backend_ids)
        if unknown_backends:
            raise ValueError(
                f"demo {demo.id!r} references unknown supported backends: "
                f"{unknown_backends}"
            )

    return GlobalManifest(
        path=manifest_path,
        schema_version=schema_version,
        backends=backends,
        decoders=decoders,
        named_variants=named_variants,
        demos=demos,
        suites=suites,
    )


def load_demo_manifest(
    path: str | Path,
    *,
    global_manifest: GlobalManifest | None = None,
    demo_id: str | None = None,
) -> DemoManifest:
    """Read and validate a per-demo manifest."""
    manifest_path = Path(path)
    data = _load_mapping(manifest_path)
    schema_version = _schema_version(data, context="demo manifest")

    raw_fixtures = _required_mapping(data, "fixtures", context="demo manifest")
    fixtures: dict[str, dict[str, Any]] = {}
    for fixture_id, fixture in raw_fixtures.items():
        if not isinstance(fixture_id, str) or not fixture_id:
            raise ValueError(f"demo manifest has invalid fixture id {fixture_id!r}")
        if not isinstance(fixture, dict):
            raise ValueError(f"fixture {fixture_id!r} must be a mapping")
        fixtures[fixture_id] = dict(fixture)

    cases = tuple(
        _parse_demo_case(item, index=i)
        for i, item in enumerate(_required_list(data, "cases"))
    )
    suites = _parse_suites(data, context="demo manifest")

    _ensure_unique("demo manifest.cases", (case.id for case in cases))

    fixture_ids = set(fixtures)
    for case in cases:
        if case.fixture not in fixture_ids:
            raise ValueError(
                f"case {case.id!r} references unknown fixture {case.fixture!r}"
            )

    case_ids = {case.id for case in cases}
    for suite_name, suite_cases in suites.items():
        unknown_cases = sorted(set(suite_cases) - case_ids)
        if unknown_cases:
            raise ValueError(
                f"demo suite {suite_name!r} references unknown cases: "
                f"{unknown_cases}"
            )

    manifest = DemoManifest(
        path=manifest_path,
        schema_version=schema_version,
        fixtures=fixtures,
        cases=cases,
        suites=suites,
    )
    if global_manifest is not None:
        global_manifest.validate_demo_manifest(manifest, demo_id=demo_id)
    return manifest


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "Loading YAML manifests requires PyYAML; install pyyaml."
            ) from exc
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _schema_version(data: dict[str, Any], *, context: str) -> int:
    version = data.get("schema_version", 1)
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(f"{context}.schema_version must be an integer")
    if version != 1:
        raise ValueError(f"Unsupported {context}.schema_version {version}; expected 1")
    return version


def _parse_backend(value: Any, *, index: int) -> BackendEntry:
    context = f"global manifest.backends[{index}]"
    data = _as_mapping(value, context=context)
    return BackendEntry(
        id=_required_str(data, "id", context=context),
        adapter=_required_str(data, "adapter", context=context),
    )


def _parse_decoder(value: Any, *, index: int) -> DecoderEntry:
    context = f"global manifest.decoders[{index}]"
    data = _as_mapping(value, context=context)
    return DecoderEntry(
        id=_required_str(data, "id", context=context),
        adapter=_required_str(data, "adapter", context=context),
    )


def _parse_named_variant(value: Any, *, index: int) -> NamedVariant:
    context = f"global manifest.named_variants[{index}]"
    data = _as_mapping(value, context=context)
    settings = _required_mapping(data, "settings", context=context)
    return NamedVariant(
        id=_required_str(data, "id", context=context),
        backend=_required_str(data, "backend", context=context),
        settings=dict(settings),
    )


def _parse_demo(value: Any, *, index: int) -> DemoEntry:
    context = f"global manifest.demos[{index}]"
    data = _as_mapping(value, context=context)
    return DemoEntry(
        id=_required_str(data, "id", context=context),
        adapter=_required_str(data, "adapter", context=context),
        config_path=_required_str(data, "config_path", context=context),
        supported_backends=_required_str_tuple(
            data, "supported_backends", context=context
        ),
    )


def _parse_demo_case(value: Any, *, index: int) -> DemoCase:
    context = f"demo manifest.cases[{index}]"
    data = _as_mapping(value, context=context)
    return DemoCase(
        id=_required_str(data, "id", context=context),
        fixture=_required_str(data, "fixture", context=context),
        variant=_required_str(data, "variant", context=context),
    )


def _parse_suites(data: dict[str, Any], *, context: str) -> dict[str, tuple[str, ...]]:
    raw_suites = _required_mapping(data, "suites", context=context)
    suites: dict[str, tuple[str, ...]] = {}
    for name, members in raw_suites.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{context}.suites contains invalid suite name {name!r}")
        if not isinstance(members, list) or not all(
            isinstance(item, str) and item for item in members
        ):
            raise ValueError(f"{context}.suites.{name} must be a string list")
        suites[name] = tuple(members)
    return suites


def _required_list(data: dict[str, Any], key: str) -> list[Any]:
    out = _get(data, key)
    if not isinstance(out, list):
        raise ValueError(f"{key} must be a list")
    return out


def _required_mapping(
    data: dict[str, Any], key: str, *, context: str
) -> dict[str, Any]:
    out = _get(data, key)
    if not isinstance(out, dict):
        raise ValueError(f"{context}.{key} must be a mapping")
    return out


def _required_str(data: dict[str, Any], key: str, *, context: str) -> str:
    out = _get(data, key)
    if not isinstance(out, str) or not out:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return out


def _required_str_tuple(
    data: dict[str, Any], key: str, *, context: str
) -> tuple[str, ...]:
    out = _get(data, key)
    if not isinstance(out, list) or not all(
        isinstance(item, str) and item for item in out
    ):
        raise ValueError(f"{context}.{key} must be a non-empty string list")
    if not out:
        raise ValueError(f"{context}.{key} must be a non-empty string list")
    return tuple(out)


def _as_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _get(data: dict[str, Any], key: str) -> Any:
    if key in data:
        return data[key]
    dashed = key.replace("_", "-")
    if dashed in data:
        return data[dashed]
    raise ValueError(f"{key} is required")


def _ensure_unique(context: str, values: Iterable[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"{context} contains duplicate ids: {sorted(duplicates)}")


def _setting_values(settings: dict[str, Any], *keys: str) -> tuple[str, ...]:
    values: list[str] = []
    for key in keys:
        if key not in settings:
            continue
        raw_value = settings[key]
        raw_values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in raw_values:
            if not isinstance(value, str):
                raise ValueError(f"settings.{key} entries must be strings")
            values.append(value)
    return tuple(values)
