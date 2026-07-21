# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for running manifest-driven demo test suites."""

from __future__ import annotations

import argparse
import importlib
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, TextIO

from flashdreams.configs.manifest import (
    DemoEntry,
    DemoManifest,
    GlobalManifest,
    NamedVariant,
    load_demo_manifest,
    load_global_manifest,
)


STATUS_INTERVAL_S = 30


def main(argv: list[str] | None = None) -> int:
    """Run manifest-defined demo test cases and write result artifacts."""
    args = _parse_args(argv)
    started_at = time.time()
    global_manifest = load_global_manifest(args.manifest)
    output_dir = args.output_root / args.manifest.stem
    mode = _mode_name(
        all_cases=args.all,
        backend_id=args.backend,
        backend_configuration_test_id=args.backend_configuration_test,
        variant_id=args.variant,
        suite=args.suite,
    )
    plan = _build_run_plan(
        global_manifest,
        manifest_path=args.manifest,
        output_root=args.output_root,
        all_cases=args.all,
        backend_id=args.backend or args.backend_configuration_test,
        backend_configuration_test_id=args.backend_configuration_test,
        variant_id=args.variant,
        suite=args.suite,
        demo_id=args.demo,
        selection_namespace=_selection_namespace(
            mode=mode,
            backend_id=args.backend,
            variant_id=args.variant,
            suite=args.suite,
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    executed = _execute_plan(
        plan,
        dry_run=args.dry_run,
        no_instantiate=args.no_instantiate,
        runner_bin=args.runner_bin,
    )
    failed_cases = [
        case
        for demo in executed["demos"]
        for case in demo["cases"]
        if case["status"] != "pass"
    ]
    run_manifest = {
        "schema_version": 1,
        "manifest_path": str(args.manifest),
        "mode": mode,
        "backend": args.backend,
        "backend_configuration_test": args.backend_configuration_test,
        "variant": args.variant,
        "suite": args.suite,
        "demo": args.demo,
        "execute": not args.dry_run,
        "dry_run": args.dry_run,
        "no_instantiate": args.no_instantiate,
        "runner_bin": args.runner_bin,
        "started_at_unix": started_at,
        "duration_s": time.time() - started_at,
        "output_dir": str(output_dir),
        "demo_count": len(executed["demos"]),
        "case_count": sum(len(demo["cases"]) for demo in executed["demos"]),
        "failed_case_count": len(failed_cases),
        "decision": "fail" if failed_cases else "pass",
        "demos": executed["demos"],
    }
    _write_yaml(output_dir / "manifest.yml", run_manifest)

    if args.backend_configuration_test is not None:
        _write_configuration_list(
            output_dir
            / f"run-all-{args.backend_configuration_test}"
            / "configuration-list.yml",
            plan["configuration_variants"],
        )

    print(
        "test-suite "
        f"{run_manifest['decision']}: "
        f"{run_manifest['case_count']} case(s) across "
        f"{run_manifest['demo_count']} demo(s); "
        f"manifest={output_dir / 'manifest.yml'}"
    )
    _print_failed_cases(executed["demos"])
    return 1 if failed_cases else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to a demo public-manifest.yml.")

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        action="store_true",
        help="Run all cases in every selected demo manifest.",
    )
    selection.add_argument(
        "--backend",
        help="Run cases for all named variants using this backend.",
    )
    selection.add_argument(
        "--backend-configuration-test",
        help=(
            "Run backend configuration coverage for this backend and store "
            "artifacts under the special run-all-<backend> location."
        ),
    )
    selection.add_argument(
        "--variant",
        help="Run cases for one named variant.",
    )
    selection.add_argument(
        "--suite",
        help="Run cases from a named suite.",
    )

    parser.add_argument("--demo", help="Restrict execution to one demo id.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/test-suite-results"),
        help="Root directory for generated test-suite artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Only materialize input/result files. By default test-suite "
            "executes flashdreams-run for each selected case."
        ),
    )
    parser.add_argument(
        "--no-instantiate",
        action="store_true",
        help="Pass --no-instantiate to flashdreams-run for config resolution only.",
    )
    parser.add_argument(
        "--runner-bin",
        default="flashdreams-run",
        help="Runner executable to invoke for each manifest case.",
    )
    return parser.parse_args(argv)


def _build_run_plan(
    global_manifest: GlobalManifest,
    *,
    manifest_path: Path,
    output_root: Path,
    all_cases: bool,
    backend_id: str | None,
    backend_configuration_test_id: str | None,
    variant_id: str | None,
    suite: str | None,
    demo_id: str | None,
    selection_namespace: str | None,
) -> dict[str, Any]:
    variants_by_id = {
        variant.id: variant for variant in global_manifest.named_variants
    }

    if backend_id is not None and backend_id not in global_manifest.backend_ids:
        raise ValueError(
            f"Unknown backend {backend_id!r}; known backends: "
            f"{sorted(global_manifest.backend_ids)}"
        )

    if backend_configuration_test_id is not None and (
        backend_configuration_test_id not in global_manifest.backend_ids
    ):
        raise ValueError(
            f"Unknown backend {backend_configuration_test_id!r}; known backends: "
            f"{sorted(global_manifest.backend_ids)}"
        )

    if variant_id is not None and variant_id not in variants_by_id:
        raise ValueError(
            f"Unknown variant {variant_id!r}; known variants: "
            f"{sorted(variants_by_id)}"
        )

    if suite is not None and suite not in global_manifest.suites:
        raise ValueError(
            f"Unknown global suite {suite!r}; known suites: "
            f"{sorted(global_manifest.suites)}"
        )

    demos = _selected_demos(
        global_manifest, backend_id=backend_id, suite=suite, demo_id=demo_id
    )
    planned_demos = []
    configuration_variants: list[dict[str, Any]] = []

    for demo in demos:
        demo_manifest = _load_demo_for_entry(
            global_manifest, demo, manifest_path=manifest_path
        )
        cases = _selected_cases(
            demo_manifest,
            variants_by_id=variants_by_id,
            all_cases=all_cases,
            backend_id=backend_id,
            suite=suite,
            variant_id=variant_id,
        )
        if not cases:
            continue

        planned_cases = []
        for case in cases:
            variant = variants_by_id[case.variant]
            fixture = demo_manifest.fixtures[case.fixture]
            case_output_dir = _case_output_dir(
                output_root=output_root,
                manifest_stem=manifest_path.stem,
                demo_id=demo.id,
                variant_id=variant.id,
                backend_configuration_test_id=backend_configuration_test_id,
                selection_namespace=selection_namespace,
            )
            planned_cases.append(
                {
                    "id": case.id,
                    "fixture": case.fixture,
                    "variant": case.variant,
                    "backend": variant.backend,
                    "settings": variant.settings,
                    "fixture_data": fixture,
                    "output_dir": str(case_output_dir),
                    "case_dir": str(case_output_dir / "cases" / case.id),
                    "status": "pending",
                }
            )
            if backend_configuration_test_id is not None:
                configuration_variants.append(_configuration_variant(variant))

        planned_demos.append(
            {
                "id": demo.id,
                "adapter": demo.adapter,
                "config_path": demo.config_path,
                "supported_backends": list(demo.supported_backends),
                "case_count": len(planned_cases),
                "cases": planned_cases,
            }
        )

    if not planned_demos:
        raise ValueError("No test-suite cases matched the requested selection")

    return {
        "demos": planned_demos,
        "configuration_variants": _unique_configuration_variants(configuration_variants),
    }


def _execute_plan(
    plan: dict[str, Any],
    *,
    dry_run: bool,
    no_instantiate: bool,
    runner_bin: str,
) -> dict[str, Any]:
    """Execute each selected manifest case."""
    executed_demos = []
    total_cases = sum(len(demo["cases"]) for demo in plan["demos"])
    case_index = 0
    for demo in plan["demos"]:
        executed_cases = []
        for case in demo["cases"]:
            case_index += 1
            executed_cases.append(
                _execute_case(
                    demo,
                    case,
                    dry_run=dry_run,
                    no_instantiate=no_instantiate,
                    runner_bin=runner_bin,
                    case_index=case_index,
                    total_cases=total_cases,
                )
            )
        executed_demo = dict(demo)
        executed_demo["cases"] = executed_cases
        executed_demo["failed_case_count"] = sum(
            1 for case in executed_cases if case["status"] != "pass"
        )
        executed_demos.append(executed_demo)
    return {**plan, "demos": executed_demos}


def _execute_case(
    demo: dict[str, Any],
    case: dict[str, Any],
    *,
    dry_run: bool,
    no_instantiate: bool,
    runner_bin: str,
    case_index: int,
    total_cases: int,
) -> dict[str, Any]:
    """Materialize one case and optionally run its model backend."""
    started_at = time.time()
    case_dir = Path(case["case_dir"])
    case_dir.mkdir(parents=True, exist_ok=True)

    fixture = case["fixture_data"]
    variant = {
        "id": case["variant"],
        "backend": case["backend"],
        "settings": case["settings"],
    }
    input_manifest = {
        "schema_version": 1,
        "demo": demo["id"],
        "demo_adapter": demo["adapter"],
        "case": case["id"],
        "fixture": case["fixture"],
        "variant": case["variant"],
        "backend": case["backend"],
        "fixture_data": fixture,
        "settings": case["settings"],
    }

    setup_commands = _setup_commands(
        demo_adapter=demo["adapter"],
        case=case,
    )
    command = _runner_command(
        runner_bin=runner_bin,
        demo_adapter=demo["adapter"],
        case=case,
        output_dir=case_dir,
        no_instantiate=no_instantiate,
        materialize_assets=not dry_run,
    )
    aux_test_info_path = case_dir / "aux-test-info.yml"
    artifacts = {"aux_test_info": str(aux_test_info_path)}
    artifacts.update(
        _case_artifacts(
            demo_adapter=demo["adapter"], case=case, output_dir=case_dir
        )
    )
    aux_test_info: dict[str, Any] = {
        "schema_version": 1,
        "input": input_manifest,
        "fixture": fixture,
        "variant": variant,
        "setup_commands": [
            {"argv": setup, "display": _command_display(setup)}
            for setup in setup_commands
        ],
        "command": {
            "argv": command,
            "display": _command_display(command),
        },
        "output_dir": str(case_dir),
        "stdout": "",
        "stderr": "",
    }

    _print_case_started(
        demo=demo,
        case=case,
        command=command,
        case_index=case_index,
        total_cases=total_cases,
        dry_run=dry_run,
    )

    if dry_run:
        result = {
            "schema_version": 1,
            "id": case["id"],
            "demo": demo["id"],
            "variant": case["variant"],
            "backend": case["backend"],
            "status": "pass",
            "step": "manifest_case_materialized",
            "execute": False,
            "command": command,
            "artifacts": artifacts,
            "started_at_unix": started_at,
            "duration_s": time.time() - started_at,
        }
    else:
        result = _run_model_case(
            case=case,
            demo=demo,
            command=command,
            setup_commands=setup_commands,
            artifacts=artifacts,
            aux_test_info=aux_test_info,
            started_at=started_at,
        )

    _write_yaml(aux_test_info_path, aux_test_info)
    _write_yaml(case_dir / "result.yml", result)
    _print_case_finished(
        demo=demo,
        case=case,
        status=result["status"],
        duration_s=result["duration_s"],
        result_path=case_dir / "result.yml",
    )

    executed_case = dict(case)
    executed_case["status"] = result["status"]
    executed_case["result_path"] = str(case_dir / "result.yml")
    executed_case["artifacts"] = result["artifacts"]
    executed_case["duration_s"] = result["duration_s"]
    executed_case["command"] = command
    return executed_case


def _case_artifacts(
    *,
    demo_adapter: str,
    case: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """Return demo-specific artifact paths for one case."""
    adapter = importlib.import_module(demo_adapter)
    build_artifacts = getattr(adapter, "build_artifacts", None)
    if build_artifacts is None:
        return {}
    return build_artifacts(case=case, output_dir=output_dir)


def _setup_commands(
    *,
    demo_adapter: str,
    case: dict[str, Any],
) -> list[list[str]]:
    """Return demo-specific setup commands required before a case runs."""
    adapter = importlib.import_module(demo_adapter)
    build_setup_commands = getattr(adapter, "build_setup_commands", None)
    if build_setup_commands is None:
        return []
    return build_setup_commands(case=case)


def _runner_command(
    *,
    runner_bin: str,
    demo_adapter: str,
    case: dict[str, Any],
    output_dir: Path,
    no_instantiate: bool,
    materialize_assets: bool,
) -> list[str]:
    adapter = importlib.import_module(demo_adapter)
    build_command = getattr(adapter, "build_command", None)
    if build_command is not None:
        return build_command(
            case=case,
            output_dir=output_dir,
            runner_bin=runner_bin,
            no_instantiate=no_instantiate,
            materialize_assets=materialize_assets,
        )

    executable = shutil.which(runner_bin) or runner_bin
    command = [executable]
    if no_instantiate:
        command.append("--no-instantiate")
    command.append(case["variant"])
    command.extend(["--output-dir", str(output_dir)])
    command.extend(_fixture_runner_args(case))
    return command


def _fixture_runner_args(case: dict[str, Any]) -> list[str]:
    fixture = case["fixture_data"]
    variant = case["variant"]
    args: list[str] = []

    prompt = _fixture_value(fixture, "prompt")
    if prompt is not None:
        args.extend(["--prompt", str(prompt)])

    if variant.startswith("omnidreams-"):
        first_frame = _fixture_value(
            fixture, "first-frame", "first_frame", "first-frame-paths"
        )
        if first_frame is not None:
            args.extend(["--first-frame-paths", str(first_frame)])
        hdmap = _fixture_value(fixture, "hdmap-video", "hdmap_video")
        if hdmap is not None:
            args.extend(["--hdmap-video-paths", str(hdmap)])
        return args

    image_path = _fixture_value(
        fixture, "first-frame", "first_frame", "image-path", "image_path"
    )
    if image_path is not None:
        args.extend(["--image-path", str(image_path)])

    input_path = _fixture_value(
        fixture, "input-video", "input_video", "input-path", "input_path"
    )
    if input_path is not None:
        args.extend(["--input-path", str(input_path)])

    return args


def _fixture_value(fixture: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in fixture:
            value = fixture[key]
            if isinstance(value, str):
                return os.environ.get(value, value)
            return value
    return None


def _run_model_case(
    *,
    case: dict[str, Any],
    demo: dict[str, Any],
    command: list[str],
    setup_commands: list[list[str]],
    artifacts: dict[str, str],
    aux_test_info: dict[str, Any],
    started_at: float,
) -> dict[str, Any]:
    step = "model_runner_executed"
    failed_command: list[str] | None = None
    try:
        with (
            tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file,
            tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file,
        ):
            returncode = 0
            for setup_command in setup_commands:
                failed_command = setup_command
                print(
                    f"test-suite setup command: {_command_display(setup_command)}",
                    flush=True,
                )
                returncode = _run_with_status_heartbeat(
                    command=setup_command,
                    stdout_file=stdout_file,
                    stderr_file=stderr_file,
                    started_at=started_at,
                    demo=demo,
                    case=case,
                )
                if returncode != 0:
                    step = "demo_dependency_setup_failed"
                    break
                failed_command = None
            else:
                failed_command = command
                returncode = _run_with_status_heartbeat(
                    command=command,
                    stdout_file=stdout_file,
                    stderr_file=stderr_file,
                    started_at=started_at,
                    demo=demo,
                    case=case,
                )
                if returncode == 0:
                    failed_command = None
            stdout_file.seek(0)
            stderr_file.seek(0)
            aux_test_info["stdout"] = stdout_file.read()
            aux_test_info["stderr"] = stderr_file.read()
    except OSError as exc:
        if failed_command is None:
            failed_command = command
        aux_test_info["stdout"] = ""
        aux_test_info["stderr"] = str(exc)
        returncode = None
        status = "fail"
    else:
        status = "pass" if returncode == 0 else "fail"

    return {
        "schema_version": 1,
        "id": case["id"],
        "demo": demo["id"],
        "variant": case["variant"],
        "backend": case["backend"],
        "status": status,
        "step": step,
        "execute": True,
        "setup_commands": setup_commands,
        "failed_command": failed_command,
        "returncode": returncode,
        "command": command,
        "command_display": _command_display(command),
        "artifacts": artifacts,
        "started_at_unix": started_at,
        "duration_s": time.time() - started_at,
    }


def _run_with_status_heartbeat(
    *,
    command: list[str],
    stdout_file: TextIO,
    stderr_file: TextIO,
    started_at: float,
    demo: dict[str, Any],
    case: dict[str, Any],
) -> int:
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        stdout=stdout_file,
        stderr=stderr_file,
        text=True,
    )
    next_status_at = time.monotonic() + STATUS_INTERVAL_S
    while True:
        returncode = process.poll()
        if returncode is not None:
            stdout_file.flush()
            stderr_file.flush()
            return returncode
        if time.monotonic() >= next_status_at:
            _print_case_heartbeat(demo=demo, case=case, started_at=started_at)
            next_status_at += STATUS_INTERVAL_S
        time.sleep(1)


def _print_case_started(
    *,
    demo: dict[str, Any],
    case: dict[str, Any],
    command: list[str],
    case_index: int,
    total_cases: int,
    dry_run: bool,
) -> None:
    mode = "materializing" if dry_run else "running"
    print(
        "test-suite "
        f"{mode} [{case_index}/{total_cases}] "
        f"demo= {demo['id']} case= {case['id']} "
        f"variant= {case['variant']} backend= {case['backend']}",
        flush=True,
    )
    if not dry_run:
        print(f"test-suite command: {_command_display(command)}", flush=True)


def _print_case_heartbeat(
    *, demo: dict[str, Any], case: dict[str, Any], started_at: float
) -> None:
    print(
        "test-suite still running "
        f"demo= {demo['id']} case= {case['id']} "
        f"variant= {case['variant']} elapsed= {time.time() - started_at:.1f}s",
        flush=True,
    )


def _print_case_finished(
    *,
    demo: dict[str, Any],
    case: dict[str, Any],
    status: str,
    duration_s: float,
    result_path: Path,
) -> None:
    print(
        "test-suite finished "
        f"demo= {demo['id']} case= {case['id']} "
        f"variant= {case['variant']} status= {status} "
        f"duration= {duration_s:.1f}s result= {result_path}",
        flush=True,
    )


def _print_failed_cases(demos: list[dict[str, Any]]) -> None:
    """Print an end-of-suite index of cases that did not run successfully."""
    failures = [
        (demo, case)
        for demo in demos
        for case in demo["cases"]
        if case["status"] != "pass"
    ]
    if not failures:
        return

    print("test-suite failed cases:")
    for demo, case in failures:
        print(
            "  - "
            f"demo={demo['id']} case={case['id']} "
            f"variant={case['variant']} backend={case['backend']} "
            f"result={case['result_path']}"
        )


def _command_display(command: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in command)


def _selected_demos(
    global_manifest: GlobalManifest,
    *,
    backend_id: str | None,
    suite: str | None,
    demo_id: str | None,
) -> tuple[DemoEntry, ...]:
    if suite is not None:
        suite_demo_ids = set(global_manifest.suites[suite])
        demos = tuple(
            demo for demo in global_manifest.demos if demo.id in suite_demo_ids
        )
    else:
        demos = global_manifest.demos

    if backend_id is not None:
        demos = tuple(demo for demo in demos if backend_id in demo.supported_backends)

    if demo_id is not None:
        global_manifest.demo(demo_id)
        demos = tuple(demo for demo in demos if demo.id == demo_id)
        if not demos:
            raise ValueError(
                f"Demo {demo_id!r} is not compatible with the requested selection"
            )

    return demos


def _load_demo_for_entry(
    global_manifest: GlobalManifest,
    demo: DemoEntry,
    *,
    manifest_path: Path,
) -> DemoManifest:
    config_path = Path(demo.config_path)
    candidates = [config_path]
    if not config_path.is_absolute():
        candidates.append(manifest_path.parent / config_path)
    for candidate in candidates:
        if candidate.is_file():
            return load_demo_manifest(
                candidate, global_manifest=global_manifest, demo_id=demo.id
            )
    raise FileNotFoundError(
        f"demo {demo.id!r} config_path {demo.config_path!r} does not exist"
    )


def _selected_cases(
    demo_manifest: DemoManifest,
    *,
    variants_by_id: dict[str, NamedVariant],
    all_cases: bool,
    backend_id: str | None,
    variant_id: str | None,
    suite: str | None,
) -> tuple[Any, ...]:
    if all_cases or backend_id is not None or variant_id is not None:
        cases = demo_manifest.cases
    else:
        assert suite is not None
        if suite not in demo_manifest.suites:
            raise ValueError(
                f"demo manifest {demo_manifest.path} has no suite {suite!r}"
            )
        suite_case_ids = set(demo_manifest.suites[suite])
        cases = tuple(
            case for case in demo_manifest.cases if case.id in suite_case_ids
        )

    if backend_id is not None:
        cases = tuple(
            case for case in cases if variants_by_id[case.variant].backend == backend_id
        )
    if variant_id is not None:
        cases = tuple(case for case in cases if case.variant == variant_id)
    return cases


def _case_output_dir(
    *,
    output_root: Path,
    manifest_stem: str,
    demo_id: str,
    variant_id: str,
    backend_configuration_test_id: str | None,
    selection_namespace: str | None,
) -> Path:
    base = output_root / manifest_stem
    if backend_configuration_test_id is not None:
        base = base / f"run-all-{backend_configuration_test_id}"
    elif selection_namespace is not None:
        base = base / selection_namespace
    return base / demo_id / variant_id


def _configuration_variant(variant: NamedVariant) -> dict[str, Any]:
    return {
        "id": variant.id,
        "backend": variant.backend,
        "settings": variant.settings,
    }


def _unique_configuration_variants(
    variants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for variant in variants:
        out[variant["id"]] = variant
    return [out[key] for key in sorted(out)]


def _write_configuration_list(path: Path, variants: list[dict[str, Any]]) -> None:
    _write_yaml(path, variants)


def _write_yaml(path: Path, value: Any) -> None:
    """Write a human-readable YAML artifact with literal multiline strings."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Writing test-suite artifacts requires PyYAML.") from exc

    class _MultilineStringDumper(yaml.SafeDumper):
        pass

    def _represent_string(
        dumper: yaml.SafeDumper, data: str
    ) -> yaml.nodes.ScalarNode:
        return dumper.represent_scalar(
            "tag:yaml.org,2002:str", data, style="|" if "\n" in data else None
        )

    _MultilineStringDumper.add_representer(str, _represent_string)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(
            value,
            f,
            Dumper=_MultilineStringDumper,
            default_flow_style=False,
            sort_keys=True,
            allow_unicode=True,
        )


def _selection_namespace(
    *,
    mode: str,
    backend_id: str | None,
    variant_id: str | None,
    suite: str | None,
) -> str | None:
    if mode == "backend":
        assert backend_id is not None
        return f"backend-{backend_id}"
    if mode == "variant":
        assert variant_id is not None
        return f"variant-{variant_id}"
    if mode == "suite":
        assert suite is not None
        return f"suite-{suite}"
    return None


def _mode_name(
    *,
    all_cases: bool,
    backend_id: str | None,
    backend_configuration_test_id: str | None,
    variant_id: str | None,
    suite: str | None,
) -> str:
    if all_cases:
        return "all"
    if backend_configuration_test_id is not None:
        return "backend_configuration_test"
    if backend_id is not None:
        return "backend"
    if variant_id is not None:
        return "variant"
    assert suite is not None
    return "suite"


def entrypoint() -> None:
    """Console script wrapper."""
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
