#!/usr/bin/env python3
"""Create a deterministic, balanced GLM-ASR benchmark run plan.

The plan is data, not an executor.  Commands are represented as argv arrays so
that a later orchestrator can use ``subprocess`` without invoking a shell.  One
task always maps to exactly one of the runner's mutually exclusive measurement
profiles and to a unique, no-overwrite result path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from faster_glm_asr.benchmarking import provenance
from faster_glm_asr.benchmarking import runner as benchmark

CONFIG_SCHEMA = "asr-benchmark-matrix-config-v0.1"
PLAN_SCHEMA = "asr-benchmark-run-plan-v0.1"
PLANNER_VERSION = "seeded-factorial-latin-rotation-v2-source-locked"
CURRENT_PYTHON_PLACEHOLDER = "${PYTHON_EXECUTABLE}"

IMPLEMENTATIONS = (
    "hf_cached",
    "hf_no_cache",
    "custom_full_prefix",
    "custom_greedy_full_prefix",
    "custom_tuple_cache",
    "custom_static_cache",
)
PROFILES = (
    "canonical_latency",
    "request_memory",
    "diagnostic_phase",
)
RUNNER_STEMS = {
    "hf_cached": "hf-cached",
    "hf_no_cache": "hf-no-cache",
    "custom_full_prefix": "custom-full-prefix",
    "custom_greedy_full_prefix": "custom-greedy-full-prefix",
    "custom_tuple_cache": "custom-tuple-cache",
    "custom_static_cache": "custom-static-cache",
}
PROFILE_STEMS = {
    "canonical_latency": "canonical",
    "request_memory": "memory",
    "diagnostic_phase": "phases",
}
CONFIG_KEYS = {
    "schema_version",
    "python_executable",
    "benchmark_script",
    "manifest",
    "environment_lock",
    "model_snapshot_root",
    "model_snapshot_manifest",
    "readiness_report",
    "output_root",
    "dataset_label",
    "artifact_scope",
    "model_name",
    "model_revision",
    "implementations",
    "profiles",
    "outer_passes",
    "seed",
    "warmup",
    "repeats",
    "max_new_tokens",
    "nvml_sample_ms",
    "include_command_argv",
}
OPAQUE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
HEX_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _write_json_atomically(output: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a plan while refusing every overwrite."""
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link is an atomic create-if-absent operation.  os.replace would
        # silently destroy evidence if another run published the same path.
        os.link(temporary, output)
        temporary.unlink()
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _require_int(configuration: Mapping[str, Any], key: str, *, minimum: int) -> int:
    value = configuration.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _require_string(configuration: Mapping[str, Any], key: str) -> str:
    value = configuration.get(key)
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{key} must be a non-empty string without NUL bytes")
    return value


def _validate_exact_sequence(value: Any, *, label: str, expected: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a JSON array of strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicate items")
    unknown = sorted(set(value) - set(expected))
    missing = sorted(set(expected) - set(value))
    if unknown or missing:
        raise ValueError(
            f"{label} must contain the exact supported set; unknown={unknown}, missing={missing}"
        )
    return tuple(value)


def _validate_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(configuration, dict):
        raise ValueError("matrix configuration must be a JSON object")
    unknown = sorted(set(configuration) - CONFIG_KEYS)
    missing = sorted(CONFIG_KEYS - set(configuration))
    if unknown or missing:
        raise ValueError(
            f"matrix configuration schema mismatch; unknown={unknown}, missing={missing}"
        )
    if configuration.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA!r}")

    validated = dict(configuration)
    for key in (
        "python_executable",
        "benchmark_script",
        "manifest",
        "environment_lock",
        "model_snapshot_root",
        "model_snapshot_manifest",
        "readiness_report",
        "output_root",
        "dataset_label",
        "artifact_scope",
        "model_name",
        "model_revision",
    ):
        validated[key] = _require_string(configuration, key)
    configured_python = validated["python_executable"]
    if configured_python == CURRENT_PYTHON_PLACEHOLDER:
        configured_python = str(Path(sys.executable).resolve())
    python_path = Path(configured_python)
    if not python_path.is_absolute():
        raise ValueError(
            f"python_executable must be an absolute path or {CURRENT_PYTHON_PLACEHOLDER!r}"
        )
    python_path = python_path.resolve()
    if not python_path.is_file():
        raise FileNotFoundError(f"python_executable does not exist: {python_path}")
    validated["python_executable"] = str(python_path)
    if not OPAQUE_LABEL_RE.fullmatch(validated["dataset_label"]):
        raise ValueError("dataset_label must be an opaque path-safe label")
    if validated["artifact_scope"] not in {"private", "public"}:
        raise ValueError("artifact_scope must be 'private' or 'public'")
    if validated["artifact_scope"] != "private":
        raise ValueError("formal matrix artifacts must remain private")
    if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", validated["model_revision"]):
        raise ValueError("model_revision must be a pinned 40- or 64-hex revision")
    validated["model_revision"] = validated["model_revision"].lower()

    validated["implementations"] = list(
        _validate_exact_sequence(
            configuration.get("implementations"),
            label="implementations",
            expected=IMPLEMENTATIONS,
        )
    )
    validated["profiles"] = list(
        _validate_exact_sequence(configuration.get("profiles"), label="profiles", expected=PROFILES)
    )
    validated["outer_passes"] = _require_int(configuration, "outer_passes", minimum=3)
    validated["seed"] = _require_int(configuration, "seed", minimum=0)
    validated["warmup"] = _require_int(configuration, "warmup", minimum=1)
    validated["repeats"] = _require_int(configuration, "repeats", minimum=1)
    validated["max_new_tokens"] = _require_int(configuration, "max_new_tokens", minimum=1)
    nvml_sample_ms = configuration.get("nvml_sample_ms")
    if (
        not isinstance(nvml_sample_ms, (int, float))
        or isinstance(nvml_sample_ms, bool)
        or not math.isfinite(float(nvml_sample_ms))
        or float(nvml_sample_ms) <= 0
    ):
        raise ValueError("nvml_sample_ms must be a finite number > 0")
    validated["nvml_sample_ms"] = float(nvml_sample_ms)
    if not isinstance(configuration.get("include_command_argv"), bool):
        raise ValueError("include_command_argv must be boolean")
    return validated


def _relative_repo_path(
    value: str,
    repository_root: Path,
    *,
    label: str,
    must_be_file: bool,
) -> tuple[str, Path]:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must be a repository-relative path without '..'")
    repository_root = repository_root.resolve()
    resolved = (repository_root / candidate).resolve()
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository root") from exc
    if must_be_file and not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return relative.as_posix(), resolved


def _repository_state(repository_root: Path) -> dict[str, Any]:
    def run(arguments: Sequence[str]) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    commit = run(("rev-parse", "HEAD")).lower()
    if not HEX_COMMIT_RE.fullmatch(commit):
        raise ValueError("git rev-parse did not return a full SHA")
    return {
        "commit": commit,
        "dirty": bool(run(("status", "--porcelain", "--untracked-files=normal"))),
    }


def _source_lock(repository_root: Path, *, allow_unavailable_git: bool = False) -> dict[str, Any]:
    """Fingerprint every package source, including uncommitted local files.

    The aggregate lock is an integrity binding, not a digital signature or an
    authenticity guarantee. ``allow_unavailable_git`` is retained as a public
    compatibility seam but source collection itself never depends on Git.
    """
    repository_root = repository_root.resolve()
    del allow_unavailable_git
    payload = benchmark._source_metadata(repository_root)
    return {**payload, "lock_sha256": _canonical_sha256(payload)}


def _validate_repository_state(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"commit", "dirty"}:
        raise ValueError("repository_state must contain exactly commit and dirty")
    commit = value.get("commit")
    if not isinstance(commit, str) or not HEX_COMMIT_RE.fullmatch(commit.lower()):
        raise ValueError("repository_state commit must be a full Git SHA")
    if not isinstance(value.get("dirty"), bool):
        raise ValueError("repository_state dirty must be boolean")
    return {"commit": commit.lower(), "dirty": value["dirty"]}


def _seeded_permutation(values: Sequence[str], seed: int, domain: str) -> tuple[str, ...]:
    """Use SHA256 ordering instead of relying on a runtime RNG implementation."""
    return tuple(
        sorted(
            values,
            key=lambda item: hashlib.sha256(f"{seed}\0{domain}\0{item}".encode()).digest(),
        )
    )


def _schedule(
    implementations: Sequence[str], profiles: Sequence[str], passes: int, seed: int
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Return complete factorial passes with cyclic position/carryover rotation."""
    implementation_order = _seeded_permutation(implementations, seed, "implementation")
    profile_order = _seeded_permutation(profiles, seed, "profile")
    result = []
    for pass_index in range(passes):
        tasks = []
        for cycle in range(len(profile_order)):
            for position in range(len(implementation_order)):
                implementation = implementation_order[
                    (position + pass_index) % len(implementation_order)
                ]
                profile = profile_order[(cycle + position + pass_index) % len(profile_order)]
                tasks.append((implementation, profile))
        expected = {
            (implementation, profile) for implementation in implementations for profile in profiles
        }
        if set(tasks) != expected or len(tasks) != len(expected):
            raise AssertionError("internal schedule is not a complete factorial pass")
        result.append(tuple(tasks))
    return tuple(result)


def _profile_arguments(profile: str, nvml_sample_ms: float) -> tuple[str, ...]:
    if profile == "canonical_latency":
        return ()
    if profile == "request_memory":
        return ("--nvml-sample-ms", format(nvml_sample_ms, "g"))
    if profile == "diagnostic_phase":
        return ("--phase-timing",)
    raise ValueError(f"unsupported measurement profile: {profile}")


def _command_argv(
    configuration: Mapping[str, Any],
    *,
    implementation: str,
    profile: str,
    output_path: str,
    benchmark_script: str,
    manifest: str,
    environment_lock: str,
    model_snapshot_root: str,
    model_snapshot_manifest: str,
    readiness_report: str,
) -> list[str]:
    argv = [
        str(configuration["python_executable"]),
        benchmark_script,
        "--repository-root",
        ".",
        "--formal",
        "--implementation",
        implementation,
        "--artifact-scope",
        str(configuration["artifact_scope"]),
        "--manifest",
        manifest,
        "--environment-lock",
        environment_lock,
        "--model-snapshot-root",
        model_snapshot_root,
        "--model-snapshot-manifest",
        model_snapshot_manifest,
        "--readiness-report",
        readiness_report,
        "--model-name",
        str(configuration["model_name"]),
        "--model-revision",
        str(configuration["model_revision"]),
        "--warmup",
        str(configuration["warmup"]),
        "--repeats",
        str(configuration["repeats"]),
        "--max-new-tokens",
        str(configuration["max_new_tokens"]),
        *_profile_arguments(profile, float(configuration["nvml_sample_ms"])),
        "--output",
        output_path,
    ]
    if "--overwrite" in argv:
        raise AssertionError("planner must never authorize benchmark output overwrite")
    return argv


def build_plan(
    configuration: Mapping[str, Any],
    *,
    repository_root: Path,
    configuration_path: str,
    configuration_file_sha256: str,
    repository_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate inputs and build a deterministic plan without executing a model."""
    repository_state_was_injected = repository_state is not None
    configuration = _validate_configuration(configuration)
    repository_root = repository_root.resolve()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", configuration_file_sha256):
        raise ValueError("configuration_file_sha256 must be a SHA256")
    configuration_file_sha256 = configuration_file_sha256.lower()
    configuration_path, _ = _relative_repo_path(
        configuration_path,
        repository_root,
        label="configuration_path",
        must_be_file=True,
    )
    benchmark_script, benchmark_path = _relative_repo_path(
        str(configuration["benchmark_script"]),
        repository_root,
        label="benchmark_script",
        must_be_file=True,
    )
    manifest, manifest_path = _relative_repo_path(
        str(configuration["manifest"]),
        repository_root,
        label="manifest",
        must_be_file=True,
    )
    environment_lock, environment_lock_path = _relative_repo_path(
        str(configuration["environment_lock"]),
        repository_root,
        label="environment_lock",
        must_be_file=True,
    )
    model_snapshot_root, model_snapshot_root_path = _relative_repo_path(
        str(configuration["model_snapshot_root"]),
        repository_root,
        label="model_snapshot_root",
        must_be_file=False,
    )
    if not model_snapshot_root_path.is_dir():
        raise FileNotFoundError(
            f"model_snapshot_root is not a directory: {model_snapshot_root_path}"
        )
    model_snapshot_manifest, model_snapshot_manifest_path = _relative_repo_path(
        str(configuration["model_snapshot_manifest"]),
        repository_root,
        label="model_snapshot_manifest",
        must_be_file=True,
    )
    readiness_report, readiness_report_path = _relative_repo_path(
        str(configuration["readiness_report"]),
        repository_root,
        label="readiness_report",
        must_be_file=True,
    )
    output_root, output_root_path = _relative_repo_path(
        str(configuration["output_root"]),
        repository_root,
        label="output_root",
        must_be_file=False,
    )
    if output_root_path.is_file():
        raise ValueError("output_root is an existing file")
    if configuration["artifact_scope"] == "private" and tuple(
        part.lower() for part in Path(output_root).parts[:2]
    ) != ("artifacts", "private"):
        raise ValueError("private artifact output_root must be under artifacts/private/")

    python_path = Path(configuration["python_executable"])
    source_lock = _source_lock(
        repository_root,
        allow_unavailable_git=repository_state_was_injected,
    )

    state = _validate_repository_state(
        repository_state if repository_state is not None else _repository_state(repository_root)
    )
    if state["dirty"]:
        raise ValueError("formal benchmark planning requires a clean Git checkout")
    model_snapshot_binding = provenance.validate_model_snapshot(
        model_snapshot_root_path,
        model_snapshot_manifest_path,
        expected_repo_id=configuration["model_name"],
        expected_revision=configuration["model_revision"],
    )
    readiness_binding = provenance.validate_readiness_report(
        readiness_report_path,
        repository_root=repository_root,
    )
    if model_snapshot_binding["contract_sha256"] != readiness_binding["contract_sha256"]:
        raise ValueError("model snapshot and readiness report contract hashes differ")
    semantic_configuration_sha256 = _canonical_sha256(configuration)
    input_bindings = {
        "manifest_path": manifest,
        "manifest_sha256": _sha256_file(manifest_path),
        "environment_lock_path": environment_lock,
        "environment_lock_sha256": _sha256_file(environment_lock_path),
        "model_snapshot_root_path": model_snapshot_root,
        "model_snapshot_manifest_path": model_snapshot_manifest,
        "model_snapshot_manifest_sha256": model_snapshot_binding["manifest_sha256"],
        "model_snapshot_files_aggregate_sha256": model_snapshot_binding["files_aggregate_sha256"],
        "model_snapshot_binding": model_snapshot_binding,
        "readiness_report_path": readiness_report,
        "readiness_report_sha256": readiness_binding["report_sha256"],
        "readiness_binding": readiness_binding,
        "model_name": configuration["model_name"],
        "model_revision": configuration["model_revision"],
        "git_commit": state["commit"],
        "git_dirty": state["dirty"],
        "benchmark_script_path": benchmark_script,
        "benchmark_script_sha256": _sha256_file(benchmark_path),
        "python_executable_path": str(python_path),
        "python_executable_sha256": _sha256_file(python_path),
        "source_map_schema": source_lock["schema_version"],
        "source_files": source_lock["files"],
        "source_map_aggregate_sha256": source_lock["aggregate_sha256"],
        "source_lock_sha256": source_lock["lock_sha256"],
        "matrix_configuration_sha256": configuration_file_sha256,
        "matrix_semantic_sha256": semantic_configuration_sha256,
    }
    schedule = _schedule(
        configuration["implementations"],
        configuration["profiles"],
        int(configuration["outer_passes"]),
        int(configuration["seed"]),
    )

    tasks = []
    output_paths = set()
    matrix_directory = f"matrix-{configuration_file_sha256[:12]}-seed-{configuration['seed']}"
    for pass_index, pass_tasks in enumerate(schedule, start=1):
        for order_index, (implementation, profile) in enumerate(pass_tasks, start=1):
            filename = (
                f"{order_index:03d}-{configuration['dataset_label']}-"
                f"{RUNNER_STEMS[implementation]}-{PROFILE_STEMS[profile]}.json"
            )
            output_path = (
                Path(output_root) / matrix_directory / f"pass-{pass_index:03d}" / filename
            ).as_posix()
            if output_path in output_paths:
                raise AssertionError(f"duplicate planned output path: {output_path}")
            output_paths.add(output_path)
            absolute_output = (repository_root / Path(output_path)).resolve()
            if absolute_output.exists():
                raise FileExistsError(f"planned benchmark output already exists: {absolute_output}")

            argv = _command_argv(
                configuration,
                implementation=implementation,
                profile=profile,
                output_path=output_path,
                benchmark_script=benchmark_script,
                manifest=manifest,
                environment_lock=environment_lock,
                model_snapshot_root=model_snapshot_root,
                model_snapshot_manifest=model_snapshot_manifest,
                readiness_report=readiness_report,
            )
            task_configuration = {
                "outer_pass": pass_index,
                "order_in_pass": order_index,
                "implementation": implementation,
                "measurement_profile": profile,
                "output_path": output_path,
                "warmup": configuration["warmup"],
                "repeats": configuration["repeats"],
                "max_new_tokens": configuration["max_new_tokens"],
                "artifact_scope": configuration["artifact_scope"],
                "bindings": input_bindings,
                "command_argv_sha256": _canonical_sha256(argv),
            }
            task = {
                "task_id": (
                    f"pass-{pass_index:03d}-order-{order_index:03d}-{implementation}-{profile}"
                ),
                "outer_pass": pass_index,
                "order_in_pass": order_index,
                "implementation": implementation,
                "measurement_profile": profile,
                "output_path": output_path,
                "bindings": {
                    **input_bindings,
                    "task_configuration_sha256": _canonical_sha256(task_configuration),
                },
                "command_argv_sha256": _canonical_sha256(argv),
            }
            if configuration["include_command_argv"]:
                task["command_argv"] = argv
            tasks.append(task)

    expected_task_count = int(configuration["outer_passes"]) * len(IMPLEMENTATIONS) * len(PROFILES)
    if len(tasks) != expected_task_count or len(output_paths) != expected_task_count:
        raise AssertionError("internal task cardinality mismatch")

    per_pass_evidence = []
    for pass_index in range(1, int(configuration["outer_passes"]) + 1):
        selected = [task for task in tasks if task["outer_pass"] == pass_index]
        per_pass_evidence.append(
            {
                "outer_pass": pass_index,
                "task_count": len(selected),
                "implementation_counts": dict(
                    sorted(Counter(task["implementation"] for task in selected).items())
                ),
                "profile_counts": dict(
                    sorted(Counter(task["measurement_profile"] for task in selected).items())
                ),
                "unique_pair_count": len(
                    {(task["implementation"], task["measurement_profile"]) for task in selected}
                ),
            }
        )

    return {
        "schema_version": PLAN_SCHEMA,
        "planner": {
            "algorithm": PLANNER_VERSION,
            "seed": configuration["seed"],
            "outer_passes": configuration["outer_passes"],
            "model_execution_performed": False,
            "shell_execution_required": False,
            "command_representation": "argv-array"
            if configuration["include_command_argv"]
            else "sha256-only",
        },
        "repository": state,
        "matrix_configuration": {
            "path": configuration_path,
            "file_sha256": configuration_file_sha256,
            "semantic_sha256": semantic_configuration_sha256,
        },
        "inputs": {
            "manifest": {"path": manifest, "sha256": input_bindings["manifest_sha256"]},
            "environment_lock": {
                "path": environment_lock,
                "sha256": input_bindings["environment_lock_sha256"],
            },
            "model_snapshot": {
                "root": model_snapshot_root,
                "manifest": model_snapshot_manifest,
                **model_snapshot_binding,
            },
            "readiness": {
                "path": readiness_report,
                **readiness_binding,
            },
            "benchmark_script": {
                "path": benchmark_script,
                "sha256": input_bindings["benchmark_script_sha256"],
            },
            "python_executable": {
                "path": input_bindings["python_executable_path"],
                "sha256": input_bindings["python_executable_sha256"],
            },
            "sources": source_lock,
            "model_name": configuration["model_name"],
            "model_revision": configuration["model_revision"],
        },
        "parameters": {
            "artifact_scope": configuration["artifact_scope"],
            "warmup_per_task": configuration["warmup"],
            "repeats_per_task": configuration["repeats"],
            "repeats_per_implementation_profile_total": (
                configuration["repeats"] * configuration["outer_passes"]
            ),
            "max_new_tokens": configuration["max_new_tokens"],
            "request_memory_nvml_sample_ms": configuration["nvml_sample_ms"],
        },
        "balance_evidence": {
            "design": "complete 6x3 factorial in every pass",
            "implementation_occurrences_per_pass": len(PROFILES),
            "profile_occurrences_per_pass": len(IMPLEMENTATIONS),
            "pair_occurrences_across_plan": configuration["outer_passes"],
            "cyclic_schedule_period_passes": len(IMPLEMENTATIONS),
            "per_pass": per_pass_evidence,
        },
        "task_count": len(tasks),
        "tasks": tasks,
    }


def plan_from_config_file(
    configuration_path: Path,
    *,
    repository_root: Path,
    repository_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    configuration_path = configuration_path.resolve()
    try:
        relative_configuration = configuration_path.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise ValueError("configuration file must be inside the repository") from exc
    raw = configuration_path.read_bytes()
    try:
        configuration = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("matrix configuration must be UTF-8 JSON") from exc
    return build_plan(
        configuration,
        repository_root=repository_root,
        configuration_path=relative_configuration,
        configuration_file_sha256=_sha256_bytes(raw),
        repository_state=repository_state,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate (but do not execute) a balanced ASR benchmark plan."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    args = parser.parse_args(argv)

    repository_root = args.repository_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = repository_root / output
    output = output.resolve()
    try:
        output.relative_to(repository_root)
    except ValueError:
        parser.error("--output must stay inside --repository-root")
    try:
        plan = plan_from_config_file(
            args.config if args.config.is_absolute() else repository_root / args.config,
            repository_root=repository_root,
        )
        _write_json_atomically(output, plan)
    except (FileExistsError, FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(f"wrote {plan['task_count']} planned tasks to {output}")
    print(f"matrix_configuration_sha256={plan['matrix_configuration']['file_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
