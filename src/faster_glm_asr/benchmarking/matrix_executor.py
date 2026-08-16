#!/usr/bin/env python3
"""Validate or serially execute an evidence-oriented ASR benchmark run plan.

Validation is the default.  Model processes are launched only when the caller
passes ``--execute``.  The executor never invokes a shell and never overwrites a
planned result.  Execution progress is an atomically checkpointed, hash-linked
event log.  Those hashes provide internal consistency checks only; they are not
an authenticity mechanism.

This module deliberately does not merge per-pass benchmark artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from faster_glm_asr.benchmarking import comparator, provenance
from faster_glm_asr.benchmarking import matrix_planner as planner

JOURNAL_SCHEMA = "asr-benchmark-run-journal-v0.1"
PLAN_KEYS = {
    "schema_version",
    "planner",
    "repository",
    "matrix_configuration",
    "inputs",
    "parameters",
    "balance_evidence",
    "task_count",
    "tasks",
}
JOURNAL_KEYS = {
    "schema_version",
    "plan_path",
    "plan_sha256",
    "created_at_utc",
    "events",
}
COMMON_EVENT_KEYS = {
    "sequence",
    "event",
    "task_id",
    "timestamp_utc",
    "previous_event_sha256",
    "event_sha256",
}
EVENT_KEYS = {
    "planned": COMMON_EVENT_KEYS | {"outer_pass", "order_in_pass"},
    "start": COMMON_EVENT_KEYS
    | {
        "attempt",
        "command_argv_sha256",
        "stdout_path",
        "stderr_path",
    },
    "end": COMMON_EVENT_KEYS
    | {
        "attempt",
        "start_event_sha256",
        "monotonic_duration_ns",
        "exit_code",
        "status",
        "stdout_path",
        "stdout_sha256",
        "stderr_path",
        "stderr_sha256",
        "output_path",
        "output_sha256",
        "error",
    },
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LOCK_SCHEMA = "asr-benchmark-run-lock-v0.1"


@dataclass(frozen=True)
class ValidatedPlan:
    path: Path
    relative_path: str
    file_sha256: str
    payload: dict[str, Any]
    configuration: dict[str, Any]
    repository_state: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _validate_utc(value: Any, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{label} must carry an explicit UTC offset")


def _load_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"cannot read {label}: {path}") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be a UTF-8 JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def _relative_path(
    value: str | Path,
    repository_root: Path,
    *,
    label: str,
    require_file: bool = False,
) -> tuple[str, Path]:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{label} must be a path string")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must be repository-relative and cannot escape")
    repository_root = repository_root.resolve()
    resolved = (repository_root / candidate).resolve()
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository root") from exc
    if require_file and not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return relative.as_posix(), resolved


def _path_inside_repository(
    path: Path, repository_root: Path, *, label: str, require_file: bool
) -> tuple[str, Path]:
    repository_root = repository_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the repository") from exc
    if require_file and not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return relative.as_posix(), resolved


def _expected_balance(tasks: Sequence[Mapping[str, Any]], outer_passes: int) -> dict[str, Any]:
    per_pass = []
    for pass_index in range(1, outer_passes + 1):
        selected = [task for task in tasks if task["outer_pass"] == pass_index]
        per_pass.append(
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
        "design": "complete 6x3 factorial in every pass",
        "implementation_occurrences_per_pass": len(planner.PROFILES),
        "profile_occurrences_per_pass": len(planner.IMPLEMENTATIONS),
        "pair_occurrences_across_plan": outer_passes,
        "cyclic_schedule_period_passes": len(planner.IMPLEMENTATIONS),
        "per_pass": per_pass,
    }


def validate_plan(
    plan_path: Path,
    *,
    repository_root: Path,
    repository_state: Mapping[str, Any] | None = None,
) -> ValidatedPlan:
    """Recompute all bindings and the full ordered task matrix."""
    repository_root = repository_root.resolve()
    relative_plan_path, plan_path = _path_inside_repository(
        plan_path, repository_root, label="plan", require_file=True
    )
    payload, plan_raw = _load_json_object(plan_path, "benchmark plan")
    if set(payload) != PLAN_KEYS:
        raise ValueError(
            "benchmark plan root schema mismatch; "
            f"unknown={sorted(set(payload) - PLAN_KEYS)}, "
            f"missing={sorted(PLAN_KEYS - set(payload))}"
        )
    if payload.get("schema_version") != planner.PLAN_SCHEMA:
        raise ValueError(f"unsupported plan schema: {payload.get('schema_version')!r}")

    matrix = payload.get("matrix_configuration")
    if not isinstance(matrix, dict) or set(matrix) != {
        "path",
        "file_sha256",
        "semantic_sha256",
    }:
        raise ValueError("matrix_configuration has an invalid schema")
    configuration_relative, configuration_path = _relative_path(
        matrix.get("path"),
        repository_root,
        label="matrix configuration path",
        require_file=True,
    )
    configuration, configuration_raw = _load_json_object(configuration_path, "matrix configuration")
    configuration = planner._validate_configuration(configuration)
    configuration_file_sha256 = planner._sha256_bytes(configuration_raw)
    configuration_semantic_sha256 = planner._canonical_sha256(configuration)
    if matrix != {
        "path": configuration_relative,
        "file_sha256": configuration_file_sha256,
        "semantic_sha256": configuration_semantic_sha256,
    }:
        raise ValueError("matrix configuration hash/path binding drift")

    actual_state = planner._validate_repository_state(
        repository_state
        if repository_state is not None
        else planner._repository_state(repository_root)
    )
    if payload.get("repository") != actual_state:
        raise ValueError("Git commit/dirty state drift from the planned repository state")
    if actual_state["dirty"]:
        raise ValueError("formal benchmark execution requires a clean Git checkout")

    benchmark_script, benchmark_path = _relative_path(
        configuration["benchmark_script"],
        repository_root,
        label="benchmark script",
        require_file=True,
    )
    manifest, manifest_path = _relative_path(
        configuration["manifest"],
        repository_root,
        label="manifest",
        require_file=True,
    )
    environment_lock, environment_path = _relative_path(
        configuration["environment_lock"],
        repository_root,
        label="environment lock",
        require_file=True,
    )
    model_snapshot_root, model_snapshot_root_path = _relative_path(
        configuration["model_snapshot_root"],
        repository_root,
        label="model snapshot root",
    )
    if not model_snapshot_root_path.is_dir():
        raise FileNotFoundError("model snapshot root is missing")
    model_snapshot_manifest, model_snapshot_manifest_path = _relative_path(
        configuration["model_snapshot_manifest"],
        repository_root,
        label="model snapshot manifest",
        require_file=True,
    )
    readiness_report, readiness_report_path = _relative_path(
        configuration["readiness_report"],
        repository_root,
        label="readiness report",
        require_file=True,
    )
    output_root, output_root_path = _relative_path(
        configuration["output_root"],
        repository_root,
        label="output root",
    )
    if output_root_path.is_file():
        raise ValueError("output root is an existing file")
    if configuration["artifact_scope"] == "private" and tuple(
        part.lower() for part in Path(output_root).parts[:2]
    ) != ("artifacts", "private"):
        raise ValueError("private artifact output root must be under artifacts/private/")

    python_path = Path(configuration["python_executable"])
    source_lock = planner._source_lock(
        repository_root,
        allow_unavailable_git=repository_state is not None,
    )
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

    input_bindings = {
        "manifest_path": manifest,
        "manifest_sha256": planner._sha256_file(manifest_path),
        "environment_lock_path": environment_lock,
        "environment_lock_sha256": planner._sha256_file(environment_path),
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
        "git_commit": actual_state["commit"],
        "git_dirty": actual_state["dirty"],
        "benchmark_script_path": benchmark_script,
        "benchmark_script_sha256": planner._sha256_file(benchmark_path),
        "python_executable_path": str(python_path),
        "python_executable_sha256": planner._sha256_file(python_path),
        "source_map_schema": source_lock["schema_version"],
        "source_files": source_lock["files"],
        "source_map_aggregate_sha256": source_lock["aggregate_sha256"],
        "source_lock_sha256": source_lock["lock_sha256"],
        "matrix_configuration_sha256": configuration_file_sha256,
        "matrix_semantic_sha256": configuration_semantic_sha256,
    }
    expected_inputs = {
        "manifest": {
            "path": manifest,
            "sha256": input_bindings["manifest_sha256"],
        },
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
    }
    if payload.get("inputs") != expected_inputs:
        raise ValueError("manifest/environment/script/model binding drift")

    expected_planner = {
        "algorithm": planner.PLANNER_VERSION,
        "seed": configuration["seed"],
        "outer_passes": configuration["outer_passes"],
        "model_execution_performed": False,
        "shell_execution_required": False,
        "command_representation": (
            "argv-array" if configuration["include_command_argv"] else "sha256-only"
        ),
    }
    if payload.get("planner") != expected_planner:
        raise ValueError("planner metadata drift")
    expected_parameters = {
        "artifact_scope": configuration["artifact_scope"],
        "warmup_per_task": configuration["warmup"],
        "repeats_per_task": configuration["repeats"],
        "repeats_per_implementation_profile_total": (
            configuration["repeats"] * configuration["outer_passes"]
        ),
        "max_new_tokens": configuration["max_new_tokens"],
        "request_memory_nvml_sample_ms": configuration["nvml_sample_ms"],
    }
    if payload.get("parameters") != expected_parameters:
        raise ValueError("plan parameter drift")

    tasks = payload.get("tasks")
    expected_count = (
        configuration["outer_passes"] * len(planner.IMPLEMENTATIONS) * len(planner.PROFILES)
    )
    if not isinstance(tasks, list) or len(tasks) != expected_count:
        raise ValueError(
            f"plan has omitted or extra tasks: expected {expected_count}, "
            f"observed {len(tasks) if isinstance(tasks, list) else 'non-list'}"
        )
    if payload.get("task_count") != expected_count:
        raise ValueError("task_count does not match the complete matrix")

    task_ids = []
    output_paths = []
    schedule = planner._schedule(
        configuration["implementations"],
        configuration["profiles"],
        configuration["outer_passes"],
        configuration["seed"],
    )
    matrix_directory = f"matrix-{configuration_file_sha256[:12]}-seed-{configuration['seed']}"
    expected_tasks = []
    position = 0
    for pass_index, pass_tasks in enumerate(schedule, start=1):
        for order_index, (implementation, profile) in enumerate(pass_tasks, start=1):
            task = tasks[position]
            position += 1
            if not isinstance(task, dict):
                raise ValueError(f"task at position {position} is not an object")
            raw_output = task.get("output_path")
            output_relative, _ = _relative_path(
                raw_output,
                repository_root,
                label=f"task {position} output path",
            )
            expected_output = (
                Path(output_root)
                / matrix_directory
                / f"pass-{pass_index:03d}"
                / (
                    f"{order_index:03d}-{configuration['dataset_label']}-"
                    f"{planner.RUNNER_STEMS[implementation]}-"
                    f"{planner.PROFILE_STEMS[profile]}.json"
                )
            ).as_posix()
            if output_relative != expected_output:
                raise ValueError(
                    f"task {position} output path/order binding drift: expected {expected_output!r}"
                )
            argv = planner._command_argv(
                configuration,
                implementation=implementation,
                profile=profile,
                output_path=expected_output,
                benchmark_script=benchmark_script,
                manifest=manifest,
                environment_lock=environment_lock,
                model_snapshot_root=model_snapshot_root,
                model_snapshot_manifest=model_snapshot_manifest,
                readiness_report=readiness_report,
            )
            argv_sha256 = planner._canonical_sha256(argv)
            task_configuration = {
                "outer_pass": pass_index,
                "order_in_pass": order_index,
                "implementation": implementation,
                "measurement_profile": profile,
                "output_path": expected_output,
                "warmup": configuration["warmup"],
                "repeats": configuration["repeats"],
                "max_new_tokens": configuration["max_new_tokens"],
                "artifact_scope": configuration["artifact_scope"],
                "bindings": input_bindings,
                "command_argv_sha256": argv_sha256,
            }
            expected_task = {
                "task_id": (
                    f"pass-{pass_index:03d}-order-{order_index:03d}-{implementation}-{profile}"
                ),
                "outer_pass": pass_index,
                "order_in_pass": order_index,
                "implementation": implementation,
                "measurement_profile": profile,
                "output_path": expected_output,
                "bindings": {
                    **input_bindings,
                    "task_configuration_sha256": planner._canonical_sha256(task_configuration),
                },
                "command_argv_sha256": argv_sha256,
            }
            if configuration["include_command_argv"]:
                expected_task["command_argv"] = argv
            if task != expected_task:
                raise ValueError(f"task {position} command argv/task hash/bindings drift")
            task_ids.append(task["task_id"])
            output_paths.append(task["output_path"])
            expected_tasks.append(expected_task)

    if len(task_ids) != len(set(task_ids)):
        raise ValueError("plan contains duplicate task IDs")
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("plan contains duplicate output paths")
    if payload.get("balance_evidence") != _expected_balance(
        expected_tasks, configuration["outer_passes"]
    ):
        raise ValueError("balance evidence does not match the ordered task matrix")

    return ValidatedPlan(
        path=plan_path,
        relative_path=relative_plan_path,
        file_sha256=planner._sha256_bytes(plan_raw),
        payload=payload,
        configuration=configuration,
        repository_state=actual_state,
    )


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if not math.isfinite(float(value)):
        return False
    return float(value) > 0 if positive else float(value) >= 0


def _expected_metric_eligibility(profile: str) -> dict[str, bool]:
    return {
        "canonical_latency": profile == "canonical_latency",
        "request_memory_observed_peak": profile == "request_memory",
        "diagnostic_cuda_phases": profile == "diagnostic_phase",
        "cold_start": False,
        "model_load": False,
    }


def _validate_numeric_summaries(payload: Mapping[str, Any], label: str) -> None:
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{label}.samples must be a non-empty list")
    all_values = {
        "generation_ms": [],
        "processor_plus_generation_ms": [],
        "request_no_file_io_ms": [],
    }
    for sample_index, sample in enumerate(samples):
        location = f"{label}.samples[{sample_index}]"
        if not isinstance(sample, dict):
            raise ValueError(f"{location} must be an object")
        observations = sample.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError(f"{location}.observations must be non-empty")
        per_sample = {
            "generation_ms": [],
            "processor_plus_generation_ms": [],
            "request_no_file_io_ms": [],
            "rtf_generation": [],
        }
        for repeat_index, observation in enumerate(observations):
            if not isinstance(observation, dict):
                raise ValueError(f"{location}.observations[{repeat_index}] must be an object")
            for metric in per_sample:
                value = observation.get(metric)
                if not _finite_number(value, positive=True):
                    raise ValueError(
                        f"{location}.observations[{repeat_index}].{metric} "
                        "must be finite and positive"
                    )
                per_sample[metric].append(float(value))
            for metric in all_values:
                all_values[metric].append(float(observation[metric]))
        summary = sample.get("summary")
        if not isinstance(summary, dict):
            raise ValueError(f"{location}.summary must be an object")
        for metric, values in per_sample.items():
            comparator._validate_summary(
                summary.get(metric), values, f"{location}.summary.{metric}"
            )
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"{label}.aggregate must be an object")
    for metric, values in all_values.items():
        comparator._validate_summary(aggregate.get(metric), values, f"{label}.aggregate.{metric}")


def _validate_profile_instrumentation(
    payload: Mapping[str, Any],
    task: Mapping[str, Any],
    configuration: Mapping[str, Any],
    label: str,
) -> None:
    run = payload["run"]
    profile = task["measurement_profile"]
    if run.get("metric_eligibility") != _expected_metric_eligibility(profile):
        raise ValueError(f"{label}.run.metric_eligibility does not match the task")
    expected_nvml = float(configuration["nvml_sample_ms"]) if profile == "request_memory" else 0.0
    if not _finite_number(run.get("nvml_sample_ms")) or not math.isclose(
        float(run["nvml_sample_ms"]), expected_nvml, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{label}.run.nvml_sample_ms does not match the task")
    expected_phase = profile == "diagnostic_phase"
    if run.get("phase_timing") is not expected_phase:
        raise ValueError(f"{label}.run.phase_timing does not match the task")
    warning = run.get("phase_timing_warning")
    if expected_phase:
        if not isinstance(warning, str) or not warning:
            raise ValueError(f"{label}.run.phase_timing_warning must be non-empty")
    elif warning is not None:
        raise ValueError(f"{label}.run.phase_timing_warning must be null")

    for sample_index, sample in enumerate(payload["samples"]):
        for repeat_index, observation in enumerate(sample["observations"]):
            location = f"{label}.samples[{sample_index}].observations[{repeat_index}]"
            memory = observation.get("nvml_process_memory")
            phases = observation.get("diagnostic_cuda_phases")
            if profile == "request_memory":
                if (
                    not isinstance(memory, dict)
                    or memory.get("enabled") is not True
                    or memory.get("available") is not True
                    or not isinstance(memory.get("sample_count"), int)
                    or isinstance(memory.get("sample_count"), bool)
                    or memory["sample_count"] <= 0
                    or not _finite_number(memory.get("observed_peak_used_bytes"))
                ):
                    raise ValueError(f"{location}.nvml_process_memory is ineligible")
                nvml_samples = memory.get("samples")
                if (
                    not isinstance(nvml_samples, list)
                    or len(nvml_samples) != memory["sample_count"]
                    or any(
                        not isinstance(row, dict)
                        or not _finite_number(row.get("elapsed_ms"))
                        or not _finite_number(row.get("used_bytes"))
                        for row in nvml_samples
                    )
                    or not math.isclose(
                        float(memory.get("interval_ms", -1)),
                        expected_nvml,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or float(memory["observed_peak_used_bytes"])
                    != max(float(row["used_bytes"]) for row in nvml_samples)
                ):
                    raise ValueError(f"{location}.nvml_process_memory samples are invalid")
                if phases is not None:
                    raise ValueError(f"{location}.diagnostic_cuda_phases must be null")
            elif profile == "diagnostic_phase":
                if not isinstance(phases, dict) or not phases:
                    raise ValueError(f"{location}.diagnostic_cuda_phases is empty")
                for phase, evidence in phases.items():
                    if not isinstance(phase, str) or not phase or not isinstance(evidence, dict):
                        raise ValueError(f"{location}.diagnostic_cuda_phases is invalid")
                    values = evidence.get("events_ms")
                    if (
                        not isinstance(values, list)
                        or not values
                        or any(not _finite_number(value) for value in values)
                    ):
                        raise ValueError(f"{location}.{phase}.events_ms is invalid")
                    comparator._validate_summary(
                        evidence.get("summary_ms"),
                        [float(value) for value in values],
                        f"{location}.{phase}.summary_ms",
                    )
                    if not _finite_number(evidence.get("total_ms")) or not math.isclose(
                        float(evidence["total_ms"]),
                        sum(float(value) for value in values),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(f"{location}.{phase}.total_ms is invalid")
                if (
                    not isinstance(memory, dict)
                    or memory.get("enabled") is not False
                    or memory.get("available") is not False
                    or not isinstance(memory.get("sample_count"), int)
                    or isinstance(memory.get("sample_count"), bool)
                    or memory["sample_count"] != 0
                    or memory.get("samples") != []
                    or not _finite_number(memory.get("interval_ms"))
                    or float(memory["interval_ms"]) != 0.0
                ):
                    raise ValueError(f"{location}.nvml_process_memory must be disabled")
            else:
                if phases is not None:
                    raise ValueError(f"{location}.diagnostic_cuda_phases must be null")
                if (
                    not isinstance(memory, dict)
                    or memory.get("enabled") is not False
                    or memory.get("available") is not False
                    or not isinstance(memory.get("sample_count"), int)
                    or isinstance(memory.get("sample_count"), bool)
                    or memory["sample_count"] != 0
                    or memory.get("samples") != []
                    or not _finite_number(memory.get("interval_ms"))
                    or float(memory["interval_ms"]) != 0.0
                ):
                    raise ValueError(f"{location}.nvml_process_memory must be disabled")


def _validate_task_result(
    output_path: Path,
    task: Mapping[str, Any],
    validated: ValidatedPlan,
    repository_root: Path,
) -> dict[str, Any]:
    """Fail closed unless one task output is complete and plan-bound."""
    payload, _ = _load_json_object(output_path, f"task {task['task_id']} result")
    label = f"task {task['task_id']} result"
    if payload.get("schema_version") != comparator.SCHEMA_VERSION:
        raise ValueError(f"{label}: expected schema {comparator.SCHEMA_VERSION}")
    run = payload.get("run")
    if not isinstance(run, dict):
        raise ValueError(f"{label}.run must be an object")
    configuration = validated.configuration
    bindings = task["bindings"]
    expected_run = {
        "implementation": task["implementation"],
        "model_name": bindings["model_name"],
        "model_revision": bindings["model_revision"],
        "manifest_sha256": bindings["manifest_sha256"],
        "warmup": configuration["warmup"],
        "repeats": configuration["repeats"],
        "max_new_tokens": configuration["max_new_tokens"],
        "artifact_scope": configuration["artifact_scope"],
        "measurement_profile": task["measurement_profile"],
    }
    for key, expected in expected_run.items():
        if run.get(key) != expected:
            raise ValueError(f"{label}.run.{key} does not match the planned task")
    manifest_value = run.get("manifest_path")
    if not isinstance(manifest_value, str) or not Path(manifest_value).is_absolute():
        raise ValueError(f"{label}.run.manifest_path must be absolute")
    _, expected_manifest = _relative_path(
        bindings["manifest_path"], repository_root, label="bound manifest", require_file=True
    )
    if Path(manifest_value).resolve() != expected_manifest:
        raise ValueError(f"{label}.run.manifest_path does not match the plan")
    environment = payload.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("environment_lock_sha256") != bindings["environment_lock_sha256"]
    ):
        raise ValueError(f"{label}.environment lock does not match the plan")
    git = payload.get("git")
    if (
        not isinstance(git, dict)
        or git.get("commit") != bindings["git_commit"]
        or git.get("dirty") is not bindings["git_dirty"]
    ):
        raise ValueError(f"{label}.git provenance does not match the plan")
    expected_sources = {
        "schema_version": bindings["source_map_schema"],
        "files": bindings["source_files"],
        "aggregate_sha256": bindings["source_map_aggregate_sha256"],
    }
    if payload.get("sources") != expected_sources:
        raise ValueError(f"{label}.sources provenance does not match the source lock")
    if payload.get("model_snapshot") != bindings["model_snapshot_binding"]:
        raise ValueError(f"{label}.model_snapshot does not match the plan")
    if payload.get("readiness") != bindings["readiness_binding"]:
        raise ValueError(f"{label}.readiness does not match the plan")
    if configuration["artifact_scope"] != "private":
        raise ValueError("matrix task result validation currently requires private evidence")

    # Reuse the canonical validator for the full common artifact/environment/
    # sample schema.  Instrumented profiles differ only in explicit run flags
    # and their per-observation diagnostic payloads, validated immediately below.
    canonical_view = copy.deepcopy(payload)
    canonical_view["run"]["measurement_profile"] = "canonical_latency"
    canonical_view["run"]["metric_eligibility"] = dict(comparator.CANONICAL_METRIC_ELIGIBILITY)
    canonical_view["run"]["nvml_sample_ms"] = 0.0
    canonical_view["run"]["phase_timing"] = False
    canonical_view["run"]["phase_timing_warning"] = None
    comparator._validate_canonical_payload(canonical_view, label)
    _validate_numeric_summaries(payload, label)
    _validate_profile_instrumentation(payload, task, configuration, label)
    return payload


def _validate_private_journal_dir(journal_dir: Path, repository_root: Path) -> tuple[str, Path]:
    if journal_dir.is_absolute():
        relative, resolved = _path_inside_repository(
            journal_dir,
            repository_root,
            label="journal directory",
            require_file=False,
        )
    else:
        relative, resolved = _relative_path(journal_dir, repository_root, label="journal directory")
    parts = Path(relative).parts
    if len(parts) < 3 or tuple(part.lower() for part in parts[:2]) != (
        "artifacts",
        "private",
    ):
        raise ValueError("journal directory must be under artifacts/private/")
    if resolved.is_file():
        raise ValueError("journal directory is an existing file")
    return relative, resolved


def _write_checkpoint(path: Path, payload: Mapping[str, Any], *, create: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if create:
            os.link(temporary, path)
            temporary.unlink()
        else:
            if not path.is_file():
                raise FileNotFoundError(f"journal checkpoint disappeared: {path}")
            os.replace(temporary, path)
        return planner._sha256_file(path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _append_event(
    journal_path: Path,
    journal: dict[str, Any],
    event_fields: Mapping[str, Any],
) -> dict[str, Any]:
    on_disk, _ = _load_json_object(journal_path, "run journal")
    if on_disk != journal:
        raise RuntimeError("journal checkpoint changed concurrently")
    previous = journal["events"][-1]["event_sha256"] if journal["events"] else "0" * 64
    event = {
        "sequence": len(journal["events"]) + 1,
        **event_fields,
        "previous_event_sha256": previous,
    }
    event["event_sha256"] = planner._canonical_sha256(event)
    journal["events"].append(event)
    _write_checkpoint(journal_path, journal, create=False)
    return event


def _new_journal(validated: ValidatedPlan, journal_path: Path) -> dict[str, Any]:
    journal = {
        "schema_version": JOURNAL_SCHEMA,
        "plan_path": validated.relative_path,
        "plan_sha256": validated.file_sha256,
        "created_at_utc": _utc_now(),
        "events": [],
    }
    _validate_utc(journal["created_at_utc"], "journal created_at_utc")
    for task in validated.payload["tasks"]:
        event = {
            "sequence": len(journal["events"]) + 1,
            "event": "planned",
            "task_id": task["task_id"],
            "timestamp_utc": _utc_now(),
            "outer_pass": task["outer_pass"],
            "order_in_pass": task["order_in_pass"],
            "previous_event_sha256": (
                journal["events"][-1]["event_sha256"] if journal["events"] else "0" * 64
            ),
        }
        event["event_sha256"] = planner._canonical_sha256(event)
        journal["events"].append(event)
    _write_checkpoint(journal_path, journal, create=True)
    return journal


def _validate_journal(
    journal_path: Path,
    validated: ValidatedPlan,
    journal_directory: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, int]]:
    journal, _ = _load_json_object(journal_path, "run journal")
    if set(journal) != JOURNAL_KEYS or journal.get("schema_version") != JOURNAL_SCHEMA:
        raise ValueError("run journal root schema is invalid")
    if journal.get("plan_path") != validated.relative_path:
        raise ValueError("run journal is bound to a different plan path")
    if journal.get("plan_sha256") != validated.file_sha256:
        raise ValueError("run journal plan SHA256 drift")
    _validate_utc(journal.get("created_at_utc"), "journal created_at_utc")
    events = journal.get("events")
    if not isinstance(events, list):
        raise ValueError("run journal events must be a list")

    tasks = validated.payload["tasks"]
    task_by_id = {task["task_id"]: task for task in tasks}
    previous = "0" * 64
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            raise ValueError(f"journal event {index} is not an object")
        kind = event.get("event")
        if kind not in EVENT_KEYS or set(event) != EVENT_KEYS[kind]:
            raise ValueError(f"journal event {index} has an invalid {kind!r} schema")
        if event.get("sequence") != index:
            raise ValueError("journal event sequence is not contiguous")
        if event.get("previous_event_sha256") != previous:
            raise ValueError("journal event hash linkage is internally inconsistent")
        unsigned = dict(event)
        event_sha256 = unsigned.pop("event_sha256")
        if not isinstance(event_sha256, str) or not SHA256_RE.fullmatch(event_sha256):
            raise ValueError("journal event SHA256 is malformed")
        if event_sha256 != planner._canonical_sha256(unsigned):
            raise ValueError("journal event content hash drift")
        previous = event_sha256
        if event.get("task_id") not in task_by_id:
            raise ValueError("journal references an unknown task")
        _validate_utc(event.get("timestamp_utc"), f"journal event {index} timestamp")

    if len(events) < len(tasks):
        raise ValueError("journal omits planned task events")
    for task, event in zip(tasks, events[: len(tasks)], strict=True):
        if event.get("event") != "planned" or event.get("task_id") != task["task_id"]:
            raise ValueError("journal planned events do not match task order")
        if (
            event.get("outer_pass") != task["outer_pass"]
            or event.get("order_in_pass") != task["order_in_pass"]
        ):
            raise ValueError("journal planned event position drift")
    if any(event.get("event") == "planned" for event in events[len(tasks) :]):
        raise ValueError("journal has duplicate/late planned events")

    successes: dict[str, str] = {}
    attempts: dict[str, int] = {task["task_id"]: 0 for task in tasks}
    starts: dict[str, Mapping[str, Any]] = {}
    ended_starts = set()
    active_start_sha256: str | None = None
    for event in events[len(tasks) :]:
        task_id = event["task_id"]
        first_pending = next(
            (task["task_id"] for task in tasks if task["task_id"] not in successes),
            None,
        )
        if event["event"] == "start":
            if active_start_sha256 is not None:
                raise ValueError("journal contains more than one active start event")
            if task_id != first_pending:
                raise ValueError("journal start events violate serial task order")
            expected_attempt = attempts[task_id] + 1
            if event["attempt"] != expected_attempt:
                raise ValueError("journal task attempt number is not contiguous")
            attempts[task_id] = expected_attempt
            if event["command_argv_sha256"] != task_by_id[task_id]["command_argv_sha256"]:
                raise ValueError("journal command argv hash drift")
            for key in ("stdout_path", "stderr_path"):
                relative, resolved = _relative_path(
                    event[key], repository_root, label=f"journal {key}"
                )
                try:
                    resolved.relative_to(journal_directory.resolve())
                except ValueError as exc:
                    raise ValueError(f"journal {key} escapes private journal dir") from exc
                if relative != event[key]:
                    raise ValueError(f"journal {key} is noncanonical")
            starts[event["event_sha256"]] = event
            active_start_sha256 = event["event_sha256"]
        elif event["event"] == "end":
            start = starts.get(event["start_event_sha256"])
            if start is None or start["task_id"] != task_id or start["attempt"] != event["attempt"]:
                raise ValueError("journal end event does not bind a matching start")
            if event["start_event_sha256"] != active_start_sha256:
                raise ValueError("journal end event is not paired with the active start")
            if event["start_event_sha256"] in ended_starts:
                raise ValueError("journal start event is consumed by multiple end events")
            ended_starts.add(event["start_event_sha256"])
            active_start_sha256 = None
            if (
                event["stdout_path"] != start["stdout_path"]
                or event["stderr_path"] != start["stderr_path"]
            ):
                raise ValueError("journal end log paths drift from start event")
            duration = event["monotonic_duration_ns"]
            if not isinstance(duration, int) or isinstance(duration, bool) or duration < 0:
                raise ValueError("journal monotonic duration is invalid")
            if event["exit_code"] is not None and (
                not isinstance(event["exit_code"], int) or isinstance(event["exit_code"], bool)
            ):
                raise ValueError("journal exit code is invalid")
            for key in ("stdout_sha256", "stderr_sha256"):
                if not isinstance(event[key], str) or not SHA256_RE.fullmatch(event[key]):
                    raise ValueError(f"journal {key} is invalid")
            for path_key, hash_key in (
                ("stdout_path", "stdout_sha256"),
                ("stderr_path", "stderr_sha256"),
            ):
                _, log_path = _relative_path(
                    event[path_key], repository_root, label=f"journal {path_key}"
                )
                if not log_path.is_file():
                    raise FileNotFoundError(f"journal log is missing: {log_path}")
                if planner._sha256_file(log_path) != event[hash_key]:
                    raise ValueError(f"journal {path_key} hash drift")
            if event["output_path"] != task_by_id[task_id]["output_path"]:
                raise ValueError("journal output path drift")
            if event["output_sha256"] is not None and (
                not isinstance(event["output_sha256"], str)
                or not SHA256_RE.fullmatch(event["output_sha256"])
            ):
                raise ValueError("journal output SHA256 is invalid")
            status = event["status"]
            if status not in {
                "success",
                "process_failed",
                "output_validation_failed",
                "launch_failed",
            }:
                raise ValueError("journal end status is invalid")
            if status == "success":
                if (
                    event["exit_code"] != 0
                    or event["output_sha256"] is None
                    or event["error"] is not None
                ):
                    raise ValueError("successful journal end lacks exit/output evidence")
                if task_id in successes:
                    raise ValueError("journal executes a task after recorded success")
                successes[task_id] = event["output_sha256"]
            elif status == "process_failed":
                if event["exit_code"] in {None, 0}:
                    raise ValueError("process_failed journal end has invalid exit code")
                if event["error"] is not None and not isinstance(event["error"], str):
                    raise ValueError("journal end error must be string or null")
            elif status == "output_validation_failed":
                if (
                    event["exit_code"] != 0
                    or not isinstance(event["error"], str)
                    or not event["error"]
                ):
                    raise ValueError("output_validation_failed journal end is inconsistent")
            elif status == "launch_failed":
                if (
                    event["exit_code"] is not None
                    or not isinstance(event["error"], str)
                    or not event["error"]
                ):
                    raise ValueError("launch_failed journal end is inconsistent")
    return journal, successes, attempts


def _active_journal_start(journal: Mapping[str, Any]) -> Mapping[str, Any] | None:
    active: Mapping[str, Any] | None = None
    for event in journal.get("events", []):
        if event.get("event") == "start":
            active = event
        elif event.get("event") == "end":
            active = None
    return active


def _validate_outputs(
    validated: ValidatedPlan,
    successful_outputs: Mapping[str, str],
    repository_root: Path,
) -> None:
    known_ids = {task["task_id"] for task in validated.payload["tasks"]}
    if set(successful_outputs) - known_ids:
        raise ValueError("successful output map contains an unknown task")
    for task in validated.payload["tasks"]:
        _, output_path = _relative_path(
            task["output_path"],
            repository_root,
            label=f"task {task['task_id']} output",
        )
        expected_hash = successful_outputs.get(task["task_id"])
        if expected_hash is None:
            if output_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite untrusted existing output: {output_path}"
                )
        else:
            if not output_path.is_file():
                raise FileNotFoundError(f"successful journal output is missing: {output_path}")
            if planner._sha256_file(output_path) != expected_hash:
                raise ValueError(f"successful output hash drift for task {task['task_id']}")
            _validate_task_result(output_path, task, validated, repository_root)


@dataclass
class _RunLock:
    path: Path
    payload_bytes: bytes

    def release(self) -> None:
        if not self.path.is_file():
            raise RuntimeError(f"run lock disappeared while held: {self.path}")
        if self.path.read_bytes() != self.payload_bytes:
            raise RuntimeError(f"run lock changed while held: {self.path}")
        self.path.unlink()


def _validate_private_lock_path(lock_file: Path, repository_root: Path) -> tuple[str, Path]:
    if lock_file.is_absolute():
        relative, resolved = _path_inside_repository(
            lock_file,
            repository_root,
            label="run lock",
            require_file=False,
        )
    else:
        relative, resolved = _relative_path(lock_file, repository_root, label="run lock")
    if tuple(part.lower() for part in Path(relative).parts[:2]) != (
        "artifacts",
        "private",
    ):
        raise ValueError("run lock must be under artifacts/private/")
    if resolved.is_dir():
        raise ValueError("run lock path is an existing directory")
    return relative, resolved


def _default_lock_file(plan_relative_path: str) -> Path:
    namespace = planner._sha256_bytes(plan_relative_path.encode("utf-8"))[:12]
    safe_stem = _safe_task_component(Path(plan_relative_path).stem)
    return Path("artifacts/private/run-locks") / f"{safe_stem}-{namespace}.lock"


def _acquire_run_lock(lock_path: Path, *, plan_sha256: str) -> _RunLock:
    if not SHA256_RE.fullmatch(plan_sha256):
        raise ValueError("run lock plan_sha256 must be a full SHA256")
    payload = {
        "schema_version": LOCK_SCHEMA,
        "plan_sha256": plan_sha256,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "created_at_utc": _utc_now(),
    }
    payload_bytes = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        created = True
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError(
            f"run lock already exists; refusing concurrent/stale-lock execution: {lock_path}"
        ) from exc
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            with suppress(FileNotFoundError):
                lock_path.unlink()
        raise
    return _RunLock(path=lock_path, payload_bytes=payload_bytes)


def _safe_task_component(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", task_id)


def _run_plan_locked(
    plan_path: Path,
    *,
    repository_root: Path,
    execute: bool = False,
    resume: bool = False,
    journal_dir: Path | None = None,
    repository_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a plan and, only when requested, execute its argv serially."""
    if resume and not execute:
        raise ValueError("--resume requires --execute")
    repository_root = repository_root.resolve()
    validated = validate_plan(
        plan_path,
        repository_root=repository_root,
        repository_state=repository_state,
    )
    if not execute:
        _validate_outputs(validated, {}, repository_root)
        return {
            "status": "validated",
            "executed": False,
            "task_count": validated.payload["task_count"],
            "plan_sha256": validated.file_sha256,
            "pass_merge_performed": False,
        }

    if not all("command_argv" in task for task in validated.payload["tasks"]):
        raise ValueError("execution requires a plan containing command argv arrays")
    if journal_dir is None:
        journal_dir = Path("artifacts/private/run-journals") / validated.path.stem
    journal_relative, journal_directory = _validate_private_journal_dir(
        journal_dir, repository_root
    )
    journal_path = journal_directory / "journal.json"

    if resume:
        if not journal_path.is_file():
            raise FileNotFoundError("--resume requires an existing journal.json")
        journal, successes, attempts = _validate_journal(
            journal_path,
            validated,
            journal_directory,
            repository_root,
        )
        active = _active_journal_start(journal)
        if active is not None:
            raise ValueError(
                "run journal ends with an active start; verify the prior process "
                "manually before creating new evidence"
            )
    else:
        if journal_path.exists():
            raise FileExistsError(
                f"journal already exists; use --resume after validation: {journal_path}"
            )
        if journal_directory.exists() and any(journal_directory.iterdir()):
            raise FileExistsError(
                f"refusing to mix evidence in non-empty journal dir: {journal_directory}"
            )
        _validate_outputs(validated, {}, repository_root)
        journal_directory.mkdir(parents=True, exist_ok=True)
        journal = _new_journal(validated, journal_path)
        journal, successes, attempts = _validate_journal(
            journal_path,
            validated,
            journal_directory,
            repository_root,
        )
    _validate_outputs(validated, successes, repository_root)

    skipped = len(successes)
    executed_count = 0
    logs_directory = journal_directory / "logs"
    logs_directory.mkdir(parents=True, exist_ok=True)
    for task in validated.payload["tasks"]:
        task_id = task["task_id"]
        if task_id in successes:
            continue

        # Recompute all bindings immediately before every process launch.  This
        # catches a manifest, lock, script, Git, config, or plan edit mid-run.
        current = validate_plan(
            validated.path,
            repository_root=repository_root,
            repository_state=repository_state,
        )
        if current.file_sha256 != validated.file_sha256:
            raise ValueError("plan changed during execution")
        _validate_outputs(current, successes, repository_root)

        attempt = attempts[task_id] + 1
        safe_id = _safe_task_component(task_id)
        stdout_path = logs_directory / f"{safe_id}.attempt-{attempt:03d}.stdout.txt"
        stderr_path = logs_directory / f"{safe_id}.attempt-{attempt:03d}.stderr.txt"
        stdout_relative = stdout_path.relative_to(repository_root).as_posix()
        stderr_relative = stderr_path.relative_to(repository_root).as_posix()
        if stdout_path.exists() or stderr_path.exists():
            raise FileExistsError("refusing to overwrite existing task stdout/stderr")

        start_event = _append_event(
            journal_path,
            journal,
            {
                "event": "start",
                "task_id": task_id,
                "timestamp_utc": _utc_now(),
                "attempt": attempt,
                "command_argv_sha256": task["command_argv_sha256"],
                "stdout_path": stdout_relative,
                "stderr_path": stderr_relative,
            },
        )
        attempts[task_id] = attempt
        started_ns = time.monotonic_ns()
        exit_code: int | None = None
        error: str | None = None
        status = "launch_failed"
        try:
            with stdout_path.open("xb") as stdout_stream, stderr_path.open("xb") as stderr_stream:
                try:
                    completed = subprocess.run(
                        task["command_argv"],
                        cwd=repository_root,
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        shell=False,
                        check=False,
                    )
                    exit_code = completed.returncode
                    status = "success" if exit_code == 0 else "process_failed"
                except OSError as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    status = "launch_failed"
                finally:
                    stdout_stream.flush()
                    stderr_stream.flush()
                    os.fsync(stdout_stream.fileno())
                    os.fsync(stderr_stream.fileno())
        finally:
            duration_ns = time.monotonic_ns() - started_ns

        _, output_path = _relative_path(
            task["output_path"], repository_root, label=f"task {task_id} output"
        )
        output_sha256 = planner._sha256_file(output_path) if output_path.is_file() else None
        if status == "success" and output_sha256 is None:
            status = "output_validation_failed"
            error = "process exited zero but did not create the planned output"
        try:
            post_task_plan = validate_plan(
                validated.path,
                repository_root=repository_root,
                repository_state=repository_state,
            )
            if post_task_plan.file_sha256 != validated.file_sha256:
                raise ValueError("plan changed during task execution")
            if status == "success" and output_sha256 is not None:
                _validate_task_result(output_path, task, post_task_plan, repository_root)
        except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
            post_error = f"post-task evidence validation failed: {exc}"
            if status == "success":
                status = "output_validation_failed"
                error = post_error
            else:
                error = f"{error}; {post_error}" if error else post_error
        stdout_sha256 = planner._sha256_file(stdout_path)
        stderr_sha256 = planner._sha256_file(stderr_path)
        _append_event(
            journal_path,
            journal,
            {
                "event": "end",
                "task_id": task_id,
                "timestamp_utc": _utc_now(),
                "attempt": attempt,
                "start_event_sha256": start_event["event_sha256"],
                "monotonic_duration_ns": duration_ns,
                "exit_code": exit_code,
                "status": status,
                "stdout_path": stdout_relative,
                "stdout_sha256": stdout_sha256,
                "stderr_path": stderr_relative,
                "stderr_sha256": stderr_sha256,
                "output_path": task["output_path"],
                "output_sha256": output_sha256,
                "error": error,
            },
        )
        executed_count += 1
        if status != "success":
            return {
                "status": "failed",
                "executed": True,
                "failed_task_id": task_id,
                "task_status": status,
                "exit_code": exit_code,
                "tasks_executed_this_invocation": executed_count,
                "tasks_skipped_from_valid_journal": skipped,
                "journal_path": (Path(journal_relative) / "journal.json").as_posix(),
                "pass_merge_performed": False,
            }
        successes[task_id] = output_sha256

    # Completion is decided only from a fresh on-disk plan, journal, logs, and
    # fully parsed result set; no in-memory success map is authoritative.
    final_plan = validate_plan(
        validated.path,
        repository_root=repository_root,
        repository_state=repository_state,
    )
    if final_plan.file_sha256 != validated.file_sha256:
        raise ValueError("plan changed before completion")
    final_journal, final_successes, _ = _validate_journal(
        journal_path,
        final_plan,
        journal_directory,
        repository_root,
    )
    if _active_journal_start(final_journal) is not None:
        raise ValueError("cannot complete with an active journal start")
    _validate_outputs(final_plan, final_successes, repository_root)
    expected_task_ids = {task["task_id"] for task in final_plan.payload["tasks"]}
    if set(final_successes) != expected_task_ids:
        raise ValueError("cannot complete before every task has a validated success")

    return {
        "status": "complete",
        "executed": True,
        "task_count": validated.payload["task_count"],
        "tasks_executed_this_invocation": executed_count,
        "tasks_skipped_from_valid_journal": skipped,
        "journal_path": (Path(journal_relative) / "journal.json").as_posix(),
        "pass_merge_performed": False,
    }


def run_plan(
    plan_path: Path,
    *,
    repository_root: Path,
    execute: bool = False,
    resume: bool = False,
    journal_dir: Path | None = None,
    repository_state: Mapping[str, Any] | None = None,
    lock_file: Path | None = None,
) -> dict[str, Any]:
    """Hold one O_EXCL run lock across validation, execution, and resume I/O."""
    if resume and not execute:
        raise ValueError("--resume requires --execute")
    repository_root = repository_root.resolve()
    plan_relative, resolved_plan = _path_inside_repository(
        plan_path,
        repository_root,
        label="plan",
        require_file=True,
    )
    initial_plan_sha256 = planner._sha256_file(resolved_plan)
    selected_lock = lock_file or _default_lock_file(plan_relative)
    _, resolved_lock = _validate_private_lock_path(selected_lock, repository_root)
    run_lock = _acquire_run_lock(resolved_lock, plan_sha256=initial_plan_sha256)
    try:
        if planner._sha256_file(resolved_plan) != initial_plan_sha256:
            raise ValueError("plan changed while acquiring the run lock")
        return _run_plan_locked(
            resolved_plan,
            repository_root=repository_root,
            execute=execute,
            resume=resume,
            journal_dir=journal_dir,
            repository_state=repository_state,
        )
    finally:
        run_lock.release()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Validate an ASR benchmark plan; execute serially only with --execute.")
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--journal-dir", type=Path)
    parser.add_argument(
        "--lock-file",
        type=Path,
        help=(
            "private O_EXCL lock path; defaults to a stable plan-path-derived "
            "file under artifacts/private/run-locks"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    repository_root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else repository_root / args.plan
    try:
        result = run_plan(
            plan_path,
            repository_root=repository_root,
            execute=args.execute,
            resume=args.resume,
            journal_dir=args.journal_dir,
            lock_file=args.lock_file,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] in {"validated", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
