#!/usr/bin/env python3
"""Aggregate completed multi-pass ASR matrix evidence without running a model.

The input plan and append-only run journal remain the source of truth.  This
tool first reuses their fail-closed validators (including current log/result
hash checks), then combines the three-or-more outer-pass artifacts for each
implementation/profile pair.  Observation values are copied verbatim; only
their positional ``repeat`` field is renumbered.  Every derived summary is
recomputed from those copied observations.

The 18 outputs have one fixed, reviewable mapping inside an explicitly selected
private output directory. Publishing is atomic and never overwrites an existing
artifact. The validator may inspect Git state, but this module never launches a
benchmark command or loads/executes a model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from faster_glm_asr.benchmarking import comparator
from faster_glm_asr.benchmarking import matrix_executor as executor
from faster_glm_asr.benchmarking import matrix_planner as planner

AGGREGATION_SCHEMA = "asr-benchmark-matrix-aggregation-v0.1"
BENCHMARK_SCHEMA = "0.3"
DEFAULT_PRIVATE_OUTPUT_DIRECTORY = Path("artifacts/private/aggregated")
AGGREGATOR_SOURCE_PATH = Path("src/faster_glm_asr/benchmarking/matrix_aggregator.py")
SUMMARY_METRICS = (
    "generation_ms",
    "processor_plus_generation_ms",
    "request_no_file_io_ms",
    "rtf_generation",
)
AGGREGATE_METRICS = SUMMARY_METRICS[:3]
OBSERVATION_KEYS = {
    "repeat",
    "preprocess_ms",
    "generation_ms",
    "decode_ms",
    "processor_plus_generation_ms",
    "request_no_file_io_ms",
    "generated_tokens",
    "hypothesis_sha256",
    "generated_token_ids_sha256",
    "rtf_generation",
    "rtf_request_no_file_io",
    "memory_bytes",
    "nvml_process_memory",
    "diagnostic_cuda_phases",
}
MEMORY_KEYS = {
    "before_prepare",
    "after_prepare",
    "after_generation",
    "generation_allocated_peak",
    "generation_reserved_peak",
    "generation_allocated_peak_delta",
}
SNAPSHOT_KEYS = {"allocated", "reserved"}
METRIC_ELIGIBILITY = {
    "canonical_latency": {
        "canonical_latency": True,
        "request_memory_observed_peak": False,
        "diagnostic_cuda_phases": False,
        "cold_start": False,
        "model_load": False,
    },
    "request_memory": {
        "canonical_latency": False,
        "request_memory_observed_peak": True,
        "diagnostic_cuda_phases": False,
        "cold_start": False,
        "model_load": False,
    },
    "diagnostic_phase": {
        "canonical_latency": False,
        "request_memory_observed_peak": False,
        "diagnostic_cuda_phases": True,
        "cold_start": False,
        "model_load": False,
    },
}


def _formal_filename(implementation: str, profile: str) -> str:
    stem = planner.RUNNER_STEMS[implementation]
    if profile == "canonical_latency":
        return f"{stem}.json"
    if profile == "request_memory":
        return f"{stem}-memory.json"
    if profile == "diagnostic_phase":
        return f"{stem}-phases.json"
    raise ValueError(f"unsupported measurement profile: {profile!r}")


FORMAL_OUTPUT_FILENAMES = {
    (implementation, profile): _formal_filename(implementation, profile)
    for implementation in planner.IMPLEMENTATIONS
    for profile in planner.PROFILES
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


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


def _repo_file(
    value: str | Path,
    repository_root: Path,
    *,
    label: str,
    require_file: bool = True,
) -> tuple[str, Path]:
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        if ".." in candidate.parts:
            raise ValueError(f"{label} cannot escape the repository")
        resolved = (repository_root / candidate).resolve()
    try:
        relative = resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the repository") from exc
    if require_file and not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return relative, resolved


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if not math.isfinite(float(value)):
        return False
    return float(value) > 0 if positive else float(value) >= 0


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty observation list")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot summarize an empty observation list")
    return {
        "n": len(materialized),
        "mean": statistics.fmean(materialized),
        "std": statistics.stdev(materialized) if len(materialized) > 1 else 0.0,
        "min": min(materialized),
        "p50": _percentile(materialized, 50),
        "p95": _percentile(materialized, 95),
        "max": max(materialized),
    }


def _equal_summary(actual: Any, values: Sequence[float], label: str) -> None:
    expected = _summary(values)
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError(f"{label} has an incomplete summary schema")
    if actual.get("n") != expected["n"]:
        raise ValueError(f"{label}.n does not match raw observations")
    for key in set(expected) - {"n"}:
        value = actual.get(key)
        if not _finite_number(value) or not math.isclose(
            float(value), float(expected[key]), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{label}.{key} does not match raw observations")


def _require_nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _validate_memory(memory: Any, label: str) -> None:
    if not isinstance(memory, dict) or set(memory) != MEMORY_KEYS:
        raise ValueError(f"{label} has an invalid allocator-memory schema")
    for snapshot_name in ("before_prepare", "after_prepare", "after_generation"):
        snapshot = memory[snapshot_name]
        if not isinstance(snapshot, dict) or set(snapshot) != SNAPSHOT_KEYS:
            raise ValueError(f"{label}.{snapshot_name} has an invalid schema")
        for key in SNAPSHOT_KEYS:
            _require_nonnegative_int(snapshot[key], f"{label}.{snapshot_name}.{key}")
    for key in (
        "generation_allocated_peak",
        "generation_reserved_peak",
        "generation_allocated_peak_delta",
    ):
        _require_nonnegative_int(memory[key], f"{label}.{key}")
    expected_delta = max(
        0,
        memory["generation_allocated_peak"] - memory["after_prepare"]["allocated"],
    )
    if memory["generation_allocated_peak_delta"] != expected_delta:
        raise ValueError(f"{label}.generation_allocated_peak_delta is inconsistent")


def _validate_phase_timings(value: Any, label: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty phase mapping")
    for phase, record in value.items():
        if not isinstance(phase, str) or not phase:
            raise ValueError(f"{label} contains an invalid phase name")
        if not isinstance(record, dict) or set(record) != {
            "events_ms",
            "summary_ms",
            "total_ms",
        }:
            raise ValueError(f"{label}.{phase} has an invalid phase schema")
        events = record["events_ms"]
        if (
            not isinstance(events, list)
            or not events
            or any(not _finite_number(value) for value in events)
        ):
            raise ValueError(f"{label}.{phase}.events_ms is invalid")
        _equal_summary(record["summary_ms"], events, f"{label}.{phase}.summary_ms")
        if not _finite_number(record["total_ms"]) or not math.isclose(
            float(record["total_ms"]),
            math.fsum(float(item) for item in events),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label}.{phase}.total_ms is inconsistent")


def _validate_observation(
    observation: Any,
    *,
    repeat: int,
    duration_s: float,
    token_count: int,
    token_hash: str,
    hypothesis_hash: str,
    profile: str,
    nvml_sample_ms: float,
    label: str,
) -> None:
    if not isinstance(observation, dict) or set(observation) != OBSERVATION_KEYS:
        raise ValueError(f"{label} has an invalid observation schema")
    if observation.get("repeat") != repeat:
        raise ValueError(f"{label}.repeat is not contiguous")
    for key in ("preprocess_ms", "decode_ms"):
        if not _finite_number(observation.get(key)):
            raise ValueError(f"{label}.{key} must be finite and non-negative")
    if not _finite_number(observation.get("generation_ms"), positive=True):
        raise ValueError(f"{label}.generation_ms must be finite and positive")
    preprocess = float(observation["preprocess_ms"])
    generation = float(observation["generation_ms"])
    decode = float(observation["decode_ms"])
    expected_relations = {
        "processor_plus_generation_ms": preprocess + generation,
        "request_no_file_io_ms": preprocess + generation + decode,
        "rtf_generation": generation / 1000.0 / duration_s,
        "rtf_request_no_file_io": (preprocess + generation + decode) / 1000.0 / duration_s,
    }
    for key, expected in expected_relations.items():
        actual = observation.get(key)
        if not _finite_number(actual) or not math.isclose(
            float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{label}.{key} is inconsistent with raw timings")
    if observation.get("generated_tokens") != token_count:
        raise ValueError(f"{label}.generated_tokens is inconsistent")
    if observation.get("generated_token_ids_sha256") != token_hash:
        raise ValueError(f"{label}.generated_token_ids_sha256 is inconsistent")
    if observation.get("hypothesis_sha256") != hypothesis_hash:
        raise ValueError(f"{label}.hypothesis_sha256 is inconsistent")
    _validate_memory(observation.get("memory_bytes"), f"{label}.memory_bytes")

    nvml = observation.get("nvml_process_memory")
    if not isinstance(nvml, dict):
        raise ValueError(f"{label}.nvml_process_memory must be an object")
    phases = observation.get("diagnostic_cuda_phases")
    if profile == "canonical_latency":
        if nvml.get("enabled") is not False or phases is not None:
            raise ValueError(f"{label} carries instrumentation in canonical latency")
    elif profile == "request_memory":
        if (
            nvml.get("enabled") is not True
            or nvml.get("available") is not True
            or nvml.get("interval_ms") != nvml_sample_ms
            or not isinstance(nvml.get("sample_count"), int)
            or isinstance(nvml.get("sample_count"), bool)
            or nvml["sample_count"] <= 0
            or not _finite_number(nvml.get("observed_peak_used_bytes"))
            or phases is not None
        ):
            raise ValueError(f"{label} lacks eligible request-memory evidence")
    elif profile == "diagnostic_phase":
        if nvml.get("enabled") is not False:
            raise ValueError(f"{label} unexpectedly enables NVML sampling")
        _validate_phase_timings(phases, f"{label}.diagnostic_cuda_phases")


def _aggregate_quality(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("errors", "substitutions", "deletions", "insertions", "reference_units")
    word = {key: 0 for key in keys}
    char = {key: 0 for key in keys}
    evaluated_samples = 0
    for sample in samples:
        quality = sample.get("quality")
        if quality is None:
            continue
        if not isinstance(quality, dict):
            raise ValueError("sample quality must be an object or null")
        evaluated_samples += 1
        for unit_name, target in (("word", word), ("char", char)):
            unit = quality.get(unit_name)
            if not isinstance(unit, dict) or set(unit) != set(keys):
                raise ValueError(f"sample quality.{unit_name} schema is invalid")
            for key in keys:
                target[key] += _require_nonnegative_int(
                    unit[key], f"sample quality.{unit_name}.{key}"
                )
    word_evaluated = word["reference_units"] > 0
    char_evaluated = char["reference_units"] > 0
    return {
        "evaluated_samples": evaluated_samples,
        "word": word,
        "char": char,
        "wer": word["errors"] / word["reference_units"] if word_evaluated else None,
        "cer": char["errors"] / char["reference_units"] if char_evaluated else None,
        "status": (
            "evaluated"
            if word_evaluated and char_evaluated
            else "partially_evaluated"
            if word_evaluated or char_evaluated
            else "not_evaluated"
        ),
        "normalization": (
            "NFKC + casefold + Unicode punctuation/symbol removal + whitespace collapse"
        ),
    }


def _stable_sample_identity(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(sample.get(key))
        for key in (
            "sample_id",
            "audio_sha256",
            "language",
            "duration_s",
            "prepared_request",
            "metadata",
            "hypothesis",
            "hypothesis_sha256",
            "generated_tokens",
            "generated_token_ids",
            "generated_token_ids_sha256",
            "quality",
        )
    }


def _validate_artifact(
    payload: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    validated: executor.ValidatedPlan,
    repository_root: Path,
    label: str,
) -> None:
    if not isinstance(payload, dict) or payload.get("schema_version") != BENCHMARK_SCHEMA:
        raise ValueError(f"{label} must use benchmark schema {BENCHMARK_SCHEMA}")
    if "matrix_aggregation" in payload:
        raise ValueError(f"{label} is already an aggregate, not a raw pass")
    run = payload.get("run")
    if not isinstance(run, dict):
        raise ValueError(f"{label}.run must be an object")
    configuration = validated.configuration
    expected_run_values = {
        "implementation": task["implementation"],
        "model_name": configuration["model_name"],
        "model_revision": configuration["model_revision"],
        "manifest_sha256": task["bindings"]["manifest_sha256"],
        "warmup": configuration["warmup"],
        "repeats": configuration["repeats"],
        "max_new_tokens": configuration["max_new_tokens"],
        "artifact_scope": configuration["artifact_scope"],
        "measurement_profile": task["measurement_profile"],
        "metric_eligibility": METRIC_ELIGIBILITY[task["measurement_profile"]],
        "runner_configuration": comparator.EXPECTED_RUNNER_CONFIGURATIONS[task["implementation"]],
    }
    for key, expected in expected_run_values.items():
        if run.get(key) != expected:
            raise ValueError(f"{label}.run.{key} does not match its planned task")
    expected_manifest_path = str((repository_root / configuration["manifest"]).resolve())
    try:
        actual_manifest_path = str(Path(run.get("manifest_path", "")).resolve())
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}.run.manifest_path is invalid") from exc
    if actual_manifest_path != expected_manifest_path:
        raise ValueError(f"{label}.run.manifest_path does not bind the planned manifest")
    profile = task["measurement_profile"]
    expected_nvml = configuration["nvml_sample_ms"] if profile == "request_memory" else 0.0
    expected_phase_timing = profile == "diagnostic_phase"
    if (
        not _finite_number(run.get("nvml_sample_ms"))
        or float(run["nvml_sample_ms"]) != float(expected_nvml)
        or run.get("phase_timing") is not expected_phase_timing
    ):
        raise ValueError(f"{label}.run instrumentation flags do not match its profile")
    if not _finite_number(run.get("runner_construction_envelope_ms")):
        raise ValueError(f"{label}.run.runner_construction_envelope_ms is invalid")
    if (
        not isinstance(run.get("runner_construction_scope"), str)
        or not run["runner_construction_scope"]
    ):
        raise ValueError(f"{label}.run.runner_construction_scope is invalid")

    git = payload.get("git")
    if not isinstance(git, dict) or (
        git.get("commit") != validated.repository_state["commit"]
        or git.get("dirty") != validated.repository_state["dirty"]
    ):
        raise ValueError(f"{label}.git does not match the planned repository state")
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        raise ValueError(f"{label}.sources must be an object")
    source_files = sources.get("files")
    if (
        not isinstance(source_files, dict)
        or not isinstance(source_files.get(task["bindings"]["benchmark_script_path"]), dict)
        or source_files[task["bindings"]["benchmark_script_path"]].get("sha256")
        != task["bindings"]["benchmark_script_sha256"]
    ):
        raise ValueError(f"{label}.sources does not bind the benchmark script")
    environment = payload.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("environment_lock_sha256") != task["bindings"]["environment_lock_sha256"]
    ):
        raise ValueError(f"{label}.environment does not bind the planned lock")

    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{label}.samples must be non-empty")
    repeats = configuration["repeats"]
    sample_ids = []
    all_values: dict[str, list[float]] = {metric: [] for metric in AGGREGATE_METRICS}
    for sample_index, sample in enumerate(samples):
        location = f"{label}.samples[{sample_index}]"
        if not isinstance(sample, dict):
            raise ValueError(f"{location} must be an object")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{location}.sample_id must be non-empty text")
        sample_ids.append(sample_id)
        duration = sample.get("duration_s")
        if not _finite_number(duration, positive=True):
            raise ValueError(f"{location}.duration_s must be finite and positive")
        audio_hash = sample.get("audio_sha256")
        if not isinstance(audio_hash, str) or comparator.SHA256_RE.fullmatch(audio_hash) is None:
            raise ValueError(f"{location}.audio_sha256 must be SHA256")
        hypothesis = sample.get("hypothesis")
        if not isinstance(hypothesis, str):
            raise ValueError(f"{location}.hypothesis must be text")
        hypothesis_hash = _sha256_bytes(hypothesis.encode("utf-8"))
        if sample.get("hypothesis_sha256") != hypothesis_hash:
            raise ValueError(f"{location}.hypothesis_sha256 is inconsistent")
        token_ids = sample.get("generated_token_ids")
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(
                not isinstance(token, int) or isinstance(token, bool) or token < 0
                for token in token_ids
            )
        ):
            raise ValueError(f"{location}.generated_token_ids is invalid")
        if sample.get("generated_tokens") != len(token_ids):
            raise ValueError(f"{location}.generated_tokens is inconsistent")
        token_hash = _sha256_bytes(json.dumps(token_ids, separators=(",", ":")).encode("utf-8"))
        if sample.get("generated_token_ids_sha256") != token_hash:
            raise ValueError(f"{location}.generated_token_ids_sha256 is inconsistent")
        comparator._validate_prepared_request(sample, location)
        observations = sample.get("observations")
        if not isinstance(observations, list) or len(observations) != repeats:
            raise ValueError(f"{location}.observations does not match run.repeats")
        values: dict[str, list[float]] = {metric: [] for metric in SUMMARY_METRICS}
        for repeat, observation in enumerate(observations):
            observation_label = f"{location}.observations[{repeat}]"
            _validate_observation(
                observation,
                repeat=repeat,
                duration_s=float(duration),
                token_count=len(token_ids),
                token_hash=token_hash,
                hypothesis_hash=hypothesis_hash,
                profile=profile,
                nvml_sample_ms=float(expected_nvml),
                label=observation_label,
            )
            for metric in SUMMARY_METRICS:
                values[metric].append(float(observation[metric]))
            for metric in AGGREGATE_METRICS:
                all_values[metric].append(float(observation[metric]))
        summary = sample.get("summary")
        if not isinstance(summary, dict) or set(summary) != set(SUMMARY_METRICS):
            raise ValueError(f"{location}.summary has an invalid schema")
        for metric in SUMMARY_METRICS:
            _equal_summary(summary[metric], values[metric], f"{location}.summary.{metric}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{label} contains duplicate sample IDs")
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict) or set(aggregate) != {
        *AGGREGATE_METRICS,
        "quality",
    }:
        raise ValueError(f"{label}.aggregate has an invalid schema")
    for metric in AGGREGATE_METRICS:
        _equal_summary(aggregate[metric], all_values[metric], f"{label}.aggregate.{metric}")
    expected_quality = _aggregate_quality(samples)
    if aggregate.get("quality") != expected_quality:
        raise ValueError(f"{label}.aggregate.quality does not match samples")
    if profile == "canonical_latency":
        comparator._validate_canonical_payload(payload, label)


def _build_pair_aggregate(
    records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], str, str]],
    *,
    validated: executor.ValidatedPlan,
    plan_relative: str,
    journal_relative: str,
    journal_sha256: str,
    script_relative: str,
    script_sha256: str,
) -> dict[str, Any]:
    if len(records) < 3:
        raise ValueError("each implementation/profile pair requires >=3 outer passes")
    ordered = sorted(records, key=lambda record: int(record[0]["outer_pass"]))
    passes = [int(task["outer_pass"]) for task, _, _, _ in ordered]
    expected_passes = list(range(1, validated.configuration["outer_passes"] + 1))
    if passes != expected_passes:
        raise ValueError(f"implementation/profile pair has missing or duplicate passes: {passes}")
    first_task, first_payload, _, _ = ordered[0]
    implementation = str(first_task["implementation"])
    profile = str(first_task["measurement_profile"])
    stable_top = {
        "git": first_payload.get("git"),
        "sources": first_payload.get("sources"),
        "model_snapshot": first_payload.get("model_snapshot"),
        "readiness": first_payload.get("readiness"),
        "environment": first_payload.get("environment"),
        "runner_configuration": first_payload["run"].get("runner_configuration"),
        "runner_construction_scope": first_payload["run"].get("runner_construction_scope"),
    }
    first_samples = first_payload["samples"]
    stable_samples = [_stable_sample_identity(sample) for sample in first_samples]
    for task, payload, _, _ in ordered[1:]:
        if task["implementation"] != implementation or task["measurement_profile"] != profile:
            raise ValueError("pair aggregation received a foreign task")
        current_top = {
            "git": payload.get("git"),
            "sources": payload.get("sources"),
            "model_snapshot": payload.get("model_snapshot"),
            "readiness": payload.get("readiness"),
            "environment": payload.get("environment"),
            "runner_configuration": payload["run"].get("runner_configuration"),
            "runner_construction_scope": payload["run"].get("runner_construction_scope"),
        }
        if current_top != stable_top:
            raise ValueError(
                f"{implementation}/{profile} environment/git/sources/runner "
                "configuration drift across passes"
            )
        current_samples = [_stable_sample_identity(sample) for sample in payload["samples"]]
        if current_samples != stable_samples:
            raise ValueError(
                f"{implementation}/{profile} sample order/audio/prepared request/"
                "token IDs/hypothesis/quality drift across passes"
            )

    result = copy.deepcopy(first_payload)
    repeats_per_pass = validated.configuration["repeats"]
    total_repeats = repeats_per_pass * len(ordered)
    result["run"]["repeats"] = total_repeats
    for sample_index, sample in enumerate(result["samples"]):
        combined = []
        for _, payload, _, _ in ordered:
            for observation in payload["samples"][sample_index]["observations"]:
                copied = copy.deepcopy(observation)
                copied["repeat"] = len(combined)
                combined.append(copied)
        sample["observations"] = combined
        sample["summary"] = {
            metric: _summary([float(row[metric]) for row in combined]) for metric in SUMMARY_METRICS
        }
    result["aggregate"] = {
        metric: _summary(
            [
                float(observation[metric])
                for sample in result["samples"]
                for observation in sample["observations"]
            ]
        )
        for metric in AGGREGATE_METRICS
    }
    result["aggregate"]["quality"] = _aggregate_quality(result["samples"])
    result["matrix_aggregation"] = {
        "schema_version": AGGREGATION_SCHEMA,
        "plan_path": plan_relative,
        "plan_sha256": validated.file_sha256,
        "journal_path": journal_relative,
        "journal_sha256": journal_sha256,
        "implementation": implementation,
        "measurement_profile": profile,
        "outer_pass_count": len(ordered),
        "repeats_per_pass": repeats_per_pass,
        "total_repeats": total_repeats,
        "pass_order": [
            {
                "outer_pass": int(task["outer_pass"]),
                "order_in_pass": int(task["order_in_pass"]),
                "task_id": task["task_id"],
                "source_artifact_path": source_path,
                "source_artifact_sha256": source_hash,
            }
            for task, _, source_path, source_hash in ordered
        ],
        "aggregation_script": {
            "path": script_relative,
            "sha256": script_sha256,
        },
        "observation_policy": (
            "source observations copied in ascending outer_pass order; only repeat "
            "renumbered contiguously; sample and aggregate summaries recomputed"
        ),
    }
    if profile == "canonical_latency":
        comparator._validate_canonical_payload(result, f"aggregate {implementation}/{profile}")
    return result


def build_aggregates(
    plan_path: Path,
    journal_path: Path,
    *,
    repository_root: Path,
    repository_state: Mapping[str, Any] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate all evidence and return the deterministic 18-payload mapping."""
    repository_root = repository_root.resolve()
    validated = executor.validate_plan(
        plan_path,
        repository_root=repository_root,
        repository_state=repository_state,
    )
    if validated.configuration["artifact_scope"] != "private":
        raise ValueError("formal matrix aggregation requires private raw artifacts")
    journal_relative, journal_path = _repo_file(journal_path, repository_root, label="run journal")
    _journal_dir_relative, journal_directory = executor._validate_private_journal_dir(
        journal_path.parent, repository_root
    )
    if journal_path != journal_directory / "journal.json":
        raise ValueError("run journal must be named journal.json in its private directory")
    _journal, successes, _ = executor._validate_journal(
        journal_path,
        validated,
        journal_directory,
        repository_root,
    )
    tasks = validated.payload["tasks"]
    expected_task_ids = {task["task_id"] for task in tasks}
    if set(successes) != expected_task_ids:
        missing = sorted(expected_task_ids - set(successes))
        raise ValueError(f"run journal is not fully successful; missing successful tasks={missing}")
    executor._validate_outputs(validated, successes, repository_root)
    journal_sha256 = _sha256_file(journal_path)
    script_relative, script_path = _repo_file(
        AGGREGATOR_SOURCE_PATH, repository_root, label="aggregation module"
    )
    if _sha256_file(Path(__file__).resolve()) != _sha256_file(script_path):
        raise ValueError("executing aggregation module differs from repository copy")
    script_sha256 = _sha256_file(script_path)

    by_pair: dict[
        tuple[str, str],
        list[tuple[Mapping[str, Any], Mapping[str, Any], str, str]],
    ] = {
        (implementation, profile): []
        for implementation in planner.IMPLEMENTATIONS
        for profile in planner.PROFILES
    }
    global_provenance: dict[str, Any] | None = None
    for task in tasks:
        source_relative, source_path = _repo_file(
            task["output_path"],
            repository_root,
            label=f"task {task['task_id']} source artifact",
        )
        source_hash = _sha256_file(source_path)
        if successes[task["task_id"]] != source_hash:
            raise ValueError(f"task {task['task_id']} output hash drift")
        payload, _ = _load_json_object(source_path, f"task {task['task_id']} artifact")
        _validate_artifact(
            payload,
            task=task,
            validated=validated,
            repository_root=repository_root,
            label=source_relative,
        )
        provenance = {
            "git": payload["git"],
            "sources": payload["sources"],
            "model_snapshot": payload["model_snapshot"],
            "readiness": payload["readiness"],
            "environment": payload["environment"],
        }
        if global_provenance is None:
            global_provenance = provenance
        elif provenance != global_provenance:
            raise ValueError("matrix-wide environment/git/source provenance drift")
        by_pair[(task["implementation"], task["measurement_profile"])].append(
            (task, payload, source_relative, source_hash)
        )

    aggregates = {
        pair: _build_pair_aggregate(
            records,
            validated=validated,
            plan_relative=validated.relative_path,
            journal_relative=journal_relative,
            journal_sha256=journal_sha256,
            script_relative=script_relative,
            script_sha256=script_sha256,
        )
        for pair, records in by_pair.items()
    }
    if set(aggregates) != set(FORMAL_OUTPUT_FILENAMES):
        raise AssertionError("internal formal output mapping is incomplete")
    return aggregates


def _write_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
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
        os.link(temporary, path)
        temporary.unlink()
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def aggregate_matrix(
    plan_path: Path,
    journal_path: Path,
    *,
    repository_root: Path,
    output_directory: Path = DEFAULT_PRIVATE_OUTPUT_DIRECTORY,
    repository_state: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Validate, build, and publish all 18 formal artifacts without overwrite."""
    repository_root = repository_root.resolve()
    aggregates = build_aggregates(
        plan_path,
        journal_path,
        repository_root=repository_root,
        repository_state=repository_state,
    )
    output_directory = (
        output_directory.resolve()
        if output_directory.is_absolute()
        else (repository_root / output_directory).resolve()
    )
    try:
        output_directory.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("output directory escapes repository") from exc
    destinations = {
        pair: output_directory / filename for pair, filename in FORMAL_OUTPUT_FILENAMES.items()
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite formal matrix artifacts: "
            + ", ".join(str(path) for path in sorted(existing))
        )
    published: dict[str, str] = {}
    for pair in (
        (implementation, profile)
        for implementation in planner.IMPLEMENTATIONS
        for profile in planner.PROFILES
    ):
        destination = destinations[pair]
        _write_json_no_overwrite(destination, aggregates[pair])
        published[f"{pair[0]}:{pair[1]}"] = destination.relative_to(repository_root).as_posix()
    return published


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate a fully successful ASR benchmark matrix into a fixed "
            "18-file private evidence mapping; never executes a model"
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PRIVATE_OUTPUT_DIRECTORY,
        help="private destination inside the repository",
    )
    args = parser.parse_args()
    try:
        published = aggregate_matrix(
            args.plan,
            args.journal,
            repository_root=args.repository_root,
            output_directory=args.output_dir,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"published": published}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
