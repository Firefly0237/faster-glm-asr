#!/usr/bin/env python3
"""Fail-closed comparison of two canonical private ASR benchmark artifacts.

The comparator is intentionally strict: it accepts only schema-v0.3 latency
passes from the same manifest/model/environment, with NVML sampling and phase
instrumentation disabled.  It requires exact generated token IDs by default and
emits no transcript or local input path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from faster_glm_asr.benchmarking import provenance

SCHEMA_VERSION = "0.3"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SUMMARY_KEYS = {"n", "mean", "std", "min", "p50", "p95", "max"}
PREPARED_REQUEST_KEYS = {
    "tensors",
    "request_batch",
    "prompt_tokens",
    "audio_placeholder_tokens",
    "audio_windows",
    "valid_feature_frames",
}
ALLOWED_PREPARED_TENSORS = {
    "input_ids",
    "input_features",
    "input_features_mask",
    "attention_mask",
}
CANONICAL_METRIC_ELIGIBILITY = {
    "canonical_latency": True,
    "request_memory_observed_peak": False,
    "diagnostic_cuda_phases": False,
    "cold_start": False,
    "model_load": False,
}
SOURCE_MAP_SCHEMA = "faster-glm-asr-source-map-v1"
SOURCE_RECORD_KEYS = {"size_bytes", "sha256"}
SOURCE_PATH_RE = re.compile(
    r"^src/faster_glm_asr/(?:modeling|kernels|benchmarking|data)/"
    r"[A-Za-z0-9_./-]+\.py$"
)
REQUIRED_PACKAGE_VERSIONS = {
    "torch",
    "triton",
    "transformers",
    "accelerate",
    "huggingface-hub",
    "safetensors",
    "numpy",
    "scipy",
    "soundfile",
    "nvidia-ml-py",
}
EXPECTED_ENVIRONMENT_KEYS = {
    "python",
    "platform",
    "torch",
    "torch_cuda",
    "cudnn",
    "device_index",
    "device_name",
    "device_uuid",
    "cuda_visible_devices",
    "compute_capability",
    "total_memory_bytes",
    "allow_tf32_matmul",
    "allow_tf32_cudnn",
    "nvidia_driver_versions",
    "package_versions",
    "environment_lock_sha256",
}
EXPECTED_RUNNER_CONFIGURATIONS = {
    "hf_cached": {
        "system": "huggingface-transformers",
        "model_source": "validated-local-snapshot",
        "network_fallback": False,
        "storage_activation_dtype": "torch.float32",
        "cache_mode": "dynamic",
        "do_sample": False,
    },
    "hf_no_cache": {
        "system": "huggingface-transformers",
        "model_source": "validated-local-snapshot",
        "network_fallback": False,
        "storage_activation_dtype": "torch.float32",
        "cache_mode": "none",
        "do_sample": False,
    },
    "custom_full_prefix": {
        "system": "custom-torch-triton-hybrid",
        "model_source": "validated-local-snapshot",
        "network_fallback": False,
        "storage_activation_dtype": "torch.float32",
        "linear_backend": "cublas",
        "mlp_fused": False,
        "kernel_dispatch_policy_version": "hybrid-auto-v1",
        "component_dispatch_policy": ("supported-norm-rope-embedding-conv-auto-triton-else-torch"),
        "attention_dispatch_policy": (
            "dense-triton-if-padded-and-head-dim-lte-256-else-torch-dense"
        ),
        "gqa_kv_expansion": "explicit-expand",
        "long_form_attention_expectation": "torch-dense-likely-for-30s-input",
        "backend_trace_status": ("policy-inferred-diagnostic-trace-required-for-observed-backend"),
        "performance_attribution": ("cache-strategy-only-no-triton-kernel-attribution"),
        "cache_mode": "none",
        "greedy_fast_path": False,
        "token_output_mode": "torch-cat",
    },
    "custom_greedy_full_prefix": {
        "system": "custom-torch-triton-hybrid",
        "model_source": "validated-local-snapshot",
        "network_fallback": False,
        "storage_activation_dtype": "torch.float32",
        "linear_backend": "cublas",
        "mlp_fused": False,
        "kernel_dispatch_policy_version": "hybrid-auto-v1",
        "component_dispatch_policy": ("supported-norm-rope-embedding-conv-auto-triton-else-torch"),
        "attention_dispatch_policy": (
            "dense-triton-if-padded-and-head-dim-lte-256-else-torch-dense"
        ),
        "gqa_kv_expansion": "explicit-expand",
        "long_form_attention_expectation": "torch-dense-likely-for-30s-input",
        "backend_trace_status": ("policy-inferred-diagnostic-trace-required-for-observed-backend"),
        "performance_attribution": ("cache-strategy-only-no-triton-kernel-attribution"),
        "cache_mode": "none",
        "greedy_fast_path": True,
        "token_output_mode": "torch-cat",
    },
    "custom_tuple_cache": {
        "system": "custom-torch-triton-hybrid",
        "model_source": "validated-local-snapshot",
        "network_fallback": False,
        "storage_activation_dtype": "torch.float32",
        "linear_backend": "cublas",
        "mlp_fused": False,
        "kernel_dispatch_policy_version": "hybrid-auto-v1",
        "component_dispatch_policy": ("supported-norm-rope-embedding-conv-auto-triton-else-torch"),
        "attention_dispatch_policy": (
            "dense-triton-if-padded-and-head-dim-lte-256-else-torch-dense"
        ),
        "gqa_kv_expansion": "explicit-expand",
        "long_form_attention_expectation": "torch-dense-likely-for-30s-input",
        "backend_trace_status": ("policy-inferred-diagnostic-trace-required-for-observed-backend"),
        "performance_attribution": ("cache-strategy-only-no-triton-kernel-attribution"),
        "cache_mode": "tuple-torch-cat",
        "greedy_fast_path": True,
        "token_output_mode": "preallocated",
    },
    "custom_static_cache": {
        "system": "custom-torch-triton-hybrid",
        "model_source": "validated-local-snapshot",
        "network_fallback": False,
        "storage_activation_dtype": "torch.float32",
        "linear_backend": "cublas",
        "mlp_fused": False,
        "kernel_dispatch_policy_version": "hybrid-auto-v1",
        "component_dispatch_policy": ("supported-norm-rope-embedding-conv-auto-triton-else-torch"),
        "attention_dispatch_policy": (
            "dense-triton-if-padded-and-head-dim-lte-256-else-torch-dense"
        ),
        "gqa_kv_expansion": "explicit-expand",
        "long_form_attention_expectation": "torch-dense-likely-for-30s-input",
        "backend_trace_status": ("policy-inferred-diagnostic-trace-required-for-observed-backend"),
        "performance_attribution": ("cache-strategy-only-no-triton-kernel-attribution"),
        "cache_mode": "static-preallocated",
        "greedy_fast_path": True,
        "token_output_mode": "preallocated",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomically(output: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    """Durably stage JSON, then atomically publish it without accidental overwrite."""
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
        if overwrite:
            os.replace(temporary, output)
        else:
            # os.link atomically fails when the destination appeared while this
            # process was running; os.replace would silently overwrite it.
            os.link(temporary, output)
            temporary.unlink()
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: expected schema {SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
        )
    run = payload.get("run")
    if not isinstance(run, dict):
        raise ValueError(f"{path}: run must be an object")
    if run.get("artifact_scope") != "private":
        raise ValueError(f"{path}: pair comparison requires private artifacts")
    if run.get("measurement_profile") != "canonical_latency":
        raise ValueError(f"{path}: measurement_profile must equal 'canonical_latency'")
    if run.get("phase_timing") is not False:
        raise ValueError(f"{path}: phase_timing must be false for canonical latency")
    if float(run.get("nvml_sample_ms", -1)) != 0.0:
        raise ValueError(f"{path}: NVML sampling must be disabled for latency")
    eligibility = run.get("metric_eligibility")
    if eligibility != CANONICAL_METRIC_ELIGIBILITY:
        raise ValueError(f"{path}: canonical latency metric is not eligible")
    if not isinstance(payload.get("samples"), list) or not payload["samples"]:
        raise ValueError(f"{path}: samples must be a non-empty list")
    _validate_canonical_payload(payload, str(path))
    return payload


def _equal_field(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    key: str,
    scope: str,
) -> None:
    if baseline.get(key) != candidate.get(key):
        raise ValueError(f"{scope}.{key} differs: {baseline.get(key)!r} != {candidate.get(key)!r}")


def _input_fingerprints(sample: Mapping[str, Any]) -> dict[str, Any]:
    request = sample.get("prepared_request", {})
    tensors = request.get("tensors", {}) if isinstance(request, dict) else {}
    result: dict[str, Any] = {}
    for name, metadata in tensors.items():
        if not isinstance(metadata, dict):
            continue
        result[name] = {
            key: metadata.get(key)
            for key in ("shape", "dtype", "content_sha256")
            if key in metadata
        }
    return result


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if not math.isfinite(float(value)):
        return False
    return float(value) > 0 if positive else float(value) >= 0


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _expected_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty observation list")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": max(values),
    }


def _validate_summary(summary: Any, values: list[float], location: str) -> None:
    if not isinstance(summary, dict) or set(summary) != SUMMARY_KEYS:
        raise ValueError(f"{location} must contain the complete summary schema")
    expected = _expected_summary(values)
    if summary.get("n") != expected["n"]:
        raise ValueError(f"{location}.n does not match raw observations")
    for key in SUMMARY_KEYS - {"n"}:
        value = summary.get(key)
        if not _finite_number(value) or not math.isclose(
            float(value), float(expected[key]), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{location}.{key} does not match raw observations")


def _validate_prepared_request(sample: Mapping[str, Any], location: str) -> None:
    request = sample.get("prepared_request")
    if not isinstance(request, dict) or set(request) != PREPARED_REQUEST_KEYS:
        raise ValueError(f"{location}.prepared_request schema is incomplete")
    tensors = request.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError(f"{location}.prepared_request.tensors must be an object")
    tensor_names = set(tensors)
    if not {"input_ids", "input_features"}.issubset(tensor_names):
        raise ValueError(
            f"{location}.prepared_request must fingerprint input_ids and input_features"
        )
    if not tensor_names.issubset(ALLOWED_PREPARED_TENSORS):
        raise ValueError(f"{location}.prepared_request has unsupported tensors")
    for name, metadata in tensors.items():
        expected_keys = {"shape", "dtype", "device", "content_sha256"}
        if not isinstance(metadata, dict) or set(metadata) != expected_keys:
            raise ValueError(f"{location}.prepared_request.tensors.{name} schema is incomplete")
        shape = metadata.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in shape
            )
        ):
            raise ValueError(f"{location}.prepared_request.tensors.{name}.shape is invalid")
        if not isinstance(metadata.get("dtype"), str) or not metadata["dtype"]:
            raise ValueError(f"{location}.prepared_request.tensors.{name}.dtype is invalid")
        if not isinstance(metadata.get("device"), str) or not metadata["device"]:
            raise ValueError(f"{location}.prepared_request.tensors.{name}.device is invalid")
        if name == "input_features" and metadata.get("dtype") != "torch.float32":
            raise ValueError(f"{location}.prepared_request.tensors.input_features must be FP32")
        if not _is_sha256(metadata.get("content_sha256")):
            raise ValueError(
                f"{location}.prepared_request.tensors.{name}.content_sha256 must be SHA256"
            )
    input_shape = tensors["input_ids"]["shape"]
    feature_shape = tensors["input_features"]["shape"]
    if len(input_shape) != 2 or input_shape[0] != 1:
        raise ValueError(f"{location}.input_ids must describe batch shape [1,S]")
    if len(feature_shape) != 3:
        raise ValueError(f"{location}.input_features must describe a rank-3 tensor")
    if request.get("request_batch") != 1:
        raise ValueError(f"{location}.prepared_request.request_batch must equal 1")
    if request.get("prompt_tokens") != input_shape[1]:
        raise ValueError(f"{location}.prepared_request.prompt_tokens is inconsistent")
    if request.get("audio_windows") != feature_shape[0]:
        raise ValueError(f"{location}.prepared_request.audio_windows is inconsistent")
    placeholder_tokens = request.get("audio_placeholder_tokens")
    if (
        not isinstance(placeholder_tokens, int)
        or isinstance(placeholder_tokens, bool)
        or placeholder_tokens <= 0
    ):
        raise ValueError(f"{location}.prepared_request.audio_placeholder_tokens is invalid")
    valid_feature_frames = request.get("valid_feature_frames")
    if valid_feature_frames is not None and (
        not isinstance(valid_feature_frames, int)
        or isinstance(valid_feature_frames, bool)
        or valid_feature_frames <= 0
    ):
        raise ValueError(f"{location}.prepared_request.valid_feature_frames is invalid")
    if "input_features_mask" in tensors and valid_feature_frames is None:
        raise ValueError(f"{location}.prepared_request.valid_feature_frames is required with mask")


def _validate_canonical_payload(payload: Mapping[str, Any], label: str) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{label}: expected schema {SCHEMA_VERSION}")
    run = payload.get("run")
    if not isinstance(run, dict):
        raise ValueError(f"{label}.run must be an object")
    if run.get("artifact_scope") != "private":
        raise ValueError(f"{label}: comparison requires private artifacts")
    if run.get("formal_run") is not True:
        raise ValueError(f"{label}: comparison requires formal_run=true")
    if run.get("comparability") != {
        "status": "formal-comparable",
        "reasons": [],
    }:
        raise ValueError(f"{label}: run is explicitly non-comparable")
    if run.get("measurement_profile") != "canonical_latency":
        raise ValueError(f"{label}: measurement_profile must be canonical_latency")
    if run.get("phase_timing") is not False or float(run.get("nvml_sample_ms", -1)) != 0.0:
        raise ValueError(f"{label}: canonical latency instrumentation is inconsistent")
    eligibility = run.get("metric_eligibility")
    if eligibility != CANONICAL_METRIC_ELIGIBILITY:
        raise ValueError(f"{label}: canonical latency metric is not eligible")
    for key in ("implementation", "model_name"):
        if not isinstance(run.get(key), str) or not run[key]:
            raise ValueError(f"{label}.run.{key} must be non-empty text")
    implementation = run["implementation"]
    expected_runner_configuration = EXPECTED_RUNNER_CONFIGURATIONS.get(implementation)
    if expected_runner_configuration is None:
        raise ValueError(f"{label}.run.implementation is unsupported")
    if not isinstance(run.get("model_revision"), str) or not re.fullmatch(
        r"[0-9a-fA-F]{40}", run["model_revision"]
    ):
        raise ValueError(f"{label}.run.model_revision must be a pinned commit SHA")
    if not _is_sha256(run.get("manifest_sha256")):
        raise ValueError(f"{label}.run.manifest_sha256 must be SHA256")
    warmup = run.get("warmup")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 1:
        raise ValueError(f"{label}.run.warmup must be an integer >= 1")
    custom = implementation.startswith("custom_")
    expected_warmup_gate = {
        "status": "pass",
        "minimum_iterations": 1,
        "observed_iterations": warmup,
        "custom_lazy_h2d_applicable": custom,
        "custom_lazy_h2d_covered": True if custom else None,
    }
    if run.get("warmup_gate") != expected_warmup_gate:
        raise ValueError(f"{label}.run.warmup_gate does not prove warmup coverage")
    max_new_tokens = run.get("max_new_tokens")
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise ValueError(f"{label}.run.max_new_tokens must be a positive integer")
    if run.get("runner_configuration") != expected_runner_configuration:
        raise ValueError(
            f"{label}.run.runner_configuration does not match implementation={implementation!r}"
        )
    git = payload.get("git")
    if (
        not isinstance(git, dict)
        or not isinstance(git.get("commit"), str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", git["commit"]) is None
        or not isinstance(git.get("branch"), str)
        or git.get("dirty") is not False
    ):
        raise ValueError(f"{label}.git provenance is incomplete")
    try:
        model_snapshot = provenance.validate_model_snapshot_binding(
            payload.get("model_snapshot"),
            expected_repo_id=run["model_name"],
            expected_revision=run["model_revision"],
        )
        readiness = provenance.validate_readiness_binding(payload.get("readiness"))
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from exc
    if model_snapshot["contract_sha256"] != readiness["contract_sha256"]:
        raise ValueError(f"{label}: model/readiness contract bindings differ")
    sources = payload.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
        "schema_version",
        "files",
        "aggregate_sha256",
    }:
        raise ValueError(f"{label}.sources provenance is incomplete")
    if sources.get("schema_version") != SOURCE_MAP_SCHEMA:
        raise ValueError(f"{label}.sources schema is unsupported")
    source_files = sources.get("files")
    if (
        not isinstance(source_files, dict)
        or not source_files
        or any(
            not isinstance(path, str)
            or SOURCE_PATH_RE.fullmatch(path) is None
            or not isinstance(record, dict)
            or set(record) != SOURCE_RECORD_KEYS
            or not isinstance(record.get("size_bytes"), int)
            or isinstance(record.get("size_bytes"), bool)
            or record["size_bytes"] < 0
            or not _is_sha256(record.get("sha256"))
            for path, record in source_files.items()
        )
        or not _is_sha256(sources.get("aggregate_sha256"))
    ):
        raise ValueError(f"{label}.sources fingerprints are incomplete")
    canonical = json.dumps(
        source_files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != sources["aggregate_sha256"]:
        raise ValueError(f"{label}.sources aggregate fingerprint is inconsistent")
    environment = payload.get("environment")
    if not isinstance(environment, dict) or set(environment) != EXPECTED_ENVIRONMENT_KEYS:
        raise ValueError(f"{label}.environment provenance is incomplete")
    for key in (
        "python",
        "platform",
        "torch",
        "torch_cuda",
        "device_name",
        "device_uuid",
    ):
        if not isinstance(environment.get(key), str) or not environment[key]:
            raise ValueError(f"{label}.environment.{key} must be non-empty text")
    if environment.get("cuda_visible_devices") is not None and not isinstance(
        environment["cuda_visible_devices"], str
    ):
        raise ValueError(f"{label}.environment.cuda_visible_devices must be null or text")
    if (
        not isinstance(environment.get("device_index"), int)
        or isinstance(environment["device_index"], bool)
        or environment["device_index"] < 0
    ):
        raise ValueError(f"{label}.environment.device_index is invalid")
    compute_capability = environment.get("compute_capability")
    if (
        not isinstance(compute_capability, list)
        or len(compute_capability) != 2
        or any(
            not isinstance(component, int) or isinstance(component, bool) or component < 0
            for component in compute_capability
        )
    ):
        raise ValueError(f"{label}.environment.compute_capability is invalid")
    if (
        not isinstance(environment.get("total_memory_bytes"), int)
        or isinstance(environment["total_memory_bytes"], bool)
        or environment["total_memory_bytes"] <= 0
    ):
        raise ValueError(f"{label}.environment.total_memory_bytes is invalid")
    if any(
        not isinstance(environment.get(key), bool)
        for key in ("allow_tf32_matmul", "allow_tf32_cudnn")
    ):
        raise ValueError(f"{label}.environment TF32 flags must be booleans")
    if not _is_sha256(environment.get("environment_lock_sha256")):
        raise ValueError(f"{label}.environment lock fingerprint is invalid")
    package_versions = environment.get("package_versions")
    if (
        not isinstance(package_versions, dict)
        or set(package_versions) != REQUIRED_PACKAGE_VERSIONS
        or any(not isinstance(version, str) or not version for version in package_versions.values())
    ):
        raise ValueError(f"{label}.environment package versions are incomplete")
    drivers = environment.get("nvidia_driver_versions")
    if (
        not isinstance(drivers, list)
        or not drivers
        or any(not isinstance(driver, str) or not driver for driver in drivers)
    ):
        raise ValueError(f"{label}.environment driver provenance is incomplete")
    repeats = run.get("repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ValueError(f"{label}.run.repeats must be a positive integer")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{label}.samples must be a non-empty list")

    all_generation_values: list[float] = []
    for sample_index, sample in enumerate(samples):
        location = f"{label}.samples[{sample_index}]"
        if not isinstance(sample, dict):
            raise ValueError(f"{location} must be an object")
        if not isinstance(sample.get("sample_id"), str) or not sample["sample_id"]:
            raise ValueError(f"{location}.sample_id must be non-empty")
        if not isinstance(sample.get("language"), str) or not sample["language"]:
            raise ValueError(f"{location}.language must be non-empty")
        if not _is_sha256(sample.get("audio_sha256")):
            raise ValueError(f"{location}.audio_sha256 must be SHA256")
        if not _finite_number(sample.get("duration_s"), positive=True):
            raise ValueError(f"{location}.duration_s must be finite and positive")
        token_ids = sample.get("generated_token_ids")
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(
                not isinstance(token, int) or isinstance(token, bool) or token < 0
                for token in token_ids
            )
        ):
            raise ValueError(f"{location}.generated_token_ids must be non-empty ints")
        if sample.get("generated_tokens") != len(token_ids):
            raise ValueError(f"{location}.generated_tokens does not match token IDs")
        expected_token_hash = hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if sample.get("generated_token_ids_sha256") != expected_token_hash:
            raise ValueError(f"{location}.generated_token_ids_sha256 does not match token IDs")
        if "quality" not in sample:
            raise ValueError(f"{location}.quality must be present (null is allowed)")
        hypothesis = sample.get("hypothesis")
        if not isinstance(hypothesis, str):
            raise ValueError(f"{location}.hypothesis must be text")
        expected_hypothesis_hash = hashlib.sha256(hypothesis.encode("utf-8")).hexdigest()
        if sample.get("hypothesis_sha256") != expected_hypothesis_hash:
            raise ValueError(f"{location}.hypothesis_sha256 does not match text")
        _validate_prepared_request(sample, location)

        observations = sample.get("observations")
        if not isinstance(observations, list) or len(observations) != repeats:
            raise ValueError(f"{location}.observations must contain run.repeats rows")
        generation_values: list[float] = []
        for repeat, observation in enumerate(observations):
            observation_location = f"{location}.observations[{repeat}]"
            if not isinstance(observation, dict):
                raise ValueError(f"{observation_location} must be an object")
            if observation.get("repeat") != repeat:
                raise ValueError(f"{observation_location}.repeat is inconsistent")
            generation_ms = observation.get("generation_ms")
            if not _finite_number(generation_ms, positive=True):
                raise ValueError(f"{observation_location}.generation_ms is invalid")
            generation_values.append(float(generation_ms))
            if observation.get("generated_tokens") != len(token_ids):
                raise ValueError(f"{observation_location}.generated_tokens is inconsistent")
            if observation.get("generated_token_ids_sha256") != expected_token_hash:
                raise ValueError(
                    f"{observation_location}.generated_token_ids_sha256 is inconsistent"
                )
            if observation.get("hypothesis_sha256") != expected_hypothesis_hash:
                raise ValueError(f"{observation_location}.hypothesis_sha256 is inconsistent")
            memory = observation.get("memory_bytes")
            if not isinstance(memory, dict) or not _finite_number(
                memory.get("generation_allocated_peak_delta")
            ):
                raise ValueError(f"{observation_location}.memory_bytes peak delta is invalid")
        summary = sample.get("summary")
        if not isinstance(summary, dict):
            raise ValueError(f"{location}.summary must be an object")
        _validate_summary(
            summary.get("generation_ms"),
            generation_values,
            f"{location}.summary.generation_ms",
        )
        all_generation_values.extend(generation_values)

    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"{label}.aggregate must be an object")
    _validate_summary(
        aggregate.get("generation_ms"),
        all_generation_values,
        f"{label}.aggregate.generation_ms",
    )


def _median(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot take median of an empty sequence")
    return float(statistics.median(materialized))


def _sample_index(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for sample in payload["samples"]:
        sample_id = str(sample.get("sample_id", ""))
        if not sample_id or sample_id in result:
            raise ValueError(f"empty or duplicate sample_id: {sample_id!r}")
        result[sample_id] = sample
    return result


def compare(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    _validate_canonical_payload(baseline, "baseline")
    _validate_canonical_payload(candidate, "candidate")
    baseline_run, candidate_run = baseline["run"], candidate["run"]
    for label, run in (("baseline", baseline_run), ("candidate", candidate_run)):
        if run.get("measurement_profile") != "canonical_latency":
            raise ValueError(f"{label} measurement_profile must equal 'canonical_latency'")
        if run.get("phase_timing") is not False:
            raise ValueError(f"{label} phase_timing must be false")
        if float(run.get("nvml_sample_ms", -1)) != 0.0:
            raise ValueError(f"{label} NVML sampling must be disabled")
    for key in (
        "model_name",
        "model_revision",
        "manifest_sha256",
        "warmup",
        "repeats",
        "max_new_tokens",
    ):
        _equal_field(baseline_run, candidate_run, key, "run")
    if baseline.get("model_snapshot") != candidate.get("model_snapshot"):
        raise ValueError("model snapshot bindings differ between runs")
    if baseline.get("readiness") != candidate.get("readiness"):
        raise ValueError("readiness bindings differ between runs")
    baseline_dtype = baseline_run["runner_configuration"]["storage_activation_dtype"]
    candidate_dtype = candidate_run["runner_configuration"]["storage_activation_dtype"]
    if baseline_dtype != candidate_dtype:
        raise ValueError("paired runs must use the same storage and activation precision")
    if baseline.get("sources") != candidate.get("sources"):
        raise ValueError("source maps differ between runs")
    if baseline.get("git") != candidate.get("git"):
        raise ValueError("Git provenance differs between runs")

    baseline_environment = baseline.get("environment", {})
    candidate_environment = candidate.get("environment", {})
    for label, environment in (
        ("baseline", baseline_environment),
        ("candidate", candidate_environment),
    ):
        lock_hash = environment.get("environment_lock_sha256")
        if not _is_sha256(lock_hash):
            raise ValueError(f"{label} environment_lock_sha256 must be a full SHA256")
    if baseline_environment != candidate_environment:
        raise ValueError("complete runtime environment records differ between runs")

    baseline_samples = _sample_index(baseline)
    candidate_samples = _sample_index(candidate)
    if set(baseline_samples) != set(candidate_samples):
        raise ValueError("sample ID sets differ")

    sample_rows = []
    for sample_id in baseline_samples:
        left, right = baseline_samples[sample_id], candidate_samples[sample_id]
        _equal_field(left, right, "audio_sha256", f"sample[{sample_id}]")
        _equal_field(left, right, "language", f"sample[{sample_id}]")
        _equal_field(left, right, "duration_s", f"sample[{sample_id}]")
        _equal_field(left, right, "generated_tokens", f"sample[{sample_id}]")
        if _input_fingerprints(left) != _input_fingerprints(right):
            raise ValueError(f"prepared input fingerprints differ for {sample_id}")
        if left.get("generated_token_ids") != right.get("generated_token_ids"):
            raise ValueError(f"generated token IDs differ for {sample_id}")
        if left.get("quality") != right.get("quality"):
            raise ValueError(f"quality records differ for {sample_id}")

        baseline_p50 = float(left["summary"]["generation_ms"]["p50"])
        candidate_p50 = float(right["summary"]["generation_ms"]["p50"])
        baseline_p95 = float(left["summary"]["generation_ms"]["p95"])
        candidate_p95 = float(right["summary"]["generation_ms"]["p95"])
        if candidate_p50 <= 0 or candidate_p95 <= 0:
            raise ValueError(f"candidate latency must be positive for {sample_id}")
        baseline_memory = _median(
            observation["memory_bytes"]["generation_allocated_peak_delta"]
            for observation in left["observations"]
        )
        candidate_memory = _median(
            observation["memory_bytes"]["generation_allocated_peak_delta"]
            for observation in right["observations"]
        )
        sample_rows.append(
            {
                "sample_id": sample_id,
                "duration_s": left["duration_s"],
                "generated_tokens": left["generated_tokens"],
                "baseline_generation_p50_ms": baseline_p50,
                "candidate_generation_p50_ms": candidate_p50,
                "p50_speedup": baseline_p50 / candidate_p50,
                "baseline_generation_p95_ms": baseline_p95,
                "candidate_generation_p95_ms": candidate_p95,
                "p95_speedup": baseline_p95 / candidate_p95,
                "baseline_generation_peak_delta_median_bytes": baseline_memory,
                "candidate_generation_peak_delta_median_bytes": candidate_memory,
                "generation_peak_delta_change_bytes": candidate_memory - baseline_memory,
            }
        )

    aggregate_baseline_p50 = float(baseline["aggregate"]["generation_ms"]["p50"])
    aggregate_candidate_p50 = float(candidate["aggregate"]["generation_ms"]["p50"])
    aggregate_baseline_p95 = float(baseline["aggregate"]["generation_ms"]["p95"])
    aggregate_candidate_p95 = float(candidate["aggregate"]["generation_ms"]["p95"])
    if aggregate_candidate_p50 <= 0 or aggregate_candidate_p95 <= 0:
        raise ValueError("candidate aggregate latency must be positive")
    return {
        "schema_version": "asr-latency-comparison-v0.1",
        "artifact_scope": "private",
        "baseline_implementation": baseline_run["implementation"],
        "candidate_implementation": candidate_run["implementation"],
        "baseline_runner_configuration": dict(baseline_run["runner_configuration"]),
        "candidate_runner_configuration": dict(candidate_run["runner_configuration"]),
        "model_name": baseline_run["model_name"],
        "model_revision": baseline_run["model_revision"],
        "manifest_sha256": baseline_run["manifest_sha256"],
        "environment": dict(baseline_environment),
        "quality_gate": (
            "exact generated token IDs and identical quality records; a null/null "
            "quality pair means quality was unavailable, not that accuracy passed"
        ),
        "interpretation_warning": (
            "overall percentiles mix sample durations; use per-sample/duration-bucket "
            "rows for claims, and verify alternating run order from the experiment log"
        ),
        "aggregate": {
            "baseline_generation_p50_ms": aggregate_baseline_p50,
            "candidate_generation_p50_ms": aggregate_candidate_p50,
            "p50_speedup": aggregate_baseline_p50 / aggregate_candidate_p50,
            "baseline_generation_p95_ms": aggregate_baseline_p95,
            "candidate_generation_p95_ms": aggregate_candidate_p95,
            "p95_speedup": aggregate_baseline_p95 / aggregate_candidate_p95,
        },
        "samples": sample_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists (use --overwrite intentionally): {args.output}")
    try:
        baseline = _load(args.baseline)
        candidate = _load(args.candidate)
        result = compare(baseline, candidate)
        result["inputs"] = {
            "baseline_artifact_sha256": _sha256(args.baseline),
            "candidate_artifact_sha256": _sha256(args.candidate),
        }
        result["comparator"] = {
            "source_sha256": _sha256(Path(__file__).resolve()),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomically(args.output, result, overwrite=args.overwrite)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(f"comparison: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
