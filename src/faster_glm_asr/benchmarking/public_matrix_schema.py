"""Strict, standard-library-only schema for public matrix summaries.

This module deliberately has no package-relative or third-party imports.  The
matrix exporter imports it normally, while the source-release guard loads this
file directly so candidate and historical result blobs use the same validator.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

PUBLIC_SCHEMA = "faster-glm-asr-public-matrix-summary-v0.1"
DEFAULT_MODEL_ID = "zai-org/GLM-ASR-Nano-2512"
DEFAULT_MODEL_REVISION = "61ba4e0b3309b6656edea3e93e419f7bd5c61957"
IMPLEMENTATIONS = (
    "hf_cached",
    "hf_no_cache",
    "custom_full_prefix",
    "custom_greedy_full_prefix",
    "custom_tuple_cache",
    "custom_static_cache",
)
PROFILES = ("canonical_latency", "request_memory", "diagnostic_phase")
BUCKETS: tuple[tuple[str, float, float, bool], ...] = (
    ("short", 0.0, 10.0, False),
    ("medium", 10.0, 20.0, False),
    ("near-30s", 20.0, 30.0, True),
)
COMPARISON_PAIRS = (
    ("hf_no_cache", "hf_cached"),
    ("custom_full_prefix", "custom_greedy_full_prefix"),
    ("custom_greedy_full_prefix", "custom_tuple_cache"),
    ("custom_tuple_cache", "custom_static_cache"),
    ("custom_full_prefix", "custom_static_cache"),
)
PACKAGE_KEYS = frozenset(
    {
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
)

ROOT_KEYS = {
    "schema_version",
    "artifact_scope",
    "evidence",
    "benchmark",
    "hardware",
    "software",
    "protocol",
    "results",
    "parity",
    "comparisons",
}
EVIDENCE_KEYS = {
    "git_commit",
    "source_map_sha256",
    "plan_sha256",
    "journal_sha256",
    "aggregate_set_sha256",
    "dataset_archive_sha256",
    "exporter_source_sha256",
    "schema_source_sha256",
}
BENCHMARK_KEYS = {"model", "dataset"}
MODEL_KEYS = {"repository", "revision"}
DATASET_KEYS = {
    "name",
    "subset",
    "selection",
    "utterance_count",
    "duration_buckets",
}
BUCKET_DEFINITION_KEYS = {
    "name",
    "minimum_seconds",
    "maximum_seconds",
    "maximum_inclusive",
}
HARDWARE_KEYS = {"accelerator_model", "nominal_memory_gib", "compute_capability"}
SOFTWARE_KEYS = {
    "python",
    "cuda_runtime",
    "cudnn_version",
    "nvidia_driver_version",
    "packages",
}
PROTOCOL_KEYS = {
    "implementations",
    "measurement_profiles",
    "outer_passes",
    "warmup_iterations_per_task",
    "measured_repeats_per_pass",
    "measured_repeats_per_bucket",
    "max_new_tokens",
    "storage_activation_dtype",
    "canonical_latency",
    "request_memory",
    "diagnostic_phase",
}
CANONICAL_PROTOCOL_KEYS = {
    "instrumentation",
    "synchronization_boundary",
    "statistics_scope",
    "rtf_definition",
}
MEMORY_PROTOCOL_KEYS = {
    "instrumentation",
    "poll_interval_ms",
    "process_peak_semantics",
    "allocator_peak_delta_semantics",
    "canonical_latency_eligible",
}
DIAGNOSTIC_PROTOCOL_KEYS = {"public_metrics_included", "canonical_latency_eligible"}
RESULT_KEYS = {"implementation", "duration_buckets"}
RESULT_BUCKET_KEYS = {"name", "canonical", "request_memory"}
CANONICAL_RESULT_KEYS = {
    "observations",
    "generation_p50_ms",
    "generation_p95_ms",
    "rtf_p50",
    "rtf_p95",
}
MEMORY_RESULT_KEYS = {
    "observations",
    "allocator_peak_delta_p50_mib",
    "allocator_peak_delta_p95_mib",
    "process_peak_observed_p50_mib",
    "process_peak_observed_p95_mib",
    "minimum_poll_samples_per_request",
    "process_peak_observed_is_lower_bound",
}
PARITY_KEYS = {
    "status",
    "criterion",
    "profile_scope",
    "implementations_checked",
    "duration_buckets_checked",
}
COMPARISON_KEYS = {
    "baseline",
    "candidate",
    "exact_token_parity",
    "speedup_definition",
    "duration_buckets",
}
COMPARISON_BUCKET_KEYS = {"name", "generation_p50_speedup", "generation_p95_speedup"}

NUMERIC_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}$")
PYTHON_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
CUDA_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+$")
NVIDIA_DRIVER_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){1,2}$")
IPV4_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])$"
)
PUBLIC_PEP440_RE = re.compile(
    r"^(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))*"
    r"(?:(?:a|b|rc)(?:0|[1-9][0-9]*))?"
    r"(?:\.post(?:0|[1-9][0-9]*))?"
    r"(?:\.dev(?:0|[1-9][0-9]*))?$"
)
LOWER_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
LOWER_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NOMINAL_MEMORY_GIB = 24
NOMINAL_MEMORY_MIB = float(NOMINAL_MEMORY_GIB * 1024)
ABSOLUTE_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/)")
GPU_UUID_RE = re.compile(r"\bGPU-[A-Za-z0-9-]{8,}\b", re.IGNORECASE)
PCI_BDF_RE = re.compile(r"\b[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]\b")
FORBIDDEN_PUBLIC_KEYS = {
    "sample_id",
    "sample_ids",
    "transcript",
    "transcripts",
    "hypothesis",
    "hypothesis_sha256",
    "generated_token_ids",
    "generated_token_ids_sha256",
    "audio",
    "audio_path",
    "audio_sha256",
    "manifest",
    "manifest_path",
    "manifest_sha256",
    "prepared_request",
    "content_sha256",
    "branch",
    "device_uuid",
    "nvml_device_uuid",
    "bdf",
    "cuda_visible_devices",
    "hostname",
    "command",
    "command_argv",
    "stdout",
    "stderr",
    "log",
    "logs",
    "path",
    "sha256",
    "wer",
    "cer",
    "quality",
}


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number: {value}")


def load_public_summary_bytes(raw: bytes) -> dict[str, Any]:
    """Parse strict UTF-8 JSON, rejecting duplicate keys and non-finite values."""
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("public summary is not UTF-8 JSON") from exc
    payload = json.loads(
        text,
        object_pairs_hook=_duplicate_rejecting_object,
        parse_constant=_reject_nonfinite_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("public summary root must be a JSON object")
    return payload


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)


def _schema_error(errors: list[str], value: Any, expected: set[str], location: str) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{location} must be an object")
        return False
    actual = set(value)
    if actual != expected:
        errors.append(
            f"{location} schema mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
        return False
    return True


def _finite(value: Any, *, positive: bool = False) -> bool:
    # Every continuous quantity emitted by the exporter is deliberately a JSON
    # float.  Requiring that exact type avoids bool/int coercion and the loss of
    # ordering beyond 2**53 when integers are converted to binary64.
    if type(value) is not float:
        return False
    return math.isfinite(value) and (value > 0 if positive else value >= 0)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _duration_in_bucket(
    duration_s: float, *, lower: float, upper: float, upper_inclusive: bool
) -> bool:
    # RTF and latency are serialized independently, so an exact source duration
    # at 10/20/30 seconds can round a few binary64 ulps to either side. Snap only
    # that representation noise; a manifest duration rounded to 1 ns remains
    # distinguishable from the boundary.
    for boundary in (lower, upper):
        if math.isclose(duration_s, boundary, rel_tol=0.0, abs_tol=1e-12):
            duration_s = boundary
    return duration_s >= lower and (duration_s <= upper if upper_inclusive else duration_s < upper)


def _validate_bucket_definitions(value: Any, errors: list[str], location: str) -> None:
    if not isinstance(value, list) or len(value) != len(BUCKETS):
        errors.append(f"{location} must contain the fixed three duration buckets")
        return
    for index, (name, lower, upper, inclusive) in enumerate(BUCKETS):
        item = value[index]
        item_location = f"{location}[{index}]"
        if not _schema_error(errors, item, BUCKET_DEFINITION_KEYS, item_location):
            continue
        expected = {
            "name": name,
            "minimum_seconds": lower,
            "maximum_seconds": upper,
            "maximum_inclusive": inclusive,
        }
        if not _strict_equal(item, expected):
            errors.append(f"{item_location} does not equal the fixed bucket definition")


def _validate_result_buckets(
    value: Any,
    errors: list[str],
    location: str,
    *,
    expected_observations: int | None,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    if not isinstance(value, list) or len(value) != len(BUCKETS):
        errors.append(f"{location} must contain the fixed three duration buckets")
        return metrics
    for index, (bucket, lower, upper, upper_inclusive) in enumerate(BUCKETS):
        item = value[index]
        item_location = f"{location}[{index}]"
        if not _schema_error(errors, item, RESULT_BUCKET_KEYS, item_location):
            continue
        if item.get("name") != bucket:
            errors.append(f"{item_location}.name is out of order")
        canonical = item.get("canonical")
        if _schema_error(errors, canonical, CANONICAL_RESULT_KEYS, f"{item_location}.canonical"):
            if not _positive_int(canonical["observations"]):
                errors.append(f"{item_location}.canonical.observations must be positive")
            elif (
                expected_observations is not None
                and canonical["observations"] != expected_observations
            ):
                errors.append(
                    f"{item_location}.canonical.observations is inconsistent with protocol"
                )
            for key in CANONICAL_RESULT_KEYS - {"observations"}:
                if not _finite(canonical[key], positive=True):
                    errors.append(f"{item_location}.canonical.{key} must be finite and positive")
            for prefix in ("generation", "rtf"):
                p50 = canonical.get(f"{prefix}_p50_ms" if prefix == "generation" else "rtf_p50")
                p95 = canonical.get(f"{prefix}_p95_ms" if prefix == "generation" else "rtf_p95")
                if (
                    _finite(p50, positive=True)
                    and _finite(p95, positive=True)
                    and float(p95) < float(p50)
                ):
                    errors.append(f"{item_location}.canonical {prefix} p95 must be >= p50")
            if all(
                _finite(canonical.get(key), positive=True)
                for key in CANONICAL_RESULT_KEYS - {"observations"}
            ):
                inferred_durations: dict[str, float] = {}
                for percentile in ("p50", "p95"):
                    generation_ms = canonical[f"generation_{percentile}_ms"]
                    rtf = canonical[f"rtf_{percentile}"]
                    duration_s = generation_ms / rtf / 1000.0
                    if not math.isfinite(duration_s) or not _duration_in_bucket(
                        duration_s,
                        lower=lower,
                        upper=upper,
                        upper_inclusive=upper_inclusive,
                    ):
                        errors.append(
                            f"{item_location}.canonical {percentile} inferred duration "
                            f"does not belong to bucket {bucket!r}"
                        )
                    if math.isfinite(duration_s):
                        inferred_durations[percentile] = duration_s
                if set(inferred_durations) == {"p50", "p95"} and not math.isclose(
                    inferred_durations["p50"],
                    inferred_durations["p95"],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    errors.append(
                        f"{item_location}.canonical p50/p95 inferred durations are inconsistent"
                    )
            if all(
                _finite(canonical.get(key), positive=True)
                for key in CANONICAL_RESULT_KEYS - {"observations"}
            ):
                metrics[bucket] = {
                    "p50": float(canonical["generation_p50_ms"]),
                    "p95": float(canonical["generation_p95_ms"]),
                }
        memory = item.get("request_memory")
        if _schema_error(errors, memory, MEMORY_RESULT_KEYS, f"{item_location}.request_memory"):
            if not _positive_int(memory["observations"]):
                errors.append(f"{item_location}.request_memory.observations must be positive")
            elif (
                expected_observations is not None
                and memory["observations"] != expected_observations
            ):
                errors.append(
                    f"{item_location}.request_memory.observations is inconsistent with protocol"
                )
            if not _positive_int(memory["minimum_poll_samples_per_request"]):
                errors.append(
                    f"{item_location}.request_memory.minimum_poll_samples_per_request must be positive"
                )
            if memory["process_peak_observed_is_lower_bound"] is not True:
                errors.append(
                    f"{item_location}.request_memory.process_peak_observed_is_lower_bound must be true"
                )
            pairs = (
                ("allocator_peak_delta_p50_mib", "allocator_peak_delta_p95_mib"),
                ("process_peak_observed_p50_mib", "process_peak_observed_p95_mib"),
            )
            for p50_key, p95_key in pairs:
                p50, p95 = memory.get(p50_key), memory.get(p95_key)
                if not _finite(p50):
                    errors.append(f"{item_location}.request_memory.{p50_key} must be finite")
                if not _finite(p95):
                    errors.append(f"{item_location}.request_memory.{p95_key} must be finite")
                if _finite(p50) and p50 > NOMINAL_MEMORY_MIB:
                    errors.append(
                        f"{item_location}.request_memory.{p50_key} exceeds the "
                        f"{NOMINAL_MEMORY_GIB} GiB hardware limit"
                    )
                if _finite(p95) and p95 > NOMINAL_MEMORY_MIB:
                    errors.append(
                        f"{item_location}.request_memory.{p95_key} exceeds the "
                        f"{NOMINAL_MEMORY_GIB} GiB hardware limit"
                    )
                if _finite(p50) and _finite(p95) and float(p95) < float(p50):
                    errors.append(f"{item_location}.request_memory {p95_key} must be >= {p50_key}")
    return metrics


def _validate_comparison_buckets(
    value: Any, errors: list[str], location: str
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    if not isinstance(value, list) or len(value) != len(BUCKETS):
        errors.append(f"{location} must contain the fixed three duration buckets")
        return metrics
    for index, (bucket, _lower, _upper, _inclusive) in enumerate(BUCKETS):
        item = value[index]
        item_location = f"{location}[{index}]"
        if not _schema_error(errors, item, COMPARISON_BUCKET_KEYS, item_location):
            continue
        if item.get("name") != bucket:
            errors.append(f"{item_location}.name is out of order")
        if all(_finite(item.get(key), positive=True) for key in COMPARISON_BUCKET_KEYS - {"name"}):
            metrics[bucket] = {
                "p50": float(item["generation_p50_speedup"]),
                "p95": float(item["generation_p95_speedup"]),
            }
        else:
            errors.append(f"{item_location} speedups must be finite and positive")
    return metrics


def _privacy_errors(payload: Any) -> list[str]:
    errors: list[str] = []

    def walk(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                folded = key.casefold()
                allowed_evidence_digest = (
                    location == "root.evidence"
                    and folded in EVIDENCE_KEYS
                    and folded.endswith("_sha256")
                )
                if (
                    folded in FORBIDDEN_PUBLIC_KEYS
                    or ("sha256" in folded and not allowed_evidence_digest)
                    or folded.endswith("_path")
                ):
                    errors.append(f"forbidden public key at {location}.{key}")
                walk(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{location}[{index}]")
        elif isinstance(value, str):
            if ABSOLUTE_PATH_RE.match(value):
                errors.append(f"absolute path at {location}")
            if GPU_UUID_RE.search(value):
                errors.append(f"GPU UUID at {location}")
            if PCI_BDF_RE.search(value):
                errors.append(f"PCI BDF at {location}")
            if "cuda_visible_devices" in value.casefold():
                errors.append(f"CUDA visibility setting at {location}")

    walk(payload, "root")
    return errors


def validate_public_summary(payload: Any) -> list[str]:
    """Return every strict schema, metric-consistency, and privacy error."""
    errors: list[str] = []
    if not _schema_error(errors, payload, ROOT_KEYS, "root") and not isinstance(payload, dict):
        return errors
    assert isinstance(payload, dict)
    if payload.get("schema_version") != PUBLIC_SCHEMA:
        errors.append(f"root.schema_version must equal {PUBLIC_SCHEMA!r}")
    if payload.get("artifact_scope") != "public":
        errors.append("root.artifact_scope must equal 'public'")

    evidence = payload.get("evidence")
    if _schema_error(errors, evidence, EVIDENCE_KEYS, "root.evidence"):
        for key in sorted(EVIDENCE_KEYS):
            value = evidence.get(key)
            pattern = LOWER_COMMIT_RE if key == "git_commit" else LOWER_SHA256_RE
            if not isinstance(value, str) or pattern.fullmatch(value) is None:
                width = "40 or 64" if key == "git_commit" else "64"
                errors.append(
                    f"root.evidence.{key} must be exactly {width} lowercase hexadecimal characters"
                )

    benchmark = payload.get("benchmark")
    if _schema_error(errors, benchmark, BENCHMARK_KEYS, "root.benchmark"):
        model = benchmark.get("model")
        if _schema_error(errors, model, MODEL_KEYS, "root.benchmark.model") and not _strict_equal(
            model, {"repository": DEFAULT_MODEL_ID, "revision": DEFAULT_MODEL_REVISION}
        ):
            errors.append("root.benchmark.model must equal the pinned GLM-ASR model")
        dataset = benchmark.get("dataset")
        if _schema_error(errors, dataset, DATASET_KEYS, "root.benchmark.dataset"):
            scalars = {
                "name": "LibriSpeech",
                "subset": "test-clean",
                "selection": "one duration-stratified utterance per bucket",
                "utterance_count": len(BUCKETS),
            }
            for key, expected in scalars.items():
                if not _strict_equal(dataset.get(key), expected):
                    errors.append(f"root.benchmark.dataset.{key} is invalid")
            _validate_bucket_definitions(
                dataset.get("duration_buckets"), errors, "root.benchmark.dataset.duration_buckets"
            )

    hardware = payload.get("hardware")
    expected_hardware = {
        "accelerator_model": "NVIDIA GeForce RTX 3090",
        "nominal_memory_gib": NOMINAL_MEMORY_GIB,
        "compute_capability": "8.6",
    }
    if _schema_error(errors, hardware, HARDWARE_KEYS, "root.hardware") and not _strict_equal(
        hardware, expected_hardware
    ):
        errors.append("root.hardware must equal the admitted 24 GiB RTX 3090 class")

    software = payload.get("software")
    if _schema_error(errors, software, SOFTWARE_KEYS, "root.software"):
        version_contracts = (
            ("python", PYTHON_VERSION_RE, "major.minor.patch"),
            ("cuda_runtime", CUDA_VERSION_RE, "major.minor"),
            (
                "nvidia_driver_version",
                NVIDIA_DRIVER_VERSION_RE,
                "two or three numeric dot-separated segments and not an IPv4 address",
            ),
        )
        for key, pattern, description in version_contracts:
            version = software.get(key)
            if (
                not isinstance(version, str)
                or pattern.fullmatch(version) is None
                or (key == "nvidia_driver_version" and IPV4_RE.fullmatch(version) is not None)
            ):
                errors.append(f"root.software.{key} must use {description}")
        if not _positive_int(software.get("cudnn_version")):
            errors.append("root.software.cudnn_version must be a positive integer")
        packages = software.get("packages")
        if _schema_error(errors, packages, set(PACKAGE_KEYS), "root.software.packages"):
            for package, version in packages.items():
                if not isinstance(version, str) or PUBLIC_PEP440_RE.fullmatch(version) is None:
                    errors.append(
                        f"root.software.packages[{package!r}] must be normalized PEP 440 without local metadata"
                    )

    protocol = payload.get("protocol")
    expected_observations: int | None = None
    if _schema_error(errors, protocol, PROTOCOL_KEYS, "root.protocol"):
        if not _strict_equal(protocol.get("implementations"), list(IMPLEMENTATIONS)):
            errors.append("root.protocol.implementations must equal the fixed six mechanisms")
        if not _strict_equal(protocol.get("measurement_profiles"), list(PROFILES)):
            errors.append("root.protocol.measurement_profiles must equal the fixed three profiles")
        integer_keys = (
            "outer_passes",
            "warmup_iterations_per_task",
            "measured_repeats_per_pass",
            "measured_repeats_per_bucket",
            "max_new_tokens",
        )
        for key in integer_keys:
            if not _positive_int(protocol.get(key)):
                errors.append(f"root.protocol.{key} must be a positive integer")
        if _positive_int(protocol.get("outer_passes")) and protocol["outer_passes"] < 3:
            errors.append("root.protocol.outer_passes must be at least three")
        if all(
            _positive_int(protocol.get(key))
            for key in ("outer_passes", "measured_repeats_per_pass", "measured_repeats_per_bucket")
        ):
            expected_observations = protocol["outer_passes"] * protocol["measured_repeats_per_pass"]
            if protocol["measured_repeats_per_bucket"] != expected_observations:
                errors.append("root.protocol.measured_repeats_per_bucket is inconsistent")
        if protocol.get("storage_activation_dtype") != "torch.float32":
            errors.append("root.protocol.storage_activation_dtype must equal 'torch.float32'")
        canonical = protocol.get("canonical_latency")
        expected_canonical = {
            "instrumentation": "NVML polling disabled; diagnostic phase timing disabled",
            "synchronization_boundary": "CUDA synchronize immediately before and after autoregressive generation",
            "statistics_scope": "one duration bucket across all outer-pass observations",
            "rtf_definition": "generation wall-clock seconds / input audio seconds",
        }
        if _schema_error(
            errors, canonical, CANONICAL_PROTOCOL_KEYS, "root.protocol.canonical_latency"
        ) and not _strict_equal(canonical, expected_canonical):
            errors.append("root.protocol.canonical_latency contract is invalid")
        memory = protocol.get("request_memory")
        if _schema_error(errors, memory, MEMORY_PROTOCOL_KEYS, "root.protocol.request_memory"):
            if not _finite(memory.get("poll_interval_ms"), positive=True):
                errors.append("root.protocol.request_memory.poll_interval_ms must be positive")
            fixed = {
                "instrumentation": "NVML process polling plus PyTorch CUDA allocator counters",
                "process_peak_semantics": "polling-observed lower bound",
                "allocator_peak_delta_semantics": "generation allocated peak minus after-prepare allocated baseline, floored at zero",
                "canonical_latency_eligible": False,
            }
            for key, expected in fixed.items():
                if not _strict_equal(memory.get(key), expected):
                    errors.append(f"root.protocol.request_memory.{key} is invalid")
        diagnostic = protocol.get("diagnostic_phase")
        if _schema_error(
            errors, diagnostic, DIAGNOSTIC_PROTOCOL_KEYS, "root.protocol.diagnostic_phase"
        ) and not _strict_equal(
            diagnostic,
            {"public_metrics_included": False, "canonical_latency_eligible": False},
        ):
            errors.append("root.protocol.diagnostic_phase contract is invalid")

    result_metrics: dict[str, dict[str, dict[str, float]]] = {}
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(IMPLEMENTATIONS):
        errors.append("root.results must contain exactly six implementation rows")
    else:
        for index, implementation in enumerate(IMPLEMENTATIONS):
            row = results[index]
            location = f"root.results[{index}]"
            if not _schema_error(errors, row, RESULT_KEYS, location):
                continue
            if row.get("implementation") != implementation:
                errors.append(f"{location}.implementation is out of order")
            result_metrics[implementation] = _validate_result_buckets(
                row.get("duration_buckets"),
                errors,
                f"{location}.duration_buckets",
                expected_observations=expected_observations,
            )

    parity = payload.get("parity")
    expected_parity = {
        "status": "pass",
        "criterion": "exact generated-token sequence equality",
        "profile_scope": "all 18 implementation/profile aggregates",
        "implementations_checked": len(IMPLEMENTATIONS),
        "duration_buckets_checked": len(BUCKETS),
    }
    if _schema_error(errors, parity, PARITY_KEYS, "root.parity") and not _strict_equal(
        parity, expected_parity
    ):
        errors.append("root.parity must prove exact matrix-wide generated-token equality")

    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != len(COMPARISON_PAIRS):
        errors.append("root.comparisons must contain the fixed five comparisons")
    else:
        for index, expected_pair in enumerate(COMPARISON_PAIRS):
            row = comparisons[index]
            location = f"root.comparisons[{index}]"
            if not _schema_error(errors, row, COMPARISON_KEYS, location):
                continue
            if (row.get("baseline"), row.get("candidate")) != expected_pair:
                errors.append(f"{location} baseline/candidate pair is out of order")
            if row.get("exact_token_parity") is not True:
                errors.append(f"{location}.exact_token_parity must be true")
            if row.get("speedup_definition") != "baseline latency / candidate latency":
                errors.append(f"{location}.speedup_definition is invalid")
            observed = _validate_comparison_buckets(
                row.get("duration_buckets"), errors, f"{location}.duration_buckets"
            )
            baseline_metrics = result_metrics.get(expected_pair[0], {})
            candidate_metrics = result_metrics.get(expected_pair[1], {})
            for bucket, _lower, _upper, _inclusive in BUCKETS:
                if (
                    bucket not in observed
                    or bucket not in baseline_metrics
                    or bucket not in candidate_metrics
                ):
                    continue
                for percentile in ("p50", "p95"):
                    expected = (
                        baseline_metrics[bucket][percentile] / candidate_metrics[bucket][percentile]
                    )
                    if observed[bucket][percentile] != expected:
                        errors.append(
                            f"{location}.{bucket}.{percentile} speedup does not match result rows"
                        )

    errors.extend(_privacy_errors(payload))
    return errors
