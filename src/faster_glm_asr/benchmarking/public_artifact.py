#!/usr/bin/env python3
"""Fail closed when a supposedly public benchmark JSON contains private data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from faster_glm_asr.benchmarking import provenance

ABSOLUTE_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/)")
FORBIDDEN_KEYS = {
    "cuda_visible_devices",
    "device_uuid",
    "manifest_path",
    "manifest_sha256",
    "reference_path",
    "reference_text",
    "reference_normalized",
    "hypothesis_normalized",
    "nvml_device_uuid",
}
PRIVATE_METADATA_ALLOWLIST = {
    "shareable",
    "duration_bucket",
    "direct_model_eligible",
    "manifest_parser_version",
}
PRIVATE_METADATA_FIXED_VALUES = {
    "shareable": False,
    "direct_model_eligible": True,
}
PRIVATE_MANIFEST_PARSER_VALUES = {"asr-inventory-v0.2", "asr-segment-v0.1"}
OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PRIVATE_SAMPLE_KEYS = {
    "sample_id",
    "language",
    "duration_s",
    "generated_tokens",
    "quality",
    "summary",
    "metadata",
}
SHAREABLE_SAMPLE_KEYS = PRIVATE_SAMPLE_KEYS | {"hypothesis"}
SHAREABLE_METADATA_ALLOWLIST = {
    "shareable",
    "authorization_id",
    "duration_bucket",
    "direct_model_eligible",
    "manifest_parser_version",
    "collection",
    "speaker_count",
    "sample_rate_hz",
    "channels",
    "license_scope",
    "split",
}
SHAREABLE_METADATA_REQUIRED = {
    "shareable",
    "authorization_id",
    "duration_bucket",
    "direct_model_eligible",
    "manifest_parser_version",
}
PRIVATE_SUMMARY_KEYS = {
    "generation_ms",
    "processor_plus_generation_ms",
    "request_no_file_io_ms",
    "rtf_generation",
}
SUMMARY_STAT_KEYS = {"n", "mean", "std", "min", "p50", "p95", "max"}
PRIVATE_QUALITY_KEYS = {"wer", "cer", "status"}
PRIVATE_QUALITY_STATUSES = {
    "evaluated",
    "not_evaluated_empty_after_normalization",
}
ROOT_KEYS = {
    "schema_version",
    "created_at_utc",
    "run",
    "git",
    "sources",
    "model_snapshot",
    "readiness",
    "environment",
    "aggregate",
    "samples",
}
PUBLIC_RUN_KEYS = {
    "implementation",
    "model_name",
    "model_revision",
    "formal_run",
    "comparability",
    "warmup_gate",
    "warmup",
    "repeats",
    "max_new_tokens",
    "runner_construction_envelope_ms",
    "runner_construction_scope",
    "artifact_scope",
    "measurement_profile",
    "metric_eligibility",
    "nvml_sample_ms",
    "phase_timing",
    "phase_timing_warning",
    "runner_configuration",
    "private_manifest_fingerprint_redacted",
}
METRIC_ELIGIBILITY_KEYS = {
    "canonical_latency",
    "request_memory_observed_peak",
    "diagnostic_cuda_phases",
    "cold_start",
    "model_load",
}
PUBLIC_GIT_KEYS = {"commit", "dirty", "branch_redacted"}
PUBLIC_SOURCE_KEYS = {"schema_version", "files", "aggregate_sha256"}
SOURCE_RECORD_KEYS = {"size_bytes", "sha256"}
SOURCE_PATH_RE = re.compile(
    r"^src/faster_glm_asr/(?:modeling|kernels|benchmarking|data)/"
    r"[A-Za-z0-9_./-]+\.py$"
)
FORBIDDEN_TEXT_RE = re.compile(
    r"(?:\bedinburgh\b|\bcourse\b|hw1-asr|inputs[\\/]private)", re.IGNORECASE
)
PUBLIC_ENVIRONMENT_KEYS = {
    "python",
    "platform",
    "torch",
    "torch_cuda",
    "cudnn",
    "device_index",
    "device_name",
    "compute_capability",
    "total_memory_bytes",
    "allow_tf32_matmul",
    "allow_tf32_cudnn",
    "nvidia_driver_versions",
    "package_versions",
    "environment_lock_sha256",
    "device_identity_redacted",
}
PACKAGE_VERSION_KEYS = {
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
PUBLIC_AGGREGATE_KEYS = {
    "generation_ms",
    "processor_plus_generation_ms",
    "request_no_file_io_ms",
    "quality",
}
AGGREGATE_QUALITY_KEYS = {
    "evaluated_samples",
    "word",
    "char",
    "wer",
    "cer",
    "status",
    "normalization",
}
EDIT_COUNT_KEYS = {
    "errors",
    "substitutions",
    "deletions",
    "insertions",
    "reference_units",
}
CUSTOM_RUNNER_CONFIGURATION_KEYS = {
    "system",
    "model_source",
    "network_fallback",
    "storage_activation_dtype",
    "linear_backend",
    "mlp_fused",
    "kernel_dispatch_policy_version",
    "component_dispatch_policy",
    "attention_dispatch_policy",
    "gqa_kv_expansion",
    "long_form_attention_expectation",
    "backend_trace_status",
    "performance_attribution",
    "cache_mode",
    "greedy_fast_path",
    "token_output_mode",
}
HF_RUNNER_CONFIGURATION_KEYS = {
    "system",
    "model_source",
    "network_fallback",
    "storage_activation_dtype",
    "cache_mode",
    "do_sample",
}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _schema_error(errors: list[str], value: Any, expected: set[str], location: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{location} must be an object")
        return
    actual = set(value)
    if actual != expected:
        errors.append(
            f"{location} schema mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _valid_sha256_or_none(value: Any) -> bool:
    return value is None or (isinstance(value, str) and SHA256_RE.fullmatch(value))


def validate_public_artifact(payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["root must be a JSON object"]
    _schema_error(errors, payload, ROOT_KEYS, "root")
    if payload.get("schema_version") != "0.3":
        errors.append("root.schema_version must equal '0.3'")
    if not isinstance(payload.get("created_at_utc"), str) or not payload.get("created_at_utc"):
        errors.append("root.created_at_utc must be a non-empty string")

    run = payload.get("run")
    if not isinstance(run, dict):
        errors.append("root.run must be an object")
        run = {}
    else:
        _schema_error(errors, run, PUBLIC_RUN_KEYS, "root.run")
    if run.get("artifact_scope") != "public":
        errors.append("run.artifact_scope must equal 'public'")
    if run.get("formal_run") is not True:
        errors.append("run.formal_run must be true")
    if run.get("comparability") != {
        "status": "formal-comparable",
        "reasons": [],
    }:
        errors.append("run.comparability must prove formal comparability")
    warmup = run.get("warmup")
    warmup_gate = run.get("warmup_gate")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 1:
        errors.append("run.warmup must be an integer >= 1")
    if (
        not isinstance(warmup_gate, dict)
        or warmup_gate.get("status") != "pass"
        or warmup_gate.get("minimum_iterations") != 1
        or warmup_gate.get("observed_iterations") != warmup
    ):
        errors.append("run.warmup_gate must prove at least one warmup")
    if run.get("private_manifest_fingerprint_redacted") is not True:
        errors.append("run.private_manifest_fingerprint_redacted must be true")
    eligibility = run.get("metric_eligibility")
    _schema_error(errors, eligibility, METRIC_ELIGIBILITY_KEYS, "root.run.metric_eligibility")
    if isinstance(eligibility, dict) and any(
        not isinstance(value, bool) for value in eligibility.values()
    ):
        errors.append("root.run.metric_eligibility values must be booleans")
    runner_configuration = run.get("runner_configuration")
    if isinstance(runner_configuration, dict):
        system = runner_configuration.get("system")
        if system == "custom-torch-triton-hybrid":
            _schema_error(
                errors,
                runner_configuration,
                CUSTOM_RUNNER_CONFIGURATION_KEYS,
                "root.run.runner_configuration",
            )
        elif system == "huggingface-transformers":
            _schema_error(
                errors,
                runner_configuration,
                HF_RUNNER_CONFIGURATION_KEYS,
                "root.run.runner_configuration",
            )
        else:
            errors.append("root.run.runner_configuration.system is unsupported")
    else:
        errors.append("root.run.runner_configuration must be an object")

    git = payload.get("git")
    _schema_error(errors, git, PUBLIC_GIT_KEYS, "root.git")
    if isinstance(git, dict) and git.get("branch_redacted") is not True:
        errors.append("root.git.branch_redacted must be true")
    if isinstance(git, dict) and git.get("dirty") is not False:
        errors.append("root.git.dirty must be false")

    try:
        model_snapshot = provenance.validate_model_snapshot_binding(
            payload.get("model_snapshot"),
            expected_repo_id=run.get("model_name"),
            expected_revision=run.get("model_revision"),
        )
    except ValueError as exc:
        errors.append(f"root.model_snapshot: {exc}")
        model_snapshot = None
    try:
        readiness = provenance.validate_readiness_binding(payload.get("readiness"))
    except ValueError as exc:
        errors.append(f"root.readiness: {exc}")
        readiness = None
    if (
        model_snapshot is not None
        and readiness is not None
        and model_snapshot["contract_sha256"] != readiness["contract_sha256"]
    ):
        errors.append("root model/readiness contract bindings differ")

    sources = payload.get("sources")
    _schema_error(errors, sources, PUBLIC_SOURCE_KEYS, "root.sources")
    if isinstance(sources, dict):
        if sources.get("schema_version") != "faster-glm-asr-source-map-v1":
            errors.append("root.sources.schema_version is unsupported")
        source_files = sources.get("files")
        if not isinstance(source_files, dict):
            errors.append("root.sources.files must be an object")
        elif not source_files:
            errors.append("root.sources.files must be non-empty")
        else:
            for name, value in source_files.items():
                if not isinstance(name, str) or SOURCE_PATH_RE.fullmatch(name) is None:
                    errors.append(f"root.sources.files contains unsafe path {name!r}")
                    continue
                _schema_error(errors, value, SOURCE_RECORD_KEYS, f"root.sources.files[{name!r}]")
                if not isinstance(value, dict):
                    continue
                if (
                    not isinstance(value.get("size_bytes"), int)
                    or isinstance(value.get("size_bytes"), bool)
                    or value["size_bytes"] < 0
                ):
                    errors.append(
                        f"root.sources.files[{name!r}].size_bytes must be non-negative integer"
                    )
                if (
                    not isinstance(value.get("sha256"), str)
                    or SHA256_RE.fullmatch(value["sha256"]) is None
                ):
                    errors.append(f"root.sources.files[{name!r}].sha256 must be SHA256")
        if (
            not isinstance(sources.get("aggregate_sha256"), str)
            or SHA256_RE.fullmatch(sources.get("aggregate_sha256", "")) is None
        ):
            errors.append("root.sources.aggregate_sha256 must be SHA256")
        elif isinstance(source_files, dict) and source_files:
            canonical = json.dumps(
                source_files,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if hashlib.sha256(canonical).hexdigest() != sources["aggregate_sha256"]:
                errors.append("root.sources.aggregate_sha256 is inconsistent")

    environment = payload.get("environment")
    _schema_error(errors, environment, PUBLIC_ENVIRONMENT_KEYS, "root.environment")
    if isinstance(environment, dict):
        if environment.get("device_identity_redacted") is not True:
            errors.append("root.environment.device_identity_redacted must be true")
        lock_hash = environment.get("environment_lock_sha256")
        if not isinstance(lock_hash, str) or not SHA256_RE.fullmatch(lock_hash):
            errors.append("root.environment.environment_lock_sha256 must be SHA256")
        package_versions = environment.get("package_versions")
        _schema_error(
            errors,
            package_versions,
            PACKAGE_VERSION_KEYS,
            "root.environment.package_versions",
        )
        if isinstance(package_versions, dict):
            for package, version in package_versions.items():
                if version is not None and not isinstance(version, str):
                    errors.append(
                        f"root.environment.package_versions[{package!r}] must be null or text"
                    )
        drivers = environment.get("nvidia_driver_versions")
        if not isinstance(drivers, list) or any(
            not isinstance(value, str) or not value for value in drivers
        ):
            errors.append("root.environment.nvidia_driver_versions must be a list of text")

    aggregate = payload.get("aggregate")
    _schema_error(errors, aggregate, PUBLIC_AGGREGATE_KEYS, "root.aggregate")
    if isinstance(aggregate, dict):
        for metric in PUBLIC_AGGREGATE_KEYS - {"quality"}:
            statistics = aggregate.get(metric)
            _schema_error(errors, statistics, SUMMARY_STAT_KEYS, f"root.aggregate.{metric}")
        aggregate_quality = aggregate.get("quality")
        _schema_error(
            errors,
            aggregate_quality,
            AGGREGATE_QUALITY_KEYS,
            "root.aggregate.quality",
        )
        if isinstance(aggregate_quality, dict):
            for unit in ("word", "char"):
                _schema_error(
                    errors,
                    aggregate_quality.get(unit),
                    EDIT_COUNT_KEYS,
                    f"root.aggregate.quality.{unit}",
                )

    def walk(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_location = f"{location}.{key}"
                if key in FORBIDDEN_KEYS and child_location != (
                    "root.model_snapshot.manifest_sha256"
                ):
                    errors.append(f"forbidden key at {child_location}")
                walk(child, child_location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{location}[{index}]")
        elif isinstance(value, str):
            if ABSOLUTE_PATH_RE.match(value):
                errors.append(f"absolute local path at {location}")
            if FORBIDDEN_TEXT_RE.search(value):
                errors.append(f"private project marker at {location}")

    walk(payload, "root")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        errors.append("root.samples must be a list")
        return errors
    if not samples:
        errors.append("root.samples must be non-empty")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            errors.append(f"samples[{index}] must be an object")
            continue
        metadata = sample.get("metadata", {})
        if not isinstance(metadata, dict):
            errors.append(f"samples[{index}].metadata must be an object")
            metadata = {}
        shareable = metadata.get("shareable")
        if not isinstance(shareable, bool):
            errors.append(f"samples[{index}].metadata.shareable must be an explicit boolean")
        if shareable is not True:
            sample_keys = set(sample)
            if sample_keys != PRIVATE_SAMPLE_KEYS:
                errors.append(
                    f"samples[{index}] private sample schema mismatch: "
                    f"missing={sorted(PRIVATE_SAMPLE_KEYS - sample_keys)}, "
                    f"unexpected={sorted(sample_keys - PRIVATE_SAMPLE_KEYS)}"
                )
            if not re.fullmatch(r"private-\d+", str(sample.get("sample_id", ""))):
                errors.append(f"samples[{index}] private ID is not opaque")
            if sample.get("language") != "redacted":
                errors.append(f"samples[{index}] private language must be redacted")
            if "hypothesis" in sample:
                errors.append(f"samples[{index}] exposes a private hypothesis")

            def private_walk(value: Any, location: str) -> None:
                if isinstance(value, dict):
                    for key, child in value.items():
                        child_location = f"{location}.{key}"
                        if key.endswith("_sha256"):
                            errors.append(f"stable private fingerprint at {child_location}")
                        if key == "generated_token_ids":
                            errors.append(f"reconstructable private token IDs at {child_location}")
                        private_walk(child, child_location)
                elif isinstance(value, list):
                    for child_index, child in enumerate(value):
                        private_walk(child, f"{location}[{child_index}]")

            private_walk(sample, f"samples[{index}]")
            unexpected = set(metadata) - PRIVATE_METADATA_ALLOWLIST
            if unexpected:
                errors.append(
                    f"samples[{index}] private metadata is not allowlisted: {sorted(unexpected)}"
                )
            missing = PRIVATE_METADATA_ALLOWLIST - set(metadata)
            if missing:
                errors.append(f"samples[{index}] private metadata is incomplete: {sorted(missing)}")
            for key, expected in PRIVATE_METADATA_FIXED_VALUES.items():
                if key in metadata and metadata[key] != expected:
                    errors.append(
                        f"samples[{index}].metadata.{key} must equal "
                        f"{expected!r} for private public output"
                    )
            if metadata.get("manifest_parser_version") not in (PRIVATE_MANIFEST_PARSER_VALUES):
                errors.append(f"samples[{index}].metadata.manifest_parser_version is invalid")
            expected_duration_bucket = {
                "asr-inventory-v0.2": "direct-model",
                "asr-segment-v0.1": "materialized-segment",
            }.get(metadata.get("manifest_parser_version"))
            if metadata.get("duration_bucket") != expected_duration_bucket:
                errors.append(
                    f"samples[{index}].metadata.duration_bucket is inconsistent "
                    "with manifest_parser_version"
                )
            duration = sample.get("duration_s")
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not math.isfinite(float(duration))
                or duration <= 0
            ):
                errors.append(f"samples[{index}].duration_s must be finite and positive")
            generated_tokens = sample.get("generated_tokens")
            if (
                not isinstance(generated_tokens, int)
                or isinstance(generated_tokens, bool)
                or generated_tokens < 0
            ):
                errors.append(f"samples[{index}].generated_tokens must be a non-negative integer")
            quality = sample.get("quality")
            if quality is not None:
                if not isinstance(quality, dict) or set(quality) != PRIVATE_QUALITY_KEYS:
                    errors.append(f"samples[{index}].quality schema mismatch")
                else:
                    for metric in ("wer", "cer"):
                        value = quality[metric]
                        if value is not None and (
                            not isinstance(value, (int, float))
                            or isinstance(value, bool)
                            or not math.isfinite(float(value))
                            or value < 0
                        ):
                            errors.append(
                                f"samples[{index}].quality.{metric} must be null or "
                                "finite and non-negative"
                            )
                    if quality["status"] not in PRIVATE_QUALITY_STATUSES:
                        errors.append(f"samples[{index}].quality.status is invalid")
            summary = sample.get("summary")
            if not isinstance(summary, dict) or set(summary) != PRIVATE_SUMMARY_KEYS:
                errors.append(f"samples[{index}].summary schema mismatch")
            else:
                for metric, statistics in summary.items():
                    if not isinstance(statistics, dict) or set(statistics) != SUMMARY_STAT_KEYS:
                        errors.append(f"samples[{index}].summary.{metric} schema mismatch")
                        continue
                    for statistic, value in statistics.items():
                        if statistic == "n":
                            valid = (
                                isinstance(value, int) and not isinstance(value, bool) and value > 0
                            )
                        else:
                            valid = (
                                isinstance(value, (int, float))
                                and not isinstance(value, bool)
                                and math.isfinite(float(value))
                                and value >= 0
                            )
                        if not valid:
                            errors.append(
                                f"samples[{index}].summary.{metric}.{statistic} "
                                "must be a finite non-negative statistic"
                            )
        else:
            sample_keys = set(sample)
            if sample_keys != SHAREABLE_SAMPLE_KEYS:
                errors.append(
                    f"samples[{index}] shareable sample schema mismatch: "
                    f"missing={sorted(SHAREABLE_SAMPLE_KEYS - sample_keys)}, "
                    f"unexpected={sorted(sample_keys - SHAREABLE_SAMPLE_KEYS)}"
                )
            if not OPAQUE_ID_RE.fullmatch(str(sample.get("sample_id", ""))):
                errors.append(f"samples[{index}] shareable ID must be opaque")
            if not isinstance(sample.get("language"), str) or not sample["language"]:
                errors.append(f"samples[{index}] shareable language must be non-empty")
            if not isinstance(sample.get("hypothesis"), str):
                errors.append(f"samples[{index}] shareable hypothesis must be text")
            authorization_id = metadata.get("authorization_id")
            if not isinstance(authorization_id, str) or not OPAQUE_ID_RE.fullmatch(
                authorization_id
            ):
                errors.append(
                    f"samples[{index}] shareable metadata requires an opaque authorization_id"
                )
            unexpected = set(metadata) - SHAREABLE_METADATA_ALLOWLIST
            missing = SHAREABLE_METADATA_REQUIRED - set(metadata)
            if unexpected:
                errors.append(
                    f"samples[{index}] shareable metadata is not allowlisted: {sorted(unexpected)}"
                )
            if missing:
                errors.append(
                    f"samples[{index}] shareable metadata is incomplete: {sorted(missing)}"
                )
            if metadata.get("shareable") is not True:
                errors.append(f"samples[{index}].metadata.shareable must be true")
            if metadata.get("direct_model_eligible") is not True:
                errors.append(f"samples[{index}].metadata.direct_model_eligible must be true")
            if metadata.get("manifest_parser_version") != "asr-inventory-v0.2":
                errors.append(f"samples[{index}] shareable parser version must be source v0.2")
            if metadata.get("duration_bucket") != "direct-model":
                errors.append(f"samples[{index}] shareable duration_bucket must be direct-model")
            duration = sample.get("duration_s")
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not math.isfinite(float(duration))
                or duration <= 0
            ):
                errors.append(f"samples[{index}].duration_s must be finite and positive")
            generated_tokens = sample.get("generated_tokens")
            if (
                not isinstance(generated_tokens, int)
                or isinstance(generated_tokens, bool)
                or generated_tokens < 0
            ):
                errors.append(f"samples[{index}].generated_tokens must be non-negative")
            quality = sample.get("quality")
            if quality is not None:
                if not isinstance(quality, dict) or set(quality) != PRIVATE_QUALITY_KEYS:
                    errors.append(f"samples[{index}].quality schema mismatch")
                else:
                    for metric in ("wer", "cer"):
                        value = quality[metric]
                        if value is not None and (
                            not isinstance(value, (int, float))
                            or isinstance(value, bool)
                            or not math.isfinite(float(value))
                            or value < 0
                        ):
                            errors.append(
                                f"samples[{index}].quality.{metric} must be null or "
                                "finite and non-negative"
                            )
                    if quality["status"] not in PRIVATE_QUALITY_STATUSES:
                        errors.append(f"samples[{index}].quality.status is invalid")
            summary = sample.get("summary")
            if not isinstance(summary, dict) or set(summary) != PRIVATE_SUMMARY_KEYS:
                errors.append(f"samples[{index}].summary schema mismatch")
            else:
                for metric, statistics in summary.items():
                    if not isinstance(statistics, dict) or set(statistics) != SUMMARY_STAT_KEYS:
                        errors.append(f"samples[{index}].summary.{metric} schema mismatch")
                        continue
                    for statistic, value in statistics.items():
                        if statistic == "n":
                            valid = (
                                isinstance(value, int) and not isinstance(value, bool) and value > 0
                            )
                        else:
                            valid = (
                                isinstance(value, (int, float))
                                and not isinstance(value, bool)
                                and math.isfinite(float(value))
                                and value >= 0
                            )
                        if not valid:
                            errors.append(
                                f"samples[{index}].summary.{metric}.{statistic} "
                                "must be a finite non-negative statistic"
                            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    try:
        payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    errors = validate_public_artifact(payload)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"public artifact check passed: {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
