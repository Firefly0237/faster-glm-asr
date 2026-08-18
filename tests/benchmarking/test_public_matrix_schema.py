from __future__ import annotations

import math
import unittest

from faster_glm_asr.benchmarking import public_matrix_schema as schema


def _valid_summary() -> dict:
    durations = {"short": 5.0, "medium": 15.0, "near-30s": 25.0}
    canonical_by_implementation: dict[str, dict[str, tuple[float, float]]] = {}
    results = []
    for implementation_index, implementation in enumerate(schema.IMPLEMENTATIONS):
        canonical_by_bucket: dict[str, tuple[float, float]] = {}
        buckets = []
        for bucket_index, (bucket, _lower, _upper, _inclusive) in enumerate(schema.BUCKETS):
            generation_p50 = 100.0 + implementation_index * 10.0 + bucket_index
            generation_p95 = generation_p50 * 1.2
            duration_s = durations[bucket]
            canonical_by_bucket[bucket] = (generation_p50, generation_p95)
            buckets.append(
                {
                    "name": bucket,
                    "canonical": {
                        "observations": 30,
                        "generation_p50_ms": generation_p50,
                        "generation_p95_ms": generation_p95,
                        "rtf_p50": generation_p50 / 1000.0 / duration_s,
                        "rtf_p95": generation_p95 / 1000.0 / duration_s,
                    },
                    "request_memory": {
                        "observations": 30,
                        "allocator_peak_delta_p50_mib": 100.0 + bucket_index,
                        "allocator_peak_delta_p95_mib": 120.0 + bucket_index,
                        "process_peak_observed_p50_mib": 1000.0 + bucket_index,
                        "process_peak_observed_p95_mib": 1100.0 + bucket_index,
                        "minimum_poll_samples_per_request": 2,
                        "process_peak_observed_is_lower_bound": True,
                    },
                }
            )
        canonical_by_implementation[implementation] = canonical_by_bucket
        results.append({"implementation": implementation, "duration_buckets": buckets})

    comparisons = []
    for baseline, candidate in schema.COMPARISON_PAIRS:
        buckets = []
        for bucket, _lower, _upper, _inclusive in schema.BUCKETS:
            baseline_p50, baseline_p95 = canonical_by_implementation[baseline][bucket]
            candidate_p50, candidate_p95 = canonical_by_implementation[candidate][bucket]
            buckets.append(
                {
                    "name": bucket,
                    "generation_p50_speedup": baseline_p50 / candidate_p50,
                    "generation_p95_speedup": baseline_p95 / candidate_p95,
                }
            )
        comparisons.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "exact_token_parity": True,
                "speedup_definition": "baseline latency / candidate latency",
                "duration_buckets": buckets,
            }
        )

    evidence = {
        "git_commit": "a" * 40,
        "source_map_sha256": "b" * 64,
        "plan_sha256": "c" * 64,
        "journal_sha256": "d" * 64,
        "aggregate_set_sha256": "e" * 64,
        "dataset_archive_sha256": "f" * 64,
        "exporter_source_sha256": "1" * 64,
        "schema_source_sha256": "2" * 64,
    }
    return {
        "schema_version": schema.PUBLIC_SCHEMA,
        "artifact_scope": "public",
        "evidence": evidence,
        "benchmark": {
            "model": {
                "repository": schema.DEFAULT_MODEL_ID,
                "revision": schema.DEFAULT_MODEL_REVISION,
            },
            "dataset": {
                "name": "LibriSpeech",
                "subset": "test-clean",
                "selection": "one duration-stratified utterance per bucket",
                "utterance_count": 3,
                "duration_buckets": [
                    {
                        "name": name,
                        "minimum_seconds": lower,
                        "maximum_seconds": upper,
                        "maximum_inclusive": inclusive,
                    }
                    for name, lower, upper, inclusive in schema.BUCKETS
                ],
            },
        },
        "hardware": {
            "accelerator_model": "NVIDIA GeForce RTX 3090",
            "nominal_memory_gib": schema.NOMINAL_MEMORY_GIB,
            "compute_capability": "8.6",
        },
        "software": {
            "python": "3.11.9",
            "cuda_runtime": "12.8",
            "cudnn_version": 91002,
            "nvidia_driver_version": "595.71.5",
            "packages": {package: "1.0.0" for package in schema.PACKAGE_KEYS},
        },
        "protocol": {
            "implementations": list(schema.IMPLEMENTATIONS),
            "measurement_profiles": list(schema.PROFILES),
            "outer_passes": 3,
            "warmup_iterations_per_task": 3,
            "measured_repeats_per_pass": 10,
            "measured_repeats_per_bucket": 30,
            "max_new_tokens": 128,
            "storage_activation_dtype": "torch.float32",
            "canonical_latency": {
                "instrumentation": "NVML polling disabled; diagnostic phase timing disabled",
                "synchronization_boundary": (
                    "CUDA synchronize immediately before and after autoregressive generation"
                ),
                "statistics_scope": "one duration bucket across all outer-pass observations",
                "rtf_definition": "generation wall-clock seconds / input audio seconds",
            },
            "request_memory": {
                "instrumentation": "NVML process polling plus PyTorch CUDA allocator counters",
                "poll_interval_ms": 5.0,
                "process_peak_semantics": "polling-observed lower bound",
                "allocator_peak_delta_semantics": (
                    "generation allocated peak minus after-prepare allocated baseline, "
                    "floored at zero"
                ),
                "canonical_latency_eligible": False,
            },
            "diagnostic_phase": {
                "public_metrics_included": False,
                "canonical_latency_eligible": False,
            },
        },
        "results": results,
        "parity": {
            "status": "pass",
            "criterion": "exact generated-token sequence equality",
            "profile_scope": "all 18 implementation/profile aggregates",
            "implementations_checked": 6,
            "duration_buckets_checked": 3,
        },
        "comparisons": comparisons,
    }


def _canonical(summary: dict, bucket_index: int = 0) -> dict:
    return summary["results"][0]["duration_buckets"][bucket_index]["canonical"]


def _set_inferred_durations(
    summary: dict, bucket_index: int, *, p50_s: float, p95_s: float
) -> None:
    canonical = _canonical(summary, bucket_index)
    canonical["rtf_p50"] = canonical["generation_p50_ms"] / 1000.0 / p50_s
    canonical["rtf_p95"] = canonical["generation_p95_ms"] / 1000.0 / p95_s


class PublicMatrixSchemaTest(unittest.TestCase):
    def test_comparison_pairs_are_fixed_within_prepared_request_families(self) -> None:
        self.assertEqual(
            schema.COMPARISON_PAIRS,
            (
                ("hf_no_cache", "hf_cached"),
                ("custom_full_prefix", "custom_greedy_full_prefix"),
                ("custom_greedy_full_prefix", "custom_tuple_cache"),
                ("custom_tuple_cache", "custom_static_cache"),
                ("custom_full_prefix", "custom_static_cache"),
            ),
        )
        self.assertNotIn(("hf_cached", "custom_static_cache"), schema.COMPARISON_PAIRS)

    def test_valid_summary_and_intended_evidence_hashes_pass(self) -> None:
        summary = _valid_summary()
        self.assertEqual(schema.validate_public_summary(summary), [])
        self.assertFalse(
            any(
                "forbidden public key at root.evidence" in error
                for error in schema._privacy_errors(summary)
            )
        )

    def test_evidence_block_is_exact_lowercase_hex_and_sha_exception_is_narrow(self) -> None:
        sha256_git = _valid_summary()
        sha256_git["evidence"]["git_commit"] = "a" * 64
        self.assertEqual(schema.validate_public_summary(sha256_git), [])

        cases = (
            ("git_commit", "A" * 40),
            ("git_commit", "a" * 39),
            ("plan_sha256", "F" * 64),
            ("plan_sha256", "f" * 63),
            ("plan_sha256", "g" * 64),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value[:8]):
                summary = _valid_summary()
                summary["evidence"][key] = value
                errors = schema.validate_public_summary(summary)
                self.assertTrue(any(f"root.evidence.{key}" in error for error in errors))

        extra = _valid_summary()
        extra["evidence"]["manifest_sha256"] = "a" * 64
        errors = schema.validate_public_summary(extra)
        self.assertTrue(any("root.evidence schema mismatch" in error for error in errors))
        self.assertTrue(any("root.evidence.manifest_sha256" in error for error in errors))

        outside = _valid_summary()
        outside["plan_sha256"] = "a" * 64
        errors = schema.validate_public_summary(outside)
        self.assertTrue(any("root schema mismatch" in error for error in errors))
        self.assertTrue(
            any("forbidden public key at root.plan_sha256" in error for error in errors)
        )

    def test_memory_metrics_are_bounded_by_public_hardware_capacity(self) -> None:
        boundary = _valid_summary()
        memory = boundary["results"][0]["duration_buckets"][0]["request_memory"]
        for key in (
            "allocator_peak_delta_p50_mib",
            "allocator_peak_delta_p95_mib",
            "process_peak_observed_p50_mib",
            "process_peak_observed_p95_mib",
        ):
            memory[key] = schema.NOMINAL_MEMORY_MIB
        self.assertEqual(schema.validate_public_summary(boundary), [])

        for key in (
            "allocator_peak_delta_p50_mib",
            "allocator_peak_delta_p95_mib",
            "process_peak_observed_p50_mib",
            "process_peak_observed_p95_mib",
        ):
            for value in (
                math.nextafter(schema.NOMINAL_MEMORY_MIB, math.inf),
                1e308,
            ):
                with self.subTest(key=key, value=value):
                    summary = _valid_summary()
                    memory = summary["results"][0]["duration_buckets"][0]["request_memory"]
                    memory[key] = value
                    errors = schema.validate_public_summary(summary)
                    self.assertTrue(
                        any(key in error and "hardware limit" in error for error in errors)
                    )

        nonfinite = _valid_summary()
        memory = nonfinite["results"][0]["duration_buckets"][0]["request_memory"]
        memory["process_peak_observed_p95_mib"] = math.inf
        self.assertTrue(
            any(
                "process_peak_observed_p95_mib must be finite" in error
                for error in schema.validate_public_summary(nonfinite)
            )
        )

    def test_rtf_percentiles_infer_one_duration_in_the_named_bucket(self) -> None:
        accepted = (
            (0, 9.999999999),
            (1, 10.0),
            (2, 20.0),
            (2, 30.0),
        )
        for bucket_index, duration_s in accepted:
            with self.subTest(bucket_index=bucket_index, duration_s=duration_s):
                summary = _valid_summary()
                _set_inferred_durations(summary, bucket_index, p50_s=duration_s, p95_s=duration_s)
                self.assertEqual(schema.validate_public_summary(summary), [])

        rejected = (
            (0, 10.0),
            (1, 9.999999999),
            (1, 20.0),
            (2, 30.000000001),
        )
        for bucket_index, duration_s in rejected:
            with self.subTest(bucket_index=bucket_index, duration_s=duration_s):
                summary = _valid_summary()
                _set_inferred_durations(summary, bucket_index, p50_s=duration_s, p95_s=duration_s)
                errors = schema.validate_public_summary(summary)
                self.assertTrue(any("inferred duration" in error for error in errors))

        inconsistent = _valid_summary()
        _set_inferred_durations(inconsistent, 0, p50_s=5.0, p95_s=6.0)
        self.assertTrue(
            any(
                "p50/p95 inferred durations are inconsistent" in error
                for error in schema.validate_public_summary(inconsistent)
            )
        )

    def test_version_fields_use_distinct_public_formats_and_reject_ipv4(self) -> None:
        two_segment_driver = _valid_summary()
        two_segment_driver["software"]["nvidia_driver_version"] = "572.83"
        self.assertEqual(schema.validate_public_summary(two_segment_driver), [])

        invalid = {
            "python": ("3.11", "3.11.9.1", "3.11.9rc1", "192.168.0.1"),
            "cuda_runtime": ("12", "12.8.0", "12.8+cu128", "192.168.0.1"),
            "nvidia_driver_version": (
                "595",
                "595.71.5.1",
                "595.71-private",
                "192.168.0.1",
            ),
        }
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    summary = _valid_summary()
                    summary["software"][key] = value
                    errors = schema.validate_public_summary(summary)
                    self.assertTrue(any(f"root.software.{key}" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
