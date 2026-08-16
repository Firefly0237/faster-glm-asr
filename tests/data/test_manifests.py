from __future__ import annotations

import copy
import csv
import hashlib
import json
import shutil
import tempfile
import unittest
import wave
from itertools import pairwise
from pathlib import Path

import numpy as np

from faster_glm_asr.benchmarking import comparator as compare_asr_latency_runs
from faster_glm_asr.benchmarking import provenance
from faster_glm_asr.benchmarking import public_artifact as check_public_artifact
from faster_glm_asr.benchmarking import runner as benchmark_asr_inference
from faster_glm_asr.data import audio_manifest as build_audio_manifest
from faster_glm_asr.data import long_audio as build_long_audio_manifest


def _source_map() -> dict:
    files = {
        "src/faster_glm_asr/benchmarking/runner.py": {
            "size_bytes": 123,
            "sha256": "d" * 64,
        }
    }
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "faster-glm-asr-source-map-v1",
        "files": files,
        "aggregate_sha256": aggregate,
    }


def _model_snapshot_binding(*, repo_id: str, revision: str) -> dict:
    return {
        "schema_version": provenance.MODEL_BINDING_SCHEMA,
        "repo_id": repo_id,
        "revision": revision,
        "file_count": provenance.MODEL_SNAPSHOT_FILE_COUNT,
        "total_size_bytes": 1_024,
        "manifest_sha256": "1" * 64,
        "files_aggregate_sha256": "2" * 64,
        "contract_sha256": "3" * 64,
    }


def _readiness_binding() -> dict:
    return {
        "schema_version": provenance.READINESS_BINDING_SCHEMA,
        "readiness_schema_version": provenance.READINESS_REPORT_SCHEMA,
        "report_sha256": "4" * 64,
        "phase": "full",
        "status": "pass",
        "training_ready": True,
        "day_rental_eligible": True,
        "contract_sha256": "3" * 64,
        "tool_sha256": "5" * 64,
        "gpu_count": 4,
        "gpu_models": ["NVIDIA GeForce RTX 3090"],
        "gpu_identity_sha256": "6" * 64,
        "topology_evidence_sha256": "7" * 64,
        "clock_power_evidence_sha256": "8" * 64,
        "hardware_evidence_sha256": "9" * 64,
    }


class AudioManifestTest(unittest.TestCase):
    def test_fixed_segment_planner_has_complete_half_open_coverage(self) -> None:
        segments = build_long_audio_manifest.plan_fixed_segments(35, 16, 4)
        self.assertEqual(segments, [(0, 16), (12, 28), (24, 35)])
        self.assertEqual(segments[0][0], 0)
        self.assertEqual(segments[-1][1], 35)
        for left, right in pairwise(segments):
            self.assertLessEqual(right[0], left[1])
            self.assertGreater(right[0], left[0])
        with self.assertRaisesRegex(ValueError, "0 <= overlap < chunk"):
            build_long_audio_manifest.plan_fixed_segments(35, 16, 16)

    def test_materializes_deterministic_long_audio_segment_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "recording.wav"
            reference = root / "recording.vtt"
            inventory = root / "inventory.csv"
            source_manifest = root / "manifests" / "sources.jsonl"
            bundle_one = root / "bundles" / "segments-v1"
            bundle_two = root / "bundles" / "segments-v2"

            pcm = np.arange(35_200, dtype=np.int32)
            pcm = ((pcm % 2048) - 1024).astype("<i2")
            with wave.open(str(audio), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16_000)
                stream.writeframes(pcm.tobytes())
            reference.write_text(
                "WEBVTT\n\n"
                "00:00:00.000 --> 00:00:00.700\n"
                "first cue\n\n"
                "00:00:00.750 --> 00:00:01.600\n"
                "second cue\n\n"
                "00:00:01.600 --> 00:00:02.200\n"
                "third cue\n",
                encoding="utf-8",
            )
            with inventory.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=list(build_audio_manifest.KNOWN_COLUMNS),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "recording-0001",
                        "audio_path": audio.name,
                        "reference_path": reference.name,
                        "language": "en",
                        "collection": "mls",
                        "speaker_count": "1",
                        "split": "domain-test",
                        "split_group_id": "session-0001",
                        "shareable": "false",
                        "authorization_id": "AUTH-001",
                        "reference_quality": "human-corrected",
                        "notes": "fixture",
                    }
                )

            self.assertEqual(build_audio_manifest.build(inventory, source_manifest), 0)
            for bundle in (bundle_one, bundle_two):
                self.assertEqual(
                    build_long_audio_manifest.build(
                        source_manifest,
                        bundle,
                        chunk_seconds="1",
                        overlap_seconds="0.25",
                    ),
                    0,
                )

            first_manifest = bundle_one / "segments.jsonl"
            second_manifest = bundle_two / "segments.jsonl"
            self.assertEqual(first_manifest.read_bytes(), second_manifest.read_bytes())
            self.assertEqual(
                (bundle_one / "segmentation-plan.json").read_bytes(),
                (bundle_two / "segmentation-plan.json").read_bytes(),
            )
            plan = json.loads((bundle_one / "segmentation-plan.json").read_text(encoding="utf-8"))
            bundle_lock = json.loads((bundle_one / "bundle-lock.json").read_text(encoding="utf-8"))
            self.assertEqual(
                bundle_lock["segmentation_plan_sha256"],
                build_long_audio_manifest._sha256_file(bundle_one / "segmentation-plan.json"),
            )
            self.assertEqual(
                bundle_lock["segment_manifest_sha256"],
                build_long_audio_manifest._sha256_file(first_manifest),
            )
            self.assertEqual(
                plan["sources"][0]["canonicalizer"]["target_sample_rate_hz"],
                16_000,
            )
            self.assertIn(
                plan["sources"][0]["canonicalizer"]["decoder"],
                {"soundfile", "python-wave"},
            )
            records = [
                json.loads(line) for line in first_manifest.read_text(encoding="utf-8").splitlines()
            ]
            bad_pcm_record = dict(records[0])
            bad_pcm_record["segment_pcm_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "segment_pcm_sha256"):
                benchmark_asr_inference._validate_materialized_segment_audio(
                    bundle_one / records[0]["audio_path"], bad_pcm_record, 1
                )
            with self.assertRaisesRegex(FileNotFoundError, "segmentation-plan"):
                benchmark_asr_inference._validate_segment_bundle_files(
                    root / "orphan" / "segments.jsonl", records
                )
            self.assertEqual(
                [(item["start_sample"], item["end_sample"]) for item in records],
                [(0, 16_000), (12_000, 28_000), (24_000, 35_200)],
            )
            self.assertEqual([item["segment_index"] for item in records], [0, 1, 2])
            self.assertTrue(
                all(
                    item["manifest_parser_version"] == "asr-segment-v0.1"
                    and item["shareable"] is False
                    and "reference_text" not in item
                    and len(item["canonical_source_audio_sha256"]) == 64
                    for item in records
                )
            )
            self.assertEqual(
                [item["owned_cue_ids"] for item in records],
                [["cue-000000"], ["cue-000001"], ["cue-000002"]],
            )
            with wave.open(str(bundle_one / records[-1]["audio_path"]), "rb") as stream:
                self.assertEqual(stream.getnframes(), 11_200)
                self.assertEqual(stream.getframerate(), 16_000)

            samples, manifest_hash = benchmark_asr_inference._load_manifest(first_manifest)
            self.assertEqual(len(samples), 3)
            self.assertEqual(len(manifest_hash), 64)
            self.assertTrue(all(sample.reference_text is None for sample in samples))
            private_references = [
                json.loads(line)
                for line in (bundle_one / "source-references.private.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(private_references), 1)
            self.assertEqual(len(private_references[0]["cues"]), 3)

            tampered_bundle = root / "bundles" / "segments-self-rehashed-tamper"
            shutil.copytree(bundle_one, tampered_bundle)
            tampered_plan_path = tampered_bundle / "segmentation-plan.json"
            tampered_plan = json.loads(tampered_plan_path.read_text(encoding="utf-8"))
            tampered_plan["chunk_samples"] = 15_000
            tampered_plan_path.write_bytes(
                build_long_audio_manifest._canonical_json_bytes(tampered_plan)
            )
            tampered_plan_hash = build_long_audio_manifest._sha256_file(tampered_plan_path)
            tampered_records = copy.deepcopy(records)
            for record in tampered_records:
                record["segmentation_plan_sha256"] = tampered_plan_hash
            tampered_manifest = tampered_bundle / "segments.jsonl"
            build_long_audio_manifest._write_jsonl(tampered_manifest, tampered_records)
            tampered_lock_path = tampered_bundle / "bundle-lock.json"
            tampered_lock = json.loads(tampered_lock_path.read_text(encoding="utf-8"))
            tampered_lock["segmentation_plan_sha256"] = tampered_plan_hash
            tampered_lock["segment_manifest_sha256"] = build_long_audio_manifest._sha256_file(
                tampered_manifest
            )
            tampered_lock_path.write_bytes(
                build_long_audio_manifest._canonical_json_bytes(tampered_lock)
            )
            with self.assertRaisesRegex(ValueError, "locked fixed plan"):
                benchmark_asr_inference._load_manifest(tampered_manifest)

            substituted_bundle = root / "bundles" / "segments-substituted-pcm"
            shutil.copytree(bundle_one, substituted_bundle)
            substituted_records = copy.deepcopy(records)
            substituted_segment_path = substituted_bundle / substituted_records[0]["audio_path"]
            with wave.open(str(substituted_segment_path), "rb") as stream:
                substituted_pcm = bytearray(stream.readframes(stream.getnframes()))
            substituted_pcm[0] ^= 1
            with wave.open(str(substituted_segment_path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16_000)
                stream.writeframes(bytes(substituted_pcm))
            substituted_records[0]["segment_pcm_sha256"] = hashlib.sha256(
                bytes(substituted_pcm)
            ).hexdigest()
            substituted_records[0]["audio_sha256"] = build_long_audio_manifest._sha256_file(
                substituted_segment_path
            )
            substituted_manifest = substituted_bundle / "segments.jsonl"
            build_long_audio_manifest._write_jsonl(substituted_manifest, substituted_records)
            substituted_lock_path = substituted_bundle / "bundle-lock.json"
            substituted_lock = json.loads(substituted_lock_path.read_text(encoding="utf-8"))
            substituted_lock["segment_manifest_sha256"] = build_long_audio_manifest._sha256_file(
                substituted_manifest
            )
            substituted_lock_path.write_bytes(
                build_long_audio_manifest._canonical_json_bytes(substituted_lock)
            )
            with self.assertRaisesRegex(ValueError, "canonical source slice"):
                benchmark_asr_inference._load_manifest(substituted_manifest)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                build_long_audio_manifest.build(source_manifest, bundle_one)
            reference.write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:00.700\nchanged cue\n",
                encoding="utf-8",
            )
            changed_bundle = root / "bundles" / "segments-after-reference-change"
            with self.assertRaisesRegex(ValueError, "reference SHA256 mismatch"):
                build_long_audio_manifest.build(source_manifest, changed_bundle)
            self.assertFalse(changed_bundle.exists())

    def test_custom_runner_routes_custom_tuple_cache_as_controlled_greedy_path(self) -> None:
        calls = []

        class FakeModel:
            def generate_tuple_cache(self, features, **kwargs):
                calls.append((features, kwargs))
                return "tuple-output"

        class FakePhaseRecorder:
            def callback(self, label, boundary):
                return (label, boundary)

        runner = benchmark_asr_inference.CustomRunner.__new__(benchmark_asr_inference.CustomRunner)
        runner.implementation = "custom_tuple_cache"
        runner.model = FakeModel()
        inputs = {
            "input_features": "features",
            "input_ids": "ids",
            "input_features_mask": "feature-mask",
            "attention_mask": "attention-mask",
        }
        phase_recorder = FakePhaseRecorder()
        output = runner.generate(inputs, max_new_tokens=7, phase_recorder=phase_recorder)
        self.assertEqual(output, "tuple-output")
        self.assertEqual(len(calls), 1)
        features, kwargs = calls[0]
        self.assertEqual(features, "features")
        self.assertFalse(kwargs["do_sample"])
        self.assertEqual(kwargs["max_new_tokens"], 7)
        self.assertIs(kwargs["phase_callback"].__self__, phase_recorder)
        self.assertIs(kwargs["phase_callback"].__func__, FakePhaseRecorder.callback)
        configuration = runner.configuration()
        self.assertEqual(configuration["cache_mode"], "tuple-torch-cat")
        self.assertTrue(configuration["greedy_fast_path"])
        self.assertEqual(configuration["token_output_mode"], "preallocated")

    def test_builds_manifest_consumed_by_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            reference = root / "sample.vtt"
            inventory = root / "inventory.csv"
            output = root / "manifests" / "domain.jsonl"

            with wave.open(str(audio), "wb") as stream:
                stream.setnchannels(2)
                stream.setsampwidth(2)
                stream.setframerate(8000)
                stream.writeframes(b"\x00\x00" * 2 * 8000)

            reference.write_text(
                "WEBVTT\n\n"
                "STYLE\n"
                "::cue { color: lime; }\n\n"
                "cue-one\n"
                "00:00:00.000 --> 00:00:00.500\n"
                "<v Lecturer>Hello world</v>\n\n"
                "00:00:00.500 --> 00:00:01.000\n"
                "<v Lecturer>Hello world</v>\n",
                encoding="utf-8",
            )
            with inventory.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=list(build_audio_manifest.KNOWN_COLUMNS),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "lecture-001",
                        "audio_path": audio.name,
                        "reference_path": reference.name,
                        "language": "en",
                        "collection": "mls",
                        "speaker_count": "1",
                        "split": "domain-test",
                        "shareable": "false",
                        "notes": "fixture",
                    }
                )

            self.assertEqual(build_audio_manifest.build(inventory, output), 0)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["sample_id"], "lecture-001")
            self.assertEqual(record["sample_rate_hz"], 8000)
            self.assertEqual(record["channels"], 2)
            self.assertEqual(record["duration_s"], 1.0)
            self.assertEqual(record["reference_text"], "Hello world Hello world")
            self.assertEqual(record["manifest_parser_version"], "asr-inventory-v0.2")
            self.assertTrue(record["direct_model_eligible"])
            self.assertEqual(len(record["audio_sha256"]), 64)
            self.assertFalse(record["shareable"])

            samples, manifest_hash = benchmark_asr_inference._load_manifest(output)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].sample_id, "lecture-001")
            self.assertEqual(samples[0].reference_text, "Hello world Hello world")
            self.assertFalse(samples[0].shareable)
            self.assertEqual(len(manifest_hash), 64)

    def test_benchmark_manifest_fails_closed_without_admission_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "unsafe.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "too-weak",
                        "audio_path": "missing.wav",
                        "audio_sha256": "a" * 64,
                        "duration_s": 1.0,
                        "language": "en",
                        "shareable": False,
                        "manifest_parser_version": "asr-inventory-v0.2",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "direct_model_eligible"):
                benchmark_asr_inference._load_manifest(manifest)
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "unapproved-public",
                        "audio_path": "missing.wav",
                        "audio_sha256": "a" * 64,
                        "duration_s": 1.0,
                        "language": "en",
                        "shareable": True,
                        "direct_model_eligible": True,
                        "manifest_parser_version": "asr-inventory-v0.2",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "authorization_id"):
                benchmark_asr_inference._load_manifest(manifest)

    def test_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            inventory = root / "inventory.csv"
            output = root / "domain.jsonl"
            with wave.open(str(audio), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(b"\x00\x00" * 160)
            inventory.write_text(
                "id,audio_path,language,shareable\n"
                "same,sample.wav,en,false\n"
                "same,sample.wav,en,false\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                build_audio_manifest.build(inventory, output)

    def test_atomic_publish_refuses_existing_output_and_cleans_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writers = (
                (
                    "domain.jsonl",
                    lambda path: build_audio_manifest._write_jsonl_atomically(
                        path, [{"sample_id": "replacement"}], overwrite=False
                    ),
                ),
                (
                    "result.json",
                    lambda path: benchmark_asr_inference._write_json_atomically(
                        path, {"replacement": True}, overwrite=False
                    ),
                ),
                (
                    "comparison.json",
                    lambda path: compare_asr_latency_runs._write_json_atomically(
                        path, {"replacement": True}, overwrite=False
                    ),
                ),
            )
            for filename, writer in writers:
                output = root / filename
                output.write_text("original\n", encoding="utf-8")

                with self.assertRaises(FileExistsError):
                    writer(output)

                self.assertEqual(output.read_text(encoding="utf-8"), "original\n")
                self.assertEqual(list(root.glob(f".{filename}.*.tmp")), [])

    def test_public_result_redacts_private_text_and_paths(self) -> None:
        summary = {
            metric: {
                "n": 2,
                "mean": 2.0,
                "std": 0.5,
                "min": 1.0,
                "p50": 2.0,
                "p95": 2.9,
                "max": 3.0,
            }
            for metric in (
                "generation_ms",
                "processor_plus_generation_ms",
                "request_no_file_io_ms",
                "rtf_generation",
            )
        }
        result = {
            "schema_version": "0.3",
            "created_at_utc": "2026-08-15T00:00:00+00:00",
            "run": {
                "formal_run": True,
                "comparability": {"status": "formal-comparable", "reasons": []},
                "warmup_gate": {
                    "status": "pass",
                    "minimum_iterations": 1,
                    "observed_iterations": 3,
                    "custom_lazy_h2d_applicable": True,
                    "custom_lazy_h2d_covered": True,
                },
                "implementation": "custom_full_prefix",
                "model_name": "zai-org/GLM-ASR-Nano-2512",
                "model_revision": "61ba4e0b3309b6656edea3e93e419f7bd5c61957",
                "manifest_sha256": "a" * 64,
                "manifest_path": "C:/private/collection/domain.jsonl",
                "warmup": 3,
                "repeats": 2,
                "max_new_tokens": 8,
                "runner_construction_envelope_ms": 10.0,
                "runner_construction_scope": "host construction envelope",
                "artifact_scope": "private",
                "measurement_profile": "canonical_latency",
                "metric_eligibility": {
                    "canonical_latency": True,
                    "request_memory_observed_peak": False,
                    "diagnostic_cuda_phases": False,
                    "cold_start": False,
                    "model_load": False,
                },
                "nvml_sample_ms": 0.0,
                "phase_timing": False,
                "phase_timing_warning": None,
                "runner_configuration": {
                    "system": "custom-torch-triton-hybrid",
                    "model_source": "validated-local-snapshot",
                    "network_fallback": False,
                    "storage_activation_dtype": "torch.float32",
                    "linear_backend": "cublas",
                    "mlp_fused": False,
                    "kernel_dispatch_policy_version": "hybrid-auto-v1",
                    "component_dispatch_policy": (
                        "supported-norm-rope-embedding-conv-auto-triton-else-torch"
                    ),
                    "attention_dispatch_policy": (
                        "dense-triton-if-padded-and-head-dim-lte-256-else-torch-dense"
                    ),
                    "gqa_kv_expansion": "explicit-expand",
                    "long_form_attention_expectation": ("torch-dense-likely-for-30s-input"),
                    "backend_trace_status": (
                        "policy-inferred-diagnostic-trace-required-for-observed-backend"
                    ),
                    "performance_attribution": ("cache-strategy-only-no-triton-kernel-attribution"),
                    "cache_mode": "none",
                    "greedy_fast_path": False,
                    "token_output_mode": "torch-cat",
                },
            },
            "git": {
                "commit": "c" * 40,
                "branch": "private-user-branch",
                "dirty": False,
            },
            "sources": _source_map(),
            "model_snapshot": _model_snapshot_binding(
                repo_id="zai-org/GLM-ASR-Nano-2512",
                revision="61ba4e0b3309b6656edea3e93e419f7bd5c61957",
            ),
            "readiness": _readiness_binding(),
            "environment": {
                "python": "3.11.9",
                "platform": "Linux-test",
                "torch": "2.10.0",
                "torch_cuda": "12.8",
                "cudnn": 99999,
                "device_index": 0,
                "device_name": "synthetic-gpu",
                "device_uuid": "GPU-private-stable-id",
                "cuda_visible_devices": "GPU-private-stable-id",
                "compute_capability": [9, 0],
                "total_memory_bytes": 1,
                "allow_tf32_matmul": True,
                "allow_tf32_cudnn": True,
                "nvidia_driver_versions": ["test-driver"],
                "package_versions": {
                    package: "test-version"
                    for package in check_public_artifact.PACKAGE_VERSION_KEYS
                },
                "environment_lock_sha256": "e" * 64,
            },
            "aggregate": {
                "generation_ms": copy.deepcopy(summary["generation_ms"]),
                "processor_plus_generation_ms": copy.deepcopy(
                    summary["processor_plus_generation_ms"]
                ),
                "request_no_file_io_ms": copy.deepcopy(summary["request_no_file_io_ms"]),
                "quality": {
                    "evaluated_samples": 1,
                    "word": {
                        "errors": 1,
                        "substitutions": 1,
                        "deletions": 0,
                        "insertions": 0,
                        "reference_units": 2,
                    },
                    "char": {
                        "errors": 1,
                        "substitutions": 1,
                        "deletions": 0,
                        "insertions": 0,
                        "reference_units": 4,
                    },
                    "wer": 0.5,
                    "cer": 0.25,
                    "status": "evaluated",
                    "normalization": "test-normalization",
                },
            },
            "samples": [
                {
                    "sample_id": "sensitive-collection-name",
                    "language": "student Alice Smith /scratch/private",
                    "duration_s": 3.0,
                    "audio_sha256": "c" * 64,
                    "hypothesis": "private transcript",
                    "hypothesis_sha256": "d" * 64,
                    "generated_token_ids": [1, 2, 3],
                    "generated_token_ids_sha256": "e" * 64,
                    "generated_tokens": 3,
                    "quality": {
                        "reference_normalized": "private reference",
                        "hypothesis_normalized": "private transcript",
                        "wer": 0.5,
                        "cer": 0.25,
                        "status": "evaluated",
                    },
                    "summary": copy.deepcopy(summary),
                    "private_note": "Alice /scratch/private",
                    "metadata": {
                        "shareable": False,
                        "manifest_parser_version": "asr-segment-v0.1",
                        "reference_path": "C:/private/collection/reference.vtt",
                        "reference_sha256": "b" * 64,
                        "notes": "student name",
                        "split": "/scratch/private/collection-name",
                    },
                }
            ],
        }
        public = benchmark_asr_inference._public_artifact(result)
        self.assertNotIn("manifest_path", public["run"])
        self.assertNotIn("manifest_sha256", public["run"])
        self.assertNotIn("branch", public["git"])
        self.assertNotIn("device_uuid", public["environment"])
        self.assertNotIn("cuda_visible_devices", public["environment"])
        self.assertEqual(public["run"]["artifact_scope"], "public")
        runner_configuration = public["run"]["runner_configuration"]
        self.assertEqual(runner_configuration["system"], "custom-torch-triton-hybrid")
        self.assertEqual(
            runner_configuration["long_form_attention_expectation"],
            "torch-dense-likely-for-30s-input",
        )
        self.assertEqual(
            runner_configuration["backend_trace_status"],
            "policy-inferred-diagnostic-trace-required-for-observed-backend",
        )
        self.assertEqual(
            runner_configuration["performance_attribution"],
            "cache-strategy-only-no-triton-kernel-attribution",
        )
        sample = public["samples"][0]
        self.assertTrue(sample["sample_id"].startswith("private-"))
        self.assertEqual(sample["language"], "redacted")
        self.assertNotIn("hypothesis", sample)
        self.assertNotIn("reference_normalized", sample["quality"])
        self.assertNotIn("hypothesis_normalized", sample["quality"])
        self.assertNotIn("reference_path", sample["metadata"])
        self.assertNotIn("notes", sample["metadata"])
        self.assertNotIn("split", sample["metadata"])
        self.assertNotIn("reference_sha256", sample["metadata"])
        self.assertNotIn("audio_sha256", sample)
        self.assertNotIn("hypothesis_sha256", sample)
        self.assertNotIn("generated_token_ids", sample)
        self.assertNotIn("private_note", sample)
        self.assertEqual(sample["metadata"]["manifest_parser_version"], "asr-segment-v0.1")
        self.assertEqual(sample["metadata"]["duration_bucket"], "materialized-segment")
        self.assertEqual(check_public_artifact.validate_public_artifact(public), [])

        leaked = dict(public)
        leaked["samples"] = [dict(public["samples"][0])]
        leaked["samples"][0]["hypothesis"] = "private transcript"
        self.assertTrue(check_public_artifact.validate_public_artifact(leaked))

        fail_open_attempt = {
            "run": {"artifact_scope": "public"},
            "samples": [
                {
                    "sample_id": "guessable",
                    "hypothesis": "private transcript",
                    "metadata": {},
                }
            ],
        }
        self.assertTrue(check_public_artifact.validate_public_artifact(fail_open_attempt))

        leaked_private_metadata = copy.deepcopy(public)
        leaked_private_metadata["samples"][0]["metadata"]["split"] = (
            "/scratch/private/collection-name"
        )
        self.assertTrue(check_public_artifact.validate_public_artifact(leaked_private_metadata))

        leaked_unknown_field = copy.deepcopy(public)
        leaked_unknown_field["samples"][0]["private_note"] = "Alice /scratch/private"
        self.assertTrue(check_public_artifact.validate_public_artifact(leaked_unknown_field))

        leaked_device_identity = copy.deepcopy(public)
        leaked_device_identity["environment"]["device_uuid"] = "GPU-private"
        self.assertTrue(check_public_artifact.validate_public_artifact(leaked_device_identity))

        shareable_result = copy.deepcopy(result)
        shareable_sample = shareable_result["samples"][0]
        shareable_sample["sample_id"] = "open-sample-0001"
        shareable_sample["language"] = "en"
        shareable_sample["metadata"].update(
            {
                "shareable": True,
                "authorization_id": "AUTH-OPEN-001",
                "duration_bucket": "direct-model",
                "direct_model_eligible": True,
                "manifest_parser_version": "asr-inventory-v0.2",
                "split": "domain-test",
            }
        )
        shareable_public = benchmark_asr_inference._public_artifact(shareable_result)
        self.assertEqual(
            set(shareable_public["samples"][0]),
            check_public_artifact.SHAREABLE_SAMPLE_KEYS,
        )
        self.assertEqual(check_public_artifact.validate_public_artifact(shareable_public), [])
        shareable_leak = copy.deepcopy(shareable_public)
        shareable_leak["samples"][0]["student_email"] = "alice@example.edu"
        self.assertTrue(check_public_artifact.validate_public_artifact(shareable_leak))

        root_leak = copy.deepcopy(public)
        root_leak["future_private_note"] = "Alice Smith"
        self.assertTrue(check_public_artifact.validate_public_artifact(root_leak))

        nested_leak = copy.deepcopy(public)
        nested_leak["run"]["future_private_note"] = "Alice Smith"
        self.assertTrue(check_public_artifact.validate_public_artifact(nested_leak))

        empty_samples = copy.deepcopy(public)
        empty_samples["samples"] = []
        self.assertTrue(check_public_artifact.validate_public_artifact(empty_samples))

    def test_quality_without_references_is_not_reported_as_zero_error(self) -> None:
        aggregate = benchmark_asr_inference._aggregate_quality([])
        self.assertEqual(aggregate["evaluated_samples"], 0)
        self.assertEqual(aggregate["status"], "not_evaluated")
        self.assertIsNone(aggregate["wer"])
        self.assertIsNone(aggregate["cer"])

    def test_edit_counts_preserve_substitution_deletion_insertion(self) -> None:
        counts = benchmark_asr_inference._edit_counts(["a", "b", "c"], ["a", "x", "c", "d"])
        self.assertEqual(counts["errors"], 2)
        self.assertEqual(counts["substitutions"], 1)
        self.assertEqual(counts["deletions"], 0)
        self.assertEqual(counts["insertions"], 1)
        self.assertEqual(counts["reference_units"], 3)

    def test_empty_normalized_reference_has_null_metrics(self) -> None:
        quality = benchmark_asr_inference._quality("!!!", "text")
        self.assertIsNotNone(quality)
        self.assertIsNone(quality["wer"])
        self.assertIsNone(quality["cer"])
        self.assertEqual(quality["status"], "not_evaluated_empty_after_normalization")

    def test_context_budget_fails_before_generation(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeding fixed model limit"):
            benchmark_asr_inference._validate_context_budget(
                {"input_ids": np.zeros((1, 8_100), dtype=np.int64)},
                max_new_tokens=128,
                sample_id="too-long",
            )

    def test_strict_latency_pair_comparison(self) -> None:
        token_ids = [7, 8]
        token_hash = hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        hypothesis = "test transcript"
        hypothesis_hash = hashlib.sha256(hypothesis.encode("utf-8")).hexdigest()
        prepared = {
            "tensors": {
                "input_ids": {
                    "shape": [1, 4],
                    "dtype": "torch.int64",
                    "device": "cuda:0",
                    "content_sha256": "1" * 64,
                },
                "input_features": {
                    "shape": [1, 128, 3000],
                    "dtype": "torch.float32",
                    "device": "cuda:0",
                    "content_sha256": "2" * 64,
                },
            },
            "request_batch": 1,
            "prompt_tokens": 4,
            "audio_placeholder_tokens": 1,
            "audio_windows": 1,
            "valid_feature_frames": None,
        }
        baseline_generation = [10.0, 30.0]
        baseline_summary = benchmark_asr_inference._summary(baseline_generation)
        baseline = {
            "schema_version": "0.3",
            "run": {
                "formal_run": True,
                "comparability": {"status": "formal-comparable", "reasons": []},
                "warmup_gate": {
                    "status": "pass",
                    "minimum_iterations": 1,
                    "observed_iterations": 3,
                    "custom_lazy_h2d_applicable": True,
                    "custom_lazy_h2d_covered": True,
                },
                "implementation": "custom_full_prefix",
                "artifact_scope": "private",
                "measurement_profile": "canonical_latency",
                "metric_eligibility": copy.deepcopy(
                    compare_asr_latency_runs.CANONICAL_METRIC_ELIGIBILITY
                ),
                "phase_timing": False,
                "nvml_sample_ms": 0.0,
                "model_name": "model",
                "model_revision": "a" * 40,
                "manifest_sha256": "a" * 64,
                "warmup": 3,
                "repeats": 2,
                "max_new_tokens": 8,
                "runner_configuration": copy.deepcopy(
                    compare_asr_latency_runs.EXPECTED_RUNNER_CONFIGURATIONS["custom_full_prefix"]
                ),
            },
            "git": {"commit": "c" * 40, "branch": "test", "dirty": False},
            "sources": _source_map(),
            "model_snapshot": _model_snapshot_binding(repo_id="model", revision="a" * 40),
            "readiness": _readiness_binding(),
            "environment": {
                "python": "3.11.9",
                "platform": "Linux-test",
                "torch": "2.10",
                "torch_cuda": "12.8",
                "cudnn": 99999,
                "device_index": 0,
                "device_uuid": "GPU-test",
                "device_name": "synthetic-gpu",
                "cuda_visible_devices": None,
                "compute_capability": [9, 0],
                "total_memory_bytes": 1,
                "allow_tf32_matmul": True,
                "allow_tf32_cudnn": True,
                "nvidia_driver_versions": ["test-driver"],
                "package_versions": {
                    package: "test-version"
                    for package in compare_asr_latency_runs.REQUIRED_PACKAGE_VERSIONS
                },
                "environment_lock_sha256": "e" * 64,
            },
            "aggregate": {"generation_ms": copy.deepcopy(baseline_summary)},
            "samples": [
                {
                    "sample_id": "sample",
                    "audio_sha256": "b" * 64,
                    "language": "en",
                    "duration_s": 3.0,
                    "generated_tokens": 2,
                    "generated_token_ids": token_ids,
                    "generated_token_ids_sha256": token_hash,
                    "hypothesis": hypothesis,
                    "hypothesis_sha256": hypothesis_hash,
                    "prepared_request": prepared,
                    "quality": {"wer": 0.0},
                    "summary": {"generation_ms": copy.deepcopy(baseline_summary)},
                    "observations": [
                        {
                            "repeat": 0,
                            "generation_ms": baseline_generation[0],
                            "generated_tokens": 2,
                            "generated_token_ids_sha256": token_hash,
                            "hypothesis_sha256": hypothesis_hash,
                            "memory_bytes": {"generation_allocated_peak_delta": 100},
                        },
                        {
                            "repeat": 1,
                            "generation_ms": baseline_generation[1],
                            "generated_tokens": 2,
                            "generated_token_ids_sha256": token_hash,
                            "hypothesis_sha256": hypothesis_hash,
                            "memory_bytes": {"generation_allocated_peak_delta": 120},
                        },
                    ],
                }
            ],
        }
        candidate = copy.deepcopy(baseline)
        candidate["run"]["implementation"] = "custom_static_cache"
        candidate["run"]["runner_configuration"] = copy.deepcopy(
            compare_asr_latency_runs.EXPECTED_RUNNER_CONFIGURATIONS["custom_static_cache"]
        )
        candidate_generation = [5.0, 15.0]
        candidate_summary = benchmark_asr_inference._summary(candidate_generation)
        candidate["aggregate"]["generation_ms"] = copy.deepcopy(candidate_summary)
        candidate["samples"][0]["summary"]["generation_ms"] = copy.deepcopy(candidate_summary)
        for observation, generation_ms in zip(
            candidate["samples"][0]["observations"],
            candidate_generation,
            strict=True,
        ):
            observation["generation_ms"] = generation_ms
        comparison = compare_asr_latency_runs.compare(baseline, candidate)
        self.assertEqual(comparison["aggregate"]["p50_speedup"], 2.0)
        self.assertEqual(comparison["samples"][0]["p95_speedup"], 2.0)
        self.assertEqual(comparison["environment"]["environment_lock_sha256"], "e" * 64)
        self.assertEqual(
            comparison["baseline_runner_configuration"],
            compare_asr_latency_runs.EXPECTED_RUNNER_CONFIGURATIONS["custom_full_prefix"],
        )
        self.assertEqual(
            comparison["candidate_runner_configuration"],
            compare_asr_latency_runs.EXPECTED_RUNNER_CONFIGURATIONS["custom_static_cache"],
        )

        candidate["samples"][0]["generated_token_ids"] = [7, 9]
        changed_token_hash = hashlib.sha256(b"[7,9]").hexdigest()
        candidate["samples"][0]["generated_token_ids_sha256"] = changed_token_hash
        for observation in candidate["samples"][0]["observations"]:
            observation["generated_token_ids_sha256"] = changed_token_hash
        with self.assertRaisesRegex(ValueError, "token IDs differ"):
            compare_asr_latency_runs.compare(baseline, candidate)

        candidate = copy.deepcopy(baseline)
        candidate["run"]["measurement_profile"] = "request_memory"
        with self.assertRaisesRegex(ValueError, "canonical_latency"):
            compare_asr_latency_runs.compare(baseline, candidate)

        candidate = copy.deepcopy(baseline)
        candidate["environment"]["nvidia_driver_versions"] = ["other-driver"]
        with self.assertRaisesRegex(ValueError, "environment records differ"):
            compare_asr_latency_runs.compare(baseline, candidate)

        invalid_configuration_left = copy.deepcopy(baseline)
        invalid_configuration_right = copy.deepcopy(baseline)
        invalid_configuration_left["run"]["runner_configuration"] = {"garbage": 1}
        invalid_configuration_right["run"]["runner_configuration"] = {"garbage": 1}
        with self.assertRaisesRegex(ValueError, "runner_configuration"):
            compare_asr_latency_runs.compare(
                invalid_configuration_left, invalid_configuration_right
            )

        missing_package_left = copy.deepcopy(baseline)
        missing_package_right = copy.deepcopy(baseline)
        missing_package_left["environment"]["package_versions"]["triton"] = None
        missing_package_right["environment"]["package_versions"]["triton"] = None
        with self.assertRaisesRegex(ValueError, "package versions"):
            compare_asr_latency_runs.compare(missing_package_left, missing_package_right)

        missing_evidence_left = copy.deepcopy(baseline)
        missing_evidence_right = copy.deepcopy(baseline)
        del missing_evidence_left["samples"][0]["prepared_request"]
        del missing_evidence_right["samples"][0]["prepared_request"]
        with self.assertRaisesRegex(ValueError, "prepared_request schema"):
            compare_asr_latency_runs.compare(missing_evidence_left, missing_evidence_right)

    def test_nvml_sampler_is_explicitly_disabled_by_default(self) -> None:
        self.assertEqual(
            benchmark_asr_inference._measurement_profile(0.0, False),
            "canonical_latency",
        )
        self.assertEqual(
            benchmark_asr_inference._measurement_profile(10.0, False),
            "request_memory",
        )
        self.assertEqual(
            benchmark_asr_inference._measurement_profile(0.0, True),
            "diagnostic_phase",
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            benchmark_asr_inference._measurement_profile(10.0, True)
        sampler = benchmark_asr_inference.NvmlProcessSampler(0, 0.0)
        sampler.start()
        summary = sampler.stop()
        self.assertFalse(summary["enabled"])
        self.assertFalse(summary["available"])
        self.assertEqual(summary["sample_count"], 0)
        self.assertIn("disabled", summary["reason"])
        bare = "12345678-1234-1234-1234-123456789abc"
        self.assertEqual(
            benchmark_asr_inference._nvml_uuid_candidates(bare),
            [f"GPU-{bare}", f"MIG-{bare}", bare],
        )
        self.assertEqual(benchmark_asr_inference._normalized_gpu_uuid(f"GPU-{bare}"), bare)
        self.assertEqual(benchmark_asr_inference._normalized_gpu_uuid(f"MIG-{bare}"), bare)

    def test_cuda_phase_recorder_pairs_repeated_events(self) -> None:
        class FakeEvent:
            clock = 0.0

            def __init__(self, enable_timing: bool) -> None:
                self.enable_timing = enable_timing
                self.timestamp = None

            def record(self) -> None:
                self.timestamp = FakeEvent.clock
                FakeEvent.clock += 1.25

            def elapsed_time(self, other: FakeEvent) -> float:
                return float(other.timestamp - self.timestamp)

        class FakeCuda:
            Event = FakeEvent

        class FakeTorch:
            cuda = FakeCuda()

        recorder = benchmark_asr_inference.CudaPhaseRecorder(FakeTorch())
        recorder.callback("decode", "start")
        recorder.callback("decode", "end")
        recorder.callback("decode", "start")
        recorder.callback("decode", "end")
        result = recorder.finalize()
        self.assertEqual(result["decode"]["events_ms"], [1.25, 1.25])
        self.assertEqual(result["decode"]["summary_ms"]["n"], 2)

        unfinished = benchmark_asr_inference.CudaPhaseRecorder(FakeTorch())
        unfinished.callback("prefill", "start")
        with self.assertRaisesRegex(RuntimeError, "unfinished"):
            unfinished.finalize()


if __name__ == "__main__":
    unittest.main()
