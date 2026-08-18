from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).parents[2] / "tools" / "public_release_guard.py"
SPEC = importlib.util.spec_from_file_location("public_release_guard", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
public_release_guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = public_release_guard
SPEC.loader.exec_module(public_release_guard)


IMPLEMENTATIONS = (
    "hf_cached",
    "hf_no_cache",
    "custom_full_prefix",
    "custom_greedy_full_prefix",
    "custom_tuple_cache",
    "custom_static_cache",
)
BUCKETS = (
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


def _valid_public_summary() -> dict:
    canonical_by_implementation: dict[str, dict[str, tuple[float, float]]] = {}
    results = []
    for implementation_index, implementation in enumerate(IMPLEMENTATIONS):
        canonical_by_bucket: dict[str, tuple[float, float]] = {}
        duration_buckets = []
        for bucket_index, (bucket, _lower, _upper, _inclusive) in enumerate(BUCKETS):
            p50 = 100.0 + implementation_index * 10.0 + bucket_index
            p95 = p50 + 20.0
            duration_s = 5.0 + bucket_index * 10.0
            canonical_by_bucket[bucket] = (p50, p95)
            duration_buckets.append(
                {
                    "name": bucket,
                    "canonical": {
                        "observations": 30,
                        "generation_p50_ms": p50,
                        "generation_p95_ms": p95,
                        "rtf_p50": p50 / 1000.0 / duration_s,
                        "rtf_p95": p95 / 1000.0 / duration_s,
                    },
                    "request_memory": {
                        "observations": 30,
                        "allocator_peak_delta_p50_mib": 10.0 + bucket_index,
                        "allocator_peak_delta_p95_mib": 12.0 + bucket_index,
                        "process_peak_observed_p50_mib": 1000.0 + bucket_index,
                        "process_peak_observed_p95_mib": 1010.0 + bucket_index,
                        "minimum_poll_samples_per_request": 2,
                        "process_peak_observed_is_lower_bound": True,
                    },
                }
            )
        canonical_by_implementation[implementation] = canonical_by_bucket
        results.append({"implementation": implementation, "duration_buckets": duration_buckets})

    comparisons = []
    for baseline, candidate in COMPARISON_PAIRS:
        duration_buckets = []
        for bucket, _lower, _upper, _inclusive in BUCKETS:
            baseline_p50, baseline_p95 = canonical_by_implementation[baseline][bucket]
            candidate_p50, candidate_p95 = canonical_by_implementation[candidate][bucket]
            duration_buckets.append(
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
                "duration_buckets": duration_buckets,
            }
        )

    return {
        "schema_version": "faster-glm-asr-public-matrix-summary-v0.1",
        "artifact_scope": "public",
        "evidence": {
            "git_commit": "d" * 40,
            "source_map_sha256": "1" * 64,
            "plan_sha256": "2" * 64,
            "journal_sha256": "3" * 64,
            "aggregate_set_sha256": "4" * 64,
            "dataset_archive_sha256": "5" * 64,
            "exporter_source_sha256": "6" * 64,
            "schema_source_sha256": "7" * 64,
        },
        "benchmark": {
            "model": {
                "repository": "zai-org/GLM-ASR-Nano-2512",
                "revision": "61ba4e0b3309b6656edea3e93e419f7bd5c61957",
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
                    for name, lower, upper, inclusive in BUCKETS
                ],
            },
        },
        "hardware": {
            "accelerator_model": "NVIDIA GeForce RTX 3090",
            "nominal_memory_gib": 24,
            "compute_capability": "8.6",
        },
        "software": {
            "python": "3.11.9",
            "cuda_runtime": "12.8",
            "cudnn_version": 91002,
            "nvidia_driver_version": "595.71.5",
            "packages": {
                package: "1.0.0"
                for package in (
                    "accelerate",
                    "huggingface-hub",
                    "numpy",
                    "nvidia-ml-py",
                    "safetensors",
                    "scipy",
                    "soundfile",
                    "torch",
                    "transformers",
                    "triton",
                )
            },
        },
        "protocol": {
            "implementations": list(IMPLEMENTATIONS),
            "measurement_profiles": [
                "canonical_latency",
                "request_memory",
                "diagnostic_phase",
            ],
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
                "instrumentation": ("NVML process polling plus PyTorch CUDA allocator counters"),
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


def _public_summary_cases() -> dict[str, bytes]:
    valid = json.dumps(_valid_public_summary(), allow_nan=False).encode("utf-8")
    schema_invalid = _valid_public_summary()
    schema_invalid["hostname"] = "private-schema-marker"
    nonfinite = _valid_public_summary()
    nonfinite["results"][0]["duration_buckets"][0]["canonical"]["generation_p50_ms"] = float("nan")
    infinity = _valid_public_summary()
    infinity["results"][0]["duration_buckets"][0]["canonical"]["generation_p50_ms"] = float("inf")
    duplicate = valid.replace(
        b'"artifact_scope": "public"',
        (b'"artifact_scope": "public", "artifact_scope": "private-duplicate-marker"'),
        1,
    )
    return {
        "valid.json": valid,
        "schema-invalid.json": json.dumps(schema_invalid, allow_nan=False).encode("utf-8"),
        "malformed.json": b'{"schema_version": "private-malformed-marker"',
        "duplicate.json": duplicate,
        "nan.json": json.dumps(nonfinite).encode("utf-8"),
        "infinity.json": json.dumps(infinity).encode("utf-8"),
    }


class PublicReleaseGuardTest(unittest.TestCase):
    def _git(self, root: Path, *arguments: str) -> None:
        subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
        )

    def _repository(self, root: Path, *, final_readme: bool = False) -> None:
        self._git(root, "init", "--quiet")
        (root / "LICENSE").write_text(
            "Apache License\n"
            "Version 2.0, January 2004\n"
            "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n"
            "END OF TERMS AND CONDITIONS\n",
            encoding="utf-8",
        )
        (root / "NOTICE").write_text(
            "edin-mls-26-spring\n"
            "CC0 1.0 Universal\n"
            "https://github.com/zai-org/GLM-ASR\n"
            "Hugging Face Transformers\n"
            "model artifacts are not bundled here\n",
            encoding="utf-8",
        )
        readme = "README.md" if final_readme else "README.draft.md"
        (root / readme).write_text("# Fixture\n", encoding="utf-8")
        (root / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
        (root / ".gitattributes").write_text("*.py text eol=lf\n", encoding="utf-8")
        schema = root / public_release_guard.PUBLIC_MATRIX_SCHEMA_RELATIVE
        schema.parent.mkdir(parents=True)
        schema.write_bytes(public_release_guard.PUBLIC_MATRIX_SCHEMA_PATH.read_bytes())
        self._git(root, "add", public_release_guard.PUBLIC_MATRIX_SCHEMA_RELATIVE)

    def test_clean_candidate_and_final_metadata_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            (root / "src" / "package").mkdir(parents=True)
            (root / "src" / "package" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            self.assertEqual(public_release_guard.scan_repository(root), [])
            findings = public_release_guard.scan_repository(root, strict_release=True)
            self.assertEqual(
                {item.code for item in findings}, {"draft-present", "missing-metadata"}
            )
            (root / "README.draft.md").rename(root / "README.md")
            self.assertEqual(public_release_guard.scan_repository(root, strict_release=True), [])

    def test_fail_closed_path_allowlist(self) -> None:
        excluded_area = "job" + "-packaging"
        cases = {
            f"docs/{excluded_area}/notes.md": "forbidden-path",
            "data/sample.json": "forbidden-top-level",
            "docs/sample.wav": "forbidden-extension",
            "misc/source.py": "top-level-not-allowlisted",
            "src/package/payload.exe": "text-type-not-allowlisted",
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                codes = {item.code for item in public_release_guard.validate_path(path)}
                self.assertIn(expected, codes)
        self.assertEqual(public_release_guard.validate_path("src/faster_glm_asr/model.py"), [])
        self.assertEqual(
            public_release_guard.validate_path("src/faster_glm_asr/data/audio_manifest.py"),
            [],
        )
        self.assertEqual(public_release_guard.validate_path("tests/data/test_manifest.py"), [])
        self.assertEqual(
            public_release_guard.validate_path("benchmarks/results/rtx3090-matrix.json"),
            [],
        )
        for path in (
            "benchmarks/result.json",
            "benchmarks/results/README.md",
            "benchmarks/results/nested/result.json",
        ):
            with self.subTest(benchmark_path=path):
                self.assertTrue(
                    any(
                        item.code == "benchmark-path-not-allowlisted"
                        for item in public_release_guard.validate_path(path)
                    )
                )
        self.assertEqual(public_release_guard.validate_path("MANIFEST.in"), [])

    def test_project_gitignore_keeps_package_data_in_release_inventory(self) -> None:
        project_root = SCRIPT.parents[1]
        candidates = set(public_release_guard.candidate_paths(project_root))
        self.assertIn(
            "src/faster_glm_asr/data/audio_manifest.py",
            candidates,
            "a root-local data ignore rule must not hide the importable data package",
        )
        top_level_findings = public_release_guard.validate_path("data/sample.json")
        self.assertTrue(any(item.code == "forbidden-top-level" for item in top_level_findings))

    def test_sensitive_content_is_rejected_without_value_in_finding(self) -> None:
        access_value = "AK" + "IA" + "A" * 16
        local_path = "/" + "ho" + "me/alice/project/file.txt"
        excluded_words = "".join(chr(value) for value in (0x6C42, 0x804C))
        excluded_english = "career" + "-preparation"
        for payload, rule in (
            (access_value, "aws-access-key"),
            (local_path, "linux-home-path"),
            (excluded_words, "career-material-zh-1"),
            (excluded_english, "career-material-en"),
        ):
            with self.subTest(rule=rule):
                findings = public_release_guard.validate_content(
                    "docs/note.md", payload.encode("utf-8")
                )
                self.assertTrue(any(rule in item.message for item in findings))
                self.assertTrue(all(payload not in item.message for item in findings))
        self.assertEqual(
            public_release_guard.validate_content("docs/example.md", b'api_key = "<redacted>"\n'),
            [],
        )

    def test_binary_oversize_and_lfs_pointer_are_rejected(self) -> None:
        self.assertEqual(
            public_release_guard.validate_content("docs/x.md", b"a\0b")[0].code,
            "binary-content",
        )
        oversized = b"a" * (public_release_guard.MAX_TEXT_BYTES + 1)
        self.assertEqual(
            public_release_guard.validate_content("docs/x.md", oversized)[0].code,
            "file-too-large",
        )
        marker = ("version https://git-" + "lfs.github.com/spec/v1\n").encode()
        self.assertEqual(
            public_release_guard.validate_content("docs/x.md", marker)[0].code,
            "git-lfs-pointer",
        )

    def test_utf8_csv_is_allowed_but_binary_csv_is_rejected(self) -> None:
        path = "configs/benchmark/inventory.example.csv"
        self.assertEqual(public_release_guard.validate_path(path), [])
        self.assertEqual(
            public_release_guard.validate_content(
                path, "sample_id,audio_path\n示例,inputs/example.wav\n".encode()
            ),
            [],
        )
        findings = public_release_guard.validate_content(path, b"sample_id\0audio_path\n")
        self.assertTrue(any(item.code == "binary-content" for item in findings))

    def test_candidate_public_results_use_strict_schema_validation(self) -> None:
        cases = _public_summary_cases()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root, final_readme=True)
            results = root / "benchmarks" / "results"
            results.mkdir(parents=True)
            for filename, payload in cases.items():
                (results / filename).write_bytes(payload)

            findings = public_release_guard.scan_repository(root, strict_release=True)
            matrix_findings = {
                item.path: item for item in findings if item.code == "public-matrix-invalid"
            }
            self.assertNotIn("benchmarks/results/valid.json", matrix_findings)
            for filename in cases.keys() - {"valid.json"}:
                self.assertIn(f"benchmarks/results/{filename}", matrix_findings)
            rendered = "\n".join(item.message for item in findings)
            for marker in (
                "private-schema-marker",
                "private-malformed-marker",
                "private-duplicate-marker",
            ):
                self.assertNotIn(marker, rendered)

    def test_public_result_validator_import_and_interface_fail_closed(self) -> None:
        path = "benchmarks/results/valid.json"
        payload = json.dumps(_valid_public_summary(), allow_nan=False).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-schema.py"
            with (
                mock.patch.object(public_release_guard, "_public_matrix_schema_module", None),
                mock.patch.object(public_release_guard, "PUBLIC_MATRIX_SCHEMA_PATH", missing),
            ):
                findings = public_release_guard.validate_content(path, payload)
            self.assertTrue(any(item.code == "public-matrix-invalid" for item in findings))

        for origin in ("candidate", "history:fixture"):
            with (
                self.subTest(origin=origin),
                mock.patch.object(
                    public_release_guard,
                    "_public_matrix_schema_module",
                    object(),
                ),
            ):
                findings = public_release_guard.validate_content(path, payload, origin=origin)
                self.assertEqual(
                    [item.code for item in findings if item.code == "public-matrix-invalid"],
                    ["public-matrix-invalid"],
                )

    def test_history_public_results_use_strict_schema_validation(self) -> None:
        cases = _public_summary_cases()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root, final_readme=True)
            self._git(root, "config", "user.name", "Release Guard Test")
            self._git(root, "config", "user.email", "guard@example.invalid")
            results = root / "benchmarks" / "results"
            results.mkdir(parents=True)
            for filename, payload in cases.items():
                (results / filename).write_bytes(payload)
            self._git(root, "add", ".")
            self._git(root, "commit", "--quiet", "-m", "public result fixtures")

            for filename in cases:
                (results / filename).unlink()
            self._git(root, "add", "-A")
            self._git(root, "commit", "--quiet", "-m", "remove result fixtures")

            self.assertEqual(public_release_guard.scan_repository(root, strict_release=True), [])
            findings = public_release_guard.scan_repository(
                root, strict_release=True, include_history=True
            )
            matrix_findings = {
                item.path: item
                for item in findings
                if item.code == "public-matrix-invalid" and item.origin.startswith("history:")
            }
            self.assertNotIn("benchmarks/results/valid.json", matrix_findings)
            for filename in cases.keys() - {"valid.json"}:
                self.assertIn(f"benchmarks/results/{filename}", matrix_findings)
            rendered = "\n".join(item.message for item in matrix_findings.values())
            for marker in (
                "private-schema-marker",
                "private-malformed-marker",
                "private-duplicate-marker",
            ):
                self.assertNotIn(marker, rendered)

    def test_history_scan_finds_content_deleted_from_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root, final_readme=True)
            self._git(root, "config", "user.name", "Release Guard Test")
            self._git(root, "config", "user.email", "guard@example.invalid")
            docs = root / "docs"
            docs.mkdir()
            leaked = "gh" + "p_" + "A" * 24
            note = docs / "note.md"
            note.write_text(leaked, encoding="utf-8")
            self._git(root, "add", ".")
            self._git(root, "commit", "--quiet", "-m", "fixture with old blob")
            note.unlink()
            (docs / "safe.md").write_text("safe\n", encoding="utf-8")
            self._git(root, "add", "-A")
            self._git(root, "commit", "--quiet", "-m", "remove old blob")

            self.assertEqual(public_release_guard.scan_repository(root, strict_release=True), [])
            findings = public_release_guard.scan_repository(
                root, strict_release=True, include_history=True
            )
            historical = [
                item
                for item in findings
                if item.path == "docs/note.md" and item.origin.startswith("history:")
            ]
            self.assertTrue(historical)
            self.assertTrue(all(leaked not in item.message for item in historical))

    def test_staged_index_blobs_are_scanned_independently_from_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root, final_readme=True)
            self._git(root, "config", "user.name", "Release Guard Test")
            self._git(root, "config", "user.email", "guard@example.invalid")
            docs = root / "docs"
            docs.mkdir()
            note = docs / "note.md"
            note.write_text("safe baseline\n", encoding="utf-8")
            results = root / "benchmarks" / "results"
            results.mkdir(parents=True)
            result = results / "rtx3090-matrix.json"
            valid_result = json.dumps(_valid_public_summary(), allow_nan=False) + "\n"
            result.write_text(valid_result, encoding="utf-8")
            valid_license = (root / "LICENSE").read_text(encoding="utf-8")
            self._git(root, "add", ".")
            self._git(root, "commit", "--quiet", "-m", "safe baseline")

            staged_secret = "pass" + "word = staged_secret_12345\n"
            note.write_text(staged_secret, encoding="utf-8")
            invalid_result = _valid_public_summary()
            invalid_result["hostname"] = "staged-private-host"
            result.write_text(json.dumps(invalid_result), encoding="utf-8")
            (root / "LICENSE").write_text("not a complete license\n", encoding="utf-8")
            self._git(root, "add", "docs/note.md", "benchmarks/results", "LICENSE")

            note.write_text("safe worktree\n", encoding="utf-8")
            result.write_text(valid_result, encoding="utf-8")
            (root / "LICENSE").write_text(valid_license, encoding="utf-8")

            findings = public_release_guard.scan_repository(
                root, strict_release=True, include_history=True
            )
            staged = [item for item in findings if item.origin.startswith("index:")]
            self.assertTrue(
                any(
                    item.path == "docs/note.md"
                    and item.code == "sensitive-content"
                    and "assigned-secret" in item.message
                    for item in staged
                )
            )
            self.assertTrue(
                any(
                    item.path == "benchmarks/results/rtx3090-matrix.json"
                    and item.code == "public-matrix-invalid"
                    for item in staged
                )
            )
            self.assertTrue(
                any(item.path == "LICENSE" and item.code == "license-invalid" for item in staged)
            )
            self.assertTrue(all("staged_secret_12345" not in item.message for item in findings))

    def test_strict_guard_binds_validator_to_stage_zero_schema_blob(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root, final_readme=True)
            self._git(root, "config", "user.name", "Release Guard Test")
            self._git(root, "config", "user.email", "guard@example.invalid")
            result = root / "benchmarks" / "results" / "rtx3090-matrix.json"
            result.parent.mkdir(parents=True)
            valid_result = json.dumps(_valid_public_summary(), allow_nan=False) + "\n"
            result.write_text(valid_result, encoding="utf-8")
            self._git(root, "add", ".")
            self._git(root, "commit", "--quiet", "-m", "safe baseline")

            invalid_result = _valid_public_summary()
            invalid_result["hostname"] = "staged-private-host"
            result.write_text(json.dumps(invalid_result), encoding="utf-8")
            self._git(root, "add", "benchmarks/results/rtx3090-matrix.json")
            result.write_text(valid_result, encoding="utf-8")

            schema = root / public_release_guard.PUBLIC_MATRIX_SCHEMA_RELATIVE
            schema_bytes = schema.read_bytes()
            schema.write_text(
                "import json\n"
                "def load_public_summary_bytes(value): return json.loads(value)\n"
                "def validate_public_summary(payload): return []\n",
                encoding="utf-8",
            )
            findings = public_release_guard.scan_repository(
                root, strict_release=True, include_history=True
            )
            self.assertEqual(
                [(item.code, item.path, item.origin) for item in findings],
                [
                    (
                        "schema-origin-drift",
                        public_release_guard.PUBLIC_MATRIX_SCHEMA_RELATIVE,
                        "index",
                    )
                ],
            )

            schema.write_bytes(schema_bytes)
            cached_permissive = mock.Mock()
            cached_permissive.load_public_summary_bytes.side_effect = json.loads
            cached_permissive.validate_public_summary.return_value = []
            with mock.patch.object(
                public_release_guard,
                "_public_matrix_schema_module",
                cached_permissive,
            ):
                findings = public_release_guard.scan_repository(
                    root, strict_release=True, include_history=True
                )
            self.assertTrue(
                any(
                    item.path == "benchmarks/results/rtx3090-matrix.json"
                    and item.code == "public-matrix-invalid"
                    and item.origin.startswith("index:")
                    for item in findings
                )
            )
            cached_permissive.load_public_summary_bytes.assert_not_called()
            cached_permissive.validate_public_summary.assert_not_called()

    def test_history_mode_rejects_a_shallow_clone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            self._repository(source, final_readme=True)
            self._git(source, "config", "user.name", "Release Guard Test")
            self._git(source, "config", "user.email", "guard@example.invalid")
            docs = source / "docs"
            docs.mkdir()
            old_note = docs / "old.md"
            old_note.write_text("gh" + "p_" + "A" * 24, encoding="utf-8")
            self._git(source, "add", ".")
            self._git(source, "commit", "--quiet", "-m", "historical sensitive blob")
            old_note.unlink()
            (docs / "safe.md").write_text("safe\n", encoding="utf-8")
            self._git(source, "add", "-A")
            self._git(source, "commit", "--quiet", "-m", "clean head")

            full_findings = public_release_guard.scan_repository(
                source, strict_release=True, include_history=True
            )
            self.assertTrue(
                any(
                    item.path == "docs/old.md"
                    and item.code == "sensitive-content"
                    and item.origin.startswith("history:")
                    for item in full_findings
                )
            )

            clone = base / "shallow"
            subprocess.run(
                ["git", "clone", "--quiet", "--depth", "1", source.as_uri(), str(clone)],
                check=True,
                capture_output=True,
            )
            shallow_findings = public_release_guard.scan_repository(
                clone, strict_release=True, include_history=True
            )
            self.assertTrue(any(item.code == "history-incomplete" for item in shallow_findings))

    def test_candidate_rejects_an_ancestor_symlink_or_windows_junction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "repository"
            root.mkdir()
            self._repository(root, final_readme=True)
            docs = root / "docs"
            docs.mkdir()
            (docs / "note.md").write_text("safe staged source\n", encoding="utf-8")
            self._git(root, "add", ".")

            original = base / "original-docs"
            docs.rename(original)
            alternate = base / "alternate-docs"
            alternate.mkdir()
            (alternate / "note.md").write_text("safe alternate source\n", encoding="utf-8")
            link_created = False
            try:
                if os.name == "nt":
                    result = subprocess.run(
                        ["cmd", "/c", "mklink", "/J", str(docs), str(alternate)],
                        check=False,
                        capture_output=True,
                    )
                    if result.returncode != 0:
                        self.skipTest("Windows junction creation is unavailable")
                else:
                    docs.symlink_to(alternate, target_is_directory=True)
                link_created = True
                findings = public_release_guard.scan_repository(root, strict_release=True)
                self.assertTrue(
                    any(
                        item.path == "docs/note.md" and item.code == "symbolic-link"
                        for item in findings
                    )
                )
            finally:
                if link_created:
                    if os.name == "nt":
                        os.rmdir(docs)
                    else:
                        docs.unlink()

    def test_release_docs_state_schema_and_evidence_authenticity_boundaries(self) -> None:
        repository = SCRIPT.parents[1]
        policy = (repository / "docs/public-release-policy.md").read_text(encoding="utf-8")
        contract = (repository / "docs/benchmark-contract.md").read_text(encoding="utf-8")
        normalized_policy = " ".join(policy.split())
        normalized_contract = " ".join(contract.split())
        self.assertIn("stage-zero schema", normalized_policy)
        self.assertIn("historical Python is never executed", normalized_policy)
        self.assertIn("audit handles", normalized_policy)
        self.assertIn("not self-authenticating proof", normalized_policy)
        self.assertIn("do not prove that a file originated from `publish`", normalized_policy)
        self.assertNotIn("it must be produced by", normalized_policy)
        self.assertIn("audit handles", normalized_contract)
        self.assertIn("not self-authenticating proof", normalized_contract)
        self.assertIn(
            "do not prove that an otherwise valid JSON file originated",
            normalized_contract,
        )

    def test_cli_does_not_echo_matched_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            (root / "docs").mkdir()
            value = "hf" + "_" + "A" * 24
            (root / "docs" / "note.md").write_text(value, encoding="utf-8")
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                status = public_release_guard.main(["--root", str(root)])
            self.assertEqual(status, 1)
            self.assertNotIn(value, captured.getvalue())
            self.assertIn("hugging-face-token", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
