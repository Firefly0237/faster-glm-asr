from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from tests.helpers.readiness import materialize_passing_readiness

from faster_glm_asr.benchmarking import comparator, provenance
from faster_glm_asr.benchmarking import matrix_aggregator as aggregation
from faster_glm_asr.benchmarking import matrix_executor as executor
from faster_glm_asr.benchmarking import matrix_planner as planner
from faster_glm_asr.data import librispeech


def _materialize_sources(root: Path) -> None:
    exact_sources = {
        "src/faster_glm_asr/benchmarking/comparator.py": Path(comparator.__file__).resolve(),
        "src/faster_glm_asr/benchmarking/matrix_aggregator.py": Path(
            aggregation.__file__
        ).resolve(),
        "src/faster_glm_asr/benchmarking/matrix_executor.py": Path(executor.__file__).resolve(),
        "src/faster_glm_asr/benchmarking/matrix_planner.py": Path(planner.__file__).resolve(),
        "src/faster_glm_asr/data/librispeech.py": Path(librispeech.__file__).resolve(),
    }
    for relative in (
        "src/faster_glm_asr/benchmarking/runner.py",
        "src/faster_glm_asr/benchmarking/comparator.py",
        "src/faster_glm_asr/benchmarking/matrix_aggregator.py",
        "src/faster_glm_asr/benchmarking/matrix_executor.py",
        "src/faster_glm_asr/benchmarking/matrix_planner.py",
        "src/faster_glm_asr/data/audio_manifest.py",
        "src/faster_glm_asr/data/librispeech.py",
        "src/faster_glm_asr/modeling/model.py",
        "src/faster_glm_asr/kernels/layers.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative in exact_sources:
            shutil.copyfile(exact_sources[relative], path)
        else:
            path.write_text(f"# fixture source: {relative}\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize_formal_provenance(root: Path, *, repo_id: str, revision: str) -> tuple[dict, dict]:
    readiness_report = materialize_passing_readiness(root)
    contract = root / provenance.READINESS_CONTRACT_PATH
    contract_hash = _sha256(contract)

    model_root = root / "artifacts/private/cache/model/glm-asr-nano-2512"
    model_root.mkdir(parents=True, exist_ok=True)
    files = []
    for index in range(provenance.MODEL_SNAPSHOT_FILE_COUNT):
        path = model_root / f"file-{index:02d}.bin"
        path.write_bytes(f"model-byte-fixture-{index}\n".encode())
        files.append(
            {
                "path": path.relative_to(model_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    model_manifest = root / "artifacts/private/cache/manifests/model.json"
    model_manifest.parent.mkdir(parents=True, exist_ok=True)
    model_manifest.write_text(
        json.dumps(
            {
                "schema_version": provenance.MODEL_MANIFEST_SCHEMA,
                "kind": provenance.MODEL_MANIFEST_KIND,
                "created_at": "2026-08-16T00:00:00+00:00",
                "root_label": model_root.name,
                "metadata": {
                    "environment_id": "rtx3090-ddp-v1",
                    "contract_sha256": contract_hash,
                    "repo_id": repo_id,
                    "revision": revision,
                },
                "file_count": len(files),
                "total_size_bytes": sum(item["size_bytes"] for item in files),
                "files": files,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return (
        provenance.validate_model_snapshot(
            model_root,
            model_manifest,
            expected_repo_id=repo_id,
            expected_revision=revision,
        ),
        provenance.validate_readiness_report(
            readiness_report,
            repository_root=root,
        ),
    )


def _summary(values: list[float]) -> dict:
    return aggregation._summary(values)


def _quality() -> dict:
    return {
        "evaluated_samples": 0,
        "word": {
            "errors": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "reference_units": 0,
        },
        "char": {
            "errors": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "reference_units": 0,
        },
        "wer": None,
        "cer": None,
        "status": "not_evaluated",
        "normalization": (
            "NFKC + casefold + Unicode punctuation/symbol removal + whitespace collapse"
        ),
    }


class MatrixFixture:
    def __init__(
        self,
        *,
        mutator: Callable[[dict, dict], None] | None = None,
        successful_tasks: int = 54,
        fail_next: bool = False,
        dataset_label: str = "domain",
        model_name: str = "fixture/model",
        model_revision: str = "a" * 40,
        manifest_relative: str = "artifacts/private/manifests/domain.jsonl",
        manifest_setup: Callable[[Path, Path], None] | None = None,
    ) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()
        (self.root / "configs").mkdir()
        (self.root / "artifacts/private/manifests").mkdir(parents=True)
        (self.root / "artifacts/private/environment").mkdir(parents=True)
        _materialize_sources(self.root)
        self.benchmark_script = self.root / "src/faster_glm_asr/benchmarking/runner.py"
        self.benchmark_script.write_text("# deterministic fixture runner\n", encoding="utf-8")
        self.manifest = self.root / manifest_relative
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.root / "artifacts/private/environment/freeze.txt"
        if manifest_setup is None:
            self.manifest.write_text('{"sample_id":"fixture"}\n', encoding="utf-8")
        else:
            manifest_setup(self.root, self.manifest)
        self.lock.write_text("torch==2.10.0\n", encoding="utf-8")
        self.repository_state = {"commit": "d" * 40, "dirty": False}
        self.model_binding, self.readiness_binding = _materialize_formal_provenance(
            self.root,
            repo_id=model_name,
            revision=model_revision,
        )
        self.configuration = {
            "schema_version": planner.CONFIG_SCHEMA,
            "python_executable": sys.executable,
            "benchmark_script": "src/faster_glm_asr/benchmarking/runner.py",
            "manifest": manifest_relative,
            "environment_lock": "artifacts/private/environment/freeze.txt",
            "model_snapshot_root": ("artifacts/private/cache/model/glm-asr-nano-2512"),
            "model_snapshot_manifest": "artifacts/private/cache/manifests/model.json",
            "readiness_report": "artifacts/private/rtx3090-readiness/full.json",
            "output_root": "artifacts/private/results/matrix/domain",
            "dataset_label": dataset_label,
            "artifact_scope": "private",
            "model_name": model_name,
            "model_revision": model_revision,
            "implementations": list(planner.IMPLEMENTATIONS),
            "profiles": list(planner.PROFILES),
            "outer_passes": 3,
            "seed": 20260815,
            "warmup": 1,
            "repeats": 1,
            "max_new_tokens": 8,
            "nvml_sample_ms": 10.0,
            "include_command_argv": True,
        }
        self.config_path = self.root / "configs/matrix.json"
        self.config_path.write_text(
            json.dumps(self.configuration, indent=2) + "\n", encoding="utf-8"
        )
        self.plan = planner.plan_from_config_file(
            self.config_path,
            repository_root=self.root,
            repository_state=self.repository_state,
        )
        self.plan_path = self.root / "artifacts/private/plans/domain-plan.json"
        self.plan_path.parent.mkdir(parents=True)
        planner._write_json_atomically(self.plan_path, self.plan)
        self.journal_path = self.root / "artifacts/private/run-journals/domain-plan/journal.json"
        self.journal_path.parent.mkdir(parents=True)
        self.log_paths: list[Path] = []
        self._write_outputs_and_journal(
            mutator=mutator,
            successful_tasks=successful_tasks,
            fail_next=fail_next,
        )

    def close(self) -> None:
        self.temporary.cleanup()

    def _environment(self) -> dict:
        return {
            "python": "3.11.9",
            "platform": "fixture-linux",
            "torch": "2.10.0",
            "torch_cuda": "13.0",
            "cudnn": 9999,
            "device_index": 0,
            "device_name": "NVIDIA synthetic-gpu",
            "device_uuid": "GPU-fixture",
            "cuda_visible_devices": "0",
            "compute_capability": [9, 0],
            "total_memory_bytes": 141_000_000_000,
            "allow_tf32_matmul": False,
            "allow_tf32_cudnn": True,
            "nvidia_driver_versions": ["999.0"],
            "package_versions": {
                package: "fixture" for package in comparator.REQUIRED_PACKAGE_VERSIONS
            },
            "environment_lock_sha256": hashlib.sha256(self.lock.read_bytes()).hexdigest(),
        }

    def _sources(self) -> dict:
        locked = self.plan["inputs"]["sources"]
        return {
            "schema_version": locked["schema_version"],
            "files": copy.deepcopy(locked["files"]),
            "aggregate_sha256": locked["aggregate_sha256"],
        }

    @staticmethod
    def _nvml(profile: str) -> dict:
        enabled = profile == "request_memory"
        result = {
            "enabled": enabled,
            "available": enabled,
            "interval_ms": 10.0 if enabled else 0.0,
            "sample_count": 2 if enabled else 0,
            "reason": None if enabled else "disabled (--nvml-sample-ms=0)",
            "device_selection": "fixture" if enabled else None,
            "nvml_device_uuid": "GPU-fixture" if enabled else None,
            "scope": "pre-prepare-sync-through-post-decode-output-hash",
            "peak_semantics": "polling-observed lower bound; not an allocator or exact peak",
            "samples": (
                [
                    {"elapsed_ms": 0.0, "used_bytes": 100},
                    {"elapsed_ms": 1.0, "used_bytes": 120},
                ]
                if enabled
                else []
            ),
        }
        if enabled:
            result.update(
                {
                    "first_used_bytes": 100,
                    "last_used_bytes": 120,
                    "observed_peak_used_bytes": 120,
                    "coverage_ms": 1.0,
                    "max_sample_gap_ms": 1.0,
                }
            )
        return result

    def _artifact(self, task: dict) -> dict:
        profile = task["measurement_profile"]
        implementation_index = list(planner.IMPLEMENTATIONS).index(task["implementation"])
        generation = float(task["outer_pass"] * 10 + implementation_index + 1)
        preprocess = 2.0
        decode = 3.0
        duration = 5.0
        token_ids = [7, 8]
        hypothesis = "fixture transcript"
        token_hash = hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        hypothesis_hash = hashlib.sha256(hypothesis.encode("utf-8")).hexdigest()
        observation = {
            "repeat": 0,
            "preprocess_ms": preprocess,
            "generation_ms": generation,
            "decode_ms": decode,
            "processor_plus_generation_ms": preprocess + generation,
            "request_no_file_io_ms": preprocess + generation + decode,
            "generated_tokens": len(token_ids),
            "hypothesis_sha256": hypothesis_hash,
            "generated_token_ids_sha256": token_hash,
            "rtf_generation": generation / 1000.0 / duration,
            "rtf_request_no_file_io": (preprocess + generation + decode) / 1000.0 / duration,
            "memory_bytes": {
                "before_prepare": {"allocated": 10, "reserved": 20},
                "after_prepare": {"allocated": 20, "reserved": 30},
                "after_generation": {"allocated": 30, "reserved": 40},
                "generation_allocated_peak": 35,
                "generation_reserved_peak": 45,
                "generation_allocated_peak_delta": 15,
            },
            "nvml_process_memory": self._nvml(profile),
            "diagnostic_cuda_phases": (
                {
                    "decode_step": {
                        "events_ms": [generation],
                        "summary_ms": _summary([generation]),
                        "total_ms": generation,
                    }
                }
                if profile == "diagnostic_phase"
                else None
            ),
        }
        sample = {
            "sample_id": "sample-001",
            "audio_sha256": "e" * 64,
            "language": "en",
            "duration_s": duration,
            "audio_load_ms": float(task["outer_pass"]),
            "prepared_request": {
                "tensors": {
                    "input_ids": {
                        "shape": [1, 4],
                        "dtype": "torch.int64",
                        "device": "cuda:0",
                        "content_sha256": "1" * 64,
                    },
                    "input_features": {
                        "shape": [1, 80, 100],
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
            },
            "metadata": {"shareable": False},
            "hypothesis": hypothesis,
            "hypothesis_sha256": hypothesis_hash,
            "generated_tokens": len(token_ids),
            "generated_token_ids": token_ids,
            "generated_token_ids_sha256": token_hash,
            "quality": None,
            "summary": {
                metric: _summary([float(observation[metric])])
                for metric in aggregation.SUMMARY_METRICS
            },
            "observations": [observation],
        }
        return {
            "schema_version": "0.3",
            "created_at_utc": f"2026-08-15T00:00:0{task['outer_pass']}+00:00",
            "run": {
                "formal_run": True,
                "comparability": {"status": "formal-comparable", "reasons": []},
                "warmup_gate": {
                    "status": "pass",
                    "minimum_iterations": 1,
                    "observed_iterations": self.configuration["warmup"],
                    "custom_lazy_h2d_applicable": task["implementation"].startswith("custom_"),
                    "custom_lazy_h2d_covered": (
                        True if task["implementation"].startswith("custom_") else None
                    ),
                },
                "implementation": task["implementation"],
                "model_name": self.configuration["model_name"],
                "model_revision": self.configuration["model_revision"],
                "manifest_path": str(self.manifest.resolve()),
                "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                "warmup": self.configuration["warmup"],
                "repeats": self.configuration["repeats"],
                "max_new_tokens": self.configuration["max_new_tokens"],
                "runner_construction_envelope_ms": 100.0 + float(task["outer_pass"]),
                "runner_construction_scope": "fixture construction envelope",
                "artifact_scope": "private",
                "measurement_profile": profile,
                "metric_eligibility": copy.deepcopy(aggregation.METRIC_ELIGIBILITY[profile]),
                "nvml_sample_ms": 10.0 if profile == "request_memory" else 0.0,
                "phase_timing": profile == "diagnostic_phase",
                "phase_timing_warning": (
                    "fixture diagnostic overhead" if profile == "diagnostic_phase" else None
                ),
                "runner_configuration": copy.deepcopy(
                    comparator.EXPECTED_RUNNER_CONFIGURATIONS[task["implementation"]]
                ),
            },
            "git": {
                "commit": self.repository_state["commit"],
                "branch": "fixture",
                "dirty": self.repository_state["dirty"],
            },
            "sources": self._sources(),
            "model_snapshot": copy.deepcopy(self.model_binding),
            "readiness": copy.deepcopy(self.readiness_binding),
            "environment": self._environment(),
            "aggregate": {
                "generation_ms": _summary([generation]),
                "processor_plus_generation_ms": _summary([preprocess + generation]),
                "request_no_file_io_ms": _summary([preprocess + generation + decode]),
                "quality": _quality(),
            },
            "samples": [sample],
        }

    @staticmethod
    def _append_event(journal: dict, fields: dict) -> dict:
        event = {
            "sequence": len(journal["events"]) + 1,
            **fields,
            "previous_event_sha256": (
                journal["events"][-1]["event_sha256"] if journal["events"] else "0" * 64
            ),
        }
        event["event_sha256"] = planner._canonical_sha256(event)
        journal["events"].append(event)
        return event

    def _write_outputs_and_journal(
        self,
        *,
        mutator: Callable[[dict, dict], None] | None,
        successful_tasks: int,
        fail_next: bool,
    ) -> None:
        timestamp = "2026-08-15T00:00:00+00:00"
        journal = {
            "schema_version": "asr-benchmark-run-journal-v0.1",
            "plan_path": self.plan_path.relative_to(self.root).as_posix(),
            "plan_sha256": hashlib.sha256(self.plan_path.read_bytes()).hexdigest(),
            "created_at_utc": timestamp,
            "events": [],
        }
        for task in self.plan["tasks"]:
            self._append_event(
                journal,
                {
                    "event": "planned",
                    "task_id": task["task_id"],
                    "timestamp_utc": timestamp,
                    "outer_pass": task["outer_pass"],
                    "order_in_pass": task["order_in_pass"],
                },
            )
        attempted_tasks = self.plan["tasks"][:successful_tasks]
        if fail_next and successful_tasks < len(self.plan["tasks"]):
            attempted_tasks = self.plan["tasks"][: successful_tasks + 1]
        for index, task in enumerate(attempted_tasks):
            success = index < successful_tasks
            safe = task["task_id"]
            stdout = self.journal_path.parent / "logs" / f"{safe}.stdout.txt"
            stderr = self.journal_path.parent / "logs" / f"{safe}.stderr.txt"
            stdout.parent.mkdir(parents=True, exist_ok=True)
            stdout.write_text("fixture stdout\n", encoding="utf-8")
            stderr.write_text("" if success else "fixture failed\n", encoding="utf-8")
            self.log_paths.extend((stdout, stderr))
            stdout_relative = stdout.relative_to(self.root).as_posix()
            stderr_relative = stderr.relative_to(self.root).as_posix()
            start = self._append_event(
                journal,
                {
                    "event": "start",
                    "task_id": task["task_id"],
                    "timestamp_utc": timestamp,
                    "attempt": 1,
                    "command_argv_sha256": task["command_argv_sha256"],
                    "stdout_path": stdout_relative,
                    "stderr_path": stderr_relative,
                },
            )
            output = self.root / task["output_path"]
            output_hash = None
            if success:
                payload = self._artifact(task)
                if mutator is not None:
                    mutator(task, payload)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                output_hash = hashlib.sha256(output.read_bytes()).hexdigest()
            self._append_event(
                journal,
                {
                    "event": "end",
                    "task_id": task["task_id"],
                    "timestamp_utc": timestamp,
                    "attempt": 1,
                    "start_event_sha256": start["event_sha256"],
                    "monotonic_duration_ns": 1,
                    "exit_code": 0 if success else 7,
                    "status": "success" if success else "process_failed",
                    "stdout_path": stdout_relative,
                    "stdout_sha256": hashlib.sha256(stdout.read_bytes()).hexdigest(),
                    "stderr_path": stderr_relative,
                    "stderr_sha256": hashlib.sha256(stderr.read_bytes()).hexdigest(),
                    "output_path": task["output_path"],
                    "output_sha256": output_hash,
                    "error": None,
                },
            )
        self.journal_path.write_text(
            json.dumps(journal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class BenchmarkAggregationTests(unittest.TestCase):
    def test_three_pass_matrix_aggregates_to_fixed_18_files(self) -> None:
        fixture = MatrixFixture()
        self.addCleanup(fixture.close)
        published = aggregation.aggregate_matrix(
            fixture.plan_path,
            fixture.journal_path,
            repository_root=fixture.root,
            repository_state=fixture.repository_state,
        )
        self.assertEqual(len(published), 18)
        self.assertEqual(
            set(published.values()),
            {
                (aggregation.DEFAULT_PRIVATE_OUTPUT_DIRECTORY / name).as_posix()
                for name in aggregation.FORMAL_OUTPUT_FILENAMES.values()
            },
        )
        canonical_path = fixture.root / published["custom_static_cache:canonical_latency"]
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        comparator._validate_canonical_payload(canonical, "aggregated canonical")
        self.assertEqual(canonical["run"]["repeats"], 3)
        observations = canonical["samples"][0]["observations"]
        self.assertEqual([row["repeat"] for row in observations], [0, 1, 2])
        self.assertEqual([row["generation_ms"] for row in observations], [16.0, 26.0, 36.0])
        provenance = canonical["matrix_aggregation"]
        self.assertEqual(provenance["outer_pass_count"], 3)
        self.assertEqual([row["outer_pass"] for row in provenance["pass_order"]], [1, 2, 3])
        self.assertTrue(
            all(len(row["source_artifact_sha256"]) == 64 for row in provenance["pass_order"])
        )

    def test_recomputes_sample_and_aggregate_summaries_from_raw_values(self) -> None:
        fixture = MatrixFixture()
        self.addCleanup(fixture.close)
        payload = aggregation.build_aggregates(
            fixture.plan_path,
            fixture.journal_path,
            repository_root=fixture.root,
            repository_state=fixture.repository_state,
        )[("custom_static_cache", "canonical_latency")]
        expected = _summary([16.0, 26.0, 36.0])
        self.assertEqual(payload["samples"][0]["summary"]["generation_ms"], expected)
        self.assertEqual(payload["aggregate"]["generation_ms"], expected)
        self.assertEqual(payload["aggregate"]["quality"], _quality())

    def test_build_is_deterministic_and_publish_refuses_overwrite(self) -> None:
        fixture = MatrixFixture()
        self.addCleanup(fixture.close)
        first = aggregation.build_aggregates(
            fixture.plan_path,
            fixture.journal_path,
            repository_root=fixture.root,
            repository_state=fixture.repository_state,
        )
        second = aggregation.build_aggregates(
            fixture.plan_path,
            fixture.journal_path,
            repository_root=fixture.root,
            repository_state=fixture.repository_state,
        )
        self.assertEqual(first, second)
        aggregation.aggregate_matrix(
            fixture.plan_path,
            fixture.journal_path,
            repository_root=fixture.root,
            repository_state=fixture.repository_state,
        )
        target = fixture.root / "artifacts/private/aggregated/custom-static-cache.json"
        before = target.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            aggregation.aggregate_matrix(
                fixture.plan_path,
                fixture.journal_path,
                repository_root=fixture.root,
                repository_state=fixture.repository_state,
            )
        self.assertEqual(target.read_bytes(), before)

    def test_rejects_missing_pass_successes(self) -> None:
        fixture = MatrixFixture(successful_tasks=36)
        self.addCleanup(fixture.close)
        with self.assertRaisesRegex(ValueError, "not fully successful"):
            aggregation.build_aggregates(
                fixture.plan_path,
                fixture.journal_path,
                repository_root=fixture.root,
                repository_state=fixture.repository_state,
            )

    def test_rejects_failed_journal(self) -> None:
        fixture = MatrixFixture(successful_tasks=53, fail_next=True)
        self.addCleanup(fixture.close)
        with self.assertRaisesRegex(ValueError, "not fully successful"):
            aggregation.build_aggregates(
                fixture.plan_path,
                fixture.journal_path,
                repository_root=fixture.root,
                repository_state=fixture.repository_state,
            )

    def test_rejects_current_output_and_log_hash_drift(self) -> None:
        fixture = MatrixFixture()
        self.addCleanup(fixture.close)
        first_output = fixture.root / fixture.plan["tasks"][0]["output_path"]
        original = first_output.read_bytes()
        first_output.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "output hash drift"):
            aggregation.build_aggregates(
                fixture.plan_path,
                fixture.journal_path,
                repository_root=fixture.root,
                repository_state=fixture.repository_state,
            )
        first_output.write_bytes(original)
        fixture.log_paths[0].write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stdout_path hash drift"):
            aggregation.build_aggregates(
                fixture.plan_path,
                fixture.journal_path,
                repository_root=fixture.root,
                repository_state=fixture.repository_state,
            )

    def test_rejects_token_drift_across_passes(self) -> None:
        def mutate(task: dict, payload: dict) -> None:
            if (
                task["outer_pass"] == 2
                and task["implementation"] == "custom_static_cache"
                and task["measurement_profile"] == "request_memory"
            ):
                sample = payload["samples"][0]
                sample["generated_token_ids"] = [7, 9]
                token_hash = hashlib.sha256(b"[7,9]").hexdigest()
                sample["generated_token_ids_sha256"] = token_hash
                sample["observations"][0]["generated_token_ids_sha256"] = token_hash

        fixture = MatrixFixture(mutator=mutate)
        self.addCleanup(fixture.close)
        with self.assertRaisesRegex(ValueError, "token IDs/hypothesis/quality drift"):
            aggregation.build_aggregates(
                fixture.plan_path,
                fixture.journal_path,
                repository_root=fixture.root,
                repository_state=fixture.repository_state,
            )

    def test_rejects_environment_drift_across_passes(self) -> None:
        def mutate(task: dict, payload: dict) -> None:
            if task["outer_pass"] == 2 and task["implementation"] == "custom_static_cache":
                payload["environment"]["device_name"] = "different synthetic-gpu"

        fixture = MatrixFixture(mutator=mutate)
        self.addCleanup(fixture.close)
        with self.assertRaisesRegex(ValueError, "provenance drift"):
            aggregation.build_aggregates(
                fixture.plan_path,
                fixture.journal_path,
                repository_root=fixture.root,
                repository_state=fixture.repository_state,
            )

    def test_rejects_prepared_request_drift_across_passes(self) -> None:
        def mutate(task: dict, payload: dict) -> None:
            if (
                task["outer_pass"] == 3
                and task["implementation"] == "custom_tuple_cache"
                and task["measurement_profile"] == "diagnostic_phase"
            ):
                payload["samples"][0]["prepared_request"]["tensors"]["input_ids"][
                    "content_sha256"
                ] = "9" * 64

        fixture = MatrixFixture(mutator=mutate)
        self.addCleanup(fixture.close)
        with self.assertRaisesRegex(ValueError, "prepared request"):
            aggregation.build_aggregates(
                fixture.plan_path,
                fixture.journal_path,
                repository_root=fixture.root,
                repository_state=fixture.repository_state,
            )


if __name__ == "__main__":
    unittest.main()
