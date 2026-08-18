from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import textwrap
import types
import unittest
from collections import Counter
from itertools import pairwise
from pathlib import Path
from unittest import mock

from tests.helpers.readiness import materialize_passing_readiness

from faster_glm_asr.benchmarking import comparator, provenance
from faster_glm_asr.benchmarking import matrix_executor as executor
from faster_glm_asr.benchmarking import matrix_planner as planner
from faster_glm_asr.benchmarking import runner as benchmark


def _materialize_relevant_sources(root: Path) -> None:
    for relative in (
        "src/faster_glm_asr/benchmarking/runner.py",
        "src/faster_glm_asr/benchmarking/comparator.py",
        "src/faster_glm_asr/data/audio_manifest.py",
        "src/faster_glm_asr/modeling/model.py",
        "src/faster_glm_asr/kernels/layers.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture source: {relative}\n", encoding="utf-8")


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


class BenchmarkMatrixPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()
        (self.root / "configs").mkdir()
        (self.root / "artifacts/private/manifests").mkdir(parents=True)
        (self.root / "artifacts/private/environment").mkdir(parents=True)
        _materialize_relevant_sources(self.root)
        _materialize_formal_provenance(
            self.root,
            repo_id="zai-org/GLM-ASR-Nano-2512",
            revision="61ba4e0b3309b6656edea3e93e419f7bd5c61957",
        )
        (self.root / "artifacts/private/manifests/domain.jsonl").write_text(
            '{"sample_id":"fixture"}\n', encoding="utf-8"
        )
        (self.root / "artifacts/private/environment/freeze.txt").write_text(
            "torch==2.10.0\n", encoding="utf-8"
        )
        self.configuration = {
            "schema_version": planner.CONFIG_SCHEMA,
            "python_executable": sys.executable,
            "benchmark_script": "src/faster_glm_asr/benchmarking/runner.py",
            "manifest": "artifacts/private/manifests/domain.jsonl",
            "environment_lock": "artifacts/private/environment/freeze.txt",
            "model_snapshot_root": ("artifacts/private/cache/model/glm-asr-nano-2512"),
            "model_snapshot_manifest": "artifacts/private/cache/manifests/model.json",
            "readiness_report": "artifacts/private/rtx3090-readiness/full.json",
            "output_root": "artifacts/private/results/matrix/domain",
            "dataset_label": "domain",
            "artifact_scope": "private",
            "model_name": "zai-org/GLM-ASR-Nano-2512",
            "model_revision": "61ba4e0b3309b6656edea3e93e419f7bd5c61957",
            "implementations": list(planner.IMPLEMENTATIONS),
            "profiles": list(planner.PROFILES),
            "outer_passes": 3,
            "seed": 20260815,
            "warmup": 3,
            "repeats": 10,
            "max_new_tokens": 128,
            "nvml_sample_ms": 10.0,
            "include_command_argv": True,
        }
        self.config_path = self.root / "configs/matrix.json"
        self.repository_state = {"commit": "c" * 40, "dirty": False}
        self._write_configuration(self.configuration)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_configuration(self, value: object) -> None:
        self.config_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def _plan(self) -> dict:
        return planner.plan_from_config_file(
            self.config_path,
            repository_root=self.root,
            repository_state=self.repository_state,
        )

    def test_plan_is_deterministic_and_does_not_execute_models(self) -> None:
        first = self._plan()
        second = self._plan()
        self.assertEqual(first, second)
        self.assertFalse(first["planner"]["model_execution_performed"])
        self.assertFalse(first["planner"]["shell_execution_required"])
        self.assertEqual(first["planner"]["command_representation"], "argv-array")
        self.assertEqual(first["task_count"], 54)

    def test_formal_matrix_uses_one_fp32_precision(self) -> None:
        configured_dtypes = {
            comparator.EXPECTED_RUNNER_CONFIGURATIONS[implementation]["storage_activation_dtype"]
            for implementation in planner.IMPLEMENTATIONS
        }
        self.assertEqual(configured_dtypes, {"torch.float32"})
        for implementation in planner.IMPLEMENTATIONS:
            if not implementation.startswith("custom_"):
                continue
            configuration = comparator.EXPECTED_RUNNER_CONFIGURATIONS[implementation]
            self.assertEqual(configuration["system"], "custom-torch-triton-hybrid")
            self.assertEqual(
                configuration["attention_dispatch_policy"],
                "dense-triton-if-padded-and-head-dim-lte-256-else-torch-dense",
            )
            self.assertEqual(configuration["gqa_kv_expansion"], "explicit-expand")
            self.assertEqual(
                configuration["performance_attribution"],
                "cache-strategy-only-no-triton-kernel-attribution",
            )

    def test_example_uses_only_supported_public_ids(self) -> None:
        example = (
            Path(__file__).resolve().parents[2] / "configs" / "benchmark" / "matrix.example.json"
        )
        configuration = json.loads(example.read_text(encoding="utf-8"))
        validated = planner._validate_configuration(configuration)
        self.assertEqual(tuple(validated["implementations"]), planner.IMPLEMENTATIONS)
        self.assertEqual(validated["artifact_scope"], "private")
        self.assertTrue(validated["output_root"].startswith("artifacts/private/"))

    def test_every_pass_is_complete_balanced_and_interleaved(self) -> None:
        plan = self._plan()
        expected_pairs = {
            (implementation, profile)
            for implementation in planner.IMPLEMENTATIONS
            for profile in planner.PROFILES
        }
        first_implementations = []
        for outer_pass in range(1, 4):
            tasks = [task for task in plan["tasks"] if task["outer_pass"] == outer_pass]
            self.assertEqual(len(tasks), 18)
            self.assertEqual(
                {(task["implementation"], task["measurement_profile"]) for task in tasks},
                expected_pairs,
            )
            self.assertEqual(
                Counter(task["implementation"] for task in tasks),
                Counter({implementation: 3 for implementation in planner.IMPLEMENTATIONS}),
            )
            self.assertEqual(
                Counter(task["measurement_profile"] for task in tasks),
                Counter({profile: 6 for profile in planner.PROFILES}),
            )
            self.assertTrue(
                all(
                    left["implementation"] != right["implementation"]
                    for left, right in pairwise(tasks)
                )
            )
            first_implementations.append(tasks[0]["implementation"])
        self.assertEqual(len(set(first_implementations)), 3)
        self.assertEqual(len({task["output_path"] for task in plan["tasks"]}), plan["task_count"])

    def test_different_seed_changes_order_and_output_namespace(self) -> None:
        first = self._plan()
        changed = copy.deepcopy(self.configuration)
        changed["seed"] += 1
        self._write_configuration(changed)
        second = self._plan()
        first_order = [
            (task["implementation"], task["measurement_profile"]) for task in first["tasks"]
        ]
        second_order = [
            (task["implementation"], task["measurement_profile"]) for task in second["tasks"]
        ]
        self.assertNotEqual(first_order, second_order)
        self.assertNotEqual(first["tasks"][0]["output_path"], second["tasks"][0]["output_path"])

    def test_rejects_duplicate_missing_and_invalid_matrix_items(self) -> None:
        duplicate = copy.deepcopy(self.configuration)
        duplicate["implementations"][-1] = duplicate["implementations"][0]
        self._write_configuration(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._plan()

        invalid = copy.deepcopy(self.configuration)
        invalid["profiles"][-1] = "combined_memory_and_phase"
        self._write_configuration(invalid)
        with self.assertRaisesRegex(ValueError, "exact supported set"):
            self._plan()

        too_few_passes = copy.deepcopy(self.configuration)
        too_few_passes["outer_passes"] = 2
        self._write_configuration(too_few_passes)
        with self.assertRaisesRegex(ValueError, "outer_passes"):
            self._plan()

    def test_tasks_bind_all_inputs_and_commands_are_safe_argv(self) -> None:
        plan = self._plan()
        expected_config_hash = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        expected_manifest_hash = hashlib.sha256(
            (self.root / self.configuration["manifest"]).read_bytes()
        ).hexdigest()
        expected_environment_hash = hashlib.sha256(
            (self.root / self.configuration["environment_lock"]).read_bytes()
        ).hexdigest()
        source_paths = benchmark.discover_source_paths(self.root)
        expected_source_records = {
            relative: {
                "size_bytes": (self.root / relative).stat().st_size,
                "sha256": hashlib.sha256((self.root / relative).read_bytes()).hexdigest(),
            }
            for relative in source_paths
        }
        expected_source_aggregate = hashlib.sha256(
            json.dumps(expected_source_records, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        task_hashes = set()
        for task in plan["tasks"]:
            bindings = task["bindings"]
            self.assertEqual(bindings["matrix_configuration_sha256"], expected_config_hash)
            self.assertEqual(bindings["manifest_sha256"], expected_manifest_hash)
            self.assertEqual(bindings["environment_lock_sha256"], expected_environment_hash)
            self.assertEqual(bindings["model_revision"], self.configuration["model_revision"])
            self.assertEqual(bindings["git_commit"], self.repository_state["commit"])
            self.assertFalse(bindings["git_dirty"])
            self.assertEqual(
                bindings["model_snapshot_manifest_sha256"],
                plan["inputs"]["model_snapshot"]["manifest_sha256"],
            )
            self.assertEqual(
                bindings["model_snapshot_files_aggregate_sha256"],
                plan["inputs"]["model_snapshot"]["files_aggregate_sha256"],
            )
            self.assertEqual(
                bindings["readiness_report_sha256"],
                plan["inputs"]["readiness"]["report_sha256"],
            )
            self.assertEqual(bindings["source_files"], expected_source_records)
            self.assertEqual(bindings["source_map_aggregate_sha256"], expected_source_aggregate)
            self.assertEqual(
                bindings["python_executable_path"], str(Path(sys.executable).absolute())
            )
            self.assertEqual(len(bindings["task_configuration_sha256"]), 64)
            task_hashes.add(bindings["task_configuration_sha256"])
            argv = task["command_argv"]
            self.assertIsInstance(argv, list)
            self.assertTrue(all(isinstance(argument, str) for argument in argv))
            self.assertNotIn("--overwrite", argv)
            self.assertIn("--formal", argv)
            self.assertIn("--model-snapshot-root", argv)
            self.assertIn("--model-snapshot-manifest", argv)
            self.assertIn("--readiness-report", argv)
            self.assertEqual(task["command_argv_sha256"], planner._canonical_sha256(argv))
            if task["measurement_profile"] == "request_memory":
                self.assertIn("--nvml-sample-ms", argv)
                self.assertNotIn("--phase-timing", argv)
            elif task["measurement_profile"] == "diagnostic_phase":
                self.assertIn("--phase-timing", argv)
                self.assertNotIn("--nvml-sample-ms", argv)
            else:
                self.assertNotIn("--phase-timing", argv)
                self.assertNotIn("--nvml-sample-ms", argv)
        self.assertEqual(len(task_hashes), plan["task_count"])

        before = self._plan()
        (self.root / self.configuration["manifest"]).write_text(
            '{"sample_id":"changed"}\n', encoding="utf-8"
        )
        after = self._plan()
        self.assertNotEqual(
            before["inputs"]["manifest"]["sha256"],
            after["inputs"]["manifest"]["sha256"],
        )
        self.assertNotEqual(
            before["tasks"][0]["bindings"]["task_configuration_sha256"],
            after["tasks"][0]["bindings"]["task_configuration_sha256"],
        )

        source = self.root / source_paths[-1]
        source.write_text("drifted relevant source\n", encoding="utf-8")
        source_drifted = self._plan()
        self.assertNotEqual(
            before["inputs"]["sources"]["lock_sha256"],
            source_drifted["inputs"]["sources"]["lock_sha256"],
        )

    def test_python_executable_must_be_absolute_or_current_placeholder(self) -> None:
        changed = copy.deepcopy(self.configuration)
        changed["python_executable"] = "python; unexpected-shell-command"
        self._write_configuration(changed)
        with self.assertRaisesRegex(ValueError, "python_executable must be an absolute"):
            self._plan()

        changed["python_executable"] = planner.CURRENT_PYTHON_PLACEHOLDER
        self._write_configuration(changed)
        task = self._plan()["tasks"][0]
        self.assertEqual(task["command_argv"][0], str(Path(sys.executable).absolute()))

    def test_python_executable_preserves_virtual_environment_symlink(self) -> None:
        venv_python = self.root / "artifacts/private/venvs/test/bin/python"
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        try:
            venv_python.symlink_to(Path(sys.executable))
        except OSError as exc:
            self.skipTest(f"file symlinks are unavailable on this platform: {exc}")

        changed = copy.deepcopy(self.configuration)
        changed["python_executable"] = str(venv_python.absolute())
        self._write_configuration(changed)
        plan = self._plan()

        task = plan["tasks"][0]
        bindings = task["bindings"]
        self.assertEqual(task["command_argv"][0], str(venv_python.absolute()))
        self.assertEqual(bindings["python_executable_path"], str(venv_python.absolute()))
        self.assertEqual(bindings["python_executable_sha256"], _sha256(venv_python))

    def test_private_output_root_cannot_leave_private_artifact_root(self) -> None:
        changed = copy.deepcopy(self.configuration)
        changed["output_root"] = "results/public-looking"
        self._write_configuration(changed)
        with self.assertRaisesRegex(ValueError, "private artifact output_root"):
            self._plan()

    def test_command_argv_can_be_omitted_without_losing_its_hash(self) -> None:
        changed = copy.deepcopy(self.configuration)
        changed["include_command_argv"] = False
        self._write_configuration(changed)
        plan = self._plan()
        self.assertEqual(plan["planner"]["command_representation"], "sha256-only")
        self.assertTrue(
            all(
                "command_argv" not in task and len(task["command_argv_sha256"]) == 64
                for task in plan["tasks"]
            )
        )

    def test_rejects_missing_forged_or_changed_model_snapshot(self) -> None:
        missing = copy.deepcopy(self.configuration)
        missing["model_snapshot_manifest"] = "artifacts/private/cache/manifests/missing.json"
        self._write_configuration(missing)
        with self.assertRaisesRegex(FileNotFoundError, "model_snapshot_manifest"):
            self._plan()

        self._write_configuration(self.configuration)
        manifest_path = self.root / self.configuration["model_snapshot_manifest"]
        original_manifest = manifest_path.read_bytes()
        forged = json.loads(original_manifest)
        forged["metadata"]["repo_id"] = "forged/model"
        manifest_path.write_text(json.dumps(forged), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "repo_id mismatch"):
            self._plan()

        manifest_path.write_bytes(original_manifest)
        model_file = self.root / self.configuration["model_snapshot_root"] / "file-00.bin"
        model_file.write_bytes(b"changed model bytes\n")
        with self.assertRaisesRegex(ValueError, "model snapshot byte mismatch"):
            self._plan()

    def test_rejects_host_or_warning_readiness_report(self) -> None:
        readiness_path = self.root / self.configuration["readiness_report"]
        original = readiness_path.read_bytes()
        host = json.loads(original)
        host["phase"] = "host"
        readiness_path.write_text(json.dumps(host), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "full-phase"):
            self._plan()

        warning = json.loads(original)
        warning["overall"]["status"] = "pass-with-warnings"
        warning["overall"]["warning_gates"] = ["topology"]
        warning["gates"][0]["status"] = "warn"
        readiness_path.write_text(json.dumps(warning), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "recomputed evidence"):
            self._plan()

    def test_rejects_dirty_git_and_zero_warmup_for_formal_plan(self) -> None:
        with self.assertRaisesRegex(ValueError, "clean Git"):
            planner.plan_from_config_file(
                self.config_path,
                repository_root=self.root,
                repository_state={"commit": "c" * 40, "dirty": True},
            )

        zero = copy.deepcopy(self.configuration)
        zero["warmup"] = 0
        self._write_configuration(zero)
        with self.assertRaisesRegex(ValueError, "warmup"):
            self._plan()

    def test_runner_root_detection_fails_fast_outside_source_checkout(self) -> None:
        with self.assertRaisesRegex(ValueError, "clean source checkout"):
            benchmark._resolve_repository_root(self.root)

    def test_model_load_helpers_are_local_only_without_network_fallback(self) -> None:
        calls: list[tuple[str, str, dict]] = []
        transformers = types.ModuleType("transformers")

        class Loader:
            @classmethod
            def from_pretrained(cls, location: str, **kwargs: object) -> object:
                calls.append((cls.__name__, location, dict(kwargs)))
                return object()

        transformers.AutoProcessor = type("AutoProcessor", (Loader,), {})
        transformers.GlmAsrForConditionalGeneration = type(
            "GlmAsrForConditionalGeneration", (Loader,), {}
        )
        fake_torch = types.SimpleNamespace(float32="fp32")
        snapshot_root = self.root / self.configuration["model_snapshot_root"]
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            benchmark._load_hf_snapshot(snapshot_root, fake_torch)
        self.assertEqual(len(calls), 2)
        self.assertTrue(
            all(
                location == str(snapshot_root)
                and kwargs.get("local_files_only") is True
                and kwargs.get("trust_remote_code") is False
                for _, location, kwargs in calls
            )
        )
        model_call = next(kwargs for name, _, kwargs in calls if name.startswith("GlmAsr"))
        self.assertIs(model_call["use_safetensors"], True)
        self.assertEqual(model_call["dtype"], "fp32")

        custom_calls = []
        modeling = types.ModuleType("faster_glm_asr.modeling")
        modeling.__path__ = []
        weight_loading = types.ModuleType("faster_glm_asr.modeling.weight_loading")

        def fake_load_model_from_hf(**kwargs: object) -> tuple[object, object]:
            custom_calls.append(dict(kwargs))
            return object(), object()

        weight_loading.load_model_from_hf = fake_load_model_from_hf
        with mock.patch.dict(
            sys.modules,
            {
                "faster_glm_asr.modeling": modeling,
                "faster_glm_asr.modeling.weight_loading": weight_loading,
            },
        ):
            benchmark._load_custom_snapshot(snapshot_root, "a" * 40)
        self.assertEqual(
            custom_calls,
            [
                {
                    "model_name": str(snapshot_root),
                    "revision": "a" * 40,
                    "local_files_only": True,
                }
            ],
        )

    def test_atomic_publish_and_task_outputs_refuse_overwrite(self) -> None:
        plan = self._plan()
        output = self.root / "artifacts/private/plans/plan.json"
        planner._write_json_atomically(output, plan)
        original = output.read_bytes()
        with self.assertRaises(FileExistsError):
            planner._write_json_atomically(output, {"replacement": True})
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

        planned_output = self.root / plan["tasks"][0]["output_path"]
        planned_output.parent.mkdir(parents=True, exist_ok=True)
        planned_output.write_text("existing evidence\n", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "planned benchmark output"):
            self._plan()


class BenchmarkPlanRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()
        (self.root / "configs").mkdir()
        (self.root / "artifacts/private/manifests").mkdir(parents=True)
        (self.root / "artifacts/private/environment").mkdir(parents=True)
        _materialize_relevant_sources(self.root)
        self.model_binding, self.readiness_binding = _materialize_formal_provenance(
            self.root,
            repo_id="fixture/model",
            revision="a" * 40,
        )
        self.fake_runner = self.root / "src/faster_glm_asr/benchmarking/runner.py"
        runner_source = textwrap.dedent(
            """\
            from __future__ import annotations
            import argparse
            import hashlib
            import json
            import sys
            from pathlib import Path

            RELEVANT = __RELEVANT__
            RUNNER_CONFIGURATIONS = {
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
                    "component_dispatch_policy": "supported-norm-rope-embedding-conv-auto-triton-else-torch",
                    "attention_dispatch_policy": "dense-triton-if-padded-and-head-dim-lte-256-else-torch-dense",
                    "gqa_kv_expansion": "explicit-expand",
                    "long_form_attention_expectation": "torch-dense-likely-for-30s-input",
                    "backend_trace_status": "policy-inferred-diagnostic-trace-required-for-observed-backend",
                    "performance_attribution": "cache-strategy-only-no-triton-kernel-attribution",
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
                    "component_dispatch_policy": "supported-norm-rope-embedding-conv-auto-triton-else-torch",
                    "attention_dispatch_policy": "dense-triton-if-padded-and-head-dim-lte-256-else-torch-dense",
                    "gqa_kv_expansion": "explicit-expand",
                    "long_form_attention_expectation": "torch-dense-likely-for-30s-input",
                    "backend_trace_status": "policy-inferred-diagnostic-trace-required-for-observed-backend",
                    "performance_attribution": "cache-strategy-only-no-triton-kernel-attribution",
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
                    "component_dispatch_policy": "supported-norm-rope-embedding-conv-auto-triton-else-torch",
                    "attention_dispatch_policy": "dense-triton-if-padded-and-head-dim-lte-256-else-torch-dense",
                    "gqa_kv_expansion": "explicit-expand",
                    "long_form_attention_expectation": "torch-dense-likely-for-30s-input",
                    "backend_trace_status": "policy-inferred-diagnostic-trace-required-for-observed-backend",
                    "performance_attribution": "cache-strategy-only-no-triton-kernel-attribution",
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
                    "component_dispatch_policy": "supported-norm-rope-embedding-conv-auto-triton-else-torch",
                    "attention_dispatch_policy": "dense-triton-if-padded-and-head-dim-lte-256-else-torch-dense",
                    "gqa_kv_expansion": "explicit-expand",
                    "long_form_attention_expectation": "torch-dense-likely-for-30s-input",
                    "backend_trace_status": "policy-inferred-diagnostic-trace-required-for-observed-backend",
                    "performance_attribution": "cache-strategy-only-no-triton-kernel-attribution",
                    "cache_mode": "static-preallocated",
                    "greedy_fast_path": True,
                    "token_output_mode": "preallocated",
                },
            }
            PACKAGE_NAMES = [
                "torch", "triton", "transformers", "accelerate",
                "huggingface-hub", "safetensors", "numpy", "scipy",
                "soundfile", "nvidia-ml-py",
            ]

            def sha_file(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            def summary(values):
                value = float(values[0])
                return {
                    "n": len(values), "mean": value, "std": 0.0,
                    "min": value, "p50": value, "p95": value, "max": value,
                }

            parser = argparse.ArgumentParser()
            parser.add_argument("--repository-root", required=True)
            parser.add_argument("--formal", action="store_true")
            parser.add_argument("--output", required=True)
            parser.add_argument("--implementation", required=True)
            parser.add_argument("--artifact-scope", required=True)
            parser.add_argument("--manifest", required=True)
            parser.add_argument("--environment-lock", required=True)
            parser.add_argument("--model-snapshot-root", required=True)
            parser.add_argument("--model-snapshot-manifest", required=True)
            parser.add_argument("--readiness-report", required=True)
            parser.add_argument("--model-name", required=True)
            parser.add_argument("--model-revision", required=True)
            parser.add_argument("--warmup", type=int, required=True)
            parser.add_argument("--repeats", type=int, required=True)
            parser.add_argument("--max-new-tokens", type=int, required=True)
            parser.add_argument("--phase-timing", action="store_true")
            parser.add_argument("--nvml-sample-ms", type=float, default=0.0)
            args = parser.parse_args()
            root = Path.cwd()
            output = Path(args.output)
            execution_log = root / "artifacts/private/fake-execution.jsonl"
            execution_log.parent.mkdir(parents=True, exist_ok=True)
            with execution_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "output": output.as_posix(),
                    "implementation": args.implementation,
                }) + "\\n")
            fail_once = root / "artifacts/private/fail-once-order-003"
            if fail_once.is_file() and output.name.startswith("003-"):
                fail_once.unlink()
                print("intentional fixture failure", file=sys.stderr)
                raise SystemExit(7)

            output.parent.mkdir(parents=True, exist_ok=True)
            if (root / "artifacts/private/emit-minimal-result").is_file():
                output.write_text(json.dumps({
                    "implementation": args.implementation,
                    "phase_timing": args.phase_timing,
                    "nvml_sample_ms": args.nvml_sample_ms,
                }) + "\\n", encoding="utf-8")
                raise SystemExit(0)

            profile = (
                "diagnostic_phase" if args.phase_timing else
                "request_memory" if args.nvml_sample_ms > 0 else
                "canonical_latency"
            )
            token_ids = [1]
            token_hash = hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            hypothesis = "fixture"
            hypothesis_hash = hashlib.sha256(hypothesis.encode("utf-8")).hexdigest()
            request_memory = profile == "request_memory"
            diagnostic_phase = profile == "diagnostic_phase"
            nvml = {
                "enabled": request_memory,
                "available": request_memory,
                "interval_ms": args.nvml_sample_ms,
                "sample_count": 1 if request_memory else 0,
                "reason": None if request_memory else "disabled",
                "samples": ([{"elapsed_ms": 1.0, "used_bytes": 1024}]
                            if request_memory else []),
            }
            if request_memory:
                nvml["observed_peak_used_bytes"] = 1024
            phase_evidence = ({
                "decode": {
                    "events_ms": [1.0],
                    "summary_ms": summary([1.0]),
                    "total_ms": 1.0,
                }
            } if diagnostic_phase else None)
            observation = {
                "repeat": 0,
                "preprocess_ms": 1.0,
                "generation_ms": 2.0,
                "decode_ms": 1.0,
                "processor_plus_generation_ms": 3.0,
                "request_no_file_io_ms": 4.0,
                "generated_tokens": 1,
                "hypothesis_sha256": hypothesis_hash,
                "generated_token_ids_sha256": token_hash,
                "rtf_generation": 0.002,
                "rtf_request_no_file_io": 0.004,
                "memory_bytes": {"generation_allocated_peak_delta": 0},
                "nvml_process_memory": nvml,
                "diagnostic_cuda_phases": phase_evidence,
            }
            sample = {
                "sample_id": "fixture",
                "audio_sha256": "e" * 64,
                "language": "en",
                "duration_s": 1.0,
                "audio_load_ms": 1.0,
                "prepared_request": {
                    "tensors": {
                        "input_ids": {
                            "shape": [1, 1], "dtype": "torch.int64",
                            "device": "cuda:0", "content_sha256": "1" * 64,
                        },
                        "input_features": {
                            "shape": [1, 1, 1], "dtype": "torch.float32",
                            "device": "cuda:0", "content_sha256": "2" * 64,
                        },
                    },
                    "request_batch": 1,
                    "prompt_tokens": 1,
                    "audio_placeholder_tokens": 1,
                    "audio_windows": 1,
                    "valid_feature_frames": None,
                },
                "metadata": {"shareable": False},
                "hypothesis": hypothesis,
                "hypothesis_sha256": hypothesis_hash,
                "generated_tokens": 1,
                "generated_token_ids": token_ids,
                "generated_token_ids_sha256": token_hash,
                "quality": None,
                "summary": {
                    "generation_ms": summary([2.0]),
                    "processor_plus_generation_ms": summary([3.0]),
                    "request_no_file_io_ms": summary([4.0]),
                    "rtf_generation": summary([0.002]),
                },
                "observations": [observation],
            }
            files = {
                relative: {
                    "size_bytes": (root / relative).stat().st_size,
                    "sha256": sha_file(root / relative),
                }
                for relative in RELEVANT
            }
            source_aggregate = hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            payload = {
                "schema_version": "0.3",
                "created_at_utc": "2026-08-15T00:00:00+00:00",
                "run": {
                    "implementation": args.implementation,
                    "model_name": args.model_name,
                    "model_revision": args.model_revision,
                    "manifest_path": str(Path(args.manifest).resolve()),
                    "manifest_sha256": sha_file(Path(args.manifest)),
                    "formal_run": args.formal,
                    "comparability": {"status": "formal-comparable", "reasons": []},
                    "warmup_gate": {
                        "status": "pass",
                        "minimum_iterations": 1,
                        "observed_iterations": args.warmup,
                        "custom_lazy_h2d_applicable": args.implementation.startswith("custom_"),
                        "custom_lazy_h2d_covered": (True if args.implementation.startswith("custom_") else None),
                    },
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "max_new_tokens": args.max_new_tokens,
                    "runner_construction_envelope_ms": 1.0,
                    "runner_construction_scope": "fixture",
                    "artifact_scope": args.artifact_scope,
                    "measurement_profile": profile,
                    "metric_eligibility": {
                        "canonical_latency": profile == "canonical_latency",
                        "request_memory_observed_peak": request_memory,
                        "diagnostic_cuda_phases": diagnostic_phase,
                        "cold_start": False,
                        "model_load": False,
                    },
                    "nvml_sample_ms": args.nvml_sample_ms,
                    "phase_timing": args.phase_timing,
                    "phase_timing_warning": ("diagnostic overhead" if diagnostic_phase else None),
                    "runner_configuration": RUNNER_CONFIGURATIONS[args.implementation],
                },
                "git": {"commit": "d" * 40, "branch": "fixture", "dirty": False},
                "sources": {
                    "schema_version": "faster-glm-asr-source-map-v1",
                    "files": files,
                    "aggregate_sha256": source_aggregate,
                },
                "model_snapshot": __MODEL_BINDING__,
                "readiness": __READINESS_BINDING__,
                "environment": {
                    "python": "3.11.0", "platform": "fixture",
                    "torch": "2.10.0", "torch_cuda": "12.8", "cudnn": "9",
                    "device_index": 0, "device_name": "synthetic-gpu", "device_uuid": "GPU-fixture",
                    "cuda_visible_devices": None, "compute_capability": [9, 0],
                    "total_memory_bytes": 1, "allow_tf32_matmul": False,
                    "allow_tf32_cudnn": False, "nvidia_driver_versions": ["fixture"],
                    "package_versions": {name: "fixture" for name in PACKAGE_NAMES},
                    "environment_lock_sha256": sha_file(Path(args.environment_lock)),
                },
                "aggregate": {
                    "generation_ms": summary([2.0]),
                    "processor_plus_generation_ms": summary([3.0]),
                    "request_no_file_io_ms": summary([4.0]),
                    "quality": {"status": "not_evaluated", "evaluated_samples": 0},
                },
                "samples": [sample],
            }
            output.write_text(json.dumps(payload) + "\\n", encoding="utf-8")
            drift = root / "artifacts/private/drift-relevant-source-after-output"
            if drift.is_file():
                (root / RELEVANT[-1]).write_text("post-output drift\\n", encoding="utf-8")
            """
        )
        runner_source = (
            runner_source.replace(
                "__RELEVANT__", repr(list(benchmark.discover_source_paths(self.root)))
            )
            .replace("__MODEL_BINDING__", repr(self.model_binding))
            .replace("__READINESS_BINDING__", repr(self.readiness_binding))
        )
        self.fake_runner.write_text(runner_source, encoding="utf-8")
        self.manifest = self.root / "artifacts/private/manifests/domain.jsonl"
        self.environment_lock = self.root / "artifacts/private/environment/freeze.txt"
        self.manifest.write_text('{"sample_id":"fixture"}\n', encoding="utf-8")
        self.environment_lock.write_text("torch==2.10.0\n", encoding="utf-8")
        self.configuration = {
            "schema_version": planner.CONFIG_SCHEMA,
            "python_executable": sys.executable,
            "benchmark_script": "src/faster_glm_asr/benchmarking/runner.py",
            "manifest": "artifacts/private/manifests/domain.jsonl",
            "environment_lock": "artifacts/private/environment/freeze.txt",
            "model_snapshot_root": ("artifacts/private/cache/model/glm-asr-nano-2512"),
            "model_snapshot_manifest": "artifacts/private/cache/manifests/model.json",
            "readiness_report": "artifacts/private/rtx3090-readiness/full.json",
            "output_root": "artifacts/private/results/matrix/domain",
            "dataset_label": "domain",
            "artifact_scope": "private",
            "model_name": "fixture/model",
            "model_revision": "a" * 40,
            "implementations": list(planner.IMPLEMENTATIONS),
            "profiles": list(planner.PROFILES),
            "outer_passes": 3,
            "seed": 20260815,
            "warmup": 1,
            "repeats": 1,
            "max_new_tokens": 1,
            "nvml_sample_ms": 10.0,
            "include_command_argv": True,
        }
        self.config_path = self.root / "configs/matrix.json"
        self.config_path.write_text(
            json.dumps(self.configuration, indent=2) + "\n", encoding="utf-8"
        )
        self.repository_state = {"commit": "d" * 40, "dirty": False}
        self.plan = planner.plan_from_config_file(
            self.config_path,
            repository_root=self.root,
            repository_state=self.repository_state,
        )
        self.plan_path = self.root / "artifacts/private/plans/domain-plan.json"
        planner._write_json_atomically(self.plan_path, self.plan)
        self.journal_dir = Path("artifacts/private/run-journals/domain-plan")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(
        self,
        *,
        execute: bool,
        resume: bool = False,
        lock_file: Path | None = None,
    ) -> dict:
        return executor.run_plan(
            self.plan_path,
            repository_root=self.root,
            execute=execute,
            resume=resume,
            journal_dir=self.journal_dir,
            repository_state=self.repository_state,
            lock_file=lock_file,
        )

    def _execution_outputs(self) -> list[str]:
        path = self.root / "artifacts/private/fake-execution.jsonl"
        if not path.is_file():
            return []
        return [
            json.loads(line)["output"] for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def _replace_plan(self, payload: object) -> None:
        self.plan_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def test_default_only_validates_and_never_executes(self) -> None:
        result = self._run(execute=False)
        self.assertEqual(result["status"], "validated")
        self.assertFalse(result["executed"])
        self.assertFalse(result["pass_merge_performed"])
        self.assertEqual(self._execution_outputs(), [])
        self.assertFalse((self.root / self.journal_dir).exists())
        self.assertTrue(
            all(not (self.root / task["output_path"]).exists() for task in self.plan["tasks"])
        )

    def test_run_lock_refuses_concurrent_and_preexisting_lock_files(self) -> None:
        lock_relative = Path("artifacts/private/run-locks/explicit-test.lock")
        _, lock_path = executor._validate_private_lock_path(lock_relative, self.root)
        plan_sha256 = hashlib.sha256(self.plan_path.read_bytes()).hexdigest()
        held = executor._acquire_run_lock(
            lock_path,
            plan_sha256=plan_sha256,
        )
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["plan_sha256"], plan_sha256)
            self.assertIn("pid", payload)
            self.assertIn("host", payload)
            with self.assertRaisesRegex(FileExistsError, "run lock already exists"):
                self._run(execute=False, lock_file=lock_relative)
        finally:
            held.release()
        self.assertFalse(lock_path.exists())
        self.assertEqual(self._run(execute=False, lock_file=lock_relative)["status"], "validated")
        self.assertFalse(lock_path.exists())

        lock_path.write_text("operator-owned stale lock\n", encoding="utf-8")
        before = lock_path.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "run lock already exists"):
            self._run(execute=False, lock_file=lock_relative)
        self.assertEqual(lock_path.read_bytes(), before)

    def test_exit_zero_with_three_field_output_is_rejected(self) -> None:
        (self.root / "artifacts/private/emit-minimal-result").write_text(
            "emit invalid result\n", encoding="utf-8"
        )
        result = self._run(execute=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["task_status"], "output_validation_failed")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["tasks_executed_this_invocation"], 1)
        journal = json.loads(
            (self.root / self.journal_dir / "journal.json").read_text(encoding="utf-8")
        )
        self.assertEqual(journal["events"][-1]["status"], "output_validation_failed")
        self.assertIn("expected schema", journal["events"][-1]["error"])

    def test_relevant_source_drift_after_output_fails_task(self) -> None:
        (self.root / "artifacts/private/drift-relevant-source-after-output").write_text(
            "drift once\n", encoding="utf-8"
        )
        result = self._run(execute=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["task_status"], "output_validation_failed")
        journal = json.loads(
            (self.root / self.journal_dir / "journal.json").read_text(encoding="utf-8")
        )
        self.assertIn("binding drift", journal["events"][-1]["error"])

    def test_execute_uses_exact_serial_task_order_and_internal_hash_linkage(self) -> None:
        result = self._run(execute=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["tasks_executed_this_invocation"], 54)
        self.assertEqual(
            self._execution_outputs(),
            [task["output_path"] for task in self.plan["tasks"]],
        )
        journal_path = self.root / self.journal_dir / "journal.json"
        journal, successes, attempts = executor._validate_journal(
            journal_path,
            executor.validate_plan(
                self.plan_path,
                repository_root=self.root,
                repository_state=self.repository_state,
            ),
            self.root / self.journal_dir,
            self.root,
        )
        self.assertEqual(len(successes), 54)
        self.assertTrue(all(attempt == 1 for attempt in attempts.values()))
        self.assertEqual(len(journal["events"]), 54 * 3)
        self.assertTrue(
            all(
                event["monotonic_duration_ns"] >= 0
                and event["exit_code"] == 0
                and len(event["stdout_sha256"]) == 64
                and len(event["stderr_sha256"]) == 64
                for event in journal["events"]
                if event["event"] == "end"
            )
        )

    def test_failure_stops_then_resume_skips_only_hash_valid_successes(self) -> None:
        (self.root / "artifacts/private/fail-once-order-003").write_text(
            "fail exactly once\n", encoding="utf-8"
        )
        failed = self._run(execute=True)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["exit_code"], 7)
        self.assertEqual(failed["tasks_executed_this_invocation"], 3)
        expected = [task["output_path"] for task in self.plan["tasks"]]
        self.assertEqual(self._execution_outputs(), expected[:3])
        self.assertTrue((self.root / expected[0]).is_file())
        self.assertTrue((self.root / expected[1]).is_file())
        self.assertFalse((self.root / expected[2]).exists())

        resumed = self._run(execute=True, resume=True)
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(resumed["tasks_skipped_from_valid_journal"], 2)
        self.assertEqual(resumed["tasks_executed_this_invocation"], 52)
        self.assertEqual(self._execution_outputs(), expected[:3] + expected[2:])
        self.assertEqual(self._execution_outputs().count(expected[0]), 1)
        self.assertEqual(self._execution_outputs().count(expected[1]), 1)
        self.assertEqual(self._execution_outputs().count(expected[2]), 2)

        journal = json.loads(
            (self.root / self.journal_dir / "journal.json").read_text(encoding="utf-8")
        )
        first_end = next(event for event in journal["events"] if event["event"] == "end")
        stdout_path = self.root / first_end["stdout_path"]
        original_stdout = stdout_path.read_bytes()
        stdout_path.write_bytes(b"tampered stdout\n")
        with self.assertRaisesRegex(ValueError, "stdout_path hash drift"):
            self._run(execute=True, resume=True)
        stdout_path.write_bytes(original_stdout)

        successful_output = self.root / expected[0]
        successful_output.write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "output hash drift"):
            self._run(execute=True, resume=True)

    def test_rejects_binding_task_hash_duplicate_and_path_tampering(self) -> None:
        original = copy.deepcopy(self.plan)

        tampered = copy.deepcopy(original)
        tampered["tasks"][0]["command_argv"][0] = "different-python"
        self._replace_plan(tampered)
        with self.assertRaisesRegex(ValueError, "argv/task hash"):
            self._run(execute=False)

        tampered = copy.deepcopy(original)
        tampered["tasks"][1] = copy.deepcopy(tampered["tasks"][0])
        self._replace_plan(tampered)
        with self.assertRaisesRegex(ValueError, "task .* drift|duplicate"):
            self._run(execute=False)

        tampered = copy.deepcopy(original)
        tampered["tasks"][0]["output_path"] = "../../outside.json"
        self._replace_plan(tampered)
        with self.assertRaisesRegex(ValueError, "cannot escape"):
            self._run(execute=False)

        self._replace_plan(original)
        self.manifest.write_text('{"sample_id":"drift"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "binding drift"):
            self._run(execute=False)

    def test_refuses_existing_output_and_existing_journal_without_resume(self) -> None:
        first_output = self.root / self.plan["tasks"][0]["output_path"]
        first_output.parent.mkdir(parents=True, exist_ok=True)
        first_output.write_text("preexisting\n", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            self._run(execute=False)

        first_output.unlink()
        (self.root / "artifacts/private/fail-once-order-003").write_text(
            "fail exactly once\n", encoding="utf-8"
        )
        self.assertEqual(self._run(execute=True)["status"], "failed")
        journal_path = self.root / self.journal_dir / "journal.json"
        journal_before = journal_path.read_bytes()
        with self.assertRaises((FileExistsError, ValueError)):
            self._run(execute=True, resume=False)
        self.assertEqual(journal_path.read_bytes(), journal_before)

    def test_journal_rejects_two_simultaneous_active_starts(self) -> None:
        (self.root / "artifacts/private/fail-once-order-003").write_text(
            "fail exactly once\n", encoding="utf-8"
        )
        self.assertEqual(self._run(execute=True)["status"], "failed")
        validated = executor.validate_plan(
            self.plan_path,
            repository_root=self.root,
            repository_state=self.repository_state,
        )
        journal_path = self.root / self.journal_dir / "journal.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        pending = self.plan["tasks"][2]
        for attempt in (2, 3):
            executor._append_event(
                journal_path,
                journal,
                {
                    "event": "start",
                    "task_id": pending["task_id"],
                    "timestamp_utc": executor._utc_now(),
                    "attempt": attempt,
                    "command_argv_sha256": pending["command_argv_sha256"],
                    "stdout_path": (
                        self.journal_dir / "logs" / f"active-{attempt}.stdout.txt"
                    ).as_posix(),
                    "stderr_path": (
                        self.journal_dir / "logs" / f"active-{attempt}.stderr.txt"
                    ).as_posix(),
                },
            )
        with self.assertRaisesRegex(ValueError, "more than one active start"):
            executor._validate_journal(
                journal_path,
                validated,
                self.root / self.journal_dir,
                self.root,
            )


if __name__ == "__main__":
    unittest.main()
