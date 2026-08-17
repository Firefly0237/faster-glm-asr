from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from experiments.distributed_training.release import (
    _sha256_file,
    _write_json_atomic,
    aggregate_matrix,
    aggregate_metrics,
    build_matrix_plan,
    build_source_lock,
    compare_trajectories,
    parameter_count_from_model_config,
    validate_source_lock,
    validate_strong_scaling_matrix,
    validate_training_config,
)
from tests.helpers.readiness import (
    materialize_passing_readiness,
    materialize_passing_training_preflight,
)
from tools.run_training_matrix import (
    _validate_evidence_pair,
    _validate_readiness_report,
    _validate_training_preflight,
)

if torch is not None:
    from experiments.distributed_training.train import main as training_main


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPOSITORY_ROOT / "configs" / "training"


def _private_temporary_directory() -> tempfile.TemporaryDirectory[str]:
    parent = REPOSITORY_ROOT / "artifacts" / "private" / "test-training-release"
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _build_matrix_fixture(root: Path) -> dict[str, object]:
    evidence_dir = root / "evidence"
    evidence_dir.mkdir(parents=True)
    readiness = evidence_dir / "full.json"
    preflight = evidence_dir / "training-preflight.json"
    readiness.write_text('{"kind":"full-readiness"}\n', encoding="utf-8")
    preflight.write_text('{"kind":"training-nccl"}\n', encoding="utf-8")
    evidence = {
        "full_readiness_report": {
            "path": str(readiness.resolve()),
            "sha256": _sha256_file(readiness),
            "status": "present",
        },
        "training_nccl_preflight": {
            "path": str(preflight.resolve()),
            "sha256": _sha256_file(preflight),
            "status": "present",
        },
    }

    source_lock = root / "source-lock.json"
    build_source_lock(
        source_lock,
        [
            CONFIG_DIR / name
            for name in (
                "strong-403m-w1.json",
                "strong-403m-w2.json",
                "strong-403m-w4.json",
            )
        ],
        CONFIG_DIR / "synthetic-data-manifest.json",
        REPOSITORY_ROOT / "configs" / "environments" / "rtx3090-ddp-v1.json",
        [
            (readiness, "readiness_report"),
            (preflight, "training_preflight"),
        ],
    )
    plan = build_matrix_plan(CONFIG_DIR, root, source_lock)
    plan["required_evidence"] = evidence
    plan_path = root / "strong-scaling" / "plan.json"
    _write_json_atomic(plan_path, plan)

    journal_runs = []
    for run in plan["runs"]:
        run_dir = Path(run["output_dir"])
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "checkpoint.pt"
        checkpoint.write_bytes(f"checkpoint:{run['name']}".encode())
        checkpoint_sha256 = _sha256_file(checkpoint)
        config = validate_training_config(Path(run["config"]))
        world = int(run["world_size"])
        run_id = f"run-{run['name']}"
        records: list[dict[str, object]] = [
            {
                "event": "run_start",
                "run_id": run_id,
                "config_path": config["path"],
                "config_sha256": config["sha256"],
                "world_size": world,
                "tokens_per_update": 32_768,
                "parameter_count": {"unique_trainable": 403_097_088},
                "start_step": 0,
                "resume_from_sha256": None,
                "model": config["document"]["model"],
                "training": config["document"]["training"],
                "data_fingerprint": "a" * 64,
                "data": {
                    "mode": "synthetic-smoke-only",
                    "token_id_contract": {
                        "model_vocab_size": 256,
                        "label_semantics": "synthetic smoke-only IDs",
                    },
                },
                "environment": {"rank": 0, "local_rank": 0},
                "rank_environments": [{"rank": rank, "local_rank": rank} for rank in range(world)],
                "source_lock": {
                    "path": str(source_lock.resolve()),
                    "sha256": _sha256_file(source_lock),
                },
                "torch_profiler": {"enabled": False},
            }
        ]
        for step in range(40):
            core_elapsed = 1.0 / world + int(run["pass"]) / 100.0 + step / 100_000.0
            records.append(
                {
                    "event": "train_update",
                    "run_id": run_id,
                    "step": step,
                    "measurement_eligible": step >= 10,
                    "canonical_timing": "core_update",
                    "critical_rank_core_update_elapsed_s": core_elapsed,
                    "critical_rank_guarded_update_elapsed_s": core_elapsed + 0.01,
                    "critical_rank_finite_guard_elapsed_s": 0.01,
                    "elapsed_s": core_elapsed,
                    "tokens": 32_768,
                    "tokens_per_second": 32_768 / core_elapsed,
                    "loss": 6.0 - step / 100.0,
                    "grad_norm": 1.0,
                }
            )
        records.extend(
            [
                {
                    "event": "checkpoint",
                    "run_id": run_id,
                    "step": 40,
                    "path": str(checkpoint.resolve()),
                    "sha256": checkpoint_sha256,
                },
                {
                    "event": "run_end",
                    "run_id": run_id,
                    "status": "complete",
                    "completed_steps": 40,
                    "measurement_eligible_updates": 30,
                },
            ]
        )
        _write_jsonl(run_dir / "metrics.jsonl", records)
        journal_runs.append(
            {
                "name": run["name"],
                "status": "complete",
                "command": run["command"],
            }
        )
    journal_path = root / "strong-scaling" / "journal.json"
    journal = {
        "schema_version": "decoder-training-matrix-journal/1",
        "status": "complete",
        "plan": {
            "path": str(plan_path.resolve()),
            "sha256": _sha256_file(plan_path),
        },
        "source_lock_sha256": _sha256_file(source_lock),
        "required_evidence": evidence,
        "runs": journal_runs,
    }
    _write_json_atomic(journal_path, journal)
    return {"plan": plan, "journal_path": journal_path}


class DecoderTrainingReleaseTest(unittest.TestCase):
    def test_parameter_counts_are_exact_for_primary_and_fallback(self) -> None:
        primary = validate_training_config(CONFIG_DIR / "strong-403m-w4.json")
        fallback = validate_training_config(CONFIG_DIR / "fallback-275m-w4.json")
        self.assertEqual(primary["parameter_count"], 403_097_088)
        self.assertEqual(fallback["parameter_count"], 275_621_120)
        self.assertEqual(
            parameter_count_from_model_config(primary["document"]["model"]),
            403_097_088,
        )
        self.assertEqual(
            parameter_count_from_model_config(fallback["document"]["model"]),
            275_621_120,
        )

    def test_strong_scaling_matrix_fixes_tokens_and_measurement_window(self) -> None:
        matrix = validate_strong_scaling_matrix(CONFIG_DIR)
        self.assertEqual([item["world_size"] for item in matrix], [1, 2, 4])
        self.assertEqual(
            [item["document"]["training"]["gradient_accumulation_steps"] for item in matrix],
            [16, 8, 4],
        )
        self.assertEqual({item["tokens_per_update"] for item in matrix}, {32_768})
        self.assertEqual({item["measurement_eligible_steps"] for item in matrix}, {30})
        for item in matrix:
            self.assertNotIn("torch_profiler", item["document"]["training"])

    def test_real_data_config_is_distinct_and_emits_periodic_validation(self) -> None:
        real_data = validate_training_config(CONFIG_DIR / "tinystories-275m-w4.json")
        training = real_data["document"]["training"]
        data = real_data["document"]["data"]
        self.assertEqual(real_data["data_mode"], "byte_files")
        self.assertEqual(real_data["world_size"], 4)
        self.assertEqual(real_data["parameter_count"], 275_621_120)
        self.assertGreaterEqual(training["steps"], 200)
        self.assertLess(training["eval_interval"], training["steps"])
        self.assertTrue(all(path.startswith("artifacts/private/") for path in data["train_files"]))
        self.assertTrue(
            all(path.startswith("artifacts/private/") for path in data["validation_files"])
        )

    def test_atomic_json_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            _write_json_atomic(path, {"value": 1})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})
            with self.assertRaises(FileExistsError):
                _write_json_atomic(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def test_source_lock_binds_code_config_data_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source-lock.json"
            build_source_lock(
                path,
                [CONFIG_DIR / "cpu-smoke.json"],
                CONFIG_DIR / "synthetic-data-manifest.json",
                REPOSITORY_ROOT / "configs" / "environments" / "rtx3090-ddp-v1.json",
            )
            validated = validate_source_lock(path)
            self.assertEqual(
                set(validated["roles"]),
                {"code", "config", "data_manifest", "environment"},
            )
            with self.assertRaises(FileExistsError):
                build_source_lock(
                    path,
                    [CONFIG_DIR / "cpu-smoke.json"],
                    CONFIG_DIR / "synthetic-data-manifest.json",
                    REPOSITORY_ROOT / "configs" / "environments" / "rtx3090-ddp-v1.json",
                )

    def test_aggregation_excludes_warmup_and_uses_critical_rank_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            records = [
                {
                    "event": "run_start",
                    "world_size": 4,
                    "parameter_count": {"unique_trainable": 403_097_088},
                    "tokens_per_update": 32_768,
                },
                {
                    "event": "train_update",
                    "step": 0,
                    "measurement_eligible": False,
                    "tokens_per_second": 1.0,
                    "elapsed_s": 10.0,
                },
                {
                    "event": "train_update",
                    "step": 1,
                    "measurement_eligible": True,
                    "tokens_per_second": 100.0,
                    "elapsed_s": 2.0,
                },
                {
                    "event": "train_update",
                    "step": 2,
                    "measurement_eligible": True,
                    "tokens_per_second": 200.0,
                    "elapsed_s": 1.0,
                },
                {"event": "run_end", "completed_steps": 3},
            ]
            (run / "metrics.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            summary = aggregate_metrics(run)
            self.assertEqual(summary["excluded_warmup_updates"], 1)
            self.assertEqual(summary["measurement_eligible_updates"], 2)
            self.assertEqual(summary["tokens_per_second"]["mean"], 150.0)

    def test_runner_validate_and_dry_run_need_no_torch_or_gpu(self) -> None:
        runner = REPOSITORY_ROOT / "tools" / "run_training_matrix.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "source-lock.json"
            validated = subprocess.run(
                [sys.executable, str(runner), "validate"],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)
            created = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "source-lock",
                    "--output",
                    str(lock),
                ],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            planned = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "matrix",
                    "--source-lock",
                    str(lock),
                    "--artifact-root",
                    str(root / "artifacts"),
                    "--dry-run",
                ],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(planned.returncode, 0, planned.stderr)
            plan = json.loads(planned.stdout)
            self.assertEqual(
                [run["world_size"] for run in plan["runs"]],
                [1, 2, 4, 4, 2, 1, 2, 4, 1],
            )
            self.assertEqual(plan["pass_orders"], [[1, 2, 4], [4, 2, 1], [2, 4, 1]])
            self.assertTrue(
                all("artifacts" in Path(run["output_dir"]).parts for run in plan["runs"])
            )

    def test_matrix_aggregate_requires_three_passes_and_ninety_samples(self) -> None:
        with _private_temporary_directory() as directory:
            root = Path(directory)
            orders = ((1, 2, 4), (4, 2, 1), (2, 4, 1))
            _build_matrix_fixture(root)
            summary = aggregate_matrix(root)
            self.assertEqual(summary["schema_version"], "decoder-training-matrix-summary/2")
            self.assertEqual(summary["pass_orders"], [list(item) for item in orders])
            self.assertEqual(len(summary["passes"]), 3)
            self.assertEqual(
                [item["measurement_eligible_updates"] for item in summary["worlds"]],
                [90, 90, 90],
            )
            self.assertEqual(
                len(
                    {
                        run["raw_artifacts"]["metrics"]["sha256"]
                        for world in summary["worlds"]
                        for run in world["runs"]
                    }
                ),
                9,
            )

    def test_matrix_aggregate_rejects_reused_or_mislabeled_evidence(self) -> None:
        def copied_metrics(root: Path, fixture: dict[str, object]) -> None:
            runs = fixture["plan"]["runs"]
            source = Path(runs[0]["output_dir"]) / "metrics.jsonl"
            target = Path(runs[1]["output_dir"]) / "metrics.jsonl"
            shutil.copyfile(source, target)

        def wrong_eligibility(root: Path, fixture: dict[str, object]) -> None:
            del root
            path = Path(fixture["plan"]["runs"][0]["output_dir"]) / "metrics.jsonl"
            records = _read_jsonl(path)
            next(record for record in records if record.get("event") == "train_update")[
                "measurement_eligible"
            ] = True
            _write_jsonl(path, records)

        def duplicate_checkpoint(root: Path, fixture: dict[str, object]) -> None:
            del root
            world_one_runs = [run for run in fixture["plan"]["runs"] if run["world_size"] == 1]
            source = Path(world_one_runs[0]["output_dir"]) / "checkpoint.pt"
            target_dir = Path(world_one_runs[1]["output_dir"])
            target = target_dir / "checkpoint.pt"
            shutil.copyfile(source, target)
            records = _read_jsonl(target_dir / "metrics.jsonl")
            final_checkpoint = next(
                record
                for record in records
                if record.get("event") == "checkpoint" and record.get("step") == 40
            )
            final_checkpoint["path"] = str(target.resolve())
            final_checkpoint["sha256"] = _sha256_file(target)
            _write_jsonl(target_dir / "metrics.jsonl", records)

        def wrong_world(root: Path, fixture: dict[str, object]) -> None:
            del root
            path = Path(fixture["plan"]["runs"][0]["output_dir"]) / "metrics.jsonl"
            records = _read_jsonl(path)
            next(record for record in records if record.get("event") == "run_start")[
                "world_size"
            ] = 4
            _write_jsonl(path, records)

        def profiler_enabled(root: Path, fixture: dict[str, object]) -> None:
            del root
            path = Path(fixture["plan"]["runs"][0]["output_dir"]) / "metrics.jsonl"
            records = _read_jsonl(path)
            next(record for record in records if record.get("event") == "run_start")[
                "torch_profiler"
            ] = {"enabled": True}
            _write_jsonl(path, records)

        def failed_journal(root: Path, fixture: dict[str, object]) -> None:
            del root
            path = Path(fixture["journal_path"])
            journal = json.loads(path.read_text(encoding="utf-8"))
            journal["status"] = "failed"
            _write_json_atomic(path, journal, overwrite=True)

        def failed_run_end(root: Path, fixture: dict[str, object]) -> None:
            del root
            path = Path(fixture["plan"]["runs"][0]["output_dir"]) / "metrics.jsonl"
            records = _read_jsonl(path)
            next(record for record in records if record.get("event") == "run_end")["status"] = (
                "failed"
            )
            _write_jsonl(path, records)

        scenarios = {
            "copied metrics": copied_metrics,
            "duplicate checkpoint": duplicate_checkpoint,
            "wrong world": wrong_world,
            "wrong eligibility": wrong_eligibility,
            "profiler enabled": profiler_enabled,
            "failed journal": failed_journal,
            "failed run_end": failed_run_end,
        }
        for name, mutate in scenarios.items():
            with self.subTest(name=name), _private_temporary_directory() as directory:
                root = Path(directory)
                fixture = _build_matrix_fixture(root)
                mutate(root, fixture)
                with self.assertRaises(ValueError):
                    aggregate_matrix(root)

    def test_formal_trajectory_and_profiler_plans_are_four_rank_and_isolated(self) -> None:
        runner = REPOSITORY_ROOT / "tools" / "run_training_matrix.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "source-lock.json"
            created = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "source-lock",
                    "--output",
                    str(lock),
                ],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            trajectory = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "trajectory",
                    "--source-lock",
                    str(lock),
                    "--artifact-root",
                    str(root / "artifacts"),
                    "--dry-run",
                ],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(trajectory.returncode, 0, trajectory.stderr)
            trajectory_plan = json.loads(trajectory.stdout)
            self.assertEqual(trajectory_plan["world_size"], 4)
            self.assertEqual(len(trajectory_plan["commands"]), 3)
            self.assertIn("--stop-after-step", trajectory_plan["commands"][1])
            self.assertIn("--resume", trajectory_plan["commands"][2])
            profile = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "profile",
                    "--source-lock",
                    str(lock),
                    "--artifact-root",
                    str(root / "artifacts"),
                ],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(profile.returncode, 0, profile.stderr)
            profile_plan = json.loads(profile.stdout)
            self.assertFalse(profile_plan["canonical_scaling_aggregate_eligible"])
            self.assertIn("profiler", Path(profile_plan["trace_dir"]).parts)
            self.assertIn("--torch-profiler-dir", profile_plan["command"])

    def test_host_phase_or_forged_full_readiness_cannot_authorize_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host = root / "host.json"
            host.write_text(
                json.dumps(
                    {
                        "schema_version": "rtx3090-readiness-v1",
                        "phase": "host",
                        "overall": {
                            "status": "pass",
                            "training_ready": False,
                            "day_rental_eligible": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "pure-pass full readiness"):
                _validate_readiness_report(host)

            forged = root / "forged-full.json"
            forged.write_text(
                json.dumps(
                    {
                        "schema_version": "rtx3090-readiness-v1",
                        "phase": "full",
                        "overall": {
                            "status": "pass",
                            "ready": True,
                            "training_ready": True,
                            "day_rental_eligible": True,
                            "failed_gates": [],
                            "warning_gates": [],
                        },
                        "gates": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "pure-pass full readiness"):
                _validate_readiness_report(forged)

    def test_short_training_preflight_is_additional_not_sufficient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forged = root / "forged-short.json"
            forged.write_text(
                json.dumps(
                    {
                        "schema_version": "decoder-training-nccl-preflight/1",
                        "pass": True,
                        "world_size": 4,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "schema keys"):
                _validate_training_preflight(forged)

            readiness = materialize_passing_readiness(root)
            short = materialize_passing_training_preflight(root / "short.json")
            self.assertTrue(_validate_training_preflight(short)["pass"])
            self.assertEqual(_validate_evidence_pair(readiness, short)[0]["world_size"], 4)

            mismatched = json.loads(short.read_text(encoding="utf-8"))
            mismatched["rank_inventory"][0]["uuid"] = "GPU-different-host-0"
            mismatched["nvidia_smi_inventory"]["stdout"] = mismatched["nvidia_smi_inventory"][
                "stdout"
            ].replace("GPU-private-fixture-0", "GPU-different-host-0")
            short.write_text(json.dumps(mismatched), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different GPU UUIDs"):
                _validate_evidence_pair(readiness, short)

            with self.assertRaisesRegex(
                ValueError, "both --readiness-report and --preflight-report"
            ):
                _validate_evidence_pair(None, short)

    def test_training_preflight_rejects_swapped_uuid_pci_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = materialize_passing_training_preflight(Path(directory) / "preflight.json")
            document = json.loads(path.read_text(encoding="utf-8"))
            first = document["rank_inventory"][0]
            second = document["rank_inventory"][1]
            first["pci_bus_id"], second["pci_bus_id"] = (
                second["pci_bus_id"],
                first["pci_bus_id"],
            )
            path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match rank identities"):
                _validate_training_preflight(path)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_planned_stop_resume_matches_uninterrupted_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "uninterrupted"
            resumed = root / "interrupted-resumed"
            common = [
                "--config",
                str(CONFIG_DIR / "cpu-smoke.json"),
                "--device",
                "cpu",
            ]
            self.assertEqual(training_main([*common, "--output-dir", str(baseline)]), 0)
            self.assertEqual(
                training_main(
                    [
                        *common,
                        "--output-dir",
                        str(resumed),
                        "--stop-after-step",
                        "2",
                    ]
                ),
                0,
            )
            self.assertEqual(
                training_main(
                    [
                        *common,
                        "--output-dir",
                        str(resumed),
                        "--resume",
                        str(resumed / "checkpoint.pt"),
                    ]
                ),
                0,
            )
            report = compare_trajectories(baseline, resumed)
            self.assertTrue(report["pass"], report["failures"])
            self.assertEqual(report["compared_steps"], 4)
            baseline_records = _read_jsonl(baseline / "metrics.jsonl")
            updates = [record for record in baseline_records if record["event"] == "train_update"]
            self.assertEqual(len(updates), 4)
            for update in updates:
                self.assertEqual(update["canonical_timing"], "core_update")
                self.assertEqual(
                    update["elapsed_s"],
                    update["critical_rank_core_update_elapsed_s"],
                )
                self.assertGreaterEqual(
                    update["critical_rank_guarded_update_elapsed_s"],
                    update["critical_rank_core_update_elapsed_s"],
                )
                self.assertGreaterEqual(update["critical_rank_finite_guard_elapsed_s"], 0.0)
            resumed_records = _read_jsonl(resumed / "metrics.jsonl")
            resumed_starts = [
                record for record in resumed_records if record["event"] == "run_start"
            ]
            resumed_starts[1]["resume_from_sha256"] = "0" * 64
            _write_jsonl(resumed / "metrics.jsonl", resumed_records)
            tampered = compare_trajectories(baseline, resumed)
            self.assertFalse(tampered["pass"])
            self.assertTrue(any("exact boundary" in failure for failure in tampered["failures"]))
            with self.assertRaises(FileExistsError):
                training_main([*common, "--output-dir", str(baseline)])


if __name__ == "__main__":
    unittest.main()
