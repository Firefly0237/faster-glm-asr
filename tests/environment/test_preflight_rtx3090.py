from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import threading
import time
import unittest
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).parents[2] / "tools" / "preflight_rtx3090.py"
SPEC = importlib.util.spec_from_file_location("preflight_rtx3090", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class PreflightRtx3090Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract_path = Path(__file__).parents[2] / ("configs/environments/rtx3090-ddp-v1.json")
        cls.contract = preflight.load_contract(cls.contract_path)

    @staticmethod
    def _gpu(index: int) -> dict[str, object]:
        return {
            "index": index,
            "name": "NVIDIA GeForce RTX 3090",
            "uuid": f"GPU-fixture-exact-{index}",
            "memory_total_mib": 24576,
            "memory_used_mib": 100,
            "driver_version": "570.26.0",
            "pci_bus_id": f"00000000:{index + 1:02X}:00.0",
            "compute_capability": "8.6",
            "temperature_c": 42,
            "utilization_percent": 0,
            "pstate": "P8",
            "power_draw_w": 25.0,
            "power_limit_w": 350.0,
            "pcie_link": {
                "gen_current": 4,
                "gen_max": 4,
                "width_current": 16,
                "width_max": 16,
            },
        }

    @staticmethod
    def _write_fixture_wheel(
        path: Path,
        *,
        name: str,
        version: str,
        requires_dist: tuple[str, ...] = (),
        requires_python: str | None = None,
    ) -> None:
        dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
        fields = [
            "Metadata-Version: 2.1",
            f"Name: {name}",
            f"Version: {version}",
            *((f"Requires-Python: {requires_python}",) if requires_python else ()),
            *(f"Requires-Dist: {requirement}" for requirement in requires_dist),
            "",
            "",
        ]
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{dist_info}/METADATA", "\n".join(fields))

    def _passing_report(self) -> dict[str, object]:
        gpus = [self._gpu(index) for index in range(4)]
        branch = copy.deepcopy(self.contract["cuda_branches"][0])
        ranks = [
            {
                "rank": index,
                "bf16": {"ok": True},
                "stress": {"requested_seconds": 300, "iterations": 1},
            }
            for index in range(4)
        ]
        return {
            "host": {
                "os_release": {"ID": "ubuntu", "VERSION_ID": "22.04"},
                "architecture": "x86_64",
                "python": "3.11.9",
                "resources": {
                    "system_ram_gib": 224.0,
                    "disk_free_gib": 1000.0,
                    "dev_shm_gib": 16.0,
                    "memlock_soft_bytes": "unlimited",
                },
            },
            "hardware": {
                "gpu_probe_before": {
                    "ok": True,
                    "gpus": gpus,
                    "pcie_query": {
                        "complete": True,
                        "available_fields": list(preflight.PCIE_QUERY_FIELDS),
                        "unavailable_fields": [],
                        "attempts": {},
                    },
                },
                "gpu_probe_after": {
                    "ok": True,
                    "gpus": copy.deepcopy(gpus),
                    "pcie_query": {"complete": True},
                },
                "cpu_topology": {"ok": True, "numa_node_count": 2, "stdout": "fixture"},
                "topology": {"ok": True, "stdout": "GPU0 GPU1 GPU2 GPU3"},
                "topology_p2p": {
                    mode: {"ok": True, "stdout": "fixture", "stderr": ""}
                    for mode in ("p", "r", "w", "n")
                },
                "xid_before": {"available": True, "matches": []},
                "xid_after": {"available": True, "matches": []},
            },
            "environment": {
                "selected_cuda_branch": branch,
                "packages": copy.deepcopy(self.contract["packages"]),
                "torch_runtime": {
                    "available": True,
                    "cuda_available": True,
                    "device_count": 4,
                    "cuda_runtime": "12.8",
                    "bf16_supported": [True, True, True, True],
                },
            },
            "network": {"probes": []},
            "provenance": {
                "cache_verification": {
                    "wheels": {"ok": True},
                    "model": {"ok": True},
                },
                "fresh_offline_install": {
                    "attempted": True,
                    "ok": True,
                    "network_disabled": True,
                },
                "freeze": {
                    "before": {
                        "freeze_ok": True,
                        "pip_check_ok": True,
                        "sha256": "a" * 64,
                    },
                    "after": {
                        "freeze_ok": True,
                        "pip_check_ok": True,
                        "sha256": "a" * 64,
                    },
                },
            },
            "distributed": {
                "ok": True,
                "payload": {
                    "world_size": 4,
                    "backend": "nccl",
                    "ranks": ranks,
                    "all_reduce": {
                        "correct": True,
                        "bus_bandwidth_min_gb_per_s": 2.0,
                    },
                    "p2p": {"query_complete": True, "full_mesh": True},
                },
            },
            "thermal": {
                "errors": [],
                "gpus": [
                    {
                        "uuid": item["uuid"],
                        "sample_count": 300,
                        "temperature_max_c": 78,
                    }
                    for item in gpus
                ],
            },
        }

    def test_contract_has_fixed_core_versions_and_four_rank_nccl(self) -> None:
        self.assertEqual(
            {name: self.contract["packages"][name] for name in preflight.REQUIRED_CORE_VERSIONS},
            preflight.REQUIRED_CORE_VERSIONS,
        )
        self.assertEqual(self.contract["packages"]["tokenizers"], "0.22.2")
        self.assertEqual(self.contract["packages"]["huggingface-hub"], "1.5.0")
        self.assertEqual(self.contract["packages"]["safetensors"], "0.8.0")
        self.assertEqual(
            {
                name: self.contract["packages"][name]
                for name in preflight.REQUIRED_BUILD_TOOL_VERSIONS
            },
            preflight.REQUIRED_BUILD_TOOL_VERSIONS,
        )
        self.assertEqual(self.contract["distributed"]["backend"], "nccl")
        self.assertEqual(self.contract["distributed"]["world_size"], 4)
        self.assertEqual(
            self.contract["xid_health_evidence"]["provider_schema_version"],
            preflight.PROVIDER_HEALTH_SCHEMA,
        )
        self.assertGreaterEqual(self.contract["xid_health_evidence"]["lookback_seconds"], 900)

    def test_contract_rejects_version_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = copy.deepcopy(self.contract)
            candidate["packages"]["tokenizers"] = "0.22.3"
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps(candidate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tokenizers"):
                preflight.load_contract(path)

            candidate = copy.deepcopy(self.contract)
            candidate["packages"]["wheel"] = "0.46.4"
            path.write_text(json.dumps(candidate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "build-tool"):
                preflight.load_contract(path)

    def test_default_artifact_is_private_and_unique(self) -> None:
        with mock.patch.object(
            preflight.uuid,
            "uuid4",
            side_effect=(mock.Mock(hex="a" * 32), mock.Mock(hex="b" * 32)),
        ):
            first = preflight.default_output_path()
            second = preflight.default_output_path()
        self.assertTrue(first.is_relative_to(preflight.DEFAULT_ARTIFACT_ROOT))
        self.assertEqual(first.name, "report.json")
        self.assertNotEqual(first.parent, second.parent)
        self.assertTrue(first.parent.name.endswith("-" + "a" * 32))
        self.assertTrue(second.parent.name.endswith("-" + "b" * 32))

    def test_gpu_csv_preserves_exact_uuid_and_rejects_bad_shape(self) -> None:
        exact_uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
        row = (
            "0, NVIDIA GeForce RTX 3090, "
            f"{exact_uuid}, 24576, 123, 570.26.02, 00000000:01:00.0, "
            "8.6, 41, 0, P8, 23.5, 350.0\n"
        )
        parsed = preflight.parse_gpu_csv(row)
        self.assertEqual(parsed[0]["uuid"], exact_uuid)
        self.assertEqual(parsed[0]["memory_total_mib"], 24576)
        with self.assertRaisesRegex(ValueError, "columns"):
            preflight.parse_gpu_csv("0, RTX 3090\n")

    def test_pcie_query_falls_back_per_field_without_hiding_unavailable_data(self) -> None:
        gpus = [self._gpu(index) for index in range(4)]
        for gpu in gpus:
            gpu.pop("pcie_link")
        values = {
            "pcie.link.gen.current": 3,
            "pcie.link.gen.max": 4,
            "pcie.link.width.current": 16,
            "pcie.link.width.max": 16,
        }

        def fake_command(command: list[str]) -> dict[str, object]:
            query = command[1]
            if query.count(",") > 2:
                return {"ok": False, "returncode": 2, "stdout": "", "stderr": "unsupported"}
            field = query.rsplit(",", 1)[-1]
            stdout = "".join(
                f"{index}, GPU-fixture-exact-{index}, {values[field]}\n" for index in range(4)
            )
            return {"ok": True, "returncode": 0, "stdout": stdout, "stderr": ""}

        with mock.patch.object(preflight, "run_command", side_effect=fake_command):
            result = preflight.collect_pcie_links(gpus)
        self.assertTrue(result["complete"])
        self.assertEqual(result["unavailable_fields"], [])
        self.assertEqual(gpus[0]["pcie_link"]["gen_current"], 3)
        self.assertEqual(gpus[0]["pcie_link"]["gen_max"], 4)

        failure = {"ok": False, "returncode": 2, "stdout": "", "stderr": "unsupported"}
        with mock.patch.object(preflight, "run_command", return_value=failure):
            unavailable = preflight.collect_pcie_links(gpus)
        self.assertFalse(unavailable["complete"])
        self.assertEqual(set(unavailable["unavailable_fields"]), set(preflight.PCIE_QUERY_FIELDS))
        self.assertTrue(all(value is None for value in gpus[0]["pcie_link"].values()))

    def test_driver_selects_only_eligible_branch(self) -> None:
        branches = self.contract["cuda_branches"]
        self.assertEqual(preflight.choose_cuda_branch("570.26.0", branches)["name"], "cu128")
        self.assertEqual(preflight.choose_cuda_branch("560.28.3", branches)["name"], "cu126")
        self.assertIsNone(preflight.choose_cuda_branch("550.99.0", branches))

    def test_xid_sources_share_an_explicit_window(self) -> None:
        window_end = datetime(2026, 8, 16, 12, 30, tzinfo=UTC)
        window_start = window_end - timedelta(minutes=15)
        calls: list[list[str]] = []

        def fake_command(
            command: list[str], timeout_s: float, environment: dict[str, str]
        ) -> dict[str, object]:
            calls.append(command)
            self.assertEqual(timeout_s, 15)
            self.assertEqual(environment["TZ"], "UTC")
            if command[0] == "journalctl":
                return {
                    "ok": True,
                    "returncode": 0,
                    "stdout": "2026-08-16 kernel: ordinary record\n",
                    "stderr": "",
                }
            return {
                "ok": False,
                "returncode": 1,
                "stdout": "",
                "stderr": "dmesg: read kernel buffer failed: Operation not permitted",
            }

        with mock.patch.object(preflight, "run_command", side_effect=fake_command):
            result = preflight.collect_xid_events(
                window_start=window_start,
                window_end=window_end,
                policy=self.contract["xid_health_evidence"],
            )
        self.assertTrue(result["available"])
        self.assertEqual(result["matches"], [])
        self.assertEqual(len(calls), 2)
        for command in calls:
            self.assertIn("--since", command)
            self.assertIn("--until", command)
            self.assertIn("2026-08-16 12:15:00.000000", command)
            self.assertIn("2026-08-16 12:30:00.000000", command)
        self.assertNotEqual(calls[1], ["dmesg", "--color=never"])

    def test_accessible_empty_window_is_available_with_zero_xids(self) -> None:
        window_end = datetime.now(UTC)
        window_start = window_end - timedelta(minutes=15)
        empty_query = {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}
        readable_probe = {
            "ok": True,
            "returncode": 0,
            "stdout": "2026-08-16 kernel: ordinary record\n",
            "stderr": "",
        }
        denied = {
            "ok": False,
            "returncode": 1,
            "stdout": "",
            "stderr": "dmesg: read kernel buffer failed: Operation not permitted",
        }
        with mock.patch.object(
            preflight, "run_command", side_effect=(empty_query, readable_probe, denied)
        ):
            result = preflight.collect_xid_events(
                window_start=window_start,
                window_end=window_end,
                policy=self.contract["xid_health_evidence"],
            )
        self.assertTrue(result["available"])
        self.assertEqual(result["matches"], [])
        self.assertTrue(result["local_sources"][0]["readability_probe"]["available"])
        self.assertEqual(preflight._xid_gate("fixture", result)["status"], "pass")

    def test_no_storage_permission_and_unproven_empty_are_unavailable(self) -> None:
        window_end = datetime.now(UTC)
        window_start = window_end - timedelta(minutes=15)
        cases = (
            {
                "ok": True,
                "returncode": 0,
                "stdout": "-- No entries --\n",
                "stderr": "No journal files were found.",
            },
            {
                "ok": True,
                "returncode": 0,
                "stdout": "ordinary kernel record\n",
                "stderr": "Permission denied",
            },
        )
        for journal_result in cases:
            with self.subTest(journal_result=journal_result):
                permission = {
                    "ok": False,
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "dmesg: read kernel buffer failed: Operation not permitted",
                }
                with mock.patch.object(
                    preflight, "run_command", side_effect=(journal_result, permission)
                ):
                    result = preflight.collect_xid_events(
                        window_start=window_start,
                        window_end=window_end,
                        policy=self.contract["xid_health_evidence"],
                    )
                self.assertFalse(result["available"])
                self.assertEqual(result["matches"], [])
                self.assertEqual(preflight._xid_gate("fixture", result)["status"], "warn")

        empty = {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}
        unproven = {
            "ok": True,
            "returncode": 0,
            "stdout": "-- No entries --\n",
            "stderr": "",
        }
        denied = {
            "ok": False,
            "returncode": 1,
            "stdout": "",
            "stderr": "Operation not permitted",
        }
        with mock.patch.object(preflight, "run_command", side_effect=(empty, unproven, denied)):
            result = preflight.collect_xid_events(
                window_start=window_start,
                window_end=window_end,
                policy=self.contract["xid_health_evidence"],
            )
        self.assertFalse(result["available"])
        self.assertEqual(preflight._xid_gate("fixture", result)["status"], "warn")

    def test_provider_xid_evidence_requires_schema_hash_and_window_coverage(self) -> None:
        now = datetime.now(UTC)
        requested_start = now - timedelta(minutes=2)
        requested_end = now - timedelta(seconds=30)
        evidence_start = now - timedelta(minutes=3)
        evidence_end = now - timedelta(seconds=20)
        policy = self.contract["xid_health_evidence"]
        document = {
            "schema_version": preflight.PROVIDER_HEALTH_SCHEMA,
            "provider": "fixture-provider",
            "instance_reference": "fixture-instance",
            "window_start": evidence_start.isoformat(),
            "window_end": evidence_end.isoformat(),
            "captured_at": (now - timedelta(seconds=10)).isoformat(),
            "nvidia_xid_events": [
                {
                    "timestamp": (now - timedelta(minutes=2, seconds=30)).isoformat(),
                    "code": 31,
                    "summary": "outside requested window",
                },
                {
                    "timestamp": (now - timedelta(minutes=1)).isoformat(),
                    "code": 79,
                    "summary": "inside requested window",
                },
            ],
        }
        unavailable = {
            "ok": False,
            "returncode": 1,
            "stdout": "",
            "stderr": "Operation not permitted",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "provider-health.json"
            checksum = root / "provider-health.sha256"
            evidence.write_text(json.dumps(document), encoding="utf-8")
            checksum.write_text(preflight.sha256_file(evidence) + "\n", encoding="ascii")
            with mock.patch.object(
                preflight, "run_command", side_effect=(unavailable, unavailable)
            ):
                result = preflight.collect_xid_events(
                    window_start=requested_start,
                    window_end=requested_end,
                    policy=policy,
                    provider_evidence_path=str(evidence),
                    provider_checksum_path=str(checksum),
                )
            self.assertTrue(result["available"])
            self.assertTrue(result["provider_evidence"]["valid"])
            self.assertEqual(len(result["matches"]), 1)
            self.assertEqual(result["matches"][0]["code"], 79)
            self.assertEqual(preflight._xid_gate("fixture", result)["status"], "fail")

            checksum.write_text("0" * 64 + "\n", encoding="ascii")
            with mock.patch.object(
                preflight, "run_command", side_effect=(unavailable, unavailable)
            ):
                bad_hash = preflight.collect_xid_events(
                    window_start=requested_start,
                    window_end=requested_end,
                    policy=policy,
                    provider_evidence_path=str(evidence),
                    provider_checksum_path=str(checksum),
                )
            self.assertFalse(bad_hash["available"])
            self.assertFalse(bad_hash["provider_evidence"]["valid"])
            self.assertIn("SHA-256 mismatch", bad_hash["provider_evidence"]["error"])
            self.assertEqual(preflight._xid_gate("fixture", bad_hash)["status"], "warn")

            document["unexpected"] = True
            evidence.write_text(json.dumps(document), encoding="utf-8")
            checksum.write_text(preflight.sha256_file(evidence) + "\n", encoding="ascii")
            with mock.patch.object(
                preflight, "run_command", side_effect=(unavailable, unavailable)
            ):
                bad_schema = preflight.collect_xid_events(
                    window_start=requested_start,
                    window_end=requested_end,
                    policy=policy,
                    provider_evidence_path=str(evidence),
                    provider_checksum_path=str(checksum),
                )
            self.assertFalse(bad_schema["provider_evidence"]["valid"])
            self.assertIn("unexpected or missing", bad_schema["provider_evidence"]["error"])

    def test_late_after_evidence_can_replace_unavailable_container_logs(self) -> None:
        requested_end = datetime.now(UTC)
        requested_start = requested_end - timedelta(minutes=5)
        unavailable = {
            "ok": False,
            "returncode": 1,
            "stdout": "",
            "stderr": "Operation not permitted",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "after.json"
            checksum = root / "after.sha256"

            def publish_after() -> None:
                time.sleep(0.1)
                observed = datetime.now(UTC)
                document = {
                    "schema_version": preflight.PROVIDER_HEALTH_SCHEMA,
                    "provider": "fixture-provider",
                    "instance_reference": "fixture-instance",
                    "window_start": (requested_start - timedelta(minutes=1)).isoformat(),
                    "window_end": observed.isoformat(),
                    "captured_at": observed.isoformat(),
                    "nvidia_xid_events": [],
                }
                evidence.write_text(json.dumps(document), encoding="utf-8")
                checksum.write_text(preflight.sha256_file(evidence) + "\n", encoding="ascii")

            publisher = threading.Thread(target=publish_after)
            publisher.start()
            with mock.patch.object(
                preflight, "run_command", side_effect=(unavailable, unavailable)
            ):
                result = preflight.collect_xid_events(
                    window_start=requested_start,
                    window_end=requested_end,
                    policy=self.contract["xid_health_evidence"],
                    provider_evidence_path=str(evidence),
                    provider_checksum_path=str(checksum),
                    provider_wait_seconds=2.0,
                )
            publisher.join(timeout=2)
            self.assertFalse(publisher.is_alive())
            self.assertTrue(result["provider_evidence"]["valid"])
            self.assertTrue(result["available"])
            self.assertEqual(preflight._xid_gate("after", result)["status"], "pass")

    def test_preexisting_and_new_xid_events_are_separate_fail_gates(self) -> None:
        report = self._passing_report()
        report["hardware"]["xid_before"]["matches"] = [{"source": "fixture"}]
        statuses = {
            item["name"]: item["status"]
            for item in preflight.evaluate_full_gates(report, self.contract)
        }
        self.assertEqual(statuses["preexisting_xid_window"], "fail")
        self.assertEqual(statuses["new_xid_during_preflight"], "pass")

        report = self._passing_report()
        report["hardware"]["xid_after"]["matches"] = [{"source": "fixture"}]
        statuses = {
            item["name"]: item["status"]
            for item in preflight.evaluate_full_gates(report, self.contract)
        }
        self.assertEqual(statuses["preexisting_xid_window"], "pass")
        self.assertEqual(statuses["new_xid_during_preflight"], "fail")

    def test_manifest_detects_mutation_unexpected_file_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            root.mkdir()
            payload = root / "package.whl"
            payload.write_bytes(b"wheel fixture")
            manifest = preflight.build_file_manifest(
                root, kind="wheel-cache", metadata={"fixture": True}
            )
            self.assertTrue(preflight.verify_file_manifest(manifest, root)["ok"])

            payload.write_bytes(b"mutated fixture")
            mutated = preflight.verify_file_manifest(manifest, root)
            self.assertFalse(mutated["ok"])
            self.assertTrue(any("mismatch" in item for item in mutated["errors"]))

            payload.write_bytes(b"wheel fixture")
            (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
            unexpected = preflight.verify_file_manifest(manifest, root)
            self.assertFalse(unexpected["ok"])
            self.assertTrue(any("unexpected" in item for item in unexpected["errors"]))

            output = Path(directory) / "manifest.json"
            preflight.atomic_write_json(output, manifest)
            with self.assertRaises(FileExistsError):
                preflight.atomic_write_json(output, manifest)

    def test_wheel_inventory_requires_unique_exact_top_level_candidates(self) -> None:
        branch = self.contract["cuda_branches"][0]

        def populate(root: Path) -> None:
            for name, version in preflight._exact_requirement_pins(self.contract, branch).items():
                encoded = name.replace("-", "_")
                resolved = version + "+cu128" if name == "torch" else version
                self._write_fixture_wheel(
                    root / f"{encoded}-{resolved}-py3-none-any.whl",
                    name=name,
                    version=resolved,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populate(root)
            inventory = preflight.wheel_cache_inventory(root, self.contract, branch)
            self.assertTrue(inventory["ok"], inventory["errors"])
            self.assertIn("triton", inventory["summary"]["top_level"])
            self.assertIn("huggingface-hub", inventory["summary"]["top_level"])
            self.assertEqual(
                inventory["summary"]["top_level"]["setuptools"]["version"],
                "80.10.2",
            )
            self.assertEqual(
                inventory["summary"]["top_level"]["wheel"]["version"],
                "0.46.3",
            )

            wheel = next(root.glob("wheel-*.whl"))
            wheel_payload = wheel.read_bytes()
            wheel.unlink()
            missing = preflight.wheel_cache_inventory(root, self.contract, branch)
            self.assertFalse(missing["ok"])
            self.assertTrue(any("wheel==0.46.3" in item for item in missing["errors"]))
            wheel.write_bytes(wheel_payload)

            hub = next(root.glob("huggingface_hub-*.whl"))
            hub_payload = hub.read_bytes()
            hub.unlink()
            missing = preflight.wheel_cache_inventory(root, self.contract, branch)
            self.assertFalse(missing["ok"])
            self.assertTrue(any("huggingface-hub" in item for item in missing["errors"]))

            hub.write_bytes(hub_payload)
            triton = next(root.glob("triton-*.whl"))
            triton.unlink()
            missing = preflight.wheel_cache_inventory(root, self.contract, branch)
            self.assertFalse(missing["ok"])
            self.assertTrue(any("triton" in item for item in missing["errors"]))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populate(root)
            self._write_fixture_wheel(
                root / "fsspec-2025.10.0-py3-none-any.whl",
                name="fsspec",
                version="2025.10.0",
            )
            self._write_fixture_wheel(
                root / "fsspec-2026.7.0-py3-none-any.whl",
                name="fsspec",
                version="2026.7.0",
            )
            ambiguous = preflight.wheel_cache_inventory(root, self.contract, branch)
            self.assertFalse(ambiguous["ok"])
            self.assertTrue(
                any("ambiguous wheel distribution fsspec" in item for item in ambiguous["errors"])
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populate(root)
            (root / "transitive-1.0-cp312-cp312-manylinux_2_28_x86_64.whl").write_bytes(b"fixture")
            incompatible = preflight.wheel_cache_inventory(root, self.contract, branch)
            self.assertFalse(incompatible["ok"])
            self.assertTrue(any("CPython 3.11" in item for item in incompatible["errors"]))

    def test_wheel_inventory_reads_real_metadata_and_rejects_missing_extra_dependency(self) -> None:
        branch = self.contract["cuda_branches"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, version in preflight._exact_requirement_pins(self.contract, branch).items():
                resolved = version + "+cu128" if name == "torch" else version
                requires = ("fsspec[http]>=1",) if name == "torch" else ()
                self._write_fixture_wheel(
                    root / f"{name.replace('-', '_')}-{resolved}-py3-none-any.whl",
                    name=name,
                    version=resolved,
                    requires_dist=requires,
                )
            self._write_fixture_wheel(
                root / "fsspec-2025.10.0-py3-none-any.whl",
                name="fsspec",
                version="2025.10.0",
                requires_dist=('aiohttp>=3; extra == "http"',),
            )

            missing = preflight.wheel_cache_inventory(root, self.contract, branch)
            self.assertFalse(missing["ok"])
            self.assertTrue(
                any("fsspec -> aiohttp>=3" in item for item in missing["errors"]),
                missing["errors"],
            )

            self._write_fixture_wheel(
                root / "aiohttp-3.14.3-py3-none-any.whl",
                name="aiohttp",
                version="3.14.3",
            )
            complete = preflight.wheel_cache_inventory(root, self.contract, branch)
            self.assertTrue(complete["ok"], complete["errors"])
            closure = complete["summary"]["deployment_closure"]
            self.assertEqual(closure["requested_extras"]["fsspec"], ["http"])
            self.assertIn("aiohttp", closure["resolved"])

    def test_wheel_inventory_rejects_requires_python_that_excludes_target(self) -> None:
        branch = self.contract["cuda_branches"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, version in preflight._exact_requirement_pins(self.contract, branch).items():
                resolved = version + "+cu128" if name == "torch" else version
                self._write_fixture_wheel(
                    root / f"{name.replace('-', '_')}-{resolved}-py3-none-any.whl",
                    name=name,
                    version=resolved,
                    requires_python=">=3.12" if name == "tokenizers" else ">=3.11",
                )

            incompatible = preflight.wheel_cache_inventory(root, self.contract, branch)
            self.assertFalse(incompatible["ok"])
            self.assertTrue(
                any(
                    "Requires-Python excludes target 3.11.0" in item
                    for item in incompatible["errors"]
                ),
                incompatible["errors"],
            )

    def test_fresh_offline_install_uses_no_index_and_records_freeze(self) -> None:
        completed = {
            "ok": True,
            "returncode": 0,
            "stdout": "fixture==1.0\n",
            "stderr": "",
        }
        commands: list[list[str]] = []

        def fake_command(command: list[str], **_: object) -> dict[str, object]:
            commands.append(command)
            return completed

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel_root = root / "wheels"
            wheel_root.mkdir()
            with (
                mock.patch.object(preflight.platform, "system", return_value="Linux"),
                mock.patch.object(preflight, "run_command", side_effect=fake_command),
            ):
                result = preflight.fresh_offline_install(
                    artifact_directory=root / "evidence",
                    wheel_root=wheel_root,
                    workdir=Path(__file__).parents[2],
                    contract=self.contract,
                    branch=self.contract["cuda_branches"][0],
                )
            self.assertTrue(result["ok"], result["errors"])
            self.assertTrue(result["network_disabled"])
            self.assertEqual(len(result["freeze_sha256"]), 64)
            build_tools_index = next(
                index for index, command in enumerate(commands) if "setuptools==80.10.2" in command
            )
            requirements_index = next(
                index for index, command in enumerate(commands) if "-r" in command
            )
            editable_index = next(
                index for index, command in enumerate(commands) if "--editable" in command
            )
            self.assertLess(build_tools_index, requirements_index)
            self.assertLess(requirements_index, editable_index)
            build_tools = commands[build_tools_index]
            self.assertIn("--no-index", build_tools)
            self.assertIn("--only-binary=:all:", build_tools)
            self.assertIn("wheel==0.46.3", build_tools)
            editable = next(command for command in commands if "--editable" in command)
            self.assertIn("--no-index", editable)
            self.assertIn("--no-build-isolation", editable)
            self.assertIn("--no-deps", editable)
            self.assertTrue((root / "evidence" / "fresh-offline-freeze.txt").is_file())

    def test_manifest_rejects_snapshot_style_symbolic_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_entry = root / "snapshot-link"
            snapshot_entry.write_text("../../blobs/fixture", encoding="utf-8")
            original = Path.is_symlink

            def report_fixture_as_link(path: Path) -> bool:
                return path == snapshot_entry or original(path)

            with (
                mock.patch.object(Path, "is_symlink", report_fixture_as_link),
                self.assertRaisesRegex(ValueError, "symbolic links"),
            ):
                preflight.build_file_manifest(
                    root, kind="model-snapshot", metadata={"fixture": True}
                )

    def test_manifest_semantics_bind_contract_branch_and_model_revision(self) -> None:
        branch = self.contract["cuda_branches"][0]
        for kind, cuda_branch in (("wheel-cache", "cu128"), ("model-snapshot", None)):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "payload.bin").write_bytes(b"fixture")
                metadata = preflight.manifest_metadata(
                    kind=kind,
                    contract=self.contract,
                    contract_path=self.contract_path,
                    cuda_branch=cuda_branch,
                )
                manifest = preflight.build_file_manifest(root, kind=kind, metadata=metadata)
                result = preflight.verify_file_manifest(manifest, root)
                ok, errors = preflight.cache_semantics_ok(
                    result,
                    kind=kind,
                    contract=self.contract,
                    contract_path=self.contract_path,
                    branch=branch if kind == "wheel-cache" else None,
                )
                self.assertTrue(ok, errors)

                result["metadata"] = dict(result["metadata"])
                result["metadata"]["contract_sha256"] = "0" * 64
                ok, errors = preflight.cache_semantics_ok(
                    result,
                    kind=kind,
                    contract=self.contract,
                    contract_path=self.contract_path,
                    branch=branch if kind == "wheel-cache" else None,
                )
                self.assertFalse(ok)
                self.assertIn("contract SHA256 mismatch", errors)

    def test_passing_full_fixture_is_ready_for_training(self) -> None:
        gates = preflight.evaluate_full_gates(self._passing_report(), self.contract)
        failures = [item for item in gates if item["status"] != "pass"]
        self.assertEqual(failures, [])
        outcome = preflight._report_outcome(gates, "full")
        self.assertEqual(outcome["status"], "pass")
        self.assertEqual(outcome["decision"], "eligible-for-day-rental-and-ready-for-training")
        self.assertTrue(outcome["ready"])
        self.assertTrue(outcome["day_rental_eligible"])

    def test_missing_topology_is_a_warning_not_a_hardware_failure(self) -> None:
        report = self._passing_report()
        report["hardware"]["gpu_probe_before"]["pcie_query"]["complete"] = False
        report["hardware"]["cpu_topology"]["numa_node_count"] = None
        gates = preflight.evaluate_host_gates(report, self.contract)
        statuses = {item["name"]: item["status"] for item in gates}
        self.assertEqual(statuses["host_topology_observation"], "warn")
        self.assertNotIn("fail", {statuses["host_topology_observation"]})

    def test_complete_partial_peer_matrix_is_an_observation_not_a_warning(self) -> None:
        report = self._passing_report()
        report["distributed"]["payload"]["p2p"] = {
            "query_complete": True,
            "full_mesh": False,
            "matrix": [
                [True, False, False, False],
                [False, True, False, False],
                [False, False, True, False],
                [False, False, False, True],
            ],
        }
        gates = preflight.evaluate_full_gates(report, self.contract)
        statuses = {item["name"]: item["status"] for item in gates}
        self.assertEqual(statuses["p2p_observation"], "pass")
        self.assertNotIn("p2p_full_mesh", statuses)
        self.assertEqual(preflight._report_outcome(gates, "full")["status"], "pass")

        report["distributed"]["payload"]["p2p"]["query_complete"] = False
        gates = preflight.evaluate_full_gates(report, self.contract)
        statuses = {item["name"]: item["status"] for item in gates}
        self.assertEqual(statuses["p2p_observation"], "warn")

    def test_freeze_drift_and_missing_cache_fail_closed(self) -> None:
        report = self._passing_report()
        report["provenance"]["freeze"]["after"]["sha256"] = "b" * 64
        report["provenance"]["cache_verification"]["model"]["ok"] = False
        gates = preflight.evaluate_full_gates(report, self.contract)
        statuses = {item["name"]: item["status"] for item in gates}
        self.assertEqual(statuses["pre_post_freeze"], "fail")
        self.assertEqual(statuses["model_snapshot_manifest"], "fail")
        outcome = preflight._report_outcome(gates, "full")
        self.assertEqual(outcome["status"], "fail")
        self.assertEqual(outcome["decision"], "not-ready-for-training")
        self.assertFalse(outcome["ready"])

    def test_host_phase_never_authorizes_day_rental(self) -> None:
        passing = preflight._report_outcome(
            [preflight.gate("diagnostic", "pass", "fixture")], "host"
        )
        self.assertEqual(passing["decision"], "continue-hourly-to-full-validation")
        self.assertTrue(passing["host_diagnostics_pass"])
        self.assertFalse(passing["ready"])
        self.assertFalse(passing["day_rental_eligible"])

        outcome = preflight._report_outcome([preflight.gate("memlock", "warn", "fixture")], "host")
        self.assertEqual(outcome["status"], "pass-with-warnings")
        self.assertEqual(outcome["decision"], "manual-review-required-during-hourly-validation")
        self.assertFalse(outcome["ready"])
        self.assertFalse(outcome["day_rental_eligible"])

    def test_distributed_launcher_rejects_any_world_size_other_than_four(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly four ranks"):
            preflight.run_distributed_workers(
                world_size=3,
                stress_seconds=30,
                buffer_mib=64,
                warmup=1,
                iterations=1,
                timeout_s=1,
            )

    def test_parser_defaults_to_host_without_explicit_output(self) -> None:
        args = preflight.make_parser().parse_args(["run"])
        self.assertEqual(args.phase, "host")
        self.assertIsNone(args.output)
        self.assertFalse(args.offline)
        self.assertEqual(args.stress_seconds, 300.0)
        self.assertEqual(args.distributed_timeout, 900.0)
        self.assertIsNone(args.provider_health_evidence_before)
        self.assertIsNone(args.provider_health_evidence_after)
        self.assertEqual(args.provider_health_evidence_wait_seconds, 30.0)


if __name__ == "__main__":
    unittest.main()
