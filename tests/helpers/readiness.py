"""Build a producer-valid RTX 3090 readiness report for integration tests."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

SOURCE_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE = Path("configs/environments/rtx3090-ddp-v1.json")
TOOL_RELATIVE = Path("tools/preflight_rtx3090.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_tool(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"_fixture_preflight_{hash(path.resolve())}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load readiness tool fixture at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gpu(index: int) -> dict[str, object]:
    return {
        "index": index,
        "name": "NVIDIA GeForce RTX 3090",
        "uuid": f"GPU-private-fixture-{index}",
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


def materialize_passing_readiness(repository_root: Path) -> Path:
    """Copy the real producer contract/tool and emit evidence they accept."""
    contract_path = repository_root / CONTRACT_RELATIVE
    tool_path = repository_root / TOOL_RELATIVE
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_ROOT / CONTRACT_RELATIVE, contract_path)
    shutil.copyfile(SOURCE_ROOT / TOOL_RELATIVE, tool_path)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    tool = _load_tool(tool_path)
    branch = copy.deepcopy(contract["cuda_branches"][0])
    gpus = [_gpu(index) for index in range(4)]
    ranks = [
        {
            "rank": index,
            "bf16": {"ok": True},
            "stress": {"requested_seconds": 300, "iterations": 1},
        }
        for index in range(4)
    ]
    report: dict[str, object] = {
        "schema_version": "rtx3090-readiness-v1",
        "phase": "full",
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
                    "available_fields": list(tool.PCIE_QUERY_FIELDS),
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
            "packages": copy.deepcopy(contract["packages"]),
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
        "contract": {"sha256": _sha256(contract_path)},
        "tool": {"sha256": _sha256(tool_path)},
    }
    report["gates"] = tool.evaluate_full_gates(report, contract)
    report["overall"] = tool._report_outcome(report["gates"], "full")
    report_path = repository_root / "artifacts/private/rtx3090-readiness/full.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def materialize_passing_training_preflight(output_path: Path) -> Path:
    """Emit the exact short NCCL report consumed by the training release gate."""
    samples_by_rank = [
        [0.010 + rank * 0.001 + index * 0.00001 for index in range(10)] for rank in range(4)
    ]
    rank_results = [
        {
            "rank": rank,
            "samples_s": samples,
            "mean_s": sum(samples) / len(samples),
            "min_s": min(samples),
            "max_s": max(samples),
        }
        for rank, samples in enumerate(samples_by_rank)
    ]
    critical = [max(samples[index] for samples in samples_by_rank) for index in range(10)]
    mean_critical = sum(critical) / len(critical)
    tensor_bytes = 64 * 1024**2
    algorithm_bandwidth = tensor_bytes / mean_critical
    environment = SOURCE_ROOT / CONTRACT_RELATIVE
    preflight = SOURCE_ROOT / "experiments/distributed_training/preflight.py"
    train = SOURCE_ROOT / "experiments/distributed_training/train.py"
    document = {
        "schema_version": "decoder-training-nccl-preflight/2",
        "created_at_utc": "2026-08-16T12:00:00+00:00",
        "pass": True,
        "scope": "single-node decoder systems experiment",
        "world_size": 4,
        "producer": {
            "preflight_path": "experiments/distributed_training/preflight.py",
            "preflight_sha256": _sha256(preflight),
            "train_path": "experiments/distributed_training/train.py",
            "train_sha256": _sha256(train),
        },
        "invocation": {
            "expected_world_size": 4,
            "expected_gpu_name": "RTX 3090",
            "minimum_memory_gib": 23.0,
            "all_reduce_mib": 64,
            "warmup": 5,
            "repetitions": 10,
        },
        "environment_candidate": {
            "path": str(environment.resolve()),
            "sha256": _sha256(environment),
        },
        "rank_inventory": [
            {
                "rank": index,
                "local_rank": index,
                "hostname": "fixture-host",
                "name": "NVIDIA GeForce RTX 3090",
                "uuid": f"GPU-private-fixture-{index}",
                "pci_bus_id": f"00000000:{index + 1:02X}:00.0",
                "total_memory_bytes": 24 * 1024**3,
                "compute_capability": [8, 6],
                "bf16_supported": True,
                "visible_device_count": 4,
            }
            for index in range(4)
        ],
        "peer_access": [[True for _ in range(4)] for _ in range(4)],
        "all_reduce": {
            "tensor_bytes": tensor_bytes,
            "warmup_repetitions": 5,
            "measured_repetitions": 10,
            "critical_rank_samples_s": critical,
            "mean_critical_rank_s": mean_critical,
            "algorithm_bandwidth_bytes_per_second": algorithm_bandwidth,
            "estimated_ring_bus_bandwidth_bytes_per_second": algorithm_bandwidth * 1.5,
            "rank_results": rank_results,
        },
        "nvidia_smi_inventory": {
            "command": [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            "returncode": 0,
            "stdout": "".join(
                f"{index}, GPU-private-fixture-{index}, "
                f"00000000:{index + 1:02X}:00.0, NVIDIA GeForce RTX 3090, "
                "24576, 570.26.0\n"
                for index in range(4)
            ),
            "stderr": "",
        },
        "nvidia_smi_topology": {
            "command": ["nvidia-smi", "topo", "-m"],
            "returncode": 0,
            "stdout": "fixture topology\n",
            "stderr": "",
        },
        "nccl_environment": {"NCCL_DEBUG": "WARN"},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
