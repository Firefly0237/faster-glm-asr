"""Fail-closed single-node NCCL preflight for the 4 x RTX 3090 experiment."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .train import (
    _distributed_context,
    _gather_rank_records,
    _runtime_environment,
    _sha256_file,
    _shared_filesystem_preflight,
    _write_json_atomic,
)

PREFLIGHT_SCHEMA = "decoder-training-nccl-preflight/2"
PREFLIGHT_SCOPE = "single-node decoder systems experiment"
PREFLIGHT_PATH = Path(__file__).resolve()
TRAIN_PATH = PREFLIGHT_PATH.with_name("train.py")
REPOSITORY_ROOT = PREFLIGHT_PATH.parents[2]


def _capture(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return {
            "command": list(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _all_reduce_probe(
    rank: int,
    world_size: int,
    device: torch.device,
    tensor_bytes: int,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    scalar = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2
    if scalar.item() != expected:
        raise RuntimeError(f"NCCL scalar all_reduce produced {scalar.item()}, expected {expected}")
    element_count = max(1, tensor_bytes // torch.tensor([], dtype=torch.float32).element_size())
    payload = torch.full((element_count,), float(rank + 1), device=device)
    for _ in range(warmup):
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        payload.fill_(float(rank + 1))
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repetitions):
        payload.fill_(float(rank + 1))
        dist.barrier()
        started = time.perf_counter()
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - started)
        if payload[0].item() != expected:
            raise RuntimeError("NCCL payload all_reduce correctness check failed")
    local = {
        "rank": rank,
        "samples_s": samples,
        "mean_s": sum(samples) / len(samples),
        "min_s": min(samples),
        "max_s": max(samples),
    }
    ranks = _gather_rank_records(local, world_size)
    critical_samples = [
        max(float(item["samples_s"][index]) for item in ranks) for index in range(repetitions)
    ]
    mean_critical = sum(critical_samples) / len(critical_samples)
    # Ring all-reduce algorithm bandwidth and the standard bus-bandwidth
    # correction. This is a diagnostic, not a model-training throughput result.
    algorithm_bandwidth = tensor_bytes / mean_critical
    bus_bandwidth = algorithm_bandwidth * (2 * (world_size - 1) / world_size)
    return {
        "tensor_bytes": tensor_bytes,
        "warmup_repetitions": warmup,
        "measured_repetitions": repetitions,
        "critical_rank_samples_s": critical_samples,
        "mean_critical_rank_s": mean_critical,
        "algorithm_bandwidth_bytes_per_second": algorithm_bandwidth,
        "estimated_ring_bus_bandwidth_bytes_per_second": bus_bandwidth,
        "rank_results": ranks,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/private/training/preflight-4x3090.json"),
    )
    parser.add_argument("--environment-candidate", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--expected-gpu-name", default="RTX 3090")
    parser.add_argument("--minimum-memory-gib", type=float, default=23.0)
    parser.add_argument("--all-reduce-mib", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args(argv)
    if (
        args.expected_world_size <= 0
        or args.minimum_memory_gib <= 0
        or args.all_reduce_mib <= 0
        or args.warmup < 0
        or args.repetitions <= 0
    ):
        raise ValueError("preflight numeric arguments are outside valid ranges")
    if not {"RANK", "LOCAL_RANK", "WORLD_SIZE"} <= set(os.environ):
        raise RuntimeError("preflight must be launched by torchrun")
    environment_candidate = args.environment_candidate.resolve()
    if not environment_candidate.is_file():
        raise FileNotFoundError(environment_candidate)

    rank, local_rank, world_size, device = _distributed_context()
    if world_size != args.expected_world_size:
        raise RuntimeError(f"expected world_size={args.expected_world_size}, observed {world_size}")
    output = args.output.resolve()
    _shared_filesystem_preflight(output.parent, rank, world_size)
    local_environment = _runtime_environment(device, rank, local_rank)
    properties = torch.cuda.get_device_properties(device)
    pci_domain = getattr(properties, "pci_domain_id", None)
    pci_bus = getattr(properties, "pci_bus_id", None)
    pci_device = getattr(properties, "pci_device_id", None)
    pci_bus_id = (
        f"{int(pci_domain):04x}:{int(pci_bus):02x}:{int(pci_device):02x}.0"
        if all(isinstance(value, int) and value >= 0 for value in (pci_domain, pci_bus, pci_device))
        else None
    )
    local_identity = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": local_environment["hostname"],
        "name": properties.name,
        "uuid": str(getattr(properties, "uuid", "")) or None,
        "pci_bus_id": pci_bus_id,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "visible_device_count": torch.cuda.device_count(),
    }
    identities = _gather_rank_records(local_identity, world_size)
    failures = []
    if len({item["hostname"] for item in identities}) != 1:
        failures.append("all ranks must be on one host")
    if sorted(int(item["local_rank"]) for item in identities) != list(range(world_size)):
        failures.append("LOCAL_RANK values must map one-to-one onto visible devices")
    device_identifiers = [item["uuid"] or item["pci_bus_id"] for item in identities]
    if not all(device_identifiers):
        failures.append("every rank must expose a CUDA UUID or PCI bus ID")
    elif len(set(device_identifiers)) != world_size:
        failures.append("CUDA device identities are not unique across ranks")
    for item in identities:
        if args.expected_gpu_name.lower() not in str(item["name"]).lower():
            failures.append(
                f"rank {item['rank']} GPU {item['name']!r} does not match "
                f"{args.expected_gpu_name!r}"
            )
        if int(item["total_memory_bytes"]) < int(args.minimum_memory_gib * 1024**3):
            failures.append(f"rank {item['rank']} has insufficient device memory")
        if item["bf16_supported"] is not True:
            failures.append(f"rank {item['rank']} does not report BF16 support")
        if int(item["visible_device_count"]) < world_size:
            failures.append(f"rank {item['rank']} sees only {item['visible_device_count']} GPUs")
    failure_records = _gather_rank_records({"rank": rank, "failures": failures}, world_size)
    combined_failures = [
        f"rank {item['rank']}: {failure}"
        for item in failure_records
        for failure in item["failures"]
    ]
    if combined_failures:
        raise RuntimeError("preflight inventory failed: " + "; ".join(combined_failures))

    peer_access = [
        [
            bool(torch.cuda.can_device_access_peer(source, target)) if source != target else True
            for target in range(world_size)
        ]
        for source in range(world_size)
    ]
    all_reduce = _all_reduce_probe(
        rank,
        world_size,
        device,
        args.all_reduce_mib * 1024**2,
        args.warmup,
        args.repetitions,
    )
    document = {
        "schema_version": PREFLIGHT_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pass": True,
        "scope": PREFLIGHT_SCOPE,
        "world_size": world_size,
        "producer": {
            "preflight_path": PREFLIGHT_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "preflight_sha256": _sha256_file(PREFLIGHT_PATH),
            "train_path": TRAIN_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "train_sha256": _sha256_file(TRAIN_PATH),
        },
        "invocation": {
            "expected_world_size": args.expected_world_size,
            "expected_gpu_name": args.expected_gpu_name,
            "minimum_memory_gib": args.minimum_memory_gib,
            "all_reduce_mib": args.all_reduce_mib,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
        },
        "environment_candidate": {
            "path": str(environment_candidate),
            "sha256": _sha256_file(environment_candidate),
        },
        "rank_inventory": identities,
        "peer_access": peer_access,
        "all_reduce": all_reduce,
        "nvidia_smi_inventory": _capture(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ),
        "nvidia_smi_topology": _capture(["nvidia-smi", "topo", "-m"]),
        "nccl_environment": {
            key: value for key, value in sorted(os.environ.items()) if key.startswith("NCCL_")
        },
    }
    write_status: list[Any] = [None]
    if rank == 0:
        try:
            if output.exists():
                raise FileExistsError(f"refusing to overwrite {output}")
            _write_json_atomic(output, document)
            write_status[0] = {"ok": True, "sha256": _sha256_file(output)}
        except Exception as exc:
            write_status[0] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    dist.broadcast_object_list(write_status, src=0)
    if write_status[0]["ok"] is not True:
        raise RuntimeError(f"preflight output failed: {write_status[0]['error']}")
    local_hash = _sha256_file(output)
    observed_hashes = _gather_rank_records({"rank": rank, "sha256": local_hash}, world_size)
    if any(item["sha256"] != write_status[0]["sha256"] for item in observed_hashes):
        raise RuntimeError(f"preflight report is not identical on every rank: {observed_hashes}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
