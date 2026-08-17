"""Fail-closed single-node NCCL preflight for the 4 x RTX 3090 experiment."""

from __future__ import annotations

import argparse
import os
import re
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
NVIDIA_SMI_INVENTORY_COMMAND = [
    "nvidia-smi",
    "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
]


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


def _canonical_gpu_uuid(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    return normalized or None


def _canonical_pci_bus_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"(?:[0-9a-fA-F]{4,8}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]",
        value.strip(),
    )
    return match.group(0).lower() if match else None


def _parse_nvidia_smi_inventory(capture: dict[str, Any]) -> list[dict[str, Any]]:
    if capture.get("returncode") != 0:
        detail = capture.get("error") or capture.get("stderr") or "unknown error"
        raise ValueError(f"nvidia-smi identity query failed: {detail}")
    stdout = capture.get("stdout")
    if not isinstance(stdout, str) or not stdout.strip():
        raise ValueError("nvidia-smi identity query returned no GPU rows")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 6:
            raise ValueError(
                f"nvidia-smi identity row {line_number} has {len(columns)} columns, expected 6"
            )
        try:
            index = int(columns[0])
        except ValueError as exc:
            raise ValueError(
                f"nvidia-smi identity row {line_number} has an invalid GPU index"
            ) from exc
        uuid = _canonical_gpu_uuid(columns[1])
        pci_bus_id = _canonical_pci_bus_id(columns[2])
        if index < 0 or uuid is None or pci_bus_id is None:
            raise ValueError(
                f"nvidia-smi identity row {line_number} has an invalid index, UUID, or PCI BDF"
            )
        rows.append(
            {
                "index": index,
                "uuid": columns[1],
                "canonical_uuid": uuid,
                "pci_bus_id": pci_bus_id,
            }
        )
    for key in ("index", "canonical_uuid", "pci_bus_id"):
        values = [row[key] for row in rows]
        if len(set(values)) != len(values):
            raise ValueError(f"nvidia-smi identity query contains duplicate {key} values")
    return rows


def _resolve_rank_gpu_binding(
    *,
    local_rank: int,
    device_index: int | None,
    selected_properties: Any,
    visible_properties: Sequence[Any],
    inventory_rows: Sequence[dict[str, Any]],
    expected_world_size: int,
) -> dict[str, Any]:
    """Bind one CUDA logical device to a physical UUID/BDF without index assumptions."""
    if device_index is None or device_index != local_rank:
        raise ValueError(
            f"LOCAL_RANK={local_rank} does not select CUDA logical device index {device_index}"
        )
    if len(visible_properties) != expected_world_size:
        raise ValueError(
            f"CUDA exposes {len(visible_properties)} logical devices, expected {expected_world_size}"
        )
    if len(inventory_rows) != expected_world_size:
        raise ValueError(
            f"nvidia-smi exposes {len(inventory_rows)} physical GPUs, expected {expected_world_size}"
        )
    if sorted(int(row["index"]) for row in inventory_rows) != list(range(expected_world_size)):
        raise ValueError("nvidia-smi physical GPU indexes are not the expected contiguous set")
    visible_uuids = [
        _canonical_gpu_uuid(getattr(item, "uuid", None)) for item in visible_properties
    ]
    if any(value is None for value in visible_uuids):
        raise ValueError("every CUDA logical device must expose a UUID")
    canonical_visible_uuids = [str(value) for value in visible_uuids]
    if len(set(canonical_visible_uuids)) != expected_world_size:
        raise ValueError("CUDA logical devices expose duplicate UUIDs")
    selected_uuid = _canonical_gpu_uuid(getattr(selected_properties, "uuid", None))
    if selected_uuid is None:
        raise ValueError("the selected CUDA device does not expose a UUID")
    if selected_uuid != canonical_visible_uuids[device_index]:
        raise ValueError("the selected CUDA device UUID does not match its logical device index")
    rows_by_uuid = {str(row["canonical_uuid"]): row for row in inventory_rows}
    if set(canonical_visible_uuids) != set(rows_by_uuid):
        raise ValueError("CUDA-visible UUIDs do not exactly match the nvidia-smi inventory")
    row = rows_by_uuid[selected_uuid]
    return {
        "nvidia_smi_index": int(row["index"]),
        "uuid": str(row["uuid"]),
        "canonical_uuid": selected_uuid,
        "pci_bus_id": str(row["pci_bus_id"]),
        "visible_cuda_uuids": canonical_visible_uuids,
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
    inventory_capture_holder: list[Any] = [
        _capture(NVIDIA_SMI_INVENTORY_COMMAND) if rank == 0 else None
    ]
    dist.broadcast_object_list(inventory_capture_holder, src=0)
    nvidia_smi_inventory = inventory_capture_holder[0]
    binding: dict[str, Any] | None = None
    binding_error: str | None = None
    try:
        if not isinstance(nvidia_smi_inventory, dict):
            raise ValueError("rank 0 did not broadcast a valid nvidia-smi capture")
        inventory_rows = _parse_nvidia_smi_inventory(nvidia_smi_inventory)
        visible_properties = [
            torch.cuda.get_device_properties(index) for index in range(torch.cuda.device_count())
        ]
        binding = _resolve_rank_gpu_binding(
            local_rank=local_rank,
            device_index=device.index,
            selected_properties=properties,
            visible_properties=visible_properties,
            inventory_rows=inventory_rows,
            expected_world_size=world_size,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        binding_error = f"{type(exc).__name__}: {exc}"
    binding_record = {
        "rank": rank,
        "local_rank": local_rank,
        "device_index": device.index,
        "error": binding_error,
        "nvidia_smi_index": binding["nvidia_smi_index"] if binding else None,
        "canonical_uuid": binding["canonical_uuid"] if binding else None,
        "pci_bus_id": binding["pci_bus_id"] if binding else None,
        "visible_cuda_uuids": binding["visible_cuda_uuids"] if binding else None,
    }
    binding_records = _gather_rank_records(binding_record, world_size)
    local_identity = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": local_environment["hostname"],
        "name": properties.name,
        "uuid": binding["uuid"] if binding else None,
        "pci_bus_id": binding["pci_bus_id"] if binding else None,
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
    for record in binding_records:
        if record["error"] is not None:
            failures.append(f"rank {record['rank']} GPU identity binding failed: {record['error']}")
    valid_bindings = [record for record in binding_records if record["error"] is None]
    if len(valid_bindings) == world_size:
        visible_orders = {tuple(record["visible_cuda_uuids"]) for record in valid_bindings}
        if len(visible_orders) != 1:
            failures.append("ranks do not share the same CUDA-visible UUID ordering")
        for key in ("nvidia_smi_index", "canonical_uuid", "pci_bus_id"):
            if len({record[key] for record in valid_bindings}) != world_size:
                failures.append(f"rank-to-GPU bindings do not have unique {key} values")
    rank_uuids = [_canonical_gpu_uuid(item["uuid"]) for item in identities]
    rank_pci_ids = [_canonical_pci_bus_id(item["pci_bus_id"]) for item in identities]
    if any(value is None for value in rank_uuids) or len(set(rank_uuids)) != world_size:
        failures.append("every rank must expose one unique nvidia-smi GPU UUID")
    if any(value is None for value in rank_pci_ids) or len(set(rank_pci_ids)) != world_size:
        failures.append("every rank must expose one unique nvidia-smi PCI BDF")
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
        "nvidia_smi_inventory": nvidia_smi_inventory,
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
