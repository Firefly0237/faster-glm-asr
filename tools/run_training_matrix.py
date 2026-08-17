"""Validate and orchestrate private decoder distributed-training artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.distributed_training.release import (  # noqa: E402
    MATRIX_CONFIG_NAMES,
    _read_json,
    _sha256_file,
    _write_json_atomic,
    aggregate_matrix,
    aggregate_metrics,
    build_matrix_plan,
    build_source_lock,
    compare_trajectories,
    training_command,
    validate_source_lock,
    validate_strong_scaling_matrix,
    validate_training_config,
)

from faster_glm_asr.benchmarking.provenance import (  # noqa: E402
    validate_readiness_report as validate_asr_readiness_report,
)

DEFAULT_CONFIG_DIR = REPOSITORY_ROOT / "configs" / "training"
DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "private" / "training"
DEFAULT_ENVIRONMENT = REPOSITORY_ROOT / "configs" / "environments" / "rtx3090-ddp-v1.json"
DEFAULT_DATA_MANIFEST = DEFAULT_CONFIG_DIR / "synthetic-data-manifest.json"
DEFAULT_READINESS_REPORT = (
    REPOSITORY_ROOT / "artifacts" / "private" / "rtx3090-readiness" / "full.json"
)
DEFAULT_TRAINING_PREFLIGHT = DEFAULT_ARTIFACT_ROOT / "preflight-4x3090.json"
TRAINING_PREFLIGHT_SCHEMA = "decoder-training-nccl-preflight/2"
TRAINING_PREFLIGHT_SCOPE = "single-node decoder systems experiment"
TRAINING_PREFLIGHT_PATH = REPOSITORY_ROOT / "experiments/distributed_training/preflight.py"
TRAINING_TRAIN_PATH = REPOSITORY_ROOT / "experiments/distributed_training/train.py"


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)


def _positive_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0


def _canonical_gpu_uuid(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().lower()
    for prefix in ("gpu-", "mig-"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized


def _canonical_pci_bus_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"([0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])$", value.strip())
    return match.group(1).lower() if match else None


def _print(document: Any) -> None:
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))


def _validate_readiness_report(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    contract = document.get("contract")
    try:
        validate_asr_readiness_report(path, repository_root=REPOSITORY_ROOT)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            f"{path}: a recomputed pure-pass full readiness report is required"
        ) from exc
    if not isinstance(contract, dict) or contract.get("sha256") != _sha256_file(
        DEFAULT_ENVIRONMENT
    ):
        raise ValueError(f"{path}: readiness environment contract differs")
    return {
        **document,
        "world_size": 4,
        "environment_candidate": {
            "path": str(DEFAULT_ENVIRONMENT.resolve()),
            "sha256": contract["sha256"],
        },
    }


def _validate_training_preflight(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    required_keys = {
        "schema_version",
        "created_at_utc",
        "pass",
        "scope",
        "world_size",
        "producer",
        "invocation",
        "environment_candidate",
        "rank_inventory",
        "peer_access",
        "all_reduce",
        "nvidia_smi_inventory",
        "nvidia_smi_topology",
        "nccl_environment",
    }
    if set(document) != required_keys:
        raise ValueError(f"{path}: training preflight schema keys are incomplete")
    if (
        document.get("schema_version") != TRAINING_PREFLIGHT_SCHEMA
        or document.get("pass") is not True
        or document.get("scope") != TRAINING_PREFLIGHT_SCOPE
        or document.get("world_size") != 4
    ):
        raise ValueError(
            f"{path}: the additional four-rank decoder-training NCCL preflight must pass"
        )
    created_at = document.get("created_at_utc")
    try:
        parsed_created_at = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{path}: training preflight timestamp is invalid") from exc
    if parsed_created_at.tzinfo is None or parsed_created_at.utcoffset() != UTC.utcoffset(None):
        raise ValueError(f"{path}: training preflight timestamp must be UTC")

    producer = document.get("producer")
    expected_producer = {
        "preflight_path": "experiments/distributed_training/preflight.py",
        "preflight_sha256": _sha256_file(TRAINING_PREFLIGHT_PATH),
        "train_path": "experiments/distributed_training/train.py",
        "train_sha256": _sha256_file(TRAINING_TRAIN_PATH),
    }
    if producer != expected_producer:
        raise ValueError(f"{path}: training preflight producer hashes differ from this checkout")
    invocation = document.get("invocation")
    expected_invocation = {
        "expected_world_size": 4,
        "expected_gpu_name": "RTX 3090",
        "minimum_memory_gib": 23.0,
        "all_reduce_mib": 64,
        "warmup": 5,
        "repetitions": 10,
    }
    if invocation != expected_invocation:
        raise ValueError(f"{path}: training preflight invocation is not canonical")

    environment = document.get("environment_candidate")
    if (
        not isinstance(environment, dict)
        or set(environment) != {"path", "sha256"}
        or environment.get("sha256") != _sha256_file(DEFAULT_ENVIRONMENT)
    ):
        raise ValueError(f"{path}: training preflight environment hash differs")

    ranks = document.get("rank_inventory")
    rank_keys = {
        "rank",
        "local_rank",
        "hostname",
        "name",
        "uuid",
        "pci_bus_id",
        "total_memory_bytes",
        "compute_capability",
        "bf16_supported",
        "visible_device_count",
    }
    if (
        not isinstance(ranks, list)
        or len(ranks) != 4
        or any(not isinstance(item, dict) or set(item) != rank_keys for item in ranks)
    ):
        raise ValueError(f"{path}: training preflight rank inventory is invalid")
    if sorted(item["rank"] for item in ranks) != list(range(4)) or sorted(
        item["local_rank"] for item in ranks
    ) != list(range(4)):
        raise ValueError(f"{path}: training preflight ranks are not one-to-one")
    if len({item["hostname"] for item in ranks}) != 1:
        raise ValueError(f"{path}: training preflight ranks are not on one host")
    rank_uuids = [_canonical_gpu_uuid(item["uuid"]) for item in ranks]
    rank_pci_ids = [_canonical_pci_bus_id(item["pci_bus_id"]) for item in ranks]
    if any(value is None for value in rank_uuids) or len(set(rank_uuids)) != 4:
        raise ValueError(f"{path}: training preflight GPU UUIDs are missing or duplicated")
    if any(value is None for value in rank_pci_ids) or len(set(rank_pci_ids)) != 4:
        raise ValueError(f"{path}: training preflight PCI bus IDs are missing or duplicated")
    rank_identity_pairs = set(zip(rank_uuids, rank_pci_ids, strict=True))
    for item in ranks:
        if (
            "rtx 3090" not in str(item["name"]).lower()
            or not isinstance(item["total_memory_bytes"], int)
            or item["total_memory_bytes"] < 23 * 1024**3
            or item["compute_capability"] != [8, 6]
            or item["bf16_supported"] is not True
            or not isinstance(item["visible_device_count"], int)
            or item["visible_device_count"] < 4
        ):
            raise ValueError(f"{path}: training preflight rank capability is invalid")

    peer_access = document.get("peer_access")
    if (
        not isinstance(peer_access, list)
        or len(peer_access) != 4
        or any(
            not isinstance(row, list)
            or len(row) != 4
            or any(type(value) is not bool for value in row)
            for row in peer_access
        )
        or any(peer_access[index][index] is not True for index in range(4))
    ):
        raise ValueError(f"{path}: training preflight peer matrix is invalid")

    all_reduce = document.get("all_reduce")
    all_reduce_keys = {
        "tensor_bytes",
        "warmup_repetitions",
        "measured_repetitions",
        "critical_rank_samples_s",
        "mean_critical_rank_s",
        "algorithm_bandwidth_bytes_per_second",
        "estimated_ring_bus_bandwidth_bytes_per_second",
        "rank_results",
    }
    if not isinstance(all_reduce, dict) or set(all_reduce) != all_reduce_keys:
        raise ValueError(f"{path}: training preflight all-reduce schema is invalid")
    if (
        all_reduce["tensor_bytes"] != 64 * 1024**2
        or all_reduce["warmup_repetitions"] != 5
        or all_reduce["measured_repetitions"] != 10
    ):
        raise ValueError(f"{path}: training preflight all-reduce parameters differ")
    rank_results = all_reduce["rank_results"]
    result_keys = {"rank", "samples_s", "mean_s", "min_s", "max_s"}
    if (
        not isinstance(rank_results, list)
        or len(rank_results) != 4
        or sorted(item.get("rank") for item in rank_results if isinstance(item, dict))
        != list(range(4))
        or any(not isinstance(item, dict) or set(item) != result_keys for item in rank_results)
    ):
        raise ValueError(f"{path}: training preflight all-reduce rank results are invalid")
    for item in rank_results:
        samples = item["samples_s"]
        if (
            not isinstance(samples, list)
            or len(samples) != 10
            or any(not _positive_finite(value) for value in samples)
            or not _close(float(item["mean_s"]), sum(samples) / len(samples))
            or not _close(float(item["min_s"]), min(samples))
            or not _close(float(item["max_s"]), max(samples))
        ):
            raise ValueError(f"{path}: training preflight all-reduce samples are invalid")
    expected_critical = [
        max(float(item["samples_s"][index]) for item in rank_results) for index in range(10)
    ]
    critical = all_reduce["critical_rank_samples_s"]
    if (
        not isinstance(critical, list)
        or len(critical) != 10
        or any(
            not _positive_finite(actual) or not _close(float(actual), expected)
            for actual, expected in zip(critical, expected_critical, strict=True)
        )
    ):
        raise ValueError(f"{path}: training preflight critical timings are invalid")
    expected_mean = sum(expected_critical) / len(expected_critical)
    expected_algorithm_bw = (64 * 1024**2) / expected_mean
    if (
        not _close(float(all_reduce["mean_critical_rank_s"]), expected_mean)
        or not _close(
            float(all_reduce["algorithm_bandwidth_bytes_per_second"]),
            expected_algorithm_bw,
        )
        or not _close(
            float(all_reduce["estimated_ring_bus_bandwidth_bytes_per_second"]),
            expected_algorithm_bw * 1.5,
        )
    ):
        raise ValueError(f"{path}: training preflight bandwidth derivation is invalid")

    expected_commands = {
        "nvidia_smi_inventory": [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        "nvidia_smi_topology": ["nvidia-smi", "topo", "-m"],
    }
    for name, command in expected_commands.items():
        capture = document.get(name)
        if (
            not isinstance(capture, dict)
            or set(capture) != {"command", "returncode", "stdout", "stderr"}
            or capture.get("command") != command
            or capture.get("returncode") != 0
            or not str(capture.get("stdout", "")).strip()
        ):
            raise ValueError(f"{path}: {name} capture is incomplete")
    inventory_rows = []
    for line in document["nvidia_smi_inventory"]["stdout"].splitlines():
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 6:
            raise ValueError(f"{path}: nvidia-smi inventory row has an invalid shape")
        try:
            inventory_rows.append(
                {
                    "index": int(columns[0]),
                    "uuid": _canonical_gpu_uuid(columns[1]),
                    "pci_bus_id": _canonical_pci_bus_id(columns[2]),
                    "name": columns[3],
                    "memory_total_mib": int(columns[4]),
                    "driver_version": columns[5],
                }
            )
        except ValueError as exc:
            raise ValueError(f"{path}: nvidia-smi inventory values are invalid") from exc
    inventory_identity_pairs = {(item["uuid"], item["pci_bus_id"]) for item in inventory_rows}
    if (
        len(inventory_rows) != 4
        or sorted(item["index"] for item in inventory_rows) != list(range(4))
        or inventory_identity_pairs != rank_identity_pairs
        or any(
            item["uuid"] is None
            or item["pci_bus_id"] is None
            or "rtx 3090" not in item["name"].lower()
            or item["memory_total_mib"] < 23000
            or not item["driver_version"]
            for item in inventory_rows
        )
    ):
        raise ValueError(f"{path}: nvidia-smi inventory does not match rank identities")
    nccl_environment = document.get("nccl_environment")
    if not isinstance(nccl_environment, dict) or any(
        not isinstance(key, str) or not key.startswith("NCCL_") for key in nccl_environment
    ):
        raise ValueError(f"{path}: training preflight NCCL environment is invalid")
    return document


def _evidence_references(
    readiness_path: Path | None, training_preflight_path: Path | None
) -> dict[str, Any]:
    def reference(path: Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        resolved = path.resolve()
        return {
            "path": str(resolved),
            "sha256": _sha256_file(resolved) if resolved.is_file() else None,
            "status": "present" if resolved.is_file() else "pending",
        }

    return {
        "full_readiness_report": reference(readiness_path),
        "training_nccl_preflight": reference(training_preflight_path),
    }


def _validate_evidence_pair(
    readiness_path: Path | None,
    training_preflight_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if readiness_path is None or training_preflight_path is None:
        raise ValueError("execution requires both --readiness-report and --preflight-report")
    readiness = _validate_readiness_report(readiness_path)
    training_preflight = _validate_training_preflight(training_preflight_path)
    readiness_environment = readiness["environment_candidate"]["sha256"]
    training_environment = training_preflight["environment_candidate"]["sha256"]
    if readiness_environment != training_environment:
        raise ValueError(
            "full readiness and training NCCL preflight bind different environment candidates"
        )
    full_gpus = readiness["hardware"]["gpu_probe_before"]["gpus"]
    short_ranks = training_preflight["rank_inventory"]
    full_uuids = {_canonical_gpu_uuid(item.get("uuid")) for item in full_gpus}
    short_uuids = {_canonical_gpu_uuid(item.get("uuid")) for item in short_ranks}
    if None in full_uuids or full_uuids != short_uuids:
        raise ValueError("full readiness and training NCCL preflight bind different GPU UUIDs")
    full_pci = {_canonical_pci_bus_id(item.get("pci_bus_id")) for item in full_gpus}
    short_pci = {_canonical_pci_bus_id(item.get("pci_bus_id")) for item in short_ranks}
    if None not in full_pci and None not in short_pci and full_pci != short_pci:
        raise ValueError("full readiness and training NCCL preflight bind different PCI devices")
    references = _evidence_references(readiness_path, training_preflight_path)
    return readiness, training_preflight, references


def _assert_evidence_source_lock(source_lock: dict[str, Any], evidence: dict[str, Any]) -> None:
    role_to_hashes: dict[str, set[str]] = {}
    for record in source_lock["files"]:
        role_to_hashes.setdefault(str(record["role"]), set()).add(str(record["sha256"]))
    expected = {
        "readiness_report": evidence["full_readiness_report"]["sha256"],
        "training_preflight": evidence["training_nccl_preflight"]["sha256"],
    }
    for role, sha256 in expected.items():
        if role_to_hashes.get(role) != {sha256}:
            raise ValueError(
                f"source lock must bind the exact {role} SHA256 used to authorize execution"
            )


def _assert_release_consensus(
    config: Path,
    source_lock: dict[str, Any],
    preflight: dict[str, Any],
    *,
    expected_data_schema: str,
) -> Path:
    config_relative = config.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    if config_relative not in source_lock["paths"]:
        raise ValueError(f"source lock does not bind selected config {config_relative}")
    environment_hash = preflight["environment_candidate"]["sha256"]
    locked_environment_hashes = {
        record["sha256"] for record in source_lock["files"] if record["role"] == "environment"
    }
    if environment_hash not in locked_environment_hashes:
        raise ValueError("preflight environment candidate is not the source-locked environment")
    manifests = [
        REPOSITORY_ROOT / record["path"]
        for record in source_lock["files"]
        if record["role"] == "data_manifest"
    ]
    if len(manifests) != 1:
        raise ValueError("source lock must bind exactly one data manifest")
    manifest = _read_json(manifests[0])
    if manifest.get("schema_version") != expected_data_schema:
        raise ValueError(f"source-locked data manifest must use {expected_data_schema}")
    return manifests[0]


def _validate_byte_corpus(config_validation: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    data = config_validation["document"]["data"]
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("byte corpus manifest outputs must be an array")
    splits = {
        record.get("split"): record
        for record in outputs
        if isinstance(record, dict) and record.get("split") in {"train", "validation"}
    }
    if set(splits) != {"train", "validation"}:
        raise ValueError("byte corpus manifest must define train and validation outputs")
    minimum_sizes = {"train": 64 * 1024**2, "validation": 4 * 1024**2}
    observed = {}
    for split in ("train", "validation"):
        record = splits[split]
        if not isinstance(record, dict):
            raise ValueError(f"invalid {split} split manifest")
        configured_paths = data[f"{split}_files"]
        if len(configured_paths) != 1:
            raise ValueError(f"config {split}_files must contain one path")
        path = (manifest_path.parent / record["filename"]).resolve()
        configured_path = (REPOSITORY_ROOT / configured_paths[0]).resolve()
        if configured_path != path:
            raise ValueError(f"config {split}_files differs from manifest output")
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != record.get("size_bytes") or size < minimum_sizes[split]:
            raise ValueError(f"{split} size mismatch or below {minimum_sizes[split]} bytes")
        digest = _sha256_file(path)
        if digest != record.get("sha256"):
            raise ValueError(f"{split} SHA256 differs from manifest")
        observed[split] = {
            "path": str(path),
            "size_bytes": size,
            "sha256": digest,
            "document_count": record.get("selected_document_count"),
        }
    if observed["train"]["sha256"] == observed["validation"]["sha256"]:
        raise ValueError("train and validation corpus bytes must differ")
    return observed


def _run_command(command: Sequence[str]) -> None:
    subprocess.run(list(command), cwd=REPOSITORY_ROOT, check=True)


def _validate_command(args: argparse.Namespace) -> int:
    matrix = validate_strong_scaling_matrix(args.config_dir)
    fallback = validate_training_config(args.config_dir / "fallback-275m-w4.json")
    smoke = validate_training_config(args.config_dir / "cpu-smoke.json")
    real_data = validate_training_config(args.config_dir / "tinystories-275m-w4.json")
    _print(
        {
            "schema_version": "decoder-training-validation/1",
            "strong_scaling": [
                {
                    key: item[key]
                    for key in (
                        "path",
                        "sha256",
                        "parameter_count",
                        "world_size",
                        "tokens_per_update",
                        "measurement_eligible_steps",
                    )
                }
                for item in matrix
            ],
            "fallback": {
                key: fallback[key]
                for key in (
                    "path",
                    "sha256",
                    "parameter_count",
                    "world_size",
                    "tokens_per_update",
                )
            },
            "cpu_smoke": {
                key: smoke[key]
                for key in (
                    "path",
                    "sha256",
                    "parameter_count",
                    "world_size",
                    "tokens_per_update",
                )
            },
            "real_data": {
                key: real_data[key]
                for key in (
                    "path",
                    "sha256",
                    "parameter_count",
                    "world_size",
                    "tokens_per_update",
                    "data_mode",
                )
            },
        }
    )
    return 0


def _source_lock_command(args: argparse.Namespace) -> int:
    config_paths = [
        args.config_dir / name
        for name in (
            *MATRIX_CONFIG_NAMES,
            "fallback-275m-w4.json",
            "cpu-smoke.json",
            "tinystories-275m-w4.json",
            "resume-275m-w4.json",
            "trajectory-275m-w4.json",
        )
    ]
    if (args.readiness_report is None) != (args.preflight_report is None):
        raise ValueError("source-lock must bind both readiness reports or neither")
    evidence_paths = []
    evidence = None
    if args.readiness_report is not None and args.preflight_report is not None:
        _readiness, _training_preflight, evidence = _validate_evidence_pair(
            args.readiness_report, args.preflight_report
        )
        evidence_paths = [
            (args.readiness_report, "readiness_report"),
            (args.preflight_report, "training_preflight"),
        ]
    document = build_source_lock(
        args.output,
        config_paths,
        args.data_manifest,
        args.environment_candidate,
        evidence_paths,
    )
    _print(
        {
            "path": str(args.output.resolve()),
            "schema_version": document["schema_version"],
            "file_count": len(document["files"]),
            "bound_runtime_evidence": evidence,
        }
    )
    return 0


def _matrix_command(args: argparse.Namespace) -> int:
    validated_lock = validate_source_lock(args.source_lock)
    plan = build_matrix_plan(args.config_dir, args.artifact_root, args.source_lock)
    plan["required_evidence"] = _evidence_references(args.readiness_report, args.preflight_report)
    if args.validate_only:
        _print(
            {
                "schema_version": plan["schema_version"],
                "validated_source_lock": validated_lock,
                "run_count": len(plan["runs"]),
                "required_evidence": plan["required_evidence"],
            }
        )
        return 0
    if args.aggregate_only:
        _readiness, _training_preflight, evidence = _validate_evidence_pair(
            args.readiness_report, args.preflight_report
        )
        _assert_evidence_source_lock(validated_lock, evidence)
        aggregate = aggregate_matrix(args.artifact_root)
        provenance = aggregate["provenance"]
        if provenance["required_evidence"] != evidence:
            raise ValueError("aggregate provenance differs from the validated readiness artifacts")
        if provenance["source_lock"] != {
            "path": str(args.source_lock.resolve()),
            "sha256": validated_lock["sha256"],
        }:
            raise ValueError("aggregate provenance differs from the selected source lock")
        output = args.artifact_root / "strong-scaling" / "aggregate.json"
        _write_json_atomic(output, aggregate)
        _print(aggregate)
        return 0
    if not args.execute:
        _print(plan)
        return 0
    readiness, training_preflight, evidence = _validate_evidence_pair(
        args.readiness_report, args.preflight_report
    )
    _assert_evidence_source_lock(validated_lock, evidence)
    plan["required_evidence"] = evidence
    for run in plan["runs"]:
        _assert_release_consensus(
            Path(run["config"]),
            validated_lock,
            readiness,
            expected_data_schema="decoder-training-data-manifest/1",
        )
    matrix_root = args.artifact_root / "strong-scaling"
    plan_path = matrix_root / "plan.json"
    journal_path = matrix_root / "journal.json"
    aggregate_path = matrix_root / "aggregate.json"
    for artifact in (plan_path, journal_path, aggregate_path):
        if artifact.exists():
            raise FileExistsError(f"refusing to overwrite {artifact}")
    _write_json_atomic(plan_path, plan)
    journal: dict[str, Any] = {
        "schema_version": "decoder-training-matrix-journal/1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "plan": {
            "path": str(plan_path.resolve()),
            "sha256": _sha256_file(plan_path),
        },
        "source_lock_sha256": validated_lock["sha256"],
        "required_evidence": evidence,
        "environment_candidate": readiness["environment_candidate"],
        "training_preflight_environment": training_preflight["environment_candidate"],
        "runs": [],
    }
    _write_json_atomic(journal_path, journal)
    try:
        for run in plan["runs"]:
            run_record = {
                "name": run["name"],
                "status": "running",
                "started_at_utc": datetime.now(UTC).isoformat(),
                "command": run["command"],
            }
            journal["runs"].append(run_record)
            _write_json_atomic(journal_path, journal, overwrite=True)
            try:
                _run_command(run["command"])
                run_record["status"] = "complete"
                run_record["summary"] = aggregate_metrics(Path(run["output_dir"]))
            except Exception as exc:
                run_record["status"] = "failed"
                run_record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                run_record["finished_at_utc"] = datetime.now(UTC).isoformat()
                _write_json_atomic(journal_path, journal, overwrite=True)
        journal["status"] = "complete"
        journal["finished_at_utc"] = datetime.now(UTC).isoformat()
        journal["aggregate_output"] = str(aggregate_path.resolve())
        _write_json_atomic(journal_path, journal, overwrite=True)
        aggregate = aggregate_matrix(args.artifact_root)
        _write_json_atomic(aggregate_path, aggregate)
        _print(aggregate)
        return 0
    except Exception:
        journal["status"] = "failed"
        journal["finished_at_utc"] = datetime.now(UTC).isoformat()
        _write_json_atomic(journal_path, journal, overwrite=True)
        raise


def _real_data_command(args: argparse.Namespace) -> int:
    config_validation = validate_training_config(args.config)
    if (
        config_validation["world_size"] != 4
        or config_validation["data_mode"] != "byte_files"
        or config_validation["parameter_count"] != 275_621_120
        or config_validation["document"]["training"]["steps"] < 200
    ):
        raise ValueError(
            "real-data config must be the >=200-update, 4-rank, 275,621,120 "
            "parameter byte-file workload"
        )
    source_lock = validate_source_lock(args.source_lock)
    output = args.artifact_root / "real-data" / "tinystories-275m-w4"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "-m",
        "experiments.distributed_training.train",
        "--config",
        str(args.config.resolve()),
        "--output-dir",
        str(output.resolve()),
        "--source-lock",
        str(args.source_lock.resolve()),
    ]
    evidence = _evidence_references(args.readiness_report, args.preflight_report)
    if not args.execute:
        _print(
            {
                "schema_version": "decoder-training-real-data-plan/1",
                "config": config_validation["path"],
                "output_dir": str(output.resolve()),
                "command": command,
                "required_evidence": evidence,
                "requirements": {
                    "minimum_train_bytes": 64 * 1024**2,
                    "minimum_validation_bytes": 4 * 1024**2,
                    "updates": config_validation["document"]["training"]["steps"],
                    "validation_interval": config_validation["document"]["training"][
                        "eval_interval"
                    ],
                },
            }
        )
        return 0
    readiness, training_preflight, evidence = _validate_evidence_pair(
        args.readiness_report, args.preflight_report
    )
    _assert_evidence_source_lock(source_lock, evidence)
    manifest_path = _assert_release_consensus(
        args.config,
        source_lock,
        readiness,
        expected_data_schema="training-corpus-manifest-v0.1",
    )
    corpus = _validate_byte_corpus(config_validation, manifest_path)
    journal_path = args.artifact_root / "real-data" / "journal.json"
    journal = {
        "schema_version": "decoder-training-real-data-journal/1",
        "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_lock_sha256": source_lock["sha256"],
        "required_evidence": evidence,
        "environment_candidate": readiness["environment_candidate"],
        "training_preflight_environment": training_preflight["environment_candidate"],
        "corpus": corpus,
        "command": command,
    }
    _write_json_atomic(journal_path, journal)
    try:
        _run_command(command)
        summary = aggregate_metrics(output)
        records = [
            json.loads(line)
            for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        validation_records = [record for record in records if record.get("event") == "validation"]
        starts = [record for record in records if record.get("event") == "run_start"]
        ends = [record for record in records if record.get("event") == "run_end"]
        updates = [record for record in records if record.get("event") == "train_update"]
        expected_steps = int(config_validation["document"]["training"]["steps"])
        eval_interval = int(config_validation["document"]["training"]["eval_interval"])
        expected_validation_steps = list(range(eval_interval, expected_steps + 1, eval_interval))
        run_id = starts[0].get("run_id") if len(starts) == 1 else None
        final_checkpoints = [
            record
            for record in records
            if record.get("event") == "checkpoint" and record.get("step") == expected_steps
        ]
        checkpoint_path = output / "checkpoint.pt"
        if (
            not isinstance(run_id, str)
            or not run_id
            or any(record.get("run_id") != run_id for record in records)
            or len(ends) != 1
            or ends[0].get("status") != "complete"
            or ends[0].get("completed_steps") != expected_steps
            or len(updates) != expected_steps
            or [record.get("step") for record in updates] != list(range(expected_steps))
            or [record.get("step") for record in validation_records] != expected_validation_steps
            or any(
                not isinstance(record.get("loss"), (int, float))
                or isinstance(record.get("loss"), bool)
                or not math.isfinite(float(record["loss"]))
                for record in validation_records
            )
            or len(final_checkpoints) != 1
            or not checkpoint_path.is_file()
            or final_checkpoints[0].get("sha256") != _sha256_file(checkpoint_path)
            or Path(str(final_checkpoints[0].get("path", ""))).resolve()
            != checkpoint_path.resolve()
        ):
            raise RuntimeError(
                "real-data run must complete every update and emit the exact finite "
                "validation/checkpoint trajectory"
            )
        journal.update(
            {
                "status": "complete",
                "summary": summary,
                "validation": validation_records,
                "raw_artifacts": {
                    "metrics": {
                        "path": str((output / "metrics.jsonl").resolve()),
                        "sha256": _sha256_file(output / "metrics.jsonl"),
                    },
                    "checkpoint": {
                        "path": str(checkpoint_path.resolve()),
                        "sha256": _sha256_file(checkpoint_path),
                    },
                },
            }
        )
        _write_json_atomic(journal_path, journal, overwrite=True)
        _print(journal)
        return 0
    except Exception as exc:
        journal["status"] = "failed"
        journal["error"] = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(journal_path, journal, overwrite=True)
        raise


def _single_process_training_command(
    config: Path,
    output: Path,
    source_lock: Path | None,
    *,
    resume: bool = False,
    stop_after_step: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.distributed_training.train",
        "--config",
        str(config.resolve()),
        "--output-dir",
        str(output.resolve()),
    ]
    if source_lock is not None:
        command.extend(["--source-lock", str(source_lock.resolve())])
    if resume:
        command.extend(["--resume", str((output / "checkpoint.pt").resolve())])
    if stop_after_step is not None:
        command.extend(["--stop-after-step", str(stop_after_step)])
    return command


def _trajectory_command(args: argparse.Namespace) -> int:
    config_validation = validate_training_config(args.config)
    training = config_validation["document"]["training"]
    if training["deterministic_algorithms"] is not True:
        raise ValueError("trajectory config must enable deterministic_algorithms")
    if (
        args.stop_after_step <= 0
        or args.stop_after_step >= training["steps"]
        or args.stop_after_step % training["checkpoint_interval"] != 0
    ):
        raise ValueError("trajectory stop must be an interior checkpoint boundary")
    validated_lock = (
        validate_source_lock(args.source_lock) if args.source_lock is not None else None
    )
    if training["require_source_lock"] and validated_lock is None:
        raise ValueError("formal trajectory config requires --source-lock")
    world_size = int(config_validation["world_size"])
    root = args.artifact_root / "trajectory" / args.config.stem
    baseline = root / "uninterrupted"
    resumed = root / "interrupted-resumed"
    report_path = root / "comparison.json"
    if args.compare_only:
        report = compare_trajectories(
            baseline,
            resumed,
            rtol=args.rtol,
            atol=args.atol,
            report_path=report_path,
        )
        _print(report)
        return 0 if report["pass"] else 1

    def build_command(
        output: Path,
        *,
        resume: bool = False,
        stop_after_step: int | None = None,
    ) -> list[str]:
        if world_size > 1:
            return training_command(
                args.config,
                output,
                world_size,
                args.source_lock,
                resume=resume,
                stop_after_step=stop_after_step,
            )
        return _single_process_training_command(
            args.config,
            output,
            args.source_lock,
            resume=resume,
            stop_after_step=stop_after_step,
        )

    commands = [
        build_command(baseline),
        build_command(resumed, stop_after_step=args.stop_after_step),
        build_command(resumed, resume=True),
    ]
    evidence = _evidence_references(args.readiness_report, args.preflight_report)
    if not args.execute:
        _print(
            {
                "schema_version": "decoder-training-trajectory-plan/1",
                "baseline_dir": str(baseline.resolve()),
                "resumed_dir": str(resumed.resolve()),
                "report": str(report_path.resolve()),
                "world_size": world_size,
                "comparison": (
                    "per-step loss plus final model, AdamW, process RNG and "
                    "data-cursor structural digests"
                ),
                "required_evidence": evidence if world_size > 1 else None,
                "commands": commands,
            }
        )
        return 0
    if world_size > 1:
        if validated_lock is None:
            raise ValueError("multi-rank trajectory requires --source-lock")
        readiness, _training_preflight, _evidence = _validate_evidence_pair(
            args.readiness_report, args.preflight_report
        )
        _assert_evidence_source_lock(validated_lock, _evidence)
        _assert_release_consensus(
            args.config,
            validated_lock,
            readiness,
            expected_data_schema="decoder-training-data-manifest/1",
        )
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite trajectory artifacts in {root}")
    journal_path = root / "journal.json"
    journal = {
        "schema_version": "decoder-training-trajectory-journal/1",
        "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_lock_sha256": (validated_lock["sha256"] if validated_lock is not None else None),
        "required_evidence": (_evidence if world_size > 1 else None),
        "commands": commands,
    }
    _write_json_atomic(journal_path, journal)
    try:
        for command in commands:
            _run_command(command)
        report = compare_trajectories(
            baseline,
            resumed,
            rtol=args.rtol,
            atol=args.atol,
            report_path=report_path,
        )
        journal["status"] = "complete" if report["pass"] else "comparison_failed"
        journal["comparison_report"] = {
            "path": str(report_path.resolve()),
            "sha256": _sha256_file(report_path),
            "pass": report["pass"],
        }
        _write_json_atomic(journal_path, journal, overwrite=True)
        _print(report)
        return 0 if report["pass"] else 1
    except Exception as exc:
        journal["status"] = "failed"
        journal["error"] = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(journal_path, journal, overwrite=True)
        raise


def _profile_command(args: argparse.Namespace) -> int:
    config_validation = validate_training_config(args.config)
    world_size = int(config_validation["world_size"])
    if world_size != 4 or config_validation["document"]["training"]["dtype"] != "bf16":
        raise ValueError("the formal profiler workload must be a four-rank BF16 config")
    source_lock = validate_source_lock(args.source_lock)
    root = args.artifact_root / "profiler" / args.config.stem
    training_output = root / "training"
    trace_output = root / "traces"
    journal_path = root / "journal.json"
    command = training_command(
        args.config,
        training_output,
        world_size,
        args.source_lock,
        profiler_dir=trace_output,
    )
    plan = {
        "schema_version": "decoder-training-profiler-plan/1",
        "canonical_scaling_aggregate_eligible": False,
        "config": config_validation["path"],
        "source_lock_sha256": source_lock["sha256"],
        "output_dir": str(training_output.resolve()),
        "trace_dir": str(trace_output.resolve()),
        "command": command,
        "required_evidence": _evidence_references(args.readiness_report, args.preflight_report),
    }
    if not args.execute:
        _print(plan)
        return 0
    readiness, training_preflight, evidence = _validate_evidence_pair(
        args.readiness_report, args.preflight_report
    )
    _assert_evidence_source_lock(source_lock, evidence)
    _assert_release_consensus(
        args.config,
        source_lock,
        readiness,
        expected_data_schema="decoder-training-data-manifest/1",
    )
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite profiler artifacts in {root}")
    journal = {
        **plan,
        "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "required_evidence": evidence,
        "environment_candidate": readiness["environment_candidate"],
        "training_preflight_environment": training_preflight["environment_candidate"],
    }
    _write_json_atomic(journal_path, journal)
    try:
        _run_command(command)
        rank_outputs = []
        for rank in range(world_size):
            rank_dir = trace_output / f"rank-{rank:03d}"
            key_averages = rank_dir / "key-averages.json"
            traces = sorted(rank_dir.glob("trace-step-*.json"))
            if not key_averages.is_file() or not traces:
                raise RuntimeError(f"rank {rank} did not emit key averages and a Chrome trace")
            rank_outputs.append(
                {
                    "rank": rank,
                    "key_averages": {
                        "path": str(key_averages.resolve()),
                        "sha256": _sha256_file(key_averages),
                    },
                    "traces": [
                        {
                            "path": str(path.resolve()),
                            "sha256": _sha256_file(path),
                        }
                        for path in traces
                    ],
                }
            )
        journal.update(
            {
                "status": "complete",
                "rank_outputs": rank_outputs,
                "diagnostic_training_summary": aggregate_metrics(training_output),
            }
        )
        _write_json_atomic(journal_path, journal, overwrite=True)
        _print(journal)
        return 0
    except Exception as exc:
        journal["status"] = "failed"
        journal["error"] = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(journal_path, journal, overwrite=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    validate.set_defaults(handler=_validate_command)

    lock = subparsers.add_parser("source-lock")
    lock.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    lock.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    lock.add_argument("--environment-candidate", type=Path, default=DEFAULT_ENVIRONMENT)
    lock.add_argument("--readiness-report", type=Path)
    lock.add_argument("--preflight-report", type=Path)
    lock.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "source-lock.json",
    )
    lock.set_defaults(handler=_source_lock_command)

    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    matrix.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    matrix.add_argument(
        "--source-lock",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "source-lock.json",
    )
    mode = matrix.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    matrix.add_argument("--readiness-report", type=Path, default=DEFAULT_READINESS_REPORT)
    matrix.add_argument("--preflight-report", type=Path, default=DEFAULT_TRAINING_PREFLIGHT)
    matrix.set_defaults(handler=_matrix_command)

    real_data = subparsers.add_parser("real-data")
    real_data.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "tinystories-275m-w4.json",
    )
    real_data.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    real_data.add_argument(
        "--source-lock",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "real-data-source-lock.json",
    )
    real_data.add_argument("--readiness-report", type=Path, default=DEFAULT_READINESS_REPORT)
    real_data.add_argument("--preflight-report", type=Path, default=DEFAULT_TRAINING_PREFLIGHT)
    real_data.add_argument("--execute", action="store_true")
    real_data.set_defaults(handler=_real_data_command)

    trajectory = subparsers.add_parser("trajectory")
    trajectory.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "trajectory-275m-w4.json",
    )
    trajectory.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    trajectory.add_argument(
        "--source-lock",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "source-lock.json",
    )
    trajectory.add_argument("--readiness-report", type=Path, default=DEFAULT_READINESS_REPORT)
    trajectory.add_argument("--preflight-report", type=Path, default=DEFAULT_TRAINING_PREFLIGHT)
    trajectory.add_argument("--stop-after-step", type=int, default=4)
    trajectory.add_argument("--rtol", type=float, default=0.0)
    trajectory.add_argument("--atol", type=float, default=0.0)
    trajectory_mode = trajectory.add_mutually_exclusive_group()
    trajectory_mode.add_argument("--execute", action="store_true")
    trajectory_mode.add_argument("--compare-only", action="store_true")
    trajectory_mode.add_argument("--dry-run", action="store_true")
    trajectory.set_defaults(handler=_trajectory_command)

    profile = subparsers.add_parser("profile")
    profile.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "resume-275m-w4.json",
    )
    profile.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    profile.add_argument(
        "--source-lock",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "source-lock.json",
    )
    profile.add_argument("--readiness-report", type=Path, default=DEFAULT_READINESS_REPORT)
    profile.add_argument("--preflight-report", type=Path, default=DEFAULT_TRAINING_PREFLIGHT)
    profile.add_argument("--execute", action="store_true")
    profile.set_defaults(handler=_profile_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
