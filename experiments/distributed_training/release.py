"""Release controls for the isolated decoder distributed-training experiment.

This module intentionally keeps configuration and orchestration checks usable
without importing PyTorch. Checkpoint trajectory inspection imports PyTorch
only when it is explicitly requested.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SOURCE_LOCK_SCHEMA = "decoder-training-source-lock/1"
TRAJECTORY_SCHEMA = "decoder-training-trajectory-comparison/1"
MATRIX_CONFIG_NAMES = (
    "strong-403m-w1.json",
    "strong-403m-w2.json",
    "strong-403m-w4.json",
)
MATRIX_PASS_ORDERS = ((1, 2, 4), (4, 2, 1), (2, 4, 1))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-standard non-finite JSON number is forbidden: {value}")

    document = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_nonstandard_constant
    )
    if not isinstance(document, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return document


def _write_json_atomic(path: Path, document: Mapping[str, Any], *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            payload = json.dumps(
                dict(document), ensure_ascii=False, indent=2, sort_keys=True
            ).encode("utf-8")
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # A hard-link publication is atomic and fails if another process
            # won the destination race; unlike os.replace it cannot overwrite
            # an artifact created after the initial existence check.
            os.link(temporary, path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def parameter_count_from_model_config(model: Mapping[str, Any]) -> int:
    """Return the exact unique parameter count for the tied-head decoder."""

    required = {
        "vocab_size",
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "hidden_dim",
    }
    missing = required - set(model)
    if missing:
        raise ValueError("model config is missing: " + ", ".join(sorted(missing)))
    values = {name: model[name] for name in required}
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in values.values()
    ):
        raise ValueError("model dimensions must be positive integers")
    dim = int(values["dim"])
    heads = int(values["n_heads"])
    kv_heads = int(values["n_kv_heads"])
    if dim % heads or heads % kv_heads:
        raise ValueError("invalid head divisibility")
    head_dim = dim // heads
    attention = dim * (heads * head_dim) + 2 * dim * (kv_heads * head_dim) + dim * dim
    feed_forward = 3 * dim * int(values["hidden_dim"])
    norms = 2 * dim
    per_layer = attention + feed_forward + norms
    embedding = int(values["vocab_size"]) * dim
    final_norm = dim
    return embedding + int(values["n_layers"]) * per_layer + final_norm


def _assert_finite_tree(value: Any, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite float")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_tree(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{location}[{index}]")


def validate_training_config(path: Path) -> dict[str, Any]:
    """Validate a config without allocating the model or importing PyTorch."""

    document = _read_json(path)
    _assert_finite_tree(document)
    if set(document) != {"model", "training", "data"}:
        raise ValueError(f"{path}: top-level keys must be exactly model, training, data")
    model = document["model"]
    training = document["training"]
    data = document["data"]
    if not all(isinstance(item, dict) for item in (model, training, data)):
        raise ValueError(f"{path}: model, training, and data must be objects")
    expected_model_keys = {
        "vocab_size",
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "hidden_dim",
        "max_seq_len",
        "rope_theta",
        "rope_fraction",
        "rms_norm_eps",
        "dropout",
    }
    if set(model) != expected_model_keys:
        raise ValueError(
            f"{path}: model keys differ from the canonical schema; "
            f"missing={sorted(expected_model_keys - set(model))}, "
            f"unknown={sorted(set(model) - expected_model_keys)}"
        )
    positive_model_ints = {
        key
        for key in expected_model_keys
        if key
        in {
            "vocab_size",
            "dim",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "hidden_dim",
            "max_seq_len",
        }
    }
    for key in positive_model_ints:
        value = model[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{path}: model.{key} must be a positive integer")
    if int(model["dim"]) % int(model["n_heads"]):
        raise ValueError(f"{path}: dim must be divisible by n_heads")
    if int(model["n_heads"]) % int(model["n_kv_heads"]):
        raise ValueError(f"{path}: n_heads must be divisible by n_kv_heads")
    head_dim = int(model["dim"]) // int(model["n_heads"])
    rope_dim = int(head_dim * float(model["rope_fraction"]))
    if rope_dim < 2 or rope_dim % 2:
        raise ValueError(f"{path}: derived RoPE dimension must be positive and even")
    if not 0.0 < float(model["rope_fraction"]) <= 1.0:
        raise ValueError(f"{path}: rope_fraction must be in (0,1]")
    if not 0.0 <= float(model["dropout"]) < 1.0:
        raise ValueError(f"{path}: dropout must be in [0,1)")
    if float(model["rope_theta"]) <= 0 or float(model["rms_norm_eps"]) <= 0:
        raise ValueError(f"{path}: rope_theta and rms_norm_eps must be positive")

    required_training = {
        "experiment_name",
        "seed",
        "steps",
        "micro_batch_size",
        "sequence_length",
        "gradient_accumulation_steps",
        "learning_rate",
        "min_lr_ratio",
        "warmup_steps",
        "weight_decay",
        "beta1",
        "beta2",
        "adam_eps",
        "optimizer_impl",
        "grad_clip",
        "eval_interval",
        "eval_batches",
        "checkpoint_interval",
        "timing_warmup_steps",
        "dtype",
        "compile_model",
        "deterministic_algorithms",
        "allow_tf32",
        "expected_world_size",
        "expected_tokens_per_update",
        "expected_parameter_count",
        "require_source_lock",
        "synthetic_train_tokens",
        "synthetic_validation_tokens",
    }
    if set(training) != required_training:
        raise ValueError(
            f"{path}: training keys differ from the canonical schema; "
            f"missing={sorted(required_training - set(training))}, "
            f"unknown={sorted(set(training) - required_training)}"
        )
    positive_training_ints = {
        "steps",
        "micro_batch_size",
        "sequence_length",
        "gradient_accumulation_steps",
        "eval_interval",
        "eval_batches",
        "checkpoint_interval",
        "synthetic_train_tokens",
        "synthetic_validation_tokens",
    }
    for key in positive_training_ints:
        value = training[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{path}: training.{key} must be a positive integer")
    if (
        not isinstance(training["seed"], int)
        or isinstance(training["seed"], bool)
        or training["seed"] < 0
    ):
        raise ValueError(f"{path}: training.seed must be a non-negative integer")
    if not isinstance(training["experiment_name"], str) or not training["experiment_name"].strip():
        raise ValueError(f"{path}: training.experiment_name must be non-empty")
    for key in ("learning_rate", "adam_eps", "grad_clip"):
        value = training[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
            raise ValueError(f"{path}: training.{key} must be positive")
    if float(training["weight_decay"]) < 0:
        raise ValueError(f"{path}: training.weight_decay must be non-negative")
    if not (0 <= float(training["beta1"]) < 1 and 0 <= float(training["beta2"]) < 1):
        raise ValueError(f"{path}: Adam betas must be in [0,1)")
    if not 0 <= float(training["min_lr_ratio"]) <= 1:
        raise ValueError(f"{path}: min_lr_ratio must be in [0,1]")
    for key in (
        "expected_world_size",
        "expected_tokens_per_update",
        "expected_parameter_count",
    ):
        value = training[key]
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ValueError(f"{path}: training.{key} must be null or positive")
    for key in (
        "compile_model",
        "deterministic_algorithms",
        "allow_tf32",
        "require_source_lock",
    ):
        if not isinstance(training[key], bool):
            raise ValueError(f"{path}: training.{key} must be boolean")
    if training["optimizer_impl"] != "single_tensor":
        raise ValueError(f"{path}: optimizer_impl must be single_tensor")
    if training["dtype"] not in {"fp32", "bf16"}:
        raise ValueError(f"{path}: dtype must be fp32 or bf16")
    if int(training["sequence_length"]) > int(model["max_seq_len"]):
        raise ValueError(f"{path}: sequence_length exceeds max_seq_len")
    if not 0 <= int(training["timing_warmup_steps"]) < int(training["steps"]):
        raise ValueError(f"{path}: invalid timing_warmup_steps")
    if not 0 <= int(training["warmup_steps"]) <= int(training["steps"]):
        raise ValueError(f"{path}: invalid warmup_steps")
    mode = data.get("mode")
    if mode == "synthetic":
        if data != {"mode": "synthetic"}:
            raise ValueError(f"{path}: synthetic data config has unknown keys")
    elif mode == "byte_files":
        if set(data) != {"mode", "train_files", "validation_files"}:
            raise ValueError(f"{path}: byte_files data config has unknown keys")
        train_files = data["train_files"]
        validation_files = data["validation_files"]
        if (
            not isinstance(train_files, list)
            or not isinstance(validation_files, list)
            or not train_files
            or not validation_files
            or any(not isinstance(item, str) or not item for item in train_files)
            or any(not isinstance(item, str) or not item for item in validation_files)
        ):
            raise ValueError(f"{path}: byte_files split paths must be non-empty arrays")
        if len(set(train_files)) != len(train_files) or len(set(validation_files)) != len(
            validation_files
        ):
            raise ValueError(f"{path}: duplicate byte_files split path")
        if set(train_files) & set(validation_files):
            raise ValueError(f"{path}: train and validation paths must be disjoint")
    else:
        raise ValueError(f"{path}: data.mode must be synthetic or byte_files")
    required_vocab = 256 if mode == "byte_files" else 128
    if int(model["vocab_size"]) < required_vocab:
        raise ValueError(
            f"{path}: model.vocab_size must cover all IDs for data.mode={mode!r}; "
            f"minimum={required_vocab}"
        )

    count = parameter_count_from_model_config(model)
    if (
        training["expected_parameter_count"] is not None
        and int(training["expected_parameter_count"]) != count
    ):
        raise ValueError(
            f"{path}: expected_parameter_count={training['expected_parameter_count']} "
            f"but analytic count is {count}"
        )
    world_size = int(training["expected_world_size"] or 1)
    tokens_per_update = (
        world_size
        * int(training["micro_batch_size"])
        * int(training["sequence_length"])
        * int(training["gradient_accumulation_steps"])
    )
    if (
        training["expected_tokens_per_update"] is not None
        and int(training["expected_tokens_per_update"]) != tokens_per_update
    ):
        raise ValueError(
            f"{path}: expected_tokens_per_update="
            f"{training['expected_tokens_per_update']} but computed "
            f"{tokens_per_update}"
        )
    if int(training["checkpoint_interval"]) > int(training["steps"]):
        raise ValueError(f"{path}: checkpoint_interval exceeds steps")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "parameter_count": count,
        "world_size": world_size,
        "tokens_per_update": tokens_per_update,
        "measurement_eligible_steps": (
            int(training["steps"]) - int(training["timing_warmup_steps"])
        ),
        "data_mode": mode,
        "document": document,
    }


def validate_strong_scaling_matrix(config_dir: Path) -> list[dict[str, Any]]:
    validated = [validate_training_config(config_dir / name) for name in MATRIX_CONFIG_NAMES]
    worlds = [item["world_size"] for item in validated]
    if worlds != [1, 2, 4]:
        raise ValueError(f"strong-scaling world sizes must be [1,2,4], got {worlds}")
    if {item["tokens_per_update"] for item in validated} != {32_768}:
        raise ValueError("strong-scaling configs must hold 32,768 tokens/update")
    if {item["parameter_count"] for item in validated} != {403_097_088}:
        raise ValueError("strong-scaling configs must use the exact 403,097,088 model")
    if {item["measurement_eligible_steps"] for item in validated} != {30}:
        raise ValueError("strong-scaling configs must expose exactly 30 eligible updates")
    model_documents = [
        json.dumps(item["document"]["model"], sort_keys=True, separators=(",", ":"))
        for item in validated
    ]
    if len(set(model_documents)) != 1:
        raise ValueError("strong-scaling configs must use identical model definitions")
    expected_accumulation = [16, 8, 4]
    observed_accumulation = [
        int(item["document"]["training"]["gradient_accumulation_steps"]) for item in validated
    ]
    if observed_accumulation != expected_accumulation:
        raise ValueError(
            f"strong-scaling gradient accumulation must be [16,8,4], got {observed_accumulation}"
        )
    return validated


def training_command(
    config: Path,
    output_dir: Path,
    world_size: int,
    source_lock: Path | None,
    *,
    resume: bool = False,
    stop_after_step: int | None = None,
    profiler_dir: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        "-m",
        "experiments.distributed_training.train",
        "--config",
        str(config.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
    ]
    if source_lock is not None:
        command.extend(["--source-lock", str(source_lock.resolve())])
    if resume:
        command.extend(["--resume", str((output_dir / "checkpoint.pt").resolve())])
    if stop_after_step is not None:
        command.extend(["--stop-after-step", str(stop_after_step)])
    if profiler_dir is not None:
        command.extend(["--torch-profiler-dir", str(profiler_dir.resolve())])
    return command


def build_matrix_plan(config_dir: Path, artifact_root: Path, source_lock: Path) -> dict[str, Any]:
    validated = validate_strong_scaling_matrix(config_dir)
    by_world = {int(item["world_size"]): item for item in validated}
    runs = []
    for pass_index, order in enumerate(MATRIX_PASS_ORDERS, 1):
        for position, world_size in enumerate(order, 1):
            item = by_world[world_size]
            config = Path(item["path"])
            pass_name = f"pass-{pass_index:02d}"
            output = artifact_root / "strong-scaling" / pass_name / f"w{world_size}"
            runs.append(
                {
                    "name": f"{pass_name}-strong-403m-w{world_size}",
                    "pass": pass_index,
                    "position_in_pass": position,
                    "config": str(config),
                    "config_sha256": item["sha256"],
                    "world_size": world_size,
                    "parameter_count": item["parameter_count"],
                    "tokens_per_update": item["tokens_per_update"],
                    "output_dir": str(output.resolve()),
                    "command": training_command(config, output, world_size, source_lock),
                }
            )
    return {
        "schema_version": "decoder-training-matrix-plan/1",
        "artifact_root": str(artifact_root.resolve()),
        "source_lock": str(source_lock.resolve()),
        "source_lock_sha256": _sha256_file(source_lock),
        "pass_orders": [list(order) for order in MATRIX_PASS_ORDERS],
        "runs": runs,
    }


def build_source_lock(
    output: Path,
    config_paths: Sequence[Path],
    data_manifest: Path,
    environment_candidate: Path,
    evidence_paths: Sequence[tuple[Path, str]] = (),
) -> dict[str, Any]:
    root = _repository_root()
    inputs: list[tuple[Path, str]] = []
    inputs.extend(
        (path, "code")
        for path in sorted((root / "experiments" / "distributed_training").glob("*.py"))
    )
    inputs.append((root / "tools" / "run_training_matrix.py", "code"))
    inputs.extend((path.resolve(), "config") for path in config_paths)
    inputs.append((data_manifest.resolve(), "data_manifest"))
    inputs.append((environment_candidate.resolve(), "environment"))
    inputs.extend((path.resolve(), role) for path, role in evidence_paths)
    records = []
    seen: set[str] = set()
    for path, role in inputs:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"source-lock input is outside repository: {path}") from exc
        if relative in seen:
            raise ValueError(f"duplicate source-lock input: {relative}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        seen.add(relative)
        records.append(
            {
                "path": relative,
                "role": role,
                "sha256": _sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    document = {
        "schema_version": SOURCE_LOCK_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repository_root_name": root.name,
        "files": sorted(records, key=lambda item: item["path"]),
    }
    _write_json_atomic(output, document)
    return document


def validate_source_lock(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    if document.get("schema_version") != SOURCE_LOCK_SCHEMA:
        raise ValueError(f"{path}: unsupported source-lock schema")
    files = document.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{path}: source-lock files must be non-empty")
    root = _repository_root()
    roles: set[str] = set()
    paths: set[str] = set()
    for index, record in enumerate(files):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "role",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"{path}: invalid files[{index}] record")
        candidate = (root / record["path"]).resolve()
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"{path}: bound path escapes repository") from exc
        if relative in paths:
            raise ValueError(f"{path}: duplicate bound path {relative}")
        if (
            not candidate.is_file()
            or candidate.stat().st_size != record["size_bytes"]
            or _sha256_file(candidate) != record["sha256"]
        ):
            raise ValueError(f"{path}: bound file changed or is missing: {relative}")
        roles.add(record["role"])
        paths.add(relative)
    required_roles = {"code", "config", "data_manifest", "environment"}
    if not required_roles <= roles:
        raise ValueError(f"{path}: source-lock missing roles {sorted(required_roles - roles)}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "roles": sorted(roles),
        "paths": sorted(paths),
        "files": files,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(record, dict) or not isinstance(record.get("event"), str):
            raise ValueError(f"{path}:{line_number}: invalid event record")
        records.append(record)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def aggregate_metrics(run_dir: Path) -> dict[str, Any]:
    records = _read_jsonl(run_dir / "metrics.jsonl")
    starts = [record for record in records if record["event"] == "run_start"]
    updates = [record for record in records if record["event"] == "train_update"]
    if not starts or not updates:
        raise ValueError(f"{run_dir}: run_start and train_update records are required")
    run_ends = [record for record in records if record["event"] == "run_end"]
    if len(run_ends) != 1:
        raise ValueError(f"{run_dir}: exactly one successful run_end is required")
    steps = [record.get("step") for record in updates]
    if (
        any(not isinstance(step, int) or isinstance(step, bool) for step in steps)
        or len(set(steps)) != len(steps)
        or steps != sorted(steps)
    ):
        raise ValueError(f"{run_dir}: train_update steps must be unique and ordered")
    measured = [record for record in updates if record.get("measurement_eligible") is True]
    if not measured:
        raise ValueError(f"{run_dir}: no measurement-eligible updates")
    throughputs = [float(record["tokens_per_second"]) for record in measured]
    latencies = [float(record["elapsed_s"]) for record in measured]
    if any(not math.isfinite(value) or value <= 0 for value in (*throughputs, *latencies)):
        raise ValueError(f"{run_dir}: timing metrics must be finite and positive")
    return {
        "schema_version": "decoder-training-run-summary/1",
        "run_dir": str(run_dir.resolve()),
        "world_size": starts[-1]["world_size"],
        "parameter_count": starts[-1]["parameter_count"]["unique_trainable"],
        "tokens_per_update": starts[-1]["tokens_per_update"],
        "observed_updates": len(updates),
        "measurement_eligible_updates": len(measured),
        "excluded_warmup_updates": len(updates) - len(measured),
        "tokens_per_second": {
            "mean": statistics.fmean(throughputs),
            "median": statistics.median(throughputs),
            "p05": _percentile(throughputs, 0.05),
            "p95": _percentile(throughputs, 0.95),
        },
        "elapsed_s": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p05": _percentile(latencies, 0.05),
            "p95": _percentile(latencies, 0.95),
        },
        "measurement_samples": [
            {
                "step": int(record["step"]),
                "tokens_per_second": float(record["tokens_per_second"]),
                "elapsed_s": float(record["elapsed_s"]),
            }
            for record in measured
        ],
    }


def _require_finite_number(value: Any, location: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{location} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or (positive and converted <= 0):
        qualifier = "positive and " if positive else ""
        raise ValueError(f"{location} must be {qualifier}finite")
    return converted


def _validate_artifact_reference(reference: Any, location: str) -> dict[str, str]:
    if (
        not isinstance(reference, dict)
        or set(reference) != {"path", "sha256", "status"}
        or reference.get("status") != "present"
        or not isinstance(reference.get("path"), str)
        or not isinstance(reference.get("sha256"), str)
        or len(reference["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in reference["sha256"])
    ):
        raise ValueError(f"{location} must be a present path/SHA256 reference")
    path = Path(reference["path"])
    if not path.is_file() or _sha256_file(path) != reference["sha256"]:
        raise ValueError(f"{location} changed or is missing: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": reference["sha256"],
        "status": "present",
    }


def _load_matrix_provenance(artifact_root: Path) -> dict[str, Any]:
    matrix_root = artifact_root / "strong-scaling"
    plan_path = matrix_root / "plan.json"
    journal_path = matrix_root / "journal.json"
    plan = _read_json(plan_path)
    journal = _read_json(journal_path)
    if plan.get("schema_version") != "decoder-training-matrix-plan/1":
        raise ValueError(f"{plan_path}: unsupported matrix plan schema")
    if journal.get("schema_version") != "decoder-training-matrix-journal/1":
        raise ValueError(f"{journal_path}: unsupported matrix journal schema")
    if journal.get("status") != "complete":
        raise ValueError(f"{journal_path}: matrix journal is not complete")
    if plan.get("artifact_root") != str(artifact_root.resolve()):
        raise ValueError("matrix plan artifact_root does not match aggregation target")
    if plan.get("pass_orders") != [list(order) for order in MATRIX_PASS_ORDERS]:
        raise ValueError("matrix plan does not contain the frozen interleaved pass orders")

    plan_reference = journal.get("plan")
    if (
        not isinstance(plan_reference, dict)
        or plan_reference.get("path") != str(plan_path.resolve())
        or plan_reference.get("sha256") != _sha256_file(plan_path)
    ):
        raise ValueError("completed journal does not bind the exact matrix plan")

    source_lock_path = Path(str(plan.get("source_lock", "")))
    source_lock_sha256 = plan.get("source_lock_sha256")
    if (
        not source_lock_path.is_file()
        or not isinstance(source_lock_sha256, str)
        or _sha256_file(source_lock_path) != source_lock_sha256
        or journal.get("source_lock_sha256") != source_lock_sha256
    ):
        raise ValueError("plan and completed journal must bind the same source lock")
    validated_source_lock = validate_source_lock(source_lock_path)
    if validated_source_lock["sha256"] != source_lock_sha256:
        raise ValueError("matrix source lock validation returned a different SHA256")
    source_lock = _read_json(source_lock_path)
    if source_lock.get("schema_version") != SOURCE_LOCK_SCHEMA:
        raise ValueError("matrix source lock has an unsupported schema")
    locked_files = source_lock.get("files")
    if not isinstance(locked_files, list):
        raise ValueError("matrix source lock has no file records")
    locked_roles = {record.get("role") for record in locked_files if isinstance(record, dict)}
    required_lock_roles = {
        "code",
        "config",
        "data_manifest",
        "environment",
        "readiness_report",
        "training_preflight",
    }
    if not required_lock_roles <= locked_roles:
        raise ValueError("matrix source lock is missing required provenance roles")

    required_evidence = plan.get("required_evidence")
    if (
        not isinstance(required_evidence, dict)
        or set(required_evidence) != {"full_readiness_report", "training_nccl_preflight"}
        or journal.get("required_evidence") != required_evidence
    ):
        raise ValueError("plan and journal must bind both frozen readiness artifacts")
    evidence = {
        name: _validate_artifact_reference(reference, f"required_evidence.{name}")
        for name, reference in required_evidence.items()
    }
    evidence_roles = {
        "full_readiness_report": "readiness_report",
        "training_nccl_preflight": "training_preflight",
    }
    for name, role in evidence_roles.items():
        matches = [
            record
            for record in locked_files
            if isinstance(record, dict) and record.get("role") == role
        ]
        if (
            len(matches) != 1
            or matches[0].get("sha256") != evidence[name]["sha256"]
            or (_repository_root() / str(matches[0].get("path", ""))).resolve()
            != Path(evidence[name]["path"]).resolve()
        ):
            raise ValueError(f"source lock does not bind the exact {name}")

    plan_runs = plan.get("runs")
    journal_runs = journal.get("runs")
    if not isinstance(plan_runs, list) or len(plan_runs) != 9:
        raise ValueError("matrix plan must contain exactly nine runs")
    if not isinstance(journal_runs, list) or len(journal_runs) != 9:
        raise ValueError("completed matrix journal must contain exactly nine runs")
    journal_by_name: dict[str, dict[str, Any]] = {}
    for record in journal_runs:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise ValueError("matrix journal contains an invalid run record")
        name = record["name"]
        if name in journal_by_name:
            raise ValueError(f"matrix journal contains duplicate run {name}")
        if record.get("status") != "complete":
            raise ValueError(f"matrix journal run is not complete: {name}")
        journal_by_name[name] = record

    expected_worlds = [world for order in MATRIX_PASS_ORDERS for world in order]
    seen_outputs: set[str] = set()
    seen_names: set[str] = set()
    canonical_config_dir = _repository_root() / "configs" / "training"
    for index, (run, expected_world) in enumerate(zip(plan_runs, expected_worlds, strict=True), 1):
        if not isinstance(run, dict):
            raise ValueError(f"matrix plan run {index} is not an object")
        pass_index = (index - 1) // 3 + 1
        position = (index - 1) % 3 + 1
        name = f"pass-{pass_index:02d}-strong-403m-w{expected_world}"
        output = (
            artifact_root / "strong-scaling" / f"pass-{pass_index:02d}" / f"w{expected_world}"
        ).resolve()
        config = (canonical_config_dir / f"strong-403m-w{expected_world}.json").resolve()
        validation = validate_training_config(config)
        expected_command = training_command(config, output, expected_world, source_lock_path)
        if (
            run.get("name") != name
            or run.get("pass") != pass_index
            or run.get("position_in_pass") != position
            or run.get("world_size") != expected_world
            or run.get("parameter_count") != 403_097_088
            or run.get("tokens_per_update") != 32_768
            or Path(str(run.get("config", ""))).resolve() != config
            or run.get("config_sha256") != validation["sha256"]
            or Path(str(run.get("output_dir", ""))).resolve() != output
            or run.get("command") != expected_command
        ):
            raise ValueError(f"matrix plan run differs from the frozen contract: {name}")
        if name in seen_names or str(output) in seen_outputs:
            raise ValueError("matrix plan reuses a run name or output directory")
        seen_names.add(name)
        seen_outputs.add(str(output))
        journal_record = journal_by_name.get(name)
        if journal_record is None or journal_record.get("command") != expected_command:
            raise ValueError(f"completed journal does not bind the planned command: {name}")
        locked_config_matches = [
            record
            for record in locked_files
            if isinstance(record, dict)
            and record.get("role") == "config"
            and record.get("sha256") == validation["sha256"]
            and (_repository_root() / str(record.get("path", ""))).resolve() == config
        ]
        if len(locked_config_matches) != 1:
            raise ValueError(f"source lock does not bind the selected config: {config}")

    if set(journal_by_name) != seen_names:
        raise ValueError("matrix journal run names differ from the frozen plan")
    return {
        "plan": plan,
        "journal": journal,
        "plan_reference": {
            "path": str(plan_path.resolve()),
            "sha256": _sha256_file(plan_path),
        },
        "journal_reference": {
            "path": str(journal_path.resolve()),
            "sha256": _sha256_file(journal_path),
        },
        "source_lock_reference": {
            "path": str(source_lock_path.resolve()),
            "sha256": source_lock_sha256,
        },
        "evidence": evidence,
    }


def _canonical_run_summary(
    expected: Mapping[str, Any], source_lock_reference: Mapping[str, str]
) -> dict[str, Any]:
    run_dir = Path(str(expected["output_dir"]))
    metrics_path = run_dir / "metrics.jsonl"
    records = _read_jsonl(metrics_path)
    starts = [record for record in records if record["event"] == "run_start"]
    updates = [record for record in records if record["event"] == "train_update"]
    ends = [record for record in records if record["event"] == "run_end"]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(f"{run_dir}: exactly one run_start and run_end are required")
    if any(record["event"] == "planned_stop" for record in records):
        raise ValueError(f"{run_dir}: planned-stop artifacts cannot enter scaling results")
    run_id = starts[0].get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{run_dir}: missing run_id")
    if any(record.get("run_id") != run_id for record in records):
        raise ValueError(f"{run_dir}: every event must bind the same run_id")

    config_path = Path(str(expected["config"]))
    config = validate_training_config(config_path)
    model = config["document"]["model"]
    training = config["document"]["training"]
    start = starts[0]
    source_lock = start.get("source_lock")
    profiler = start.get("torch_profiler")
    parameter_count = start.get("parameter_count")
    data = start.get("data")
    data_fingerprint = start.get("data_fingerprint")
    rank_environments = start.get("rank_environments")
    if (
        start.get("config_path") != str(config_path.resolve())
        or start.get("config_sha256") != expected["config_sha256"]
        or config["sha256"] != expected["config_sha256"]
        or start.get("world_size") != expected["world_size"]
        or start.get("tokens_per_update") != 32_768
        or not isinstance(parameter_count, dict)
        or parameter_count.get("unique_trainable") != 403_097_088
        or start.get("start_step") != 0
        or start.get("resume_from_sha256") is not None
        or start.get("model") != model
        or start.get("training") != training
        or not isinstance(source_lock, dict)
        or source_lock.get("path") != source_lock_reference["path"]
        or source_lock.get("sha256") != source_lock_reference["sha256"]
        or not isinstance(profiler, dict)
        or profiler.get("enabled") is not False
        or not isinstance(data_fingerprint, str)
        or len(data_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in data_fingerprint)
        or not isinstance(data, dict)
        or data.get("mode") != "synthetic-smoke-only"
        or not isinstance(data.get("token_id_contract"), dict)
        or data["token_id_contract"].get("model_vocab_size") != 256
        or data["token_id_contract"].get("label_semantics") != "synthetic smoke-only IDs"
        or not isinstance(rank_environments, list)
        or len(rank_environments) != expected["world_size"]
        or {item.get("rank") for item in rank_environments if isinstance(item, dict)}
        != set(range(int(expected["world_size"])))
        or start.get("environment") != rank_environments[0]
    ):
        raise ValueError(f"{run_dir}: run_start differs from the canonical matrix contract")
    if (
        training["steps"] != 40
        or training["timing_warmup_steps"] != 10
        or training["expected_world_size"] != expected["world_size"]
        or training["expected_tokens_per_update"] != 32_768
        or training["expected_parameter_count"] != 403_097_088
    ):
        raise ValueError(f"{run_dir}: selected config is not a canonical scaling config")

    if len(updates) != 40 or [record.get("step") for record in updates] != list(range(40)):
        raise ValueError(f"{run_dir}: canonical run must contain update steps 0..39 exactly")
    for step, record in enumerate(updates):
        if record.get("measurement_eligible") is not (step >= 10):
            raise ValueError(f"{run_dir}: incorrect measurement eligibility at step={step}")
        if record.get("canonical_timing") != "core_update":
            raise ValueError(f"{run_dir}: canonical timing must be core_update")
        if record.get("tokens") != 32_768:
            raise ValueError(f"{run_dir}: update tokens differ from the fixed global budget")
        core_elapsed = _require_finite_number(
            record.get("critical_rank_core_update_elapsed_s"),
            f"{run_dir}: step={step} core timing",
            positive=True,
        )
        guarded_elapsed = _require_finite_number(
            record.get("critical_rank_guarded_update_elapsed_s"),
            f"{run_dir}: step={step} guarded timing",
            positive=True,
        )
        guard_elapsed = _require_finite_number(
            record.get("critical_rank_finite_guard_elapsed_s"),
            f"{run_dir}: step={step} finite-guard timing",
        )
        elapsed = _require_finite_number(
            record.get("elapsed_s"), f"{run_dir}: step={step} elapsed_s", positive=True
        )
        throughput = _require_finite_number(
            record.get("tokens_per_second"),
            f"{run_dir}: step={step} tokens_per_second",
            positive=True,
        )
        _require_finite_number(record.get("loss"), f"{run_dir}: step={step} loss")
        _require_finite_number(record.get("grad_norm"), f"{run_dir}: step={step} grad_norm")
        if (
            elapsed != core_elapsed
            or guarded_elapsed < core_elapsed
            or guard_elapsed < 0
            or not math.isclose(throughput, 32_768 / core_elapsed, rel_tol=1e-12)
        ):
            raise ValueError(f"{run_dir}: inconsistent timing boundary at step={step}")

    end = ends[0]
    if (
        end.get("status") != "complete"
        or end.get("completed_steps") != 40
        or end.get("measurement_eligible_updates") != 30
    ):
        raise ValueError(f"{run_dir}: run_end does not prove all canonical updates completed")
    final_checkpoints = [
        record for record in records if record["event"] == "checkpoint" and record.get("step") == 40
    ]
    checkpoint_path = run_dir / "checkpoint.pt"
    if len(final_checkpoints) != 1 or not checkpoint_path.is_file():
        raise ValueError(f"{run_dir}: exactly one final checkpoint event is required")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    checkpoint_event = final_checkpoints[0]
    if (
        Path(str(checkpoint_event.get("path", ""))).resolve() != checkpoint_path.resolve()
        or checkpoint_event.get("sha256") != checkpoint_sha256
    ):
        raise ValueError(f"{run_dir}: final checkpoint path/SHA256 mismatch")

    summary = aggregate_metrics(run_dir)
    summary.update(
        {
            "run_id": run_id,
            "config_sha256": expected["config_sha256"],
            "data_fingerprint": data_fingerprint,
            "raw_artifacts": {
                "metrics": {
                    "path": str(metrics_path.resolve()),
                    "sha256": _sha256_file(metrics_path),
                },
                "checkpoint": {
                    "path": str(checkpoint_path.resolve()),
                    "sha256": checkpoint_sha256,
                },
            },
        }
    )
    return summary


def aggregate_matrix(artifact_root: Path) -> dict[str, Any]:
    provenance = _load_matrix_provenance(artifact_root)
    plan_runs = provenance["plan"]["runs"]
    pass_summaries = []
    grouped: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 4: []}
    seen_run_ids: set[str] = set()
    seen_metrics: set[str] = set()
    seen_checkpoints: set[str] = set()
    data_fingerprints: set[str] = set()
    run_index = 0
    for pass_index, order in enumerate(MATRIX_PASS_ORDERS, 1):
        runs = []
        for world in order:
            expected = plan_runs[run_index]
            run_index += 1
            summary = _canonical_run_summary(expected, provenance["source_lock_reference"])
            if (
                int(summary["world_size"]) != world
                or int(summary["parameter_count"]) != 403_097_088
                or int(summary["tokens_per_update"]) != 32_768
                or int(summary["observed_updates"]) != 40
                or int(summary["measurement_eligible_updates"]) != 30
            ):
                raise ValueError(
                    "each canonical scaling run must contain 40 updates with "
                    "exactly 30 measurement-eligible observations"
                )
            metrics_sha = summary["raw_artifacts"]["metrics"]["sha256"]
            checkpoint_sha = summary["raw_artifacts"]["checkpoint"]["sha256"]
            if (
                summary["run_id"] in seen_run_ids
                or metrics_sha in seen_metrics
                or checkpoint_sha in seen_checkpoints
            ):
                raise ValueError("canonical matrix runs must use distinct raw artifacts")
            seen_run_ids.add(summary["run_id"])
            seen_metrics.add(metrics_sha)
            seen_checkpoints.add(checkpoint_sha)
            data_fingerprints.add(summary["data_fingerprint"])
            grouped[world].append(summary)
            runs.append(summary)
        pass_summaries.append({"pass": pass_index, "order": list(order), "runs": runs})
    if any(len(items) != 3 for items in grouped.values()):
        raise ValueError("each world size must have exactly three successful runs")
    if len(data_fingerprints) != 1:
        raise ValueError("all canonical scaling runs must use one identical data fingerprint")
    world_summaries = []
    for world in (1, 2, 4):
        samples = [sample for run in grouped[world] for sample in run["measurement_samples"]]
        if len(samples) != 90:
            raise ValueError(f"world_size={world} must contribute exactly 90 measured updates")
        throughputs = [float(item["tokens_per_second"]) for item in samples]
        latencies = [float(item["elapsed_s"]) for item in samples]
        world_summaries.append(
            {
                "world_size": world,
                "successful_runs": 3,
                "measurement_eligible_updates": 90,
                "tokens_per_second": {
                    "mean": statistics.fmean(throughputs),
                    "median": statistics.median(throughputs),
                    "p05": _percentile(throughputs, 0.05),
                    "p95": _percentile(throughputs, 0.95),
                },
                "elapsed_s": {
                    "mean": statistics.fmean(latencies),
                    "median": statistics.median(latencies),
                    "p05": _percentile(latencies, 0.05),
                    "p95": _percentile(latencies, 0.95),
                },
                "runs": grouped[world],
            }
        )
    baseline = float(world_summaries[0]["tokens_per_second"]["median"])
    for summary in world_summaries:
        world = int(summary["world_size"])
        median = float(summary["tokens_per_second"]["median"])
        summary["strong_scaling"] = {
            "speedup_vs_w1": median / baseline,
            "efficiency_vs_w1": median / (baseline * world),
        }
    return {
        "schema_version": "decoder-training-matrix-summary/2",
        "aggregation_rule": (
            "three predeclared interleaved passes; per-world median synchronized "
            "critical-rank core-update tokens/s over 90 measurement-eligible "
            "updates; finite-guard diagnostics and profiler runs are excluded"
        ),
        "provenance": {
            "plan": provenance["plan_reference"],
            "journal": provenance["journal_reference"],
            "source_lock": provenance["source_lock_reference"],
            "required_evidence": provenance["evidence"],
        },
        "pass_orders": [list(order) for order in MATRIX_PASS_ORDERS],
        "passes": pass_summaries,
        "worlds": world_summaries,
    }


def _digest_tree(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if item is None:
            digest.update(b"N;")
        elif isinstance(item, bool):
            digest.update(b"B1;" if item else b"B0;")
        elif isinstance(item, int):
            digest.update(f"I{item};".encode("ascii"))
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("checkpoint contains a non-finite Python float")
            digest.update(f"F{item.hex()};".encode("ascii"))
        elif isinstance(item, str):
            encoded = item.encode("utf-8")
            digest.update(f"S{len(encoded)}:".encode("ascii"))
            digest.update(encoded)
        elif isinstance(item, bytes):
            digest.update(f"Y{len(item)}:".encode("ascii"))
            digest.update(item)
        elif isinstance(item, Mapping):
            digest.update(f"M{len(item)}[".encode("ascii"))
            ordered = sorted(item.items(), key=lambda pair: (type(pair[0]).__name__, repr(pair[0])))
            for key, child in ordered:
                visit(key)
                visit(child)
            digest.update(b"]")
        elif isinstance(item, tuple):
            digest.update(f"T{len(item)}[".encode("ascii"))
            for child in item:
                visit(child)
            digest.update(b"]")
        elif isinstance(item, list):
            digest.update(f"L{len(item)}[".encode("ascii"))
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("PyTorch is required to digest checkpoint tensors") from exc
            if not torch.is_tensor(item):
                raise TypeError(f"unsupported checkpoint value {type(item).__name__}")
            tensor = item.detach().cpu().contiguous()
            digest.update(
                (f"X{tensor.dtype}:{','.join(map(str, tensor.shape))}:{tensor.numel()}:").encode(
                    "ascii"
                )
            )
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(raw)

    visit(value)
    return digest.hexdigest()


def checkpoint_summary(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for checkpoint comparison") from exc
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != "0.4":
        raise ValueError(f"{path}: unsupported checkpoint")
    process_states = checkpoint.get("process_states")
    if not isinstance(process_states, list):
        raise ValueError(f"{path}: invalid process_states")
    cursors = [
        {
            "train_batcher": state["train_batcher"],
            "validation_batcher": state["validation_batcher"],
        }
        for state in process_states
    ]
    rng_states = [
        {
            "python_random_state": state["python_random_state"],
            "torch_cpu_rng_state": state["torch_cpu_rng_state"],
            "torch_cuda_rng_state": state["torch_cuda_rng_state"],
        }
        for state in process_states
    ]
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256_file(path),
        "next_step": checkpoint["next_step"],
        "world_size": checkpoint["world_size"],
        "device_type": checkpoint["device_type"],
        "config_sha256": checkpoint["config_sha256"],
        "data_fingerprint": checkpoint["data_fingerprint"],
        "source_lock_sha256": checkpoint.get("source_lock_sha256"),
        "model_sha256": _digest_tree(checkpoint["model"]),
        "adamw_sha256": _digest_tree(checkpoint["optimizer"]),
        "process_state_sha256": _digest_tree(process_states),
        "cursor_sha256": _digest_tree(cursors),
        "rng_sha256": _digest_tree(rng_states),
    }


def _trajectory_updates(records: Iterable[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for record in records:
        if record.get("event") != "train_update":
            continue
        step = record.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("trajectory contains an invalid train_update step")
        if step in result:
            raise ValueError(f"trajectory contains duplicate train_update step {step}")
        loss = record.get("loss")
        if not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
            raise ValueError(f"trajectory step {step} has invalid loss")
        result[step] = record
    return result


def compare_trajectories(
    uninterrupted_dir: Path,
    resumed_dir: Path,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    report_path: Path | None = None,
) -> dict[str, Any]:
    if rtol < 0 or atol < 0 or not math.isfinite(rtol) or not math.isfinite(atol):
        raise ValueError("trajectory tolerances must be finite and non-negative")
    baseline_records = _read_jsonl(uninterrupted_dir / "metrics.jsonl")
    resumed_records = _read_jsonl(resumed_dir / "metrics.jsonl")
    baseline_updates = _trajectory_updates(baseline_records)
    resumed_updates = _trajectory_updates(resumed_records)
    baseline_checkpoint = checkpoint_summary(uninterrupted_dir / "checkpoint.pt")
    resumed_checkpoint = checkpoint_summary(resumed_dir / "checkpoint.pt")
    expected_steps = list(range(int(baseline_checkpoint["next_step"])))
    failures: list[str] = []
    if sorted(baseline_updates) != expected_steps:
        failures.append("uninterrupted trajectory is not contiguous through final step")
    if sorted(resumed_updates) != expected_steps:
        failures.append("resumed trajectory is not contiguous through final step")
    baseline_ends = [record for record in baseline_records if record.get("event") == "run_end"]
    resumed_ends = [record for record in resumed_records if record.get("event") == "run_end"]
    baseline_starts = [record for record in baseline_records if record.get("event") == "run_start"]
    planned_stops = [record for record in resumed_records if record.get("event") == "planned_stop"]
    resumed_starts = [record for record in resumed_records if record.get("event") == "run_start"]
    final_step = int(baseline_checkpoint["next_step"])
    if (
        len(baseline_ends) != 1
        or baseline_ends[0].get("status") != "complete"
        or baseline_ends[0].get("completed_steps") != final_step
    ):
        failures.append("uninterrupted trajectory must contain exactly one run_end")
    if (
        len(resumed_ends) != 1
        or resumed_ends[0].get("status") != "complete"
        or resumed_ends[0].get("completed_steps") != final_step
    ):
        failures.append("resumed trajectory must contain exactly one run_end")
    if len(baseline_starts) != 1:
        failures.append("uninterrupted trajectory must contain exactly one run_start")
    else:
        baseline_run_id = baseline_starts[0].get("run_id")
        if (
            not isinstance(baseline_run_id, str)
            or not baseline_run_id
            or any(record.get("run_id") != baseline_run_id for record in baseline_records)
        ):
            failures.append("uninterrupted events must bind one non-empty run ID")
    if len(planned_stops) != 1 or len(resumed_starts) != 2:
        failures.append("resumed trajectory must contain one planned_stop and two run_start events")
    else:
        stop_step = planned_stops[0].get("completed_steps")
        first_start, second_start = resumed_starts
        if (
            not isinstance(stop_step, int)
            or isinstance(stop_step, bool)
            or not 0 < stop_step < final_step
            or first_start.get("start_step") != 0
            or first_start.get("resume_from_sha256") is not None
            or second_start.get("start_step") != stop_step
            or second_start.get("resume_from_sha256") != planned_stops[0].get("checkpoint_sha256")
        ):
            failures.append("planned stop and resume metadata do not form one exact boundary")
        start_worlds = {
            first_start.get("world_size"),
            second_start.get("world_size"),
            baseline_starts[0].get("world_size") if len(baseline_starts) == 1 else None,
        }
        if start_worlds != {baseline_checkpoint["world_size"]}:
            failures.append("trajectory run_start records do not use one world size")
        start_config_hashes = {
            first_start.get("config_sha256"),
            second_start.get("config_sha256"),
            baseline_starts[0].get("config_sha256") if len(baseline_starts) == 1 else None,
        }
        if start_config_hashes != {baseline_checkpoint["config_sha256"]}:
            failures.append("trajectory run_start records do not bind one config")
        first_run_id = first_start.get("run_id")
        second_run_id = second_start.get("run_id")
        if (
            not isinstance(first_run_id, str)
            or not first_run_id
            or not isinstance(second_run_id, str)
            or not second_run_id
            or first_run_id == second_run_id
        ):
            failures.append("trajectory segments must use two distinct non-empty run IDs")
        else:
            second_start_index = resumed_records.index(second_start)
            if any(
                record.get("run_id") != first_run_id
                for record in resumed_records[:second_start_index]
            ) or any(
                record.get("run_id") != second_run_id
                for record in resumed_records[second_start_index:]
            ):
                failures.append("trajectory events cross or omit their segment run ID")
    loss_mismatches = []
    for step in sorted(set(baseline_updates) & set(resumed_updates)):
        baseline_loss = float(baseline_updates[step]["loss"])
        resumed_loss = float(resumed_updates[step]["loss"])
        if not math.isclose(baseline_loss, resumed_loss, rel_tol=rtol, abs_tol=atol):
            loss_mismatches.append(
                {
                    "step": step,
                    "uninterrupted": baseline_loss,
                    "resumed": resumed_loss,
                }
            )
    if loss_mismatches:
        failures.append(f"{len(loss_mismatches)} step losses differ")
    digest_fields = (
        "next_step",
        "world_size",
        "device_type",
        "config_sha256",
        "data_fingerprint",
        "source_lock_sha256",
        "model_sha256",
        "adamw_sha256",
        "process_state_sha256",
        "cursor_sha256",
        "rng_sha256",
    )
    digest_mismatches = {
        field: {
            "uninterrupted": baseline_checkpoint[field],
            "resumed": resumed_checkpoint[field],
        }
        for field in digest_fields
        if baseline_checkpoint[field] != resumed_checkpoint[field]
    }
    if digest_mismatches:
        failures.append("final checkpoint state differs: " + ", ".join(digest_mismatches))
    report = {
        "schema_version": TRAJECTORY_SCHEMA,
        "pass": not failures,
        "uninterrupted_dir": str(uninterrupted_dir.resolve()),
        "resumed_dir": str(resumed_dir.resolve()),
        "rtol": rtol,
        "atol": atol,
        "compared_steps": len(set(baseline_updates) & set(resumed_updates)),
        "loss_mismatches": loss_mismatches,
        "checkpoint_mismatches": digest_mismatches,
        "failures": failures,
        "uninterrupted_checkpoint": baseline_checkpoint,
        "resumed_checkpoint": resumed_checkpoint,
    }
    if report_path is not None:
        _write_json_atomic(report_path, report)
    return report
