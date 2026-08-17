"""Run the isolated decoder systems experiment with one process or DDP."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .data import RandomWindowBatcher, load_byte_stream, synthetic_stream
from .model import CausalLM, ModelConfig

DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


@dataclass(frozen=True)
class TrainingConfig:
    experiment_name: str = "decoder-systems-smoke"
    seed: int = 1234
    steps: int = 20
    micro_batch_size: int = 4
    sequence_length: int = 128
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 2
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    optimizer_impl: str = "single_tensor"
    grad_clip: float = 1.0
    eval_interval: int = 10
    eval_batches: int = 4
    checkpoint_interval: int = 10
    timing_warmup_steps: int = 0
    dtype: str = "fp32"
    compile_model: bool = False
    deterministic_algorithms: bool = False
    allow_tf32: bool = False
    expected_world_size: int | None = None
    expected_tokens_per_update: int | None = None
    expected_parameter_count: int | None = None
    require_source_lock: bool = False
    synthetic_train_tokens: int = 131_072
    synthetic_validation_tokens: int = 16_384

    def __post_init__(self) -> None:
        positive_ints = {
            "steps": self.steps,
            "micro_batch_size": self.micro_batch_size,
            "sequence_length": self.sequence_length,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "eval_interval": self.eval_interval,
            "eval_batches": self.eval_batches,
            "checkpoint_interval": self.checkpoint_interval,
            "synthetic_train_tokens": self.synthetic_train_tokens,
            "synthetic_validation_tokens": self.synthetic_validation_tokens,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.warmup_steps, int) or isinstance(self.warmup_steps, bool):
            raise ValueError("warmup_steps must be an integer")
        if not isinstance(self.timing_warmup_steps, int) or isinstance(
            self.timing_warmup_steps, bool
        ):
            raise ValueError("timing_warmup_steps must be an integer")
        for name, value in {
            "learning_rate": self.learning_rate,
            "adam_eps": self.adam_eps,
            "grad_clip": self.grad_clip,
        }.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        if (
            not isinstance(self.weight_decay, (int, float))
            or isinstance(self.weight_decay, bool)
            or not math.isfinite(float(self.weight_decay))
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be non-negative")
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in (self.beta1, self.beta2)
        ) or not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise ValueError("Adam betas must be in [0,1)")
        if self.dtype not in {"fp32", "bf16"}:
            raise ValueError("dtype must be fp32 or bf16")
        if self.optimizer_impl != "single_tensor":
            raise ValueError("optimizer_impl must be single_tensor")
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string")
        for name, value in {
            "compile_model": self.compile_model,
            "deterministic_algorithms": self.deterministic_algorithms,
            "allow_tf32": self.allow_tf32,
            "require_source_lock": self.require_source_lock,
        }.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        if (
            not isinstance(self.min_lr_ratio, (int, float))
            or isinstance(self.min_lr_ratio, bool)
            or not 0.0 <= self.min_lr_ratio <= 1.0
        ):
            raise ValueError("min_lr_ratio must be in [0,1]")
        if self.warmup_steps < 0 or self.warmup_steps > self.steps:
            raise ValueError("warmup_steps must be in [0, steps]")
        if not 0 <= self.timing_warmup_steps < self.steps:
            raise ValueError("timing_warmup_steps must be in [0, steps)")
        for name, value in {
            "expected_world_size": self.expected_world_size,
            "expected_tokens_per_update": self.expected_tokens_per_update,
            "expected_parameter_count": self.expected_parameter_count,
        }.items():
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be null or a positive integer")


def _load_config(path: Path) -> tuple[ModelConfig, TrainingConfig, dict[str, Any]]:
    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-standard non-finite JSON number is forbidden: {value}")

    raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonstandard_constant)
    unknown = set(raw) - {"model", "training", "data"}
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    model = ModelConfig(**raw.get("model", {}))
    training = TrainingConfig(**raw.get("training", {}))
    data_config = dict(raw.get("data", {"mode": "synthetic"}))
    if training.sequence_length > model.max_seq_len:
        raise ValueError("training.sequence_length exceeds model.max_seq_len")
    if (
        min(
            training.synthetic_train_tokens,
            training.synthetic_validation_tokens,
        )
        < training.sequence_length + 1
    ):
        raise ValueError("synthetic token streams must contain at least sequence_length + 1 tokens")
    required_vocab = 256 if data_config.get("mode") == "byte_files" else 128
    if model.vocab_size < required_vocab:
        raise ValueError(
            f"model.vocab_size must be >= {required_vocab} for "
            f"data.mode={data_config.get('mode', 'synthetic')!r}"
        )
    return model, training, data_config


def _distributed_context(
    requested_device: str = "auto",
) -> tuple[int, int, int, torch.device]:
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("requested_device must be auto, cpu, or cuda")
    distributed_keys = {"RANK", "LOCAL_RANK", "WORLD_SIZE"}
    if not (distributed_keys & set(os.environ)):
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        use_cuda = requested_device == "cuda" or (
            requested_device == "auto" and torch.cuda.is_available()
        )
        device = torch.device("cuda" if use_cuda else "cpu")
        return 0, 0, 1, device
    missing = distributed_keys - set(os.environ)
    if missing:
        raise RuntimeError("partial torchrun environment; missing " + ", ".join(sorted(missing)))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size <= 0 or not 0 <= rank < world_size:
        raise RuntimeError(f"invalid torchrun rank tuple rank={rank}, world_size={world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("multi-process execution requires CUDA/NCCL")
    if requested_device == "cpu":
        raise RuntimeError("torchrun execution cannot be forced onto CPU")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside visible CUDA device count "
            f"{torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_fingerprint(description: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(description), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _streams(
    data: Mapping[str, Any], training: TrainingConfig
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    mode = data.get("mode", "synthetic")
    if mode == "synthetic":
        unknown = set(data) - {"mode"}
        if unknown:
            raise ValueError(f"unknown synthetic data keys: {sorted(unknown)}")
        train = synthetic_stream(training.synthetic_train_tokens, training.seed + 10)
        validation = synthetic_stream(training.synthetic_validation_tokens, training.seed + 20)
        return (
            train,
            validation,
            {
                "mode": "synthetic-smoke-only",
                "train_tokens": training.synthetic_train_tokens,
                "validation_tokens": training.synthetic_validation_tokens,
                "train_seed": training.seed + 10,
                "validation_seed": training.seed + 20,
            },
        )
    if mode != "byte_files":
        raise ValueError("data.mode must be synthetic or byte_files")
    unknown = set(data) - {"mode", "train_files", "validation_files"}
    if unknown:
        raise ValueError(f"unknown byte_files data keys: {sorted(unknown)}")
    train_paths = [Path(item).resolve() for item in data.get("train_files", [])]
    validation_paths = [Path(item).resolve() for item in data.get("validation_files", [])]
    if set(train_paths) & set(validation_paths):
        raise ValueError("train_files and validation_files must be disjoint")
    if len(set(train_paths)) != len(train_paths) or len(set(validation_paths)) != len(
        validation_paths
    ):
        raise ValueError("duplicate paths within a data split are not allowed")
    train_records = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in train_paths
    ]
    validation_records = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in validation_paths
    ]
    train_hashes = {record["sha256"] for record in train_records}
    validation_hashes = {record["sha256"] for record in validation_records}
    overlap = sorted(train_hashes & validation_hashes)
    if overlap:
        raise ValueError(
            "train and validation contain identical file content: " + ", ".join(overlap)
        )
    train = load_byte_stream(train_paths)
    validation = load_byte_stream(validation_paths)
    return (
        train,
        validation,
        {
            "mode": "byte_files",
            "train_files": train_records,
            "validation_files": validation_records,
            "document_boundary_policy": "concatenate_without_separator",
        },
    )


def _validate_stream_token_domain(
    train: torch.Tensor,
    validation: torch.Tensor,
    model_vocab_size: int,
    data_mode: str,
) -> dict[str, Any]:
    observed: dict[str, dict[str, int]] = {}
    for split, tokens in (("train", train), ("validation", validation)):
        if tokens.ndim != 1 or tokens.dtype != torch.long or tokens.numel() == 0:
            raise ValueError(f"{split} token stream must be non-empty 1-D torch.long")
        minimum, maximum = torch.aminmax(tokens)
        split_minimum = int(minimum.item())
        split_maximum = int(maximum.item())
        if split_minimum < 0 or split_maximum >= model_vocab_size:
            raise ValueError(
                f"{split} token IDs [{split_minimum},{split_maximum}] fall outside "
                f"model vocabulary [0,{model_vocab_size - 1}]"
            )
        if data_mode == "byte_files" and split_maximum >= 256:
            raise ValueError(f"{split} byte-file labels must be raw byte IDs in [0,255]")
        observed[split] = {"minimum": split_minimum, "maximum": split_maximum}
    return {
        "model_vocab_size": model_vocab_size,
        "observed": observed,
        "label_semantics": (
            "raw next-byte IDs 0..255" if data_mode == "byte_files" else "synthetic smoke-only IDs"
        ),
        "reserved_unused_output_id_range": (
            [256, model_vocab_size - 1]
            if data_mode == "byte_files" and model_vocab_size > 256
            else None
        ),
    }


def _lr_for_step(step: int, config: TrainingConfig) -> float:
    if config.warmup_steps and step < config.warmup_steps:
        return config.learning_rate * float(step + 1) / config.warmup_steps
    decay_steps = config.steps - config.warmup_steps
    if decay_steps <= 1:
        progress = 1.0
    else:
        progress = min(
            1.0,
            max(0.0, (step - config.warmup_steps) / (decay_steps - 1)),
        )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine
    return config.learning_rate * multiplier


def _optimizer(model: CausalLM, config: TrainingConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for _name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # Matrices receive AdamW decay; vectors such as RMSNorm scales do not.
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
        foreach=False,
        fused=False,
    )


def _configure_deterministic_runtime(enabled: bool) -> str | None:
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if not enabled:
        return observed
    if observed is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        return DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
    if observed != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "deterministic training requires CUBLAS_WORKSPACE_CONFIG="
            f"{DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG}, observed {observed!r}"
        )
    return observed


def _optimizer_parameter_names(
    raw_model: CausalLM, optimizer: torch.optim.Optimizer
) -> list[list[str]]:
    names_by_identity = {id(parameter): name for name, parameter in raw_model.named_parameters()}
    groups: list[list[str]] = []
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            name = names_by_identity.get(id(parameter))
            if name is None:
                raise ValueError("optimizer contains a parameter not owned by the model")
            names.append(name)
        groups.append(names)
    return groups


def _resume_checkpoint_sha256(path: Path, world_size: int) -> str:
    """Require every rank to see the exact same checkpoint bytes."""
    resolved = path.resolve()
    try:
        local_status: dict[str, Any] = {
            "ok": True,
            "path": str(resolved),
            "sha256": _sha256_file(resolved),
        }
    except Exception as exc:
        local_status = {
            "ok": False,
            "path": str(resolved),
            "error": f"{type(exc).__name__}: {exc}",
        }
    statuses: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather_object(statuses, local_status)
    else:
        statuses[0] = local_status
    if any(status is None or status.get("ok") is not True for status in statuses):
        raise RuntimeError(
            f"every rank must be able to read the resume checkpoint; observed={statuses}"
        )
    hashes = {str(status["sha256"]) for status in statuses if status is not None}
    if len(hashes) != 1:
        raise RuntimeError(
            f"all ranks must read identical resume checkpoint bytes; observed={statuses}"
        )
    return hashes.pop()


def _local_process_state(
    train_batcher: RandomWindowBatcher,
    validation_batcher: RandomWindowBatcher,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        # One DDP process consumes randomness only from its selected CUDA
        # device.  Capturing every visible GPU would initialize unrelated
        # devices and makes rank-local replay depend on visibility/order.
        "torch_cuda_rng_state": (
            torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None
        ),
        "train_batcher": dict(train_batcher.state_dict()),
        "validation_batcher": dict(validation_batcher.state_dict()),
    }


def _save_checkpoint(
    path: Path,
    raw_model: CausalLM,
    optimizer: torch.optim.Optimizer,
    next_step: int,
    rank: int,
    world_size: int,
    config_sha256: str,
    data_fingerprint: str,
    train_batcher: RandomWindowBatcher,
    validation_batcher: RandomWindowBatcher,
    device: torch.device,
    source_lock_sha256: str | None = None,
    run_id: str | None = None,
) -> str:
    local_state = _local_process_state(train_batcher, validation_batcher, device)
    process_states = [None for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather_object(process_states, local_state)
    else:
        process_states[0] = local_state
    status: list[dict[str, Any] | None] = [None]
    if rank == 0:
        temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": "0.4",
                    "next_step": next_step,
                    "world_size": world_size,
                    "device_type": device.type,
                    "config_sha256": config_sha256,
                    "data_fingerprint": data_fingerprint,
                    "source_lock_sha256": source_lock_sha256,
                    "artifact_run_id": run_id,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "optimizer_parameter_names": _optimizer_parameter_names(raw_model, optimizer),
                    "process_states": process_states,
                },
                temporary,
            )
            os.replace(temporary, path)
            status[0] = {"ok": True, "sha256": _sha256_file(path)}
        except Exception as exc:
            status[0] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            temporary.unlink(missing_ok=True)
    if world_size > 1:
        dist.broadcast_object_list(status, src=0)
    assert status[0] is not None
    if not status[0]["ok"]:
        raise RuntimeError(f"rank-0 checkpoint write failed: {status[0]['error']}")
    expected_sha256 = str(status[0]["sha256"])
    try:
        local_verification: dict[str, Any] = {
            "ok": True,
            "rank": rank,
            "sha256": _sha256_file(path),
        }
    except Exception as exc:
        local_verification = {
            "ok": False,
            "rank": rank,
            "error": f"{type(exc).__name__}: {exc}",
        }
    verification: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather_object(verification, local_verification)
    else:
        verification[0] = local_verification
    if any(
        item is None or item.get("ok") is not True or item.get("sha256") != expected_sha256
        for item in verification
    ):
        raise RuntimeError(
            "checkpoint must be readable with identical SHA256 on every rank; "
            f"observed={verification}"
        )
    return expected_sha256


def _load_checkpoint(
    path: Path,
    raw_model: CausalLM,
    optimizer: torch.optim.Optimizer,
    rank: int,
    world_size: int,
    device: torch.device,
    config_sha256: str,
    data_fingerprint: str,
    train_batcher: RandomWindowBatcher,
    validation_batcher: RandomWindowBatcher,
    source_lock_sha256: str | None = None,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a mapping")
    if checkpoint.get("schema_version") != "0.4":
        raise ValueError(
            f"unsupported checkpoint schema {checkpoint.get('schema_version')!r}; expected '0.4'"
        )
    if checkpoint["world_size"] != world_size:
        raise ValueError("resume requires the same world_size for exact data/RNG replay")
    if checkpoint["config_sha256"] != config_sha256:
        raise ValueError("checkpoint config hash does not match this run")
    if checkpoint["data_fingerprint"] != data_fingerprint:
        raise ValueError("checkpoint data fingerprint does not match this run")
    if checkpoint.get("source_lock_sha256") != source_lock_sha256:
        raise ValueError("checkpoint source-lock hash does not match this run")
    if checkpoint["device_type"] != device.type:
        raise ValueError(
            "exact resume requires the same device type; "
            f"checkpoint={checkpoint['device_type']}, current={device.type}"
        )
    next_step = checkpoint.get("next_step")
    if not isinstance(next_step, int) or isinstance(next_step, bool) or next_step < 0:
        raise ValueError("checkpoint next_step must be a non-negative integer")
    process_states = checkpoint.get("process_states")
    if not isinstance(process_states, list) or len(process_states) != world_size:
        raise ValueError("checkpoint process_states must contain exactly one entry per rank")
    state = process_states[rank]
    required_process_keys = {
        "python_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
        "train_batcher",
        "validation_batcher",
    }
    if not isinstance(state, Mapping) or not required_process_keys <= set(state):
        raise ValueError("checkpoint rank-local process state is incomplete")
    if device.type == "cuda" and state["torch_cuda_rng_state"] is None:
        raise ValueError("CUDA checkpoint is missing this rank's CUDA RNG state")
    saved_parameter_names = checkpoint.get("optimizer_parameter_names")
    current_parameter_names = _optimizer_parameter_names(raw_model, optimizer)
    if saved_parameter_names != current_parameter_names:
        raise ValueError("optimizer parameter names/order do not match the current model")
    raw_model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    random.setstate(state["python_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
    if device.type == "cuda" and state["torch_cuda_rng_state"] is not None:
        # RNG APIs require a CPU ByteTensor even though the remaining
        # checkpoint tensors were mapped directly to the target device.
        torch.cuda.set_rng_state(state["torch_cuda_rng_state"].cpu(), device=device)
    train_batcher.load_state_dict(state["train_batcher"])
    validation_batcher.load_state_dict(state["validation_batcher"])
    return next_step


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    batcher: RandomWindowBatcher,
    batches: int,
    device: torch.device,
    use_bf16: bool,
    world_size: int,
) -> float:
    model.eval()
    total = torch.zeros((), device=device)
    for _ in range(batches):
        inputs, labels = batcher.next(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            _, loss = model(inputs, labels)
        if loss is None:
            raise RuntimeError("evaluation loss was not produced")
        total += loss.detach()
    total /= batches
    if world_size > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        total /= world_size
    model.train()
    return float(total.item())


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"short JSONL append: {written} != {len(encoded)}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    encoded = json.dumps(dict(record), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Publish without overwrite: a hard link is atomic and fails if the
        # destination appeared after the caller's freshness checks.
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_and_verify_source_lock(path: Path) -> dict[str, Any]:
    """Verify every file bound by a source lock before a formal run starts."""

    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-standard non-finite JSON number is forbidden: {value}")

    document = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_nonstandard_constant
    )
    if not isinstance(document, dict):
        raise ValueError("source-lock root must be a JSON object")
    if document.get("schema_version") != "decoder-training-source-lock/1":
        raise ValueError("unsupported source-lock schema_version")
    files = document.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("source-lock files must be a non-empty array")
    repository_root = Path(__file__).resolve().parents[2]
    verified_roles: set[str] = set()
    verified_paths: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise ValueError(f"source-lock files[{index}] must be an object")
        if set(item) != {"path", "role", "sha256", "size_bytes"}:
            raise ValueError(f"source-lock files[{index}] has unexpected or missing fields")
        relative = item["path"]
        role = item["role"]
        expected_hash = item["sha256"]
        expected_size = item["size_bytes"]
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"source-lock files[{index}].path must be non-empty")
        if not isinstance(role, str) or not role:
            raise ValueError(f"source-lock files[{index}].role must be non-empty")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"source-lock files[{index}].sha256 is invalid")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise ValueError(f"source-lock files[{index}].size_bytes is invalid")
        candidate = (repository_root / relative).resolve()
        try:
            candidate.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError(f"source-lock path escapes repository root: {relative!r}") from exc
        normalized = candidate.relative_to(repository_root).as_posix()
        if normalized in verified_paths:
            raise ValueError(f"duplicate source-lock path: {normalized}")
        if not candidate.is_file():
            raise FileNotFoundError(f"source-lock file is missing: {candidate}")
        if candidate.stat().st_size != expected_size:
            raise ValueError(f"source-lock size mismatch: {normalized}")
        if _sha256_file(candidate) != expected_hash:
            raise ValueError(f"source-lock SHA256 mismatch: {normalized}")
        verified_paths.add(normalized)
        verified_roles.add(role)
    required_roles = {"code", "config", "data_manifest", "environment"}
    missing_roles = required_roles - verified_roles
    if missing_roles:
        raise ValueError(
            "source-lock does not bind all required roles: " + ", ".join(sorted(missing_roles))
        )
    return {
        "schema_version": document["schema_version"],
        "file_count": len(files),
        "roles": sorted(verified_roles),
        "paths": sorted(verified_paths),
    }


def _append_jsonl_distributed(
    path: Path,
    record: Mapping[str, Any],
    rank: int,
    world_size: int,
) -> None:
    """Make rank-0 log I/O failure visible to every DDP process."""
    status: list[str | None] = [None]
    if rank == 0:
        try:
            _append_jsonl(path, record)
        except Exception as exc:
            status[0] = f"{type(exc).__name__}: {exc}"
    if world_size > 1:
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(f"rank-0 metrics write failed: {status[0]}")


def _json_safe_cuda_uuid(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    cuuuid_type = getattr(torch._C, "_CUuuid", None)
    if isinstance(cuuuid_type, type) and type(value) is cuuuid_type:
        return str(value)
    value_type = type(value)
    raise TypeError(
        f"unsupported CUDA UUID observation type: {value_type.__module__}.{value_type.__qualname__}"
    )


def _runtime_environment(
    device: torch.device, rank: int = 0, local_rank: int = 0
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        git_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_dirty = None, None
    result: dict[str, Any] = {
        "rank": rank,
        "local_rank": local_rank,
        "pid": os.getpid(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nccl_environment": {
            key: value for key, value in sorted(os.environ.items()) if key.startswith("NCCL_")
        },
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu_name": properties.name,
                "gpu_uuid": _json_safe_cuda_uuid(getattr(properties, "uuid", None)),
                "pci_bus_id": getattr(properties, "pci_bus_id", None),
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
                "cudnn": torch.backends.cudnn.version(),
                "nccl": torch.cuda.nccl.version(),
                "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return result


def _memory_diagnostics(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "device_free_bytes": None,
            "device_total_bytes": None,
        }
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
    }


def _gather_rank_records(
    local_record: Mapping[str, Any], world_size: int
) -> list[Mapping[str, Any]]:
    if world_size == 1:
        return [dict(local_record)]
    gathered: list[Mapping[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, dict(local_record))
    if any(item is None for item in gathered):
        raise RuntimeError(f"rank record gather was incomplete: {gathered}")
    return [dict(item) for item in gathered if item is not None]


def _shared_filesystem_preflight(output_dir: Path, rank: int, world_size: int) -> None:
    """Require every rank to observe one rank-0 sentinel with identical bytes."""
    token_box = [uuid.uuid4().hex if rank == 0 else None]
    if world_size > 1:
        dist.broadcast_object_list(token_box, src=0)
    token = str(token_box[0])
    sentinel = output_dir / f".shared-fs-{token}.sentinel"
    create_error: list[str | None] = [None]
    if rank == 0:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with sentinel.open("xb") as stream:
                stream.write(token.encode("ascii"))
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as exc:
            create_error[0] = f"{type(exc).__name__}: {exc}"
    if world_size > 1:
        dist.broadcast_object_list(create_error, src=0)
    if create_error[0] is not None:
        raise RuntimeError(f"rank-0 shared-filesystem probe failed: {create_error[0]}")
    if world_size > 1:
        dist.barrier()
    try:
        local = {"rank": rank, "ok": sentinel.read_text("ascii") == token}
    except Exception as exc:
        local = {"rank": rank, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    observed = _gather_rank_records(local, world_size)
    if rank == 0:
        sentinel.unlink(missing_ok=True)
    if world_size > 1:
        dist.barrier()
    if any(item.get("ok") is not True for item in observed):
        raise RuntimeError(
            f"output_dir is not a shared filesystem visible to every rank; observed={observed}"
        )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--torch-profiler-dir", type=Path)
    parser.add_argument("--profiler-wait", type=int, default=1)
    parser.add_argument("--profiler-warmup", type=int, default=2)
    parser.add_argument("--profiler-active", type=int, default=3)
    parser.add_argument("--profiler-repeat", type=int, default=1)
    args = parser.parse_args(argv)

    config_bytes = args.config.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    model_config, training, data_config = _load_config(args.config)
    # cuBLAS reads this process setting when it creates a workspace.  Apply the
    # deterministic recipe before process-group setup can initialize CUDA.
    _configure_deterministic_runtime(training.deterministic_algorithms)
    rank, local_rank, world_size, device = _distributed_context(args.device)
    if training.expected_world_size is not None and world_size != training.expected_world_size:
        raise RuntimeError(
            f"config requires world_size={training.expected_world_size}, "
            f"but launch provided {world_size}"
        )
    if training.dtype == "bf16" and device.type != "cuda":
        raise RuntimeError("the bf16 recipe is intentionally restricted to CUDA")
    if training.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not report BF16 support")
    if training.require_source_lock and args.source_lock is None:
        raise RuntimeError("this formal config requires --source-lock")
    source_lock_path = args.source_lock.resolve() if args.source_lock else None
    source_lock_sha256 = None
    source_lock_summary = None
    if source_lock_path is not None:
        try:
            source_lock_summary = _load_and_verify_source_lock(source_lock_path)
            source_lock_sha256 = _sha256_file(source_lock_path)
            repository_root = Path(__file__).resolve().parents[2]
            config_relative = args.config.resolve().relative_to(repository_root).as_posix()
            if config_relative not in source_lock_summary["paths"]:
                raise ValueError(f"source lock does not bind selected config: {config_relative}")
            local_source_status: dict[str, Any] = {
                "ok": True,
                "rank": rank,
                "sha256": source_lock_sha256,
            }
        except Exception as exc:
            local_source_status = {
                "ok": False,
                "rank": rank,
                "error": f"{type(exc).__name__}: {exc}",
            }
        source_statuses = _gather_rank_records(local_source_status, world_size)
        if (
            any(item["ok"] is not True for item in source_statuses)
            or len({item.get("sha256") for item in source_statuses if item["ok"] is True}) != 1
        ):
            raise RuntimeError(
                "source-lock verification must succeed identically on every rank; "
                f"observed={source_statuses}"
            )
    if args.stop_after_step is not None and (
        isinstance(args.stop_after_step, bool)
        or args.stop_after_step <= 0
        or args.stop_after_step >= training.steps
        or args.stop_after_step % training.checkpoint_interval != 0
    ):
        raise ValueError(
            "--stop-after-step must be > 0, < training.steps, and exactly "
            "divisible by checkpoint_interval"
        )
    profiler_values = (
        args.profiler_wait,
        args.profiler_warmup,
        args.profiler_active,
        args.profiler_repeat,
    )
    if (
        any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in profiler_values
        )
        or args.profiler_active <= 0
        or args.profiler_repeat <= 0
    ):
        raise ValueError("profiler wait/warmup must be non-negative and active/repeat positive")

    torch.use_deterministic_algorithms(training.deterministic_algorithms)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = training.allow_tf32
        torch.backends.cudnn.allow_tf32 = training.allow_tf32
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = training.deterministic_algorithms

    output_dir = args.output_dir.resolve()
    _shared_filesystem_preflight(output_dir, rank, world_size)
    log_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "checkpoint.pt"
    if args.resume is not None and args.resume.resolve() != checkpoint_path.resolve():
        raise ValueError(
            "--resume must point to OUTPUT_DIR/checkpoint.pt so interrupted and "
            "resumed records form one fail-closed trajectory"
        )
    if args.resume is None:
        artifact_error = [None]
        if rank == 0:
            existing = sorted(output_dir.iterdir()) if output_dir.exists() else []
            if existing:
                artifact_error[0] = "refusing to overwrite a fresh run's artifacts: " + ", ".join(
                    map(str, existing)
                )
        if world_size > 1:
            dist.broadcast_object_list(artifact_error, src=0)
        if artifact_error[0] is not None:
            raise FileExistsError(str(artifact_error[0]))
    tokens_per_update = (
        world_size
        * training.micro_batch_size
        * training.sequence_length
        * training.gradient_accumulation_steps
    )
    if (
        training.expected_tokens_per_update is not None
        and tokens_per_update != training.expected_tokens_per_update
    ):
        raise RuntimeError(
            f"config requires {training.expected_tokens_per_update} tokens/update, "
            f"but launch computes {tokens_per_update}"
        )

    # All ranks must construct identical initial parameters.
    random.seed(training.seed)
    torch.manual_seed(training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training.seed)

    train_tokens = None
    validation_tokens = None
    data_description = None
    try:
        train_tokens, validation_tokens, data_description = _streams(data_config, training)
        data_description["token_id_contract"] = _validate_stream_token_domain(
            train_tokens,
            validation_tokens,
            model_config.vocab_size,
            str(data_config.get("mode", "synthetic")),
        )
        local_data_status: dict[str, Any] = {
            "ok": True,
            "rank": rank,
            "data_fingerprint": _data_fingerprint(data_description),
        }
    except Exception as exc:
        local_data_status = {
            "ok": False,
            "rank": rank,
            "error": f"{type(exc).__name__}: {exc}",
        }
    data_statuses = _gather_rank_records(local_data_status, world_size)
    if any(item["ok"] is not True for item in data_statuses):
        raise RuntimeError(f"data loading must succeed on every rank; observed={data_statuses}")
    assert train_tokens is not None
    assert validation_tokens is not None
    assert data_description is not None
    data_fingerprint = _data_fingerprint(data_description)
    if world_size > 1:
        # A shared launch can still expose different config files or local
        # datasets to different ranks.  Detect that before constructing DDP;
        # otherwise gradients would silently combine different experiments and
        # rank 0's checkpoint metadata would misdescribe the other ranks.
        local_inputs = {
            "config_sha256": config_sha256,
            "data_fingerprint": data_fingerprint,
            "source_lock_sha256": source_lock_sha256,
        }
        gathered_inputs: list[dict[str, Any] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_inputs, local_inputs)
        if any(item != local_inputs for item in gathered_inputs):
            raise RuntimeError(
                "all DDP ranks must use identical config and data fingerprints; "
                f"observed={gathered_inputs}"
            )
    train_batcher = RandomWindowBatcher(
        train_tokens,
        training.micro_batch_size,
        training.sequence_length,
        training.seed + 1_000 + rank,
    )
    validation_batcher = RandomWindowBatcher(
        validation_tokens,
        training.micro_batch_size,
        training.sequence_length,
        training.seed + 2_000 + rank,
    )

    raw_model = CausalLM(model_config).to(device)
    parameter_count = raw_model.parameter_count()
    if (
        training.expected_parameter_count is not None
        and parameter_count["unique_trainable"] != training.expected_parameter_count
    ):
        raise RuntimeError(
            f"config requires {training.expected_parameter_count} parameters, "
            f"but model has {parameter_count['unique_trainable']}"
        )
    non_fp32_parameters = [
        name for name, parameter in raw_model.named_parameters() if parameter.dtype != torch.float32
    ]
    if non_fp32_parameters:
        raise RuntimeError(
            "the precision contract requires FP32 parameters; observed "
            + ", ".join(non_fp32_parameters[:8])
        )
    optimizer = _optimizer(raw_model, training)
    # After identical initialization, give each rank an independent runtime RNG
    # stream (notably for dropout).  Resume below replaces these states exactly.
    runtime_seed = training.seed + 10_000 + rank
    random.seed(runtime_seed)
    torch.manual_seed(runtime_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(runtime_seed)
    start_step = 0
    resume_sha256 = None
    if args.resume is not None:
        resume_sha256 = _resume_checkpoint_sha256(args.resume, world_size)
        try:
            start_step = _load_checkpoint(
                args.resume,
                raw_model,
                optimizer,
                rank,
                world_size,
                device,
                config_sha256,
                data_fingerprint,
                train_batcher,
                validation_batcher,
                source_lock_sha256,
            )
            local_load_status: dict[str, Any] = {
                "ok": True,
                "next_step": start_step,
            }
        except Exception as exc:
            local_load_status = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        load_statuses: list[dict[str, Any] | None] = [None for _ in range(world_size)]
        if world_size > 1:
            dist.all_gather_object(load_statuses, local_load_status)
        else:
            load_statuses[0] = local_load_status
        if any(status is None or status.get("ok") is not True for status in load_statuses):
            raise RuntimeError(
                f"checkpoint load/validation must succeed on every rank; observed={load_statuses}"
            )
        restored_steps = {
            int(status["next_step"]) for status in load_statuses if status is not None
        }
        if len(restored_steps) != 1:
            raise RuntimeError(
                f"checkpoint next_step differs across ranks; observed={load_statuses}"
            )
        start_step = restored_steps.pop()
        if start_step < 0 or start_step > training.steps:
            raise ValueError(f"checkpoint next_step={start_step} is outside [0,{training.steps}]")
    if args.stop_after_step is not None and start_step >= args.stop_after_step:
        raise ValueError("--stop-after-step must be greater than the resumed next_step")
    train_model: torch.nn.Module = raw_model
    if world_size > 1:
        train_model = DistributedDataParallel(
            train_model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    if training.compile_model:
        # PyTorch 2.10's DDP design note places the DDP wrapper before
        # compilation so DDPOptimizer can split graphs at bucket boundaries.
        # Module.compile() keeps no_sync/checkpoint ownership on the same object.
        train_model.compile()

    profiler_root = (
        args.torch_profiler_dir.resolve() if args.torch_profiler_dir is not None else None
    )
    profiler_error: list[str | None] = [None]
    if rank == 0 and profiler_root is not None:
        try:
            if profiler_root.exists() and any(profiler_root.iterdir()):
                profiler_error[0] = (
                    f"refusing to overwrite a non-empty torch profiler directory: {profiler_root}"
                )
            else:
                profiler_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            profiler_error[0] = f"{type(exc).__name__}: {exc}"
    if world_size > 1:
        dist.broadcast_object_list(profiler_error, src=0)
    if profiler_error[0] is not None:
        raise FileExistsError(str(profiler_error[0]))
    if world_size > 1:
        dist.barrier()

    use_bf16 = training.dtype == "bf16"
    run_id_box = [str(uuid.uuid4()) if rank == 0 else None]
    if world_size > 1:
        dist.broadcast_object_list(run_id_box, src=0)
    run_id = str(run_id_box[0])
    local_environment = _runtime_environment(device, rank, local_rank)
    local_environment["memory_after_model_init"] = _memory_diagnostics(device)
    rank_environments = _gather_rank_records(local_environment, world_size)
    profiler_schedule_steps = (
        args.profiler_wait + args.profiler_warmup + args.profiler_active
    ) * args.profiler_repeat
    profiler_available_steps = (args.stop_after_step or training.steps) - start_step
    if profiler_root is not None and profiler_available_steps < profiler_schedule_steps:
        raise ValueError(
            "the remaining run is shorter than the requested profiler schedule: "
            f"remaining={profiler_available_steps}, "
            f"schedule={profiler_schedule_steps}"
        )
    _append_jsonl_distributed(
        log_path,
        {
            "event": "run_start",
            "run_id": run_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "experiment_name": training.experiment_name,
            "experiment_scope": ("standalone decoder systems experiment; not GLM-ASR training"),
            "config_path": str(args.config.resolve()),
            "config_sha256": config_sha256,
            "data_fingerprint": data_fingerprint,
            "source_lock": (
                {
                    "path": str(source_lock_path),
                    "sha256": source_lock_sha256,
                    "verification": source_lock_summary,
                }
                if source_lock_path is not None
                else None
            ),
            "resume_from_sha256": resume_sha256,
            "start_step": start_step,
            "model": model_config.to_dict(),
            "training": asdict(training),
            "data": data_description,
            "environment": rank_environments[0],
            "rank_environments": rank_environments,
            "world_size": world_size,
            "runtime_seed_rank0": training.seed + 10_000,
            "local_tokens_per_update": (
                training.micro_batch_size
                * training.sequence_length
                * training.gradient_accumulation_steps
            ),
            "tokens_per_update": tokens_per_update,
            "parameter_count": parameter_count,
            "measurement_window": {
                "timing_warmup_steps": training.timing_warmup_steps,
                "eligible_step_start_inclusive": training.timing_warmup_steps,
                "eligible_step_end_exclusive": training.steps,
            },
            "torch_profiler": {
                "enabled": profiler_root is not None,
                "directory": str(profiler_root) if profiler_root is not None else None,
                "wait": args.profiler_wait,
                "warmup": args.profiler_warmup,
                "active": args.profiler_active,
                "repeat": args.profiler_repeat,
            },
            "attention_backend": (
                "torch-sdpa-native-gqa"
                if device.type == "cuda"
                else "torch-sdpa-cpu-explicit-kv-repeat-correctness-only"
            ),
            "precision_recipe": (
                "fp32 parameters/AdamW states/gradients and FP32 DDP reductions "
                "when distributed, with BF16 autocast compute and FP32 loss/reductions"
                if use_bf16
                else "fp32 parameters/AdamW states/compute/gradients"
            ),
            "throughput_scope": (
                "canonical core_update is synchronized forward/backward including "
                "DDP gradient reduction plus clipping and optimizer time; the "
                "cross-rank finite guard is reported separately, and mean-loss "
                "reduction, validation, checkpoint, metrics I/O and profiler runs "
                "are outside canonical measurements"
            ),
            "status": "decoder-systems-experiment",
        },
        rank,
        world_size,
    )

    profiler = None
    profiler_rank_dir = None
    if profiler_root is not None:
        profiler_rank_dir = profiler_root / f"rank-{rank:03d}"
        try:
            profiler_rank_dir.mkdir(parents=True, exist_ok=False)
            local_profiler_status: dict[str, Any] = {"ok": True, "rank": rank}
        except Exception as exc:
            local_profiler_status = {
                "ok": False,
                "rank": rank,
                "error": f"{type(exc).__name__}: {exc}",
            }
        profiler_statuses = _gather_rank_records(local_profiler_status, world_size)
        if any(item["ok"] is not True for item in profiler_statuses):
            raise RuntimeError(
                "profiler rank-directory creation must succeed on every rank; "
                f"observed={profiler_statuses}"
            )

        def export_trace(active_profiler: Any) -> None:
            destination = profiler_rank_dir / f"trace-step-{int(active_profiler.step_num):06d}.json"
            temporary = destination.with_suffix(destination.suffix + f".{uuid.uuid4().hex}.tmp")
            try:
                active_profiler.export_chrome_trace(str(temporary))
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=args.profiler_wait,
                warmup=args.profiler_warmup,
                active=args.profiler_active,
                repeat=args.profiler_repeat,
            ),
            on_trace_ready=export_trace,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        profiler.start()

    train_model.train()
    run_started = time.perf_counter()
    planned_stop = False
    measured_updates = 0
    run_peak_allocated = 0
    run_peak_reserved = 0
    for step in range(start_step, training.steps):
        if world_size > 1:
            dist.barrier()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        local_loss_sum = torch.zeros((), device=device)
        local_losses_finite = torch.ones((), dtype=torch.bool, device=device)
        for micro_step in range(training.gradient_accumulation_steps):
            inputs, labels = train_batcher.next(device)
            should_sync = micro_step == training.gradient_accumulation_steps - 1
            sync_context = (
                contextlib.nullcontext()
                if should_sync or world_size == 1
                else train_model.no_sync()
            )
            # DDP no_sync must wrap both forward and backward.
            with sync_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    _, loss = train_model(inputs, labels)
                    if loss is None:
                        raise RuntimeError("training loss was not produced")
                    local_losses_finite &= torch.isfinite(loss.detach())
                    scaled_loss = loss / training.gradient_accumulation_steps
                scaled_loss.backward()
            local_loss_sum += loss.detach()

        if step == start_step:
            invalid_gradient_dtypes = [
                name
                for name, parameter in raw_model.named_parameters()
                if parameter.grad is not None and parameter.grad.dtype != torch.float32
            ]
            local_precision_status = {
                "rank": rank,
                "ok": not invalid_gradient_dtypes,
                "invalid_gradient_dtypes": invalid_gradient_dtypes[:8],
            }
            precision_statuses = _gather_rank_records(local_precision_status, world_size)
            if any(item["ok"] is not True for item in precision_statuses):
                raise RuntimeError(
                    "the precision contract requires FP32 gradients and DDP "
                    f"reduction buckets; observed={precision_statuses}"
                )

        learning_rate = _lr_for_step(step, training)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), training.grad_clip)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pre_guard_elapsed = time.perf_counter() - step_started

        guard_started = time.perf_counter()
        step_finite = local_losses_finite & torch.isfinite(grad_norm.detach())
        if world_size > 1:
            finite_flag = step_finite.to(dtype=torch.int32)
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            step_finite = finite_flag.bool()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if not bool(step_finite.item()):
            raise FloatingPointError(
                f"at least one rank observed non-finite loss or gradients at "
                f"step={step}; optimizer.step was not executed"
            )
        finite_guard_elapsed = time.perf_counter() - guard_started

        optimizer_started = time.perf_counter()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimizer_elapsed = time.perf_counter() - optimizer_started
        local_core_elapsed = pre_guard_elapsed + optimizer_elapsed
        local_guarded_elapsed = time.perf_counter() - step_started
        if step == start_step:
            invalid_optimizer_dtypes = []
            for parameter_state in optimizer.state.values():
                for state_name, value in parameter_state.items():
                    if (
                        torch.is_tensor(value)
                        and value.is_floating_point()
                        and value.dtype != torch.float32
                    ):
                        invalid_optimizer_dtypes.append(f"{state_name}:{value.dtype}")
            local_optimizer_status = {
                "rank": rank,
                "ok": not invalid_optimizer_dtypes,
                "invalid_optimizer_dtypes": invalid_optimizer_dtypes[:8],
            }
            optimizer_statuses = _gather_rank_records(local_optimizer_status, world_size)
            if any(item["ok"] is not True for item in optimizer_statuses):
                raise RuntimeError(
                    "the precision contract requires FP32 AdamW state; "
                    f"observed={optimizer_statuses}"
                )

        local_mean_loss = local_loss_sum / training.gradient_accumulation_steps
        mean_loss = local_mean_loss.clone()
        if world_size > 1:
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= world_size

        memory = _memory_diagnostics(device)
        if memory["peak_allocated_bytes"] is not None:
            run_peak_allocated = max(run_peak_allocated, int(memory["peak_allocated_bytes"]))
        if memory["peak_reserved_bytes"] is not None:
            run_peak_reserved = max(run_peak_reserved, int(memory["peak_reserved_bytes"]))
        local_tokens = (
            training.micro_batch_size
            * training.sequence_length
            * training.gradient_accumulation_steps
        )
        rank_diagnostics = _gather_rank_records(
            {
                "rank": rank,
                "local_rank": local_rank,
                "core_update_elapsed_s": local_core_elapsed,
                "guarded_update_elapsed_s": local_guarded_elapsed,
                "finite_guard_elapsed_s": finite_guard_elapsed,
                "tokens": local_tokens,
                "core_tokens_per_second": local_tokens / local_core_elapsed,
                "guarded_tokens_per_second": local_tokens / local_guarded_elapsed,
                "loss": float(local_mean_loss.item()),
                "grad_norm": float(grad_norm.item()),
                "memory": memory,
            },
            world_size,
        )
        critical_core_elapsed = max(
            float(item["core_update_elapsed_s"]) for item in rank_diagnostics
        )
        critical_guarded_elapsed = max(
            float(item["guarded_update_elapsed_s"]) for item in rank_diagnostics
        )
        critical_guard_elapsed = max(
            float(item["finite_guard_elapsed_s"]) for item in rank_diagnostics
        )
        measurement_eligible = step >= training.timing_warmup_steps
        if measurement_eligible:
            measured_updates += 1
        _append_jsonl_distributed(
            log_path,
            {
                "event": "train_update",
                "run_id": run_id,
                "step": step,
                "loss": float(mean_loss.item()),
                "learning_rate": learning_rate,
                "grad_norm": float(grad_norm.item()),
                "measurement_eligible": measurement_eligible,
                "canonical_timing": "core_update",
                "critical_rank_core_update_elapsed_s": critical_core_elapsed,
                "critical_rank_guarded_update_elapsed_s": critical_guarded_elapsed,
                "critical_rank_finite_guard_elapsed_s": critical_guard_elapsed,
                "elapsed_s": critical_core_elapsed,
                "tokens": tokens_per_update,
                "tokens_per_second": tokens_per_update / critical_core_elapsed,
                "rank_diagnostics": rank_diagnostics,
            },
            rank,
            world_size,
        )
        if profiler is not None:
            profiler.step()

        next_step = step + 1
        if next_step % training.eval_interval == 0 or next_step == training.steps:
            validation_loss = _evaluate(
                train_model,
                validation_batcher,
                training.eval_batches,
                device,
                use_bf16,
                world_size,
            )
            if not math.isfinite(validation_loss):
                raise FloatingPointError(
                    f"non-finite validation loss at step={next_step}: {validation_loss}"
                )
            log_perplexity = validation_loss
            perplexity = math.exp(validation_loss) if validation_loss < 80.0 else None
            _append_jsonl_distributed(
                log_path,
                {
                    "event": "validation",
                    "run_id": run_id,
                    "step": next_step,
                    "loss": validation_loss,
                    "log_perplexity": log_perplexity,
                    "perplexity": perplexity,
                    "perplexity_status": (
                        "exact" if perplexity is not None else "omitted_overflow_guard"
                    ),
                },
                rank,
                world_size,
            )

        if next_step % training.checkpoint_interval == 0 or next_step == training.steps:
            checkpoint_sha256 = _save_checkpoint(
                checkpoint_path,
                raw_model,
                optimizer,
                next_step,
                rank,
                world_size,
                config_sha256,
                data_fingerprint,
                train_batcher,
                validation_batcher,
                device,
                source_lock_sha256,
                run_id,
            )
            _append_jsonl_distributed(
                log_path,
                {
                    "event": "checkpoint",
                    "run_id": run_id,
                    "step": next_step,
                    "path": str(checkpoint_path),
                    "sha256": checkpoint_sha256,
                },
                rank,
                world_size,
            )
            if args.stop_after_step == next_step:
                planned_stop = True
                _append_jsonl_distributed(
                    log_path,
                    {
                        "event": "planned_stop",
                        "run_id": run_id,
                        "completed_steps": next_step,
                        "checkpoint_sha256": checkpoint_sha256,
                        "reason": "requested checkpoint-boundary interruption",
                    },
                    rank,
                    world_size,
                )
                break

    if profiler is not None:
        profiler.stop()
        assert profiler_rank_dir is not None
        profiler_events = []
        for event in profiler.key_averages():
            profiler_events.append(
                {
                    "key": event.key,
                    "count": int(event.count),
                    "self_cpu_time_total_us": float(event.self_cpu_time_total),
                    "cpu_time_total_us": float(event.cpu_time_total),
                    "self_device_time_total_us": float(
                        getattr(event, "self_device_time_total", 0.0)
                    ),
                    "device_time_total_us": float(getattr(event, "device_time_total", 0.0)),
                    "cpu_memory_usage_bytes": int(event.cpu_memory_usage),
                    "device_memory_usage_bytes": int(getattr(event, "device_memory_usage", 0)),
                }
            )
        _write_json_atomic(
            profiler_rank_dir / "key-averages.json",
            {
                "schema_version": "decoder-training-profiler/1",
                "rank": rank,
                "events": profiler_events,
            },
        )
    if world_size > 1:
        dist.barrier()

    final_rank_diagnostics = _gather_rank_records(
        {
            "rank": rank,
            "peak_allocated_bytes": (run_peak_allocated if device.type == "cuda" else None),
            "peak_reserved_bytes": (run_peak_reserved if device.type == "cuda" else None),
            "elapsed_s": time.perf_counter() - run_started,
        },
        world_size,
    )
    if not planned_stop:
        _append_jsonl_distributed(
            log_path,
            {
                "event": "run_end",
                "run_id": run_id,
                "status": "complete",
                "completed_steps": training.steps,
                "elapsed_s": max(float(item["elapsed_s"]) for item in final_rank_diagnostics),
                "measurement_eligible_updates": measured_updates,
                "rank_diagnostics": final_rank_diagnostics,
            },
            rank,
            world_size,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
