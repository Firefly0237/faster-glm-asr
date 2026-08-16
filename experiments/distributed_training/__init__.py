"""Single-node decoder training experiment for DDP systems validation.

Imports stay lazy so release configuration checks remain usable on hosts where
PyTorch is intentionally absent.
"""

from __future__ import annotations

from typing import Any

__all__ = ["CausalLM", "ModelConfig", "TrainingConfig"]


def __getattr__(name: str) -> Any:
    if name in {"CausalLM", "ModelConfig"}:
        from .model import CausalLM, ModelConfig

        return {"CausalLM": CausalLM, "ModelConfig": ModelConfig}[name]
    if name == "TrainingConfig":
        from .train import TrainingConfig

        return TrainingConfig
    raise AttributeError(name)
