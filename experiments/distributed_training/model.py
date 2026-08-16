"""Minimal decoder-only Transformer with explicit RMSNorm, RoPE and GQA.

The implementation favors readable tensor shapes and testable invariants over
feature count.  It is a training model, not the GLM-ASR serving implementation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2
    hidden_dim: int = 768
    max_seq_len: int = 512
    rope_theta: float = 10_000.0
    rope_fraction: float = 1.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive_ints = {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "hidden_dim": self.hidden_dim,
            "max_seq_len": self.max_seq_len,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        if (
            not isinstance(self.rope_fraction, (int, float))
            or isinstance(self.rope_fraction, bool)
            or not math.isfinite(float(self.rope_fraction))
            or not 0.0 < self.rope_fraction <= 1.0
        ):
            raise ValueError("rope_fraction must be in (0, 1]")
        if (
            not isinstance(self.rope_theta, (int, float))
            or isinstance(self.rope_theta, bool)
            or not math.isfinite(float(self.rope_theta))
            or self.rope_theta <= 0
        ):
            raise ValueError("rope_theta must be finite and positive")
        if (
            not isinstance(self.rms_norm_eps, (int, float))
            or isinstance(self.rms_norm_eps, bool)
            or not math.isfinite(float(self.rms_norm_eps))
            or self.rms_norm_eps <= 0
        ):
            raise ValueError("rms_norm_eps must be positive")
        if (
            not isinstance(self.dropout, (int, float))
            or isinstance(self.dropout, bool)
            or not math.isfinite(float(self.dropout))
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_dim < 2 or self.rope_dim % 2:
            raise ValueError(
                f"int(head_dim * rope_fraction) must be an even integer >= 2; got {self.rope_dim}"
            )

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def rope_dim(self) -> int:
        return int(self.head_dim * self.rope_fraction)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RMSNorm(nn.Module):
    """RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps).

    The reduction is explicitly performed in FP32.  The normalized activation
    and scale are cast to the input dtype before the final multiplication, so
    mixed-precision behavior is visible rather than delegated to autocast.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        inverse_rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = (x_float * inverse_rms).to(dtype=x.dtype)
        return normalized * self.weight.to(dtype=x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    """Precompute cos/sin for Llama-style rotary pairs.

    ``rotary_dim`` may be smaller than the attention head dimension.  Only the
    leading rotary dimensions are transformed; the remaining dimensions pass
    through unchanged (partial RoPE).
    """

    def __init__(self, rotary_dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        if rotary_dim < 2 or rotary_dim % 2:
            raise ValueError("rotary_dim must be an even integer >= 2")
        if max_seq_len <= 0 or theta <= 0:
            raise ValueError("max_seq_len and theta must be positive")
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.rotary_dim = rotary_dim
        self.max_seq_len = max_seq_len
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q: [B,H_q,T,D_h], k: [B,H_kv,T,D_h]
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError("q and k must be rank-4 [B,H,T,D]")
        if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
            raise ValueError("q and k must share batch and sequence dimensions")
        if q.shape[-1] < self.rotary_dim or k.shape[-1] < self.rotary_dim:
            raise ValueError("head dimension is smaller than rotary_dim")
        sequence_length = q.shape[2]
        end = position_offset + sequence_length
        if position_offset < 0 or end > self.max_seq_len:
            raise ValueError(
                f"position range [{position_offset}, {end}) exceeds [0, {self.max_seq_len})"
            )
        cos = self.cos[position_offset:end].to(device=q.device, dtype=q.dtype)[None, None, :, :]
        sin = self.sin[position_offset:end].to(device=q.device, dtype=q.dtype)[None, None, :, :]

        def apply(x: torch.Tensor) -> torch.Tensor:
            rotated, passthrough = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
            rotated = rotated * cos + _rotate_half(rotated) * sin
            return torch.cat((rotated, passthrough), dim=-1)

        return apply(q), apply(k)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.rope = RotaryEmbedding(config.rope_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, _ = x.shape
        q = self.q_proj(x).view(batch, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(x)
            .view(batch, sequence_length, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch, sequence_length, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q, k = self.rope(q, k)
        use_native_gqa = self.n_heads != self.n_kv_heads
        if use_native_gqa and q.device.type != "cuda":
            # PyTorch documents native enable_gqa support for CUDA Flash/math
            # backends.  CPU is a correctness-only fallback that deliberately
            # materializes repeated K/V heads; performance claims must never be
            # taken from this path.
            groups = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
            use_native_gqa = False
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            enable_gqa=use_native_gqa,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch, sequence_length, self.n_heads * self.head_dim)
        )
        return self.o_proj(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.feed_forward = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.ffn_norm(x))


class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.apply(self._initialize_module)
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.layers:
            nn.init.normal_(block.attention.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.feed_forward.down_proj.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, token_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be [B,T]")
        if token_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        if labels is not None and labels.shape != token_ids.shape:
            raise ValueError("labels must have the same [B,T] shape as token_ids")
        hidden = self.token_embedding(token_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_norm(hidden)
        # Exact weight tying: there is no separately stored lm-head parameter.
        logits = F.linear(hidden, self.token_embedding.weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def parameter_count(self) -> dict[str, int]:
        unique = sum(parameter.numel() for parameter in self.parameters())
        return {"unique_trainable": unique, "stored_lm_head_extra": 0}
