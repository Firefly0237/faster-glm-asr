# Provenance: adapted from the ed-aisys/edin-mls-26-spring GLM-ASR Triton
# reference at commit 525c3c4c3c584ab4c9e0ec7ff8d8ee202933d374. Upstream
# contributors include Yangshen Deng and Yeqi Huang (Chivier Humber); preserve
# that contribution boundary when describing or redistributing this module.

"""Attention with unmasked dense Torch/Triton paths and a masked SDPA fallback.

The custom unmasked paths materialize the attention score matrix and make no
FlashAttention or IO-complexity claim. Masked requests delegate correctness to
PyTorch SDPA, whose CUDA backend is selected by PyTorch rather than this module.
"""

import numpy as np
import torch

from ._triton import TRITON_AVAILABLE, tl, triton


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for Attention
# ============================================================================


@triton.jit
def attention_scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    scale,
    seq_k,
    head_dim,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute scaled attention scores for a single query position.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim,
        other=0.0,
    )
    k = tl.load(
        k_ptr + pid_bh * stride_k0 + offs_k[:, None] * stride_k1 + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    scores = tl.sum(k * q[None, :], axis=1) * scale
    tl.store(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        scores,
        mask=offs_k < seq_k,
    )


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """
    Apply softmax along the last dimension (seq_k).
    Grid: (batch_heads * seq_q,)
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_k

    s = tl.load(scores_ptr + row * stride_s + offs, mask=mask, other=-float("inf"))
    s = s - tl.max(s, axis=0)
    exp_s = tl.exp(s)
    denom = tl.sum(exp_s, axis=0)
    out = exp_s / denom

    tl.store(scores_ptr + row * stride_s + offs, out, mask=mask)


@triton.jit
def attention_output_kernel(
    attn_ptr,
    v_ptr,
    output_ptr,
    seq_k,
    head_dim,
    stride_w0,
    stride_w1,
    stride_w2,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute attention output: attn_weights @ V
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    w = tl.load(
        attn_ptr + pid_bh * stride_w0 + pid_q * stride_w1 + offs_k * stride_w2,
        mask=offs_k < seq_k,
        other=0.0,
    )
    v = tl.load(
        v_ptr + pid_bh * stride_v0 + offs_k[:, None] * stride_v1 + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    out = tl.sum(v * w[:, None], axis=0)
    tl.store(
        output_ptr + pid_bh * stride_o0 + pid_q * stride_o1 + offs_d * stride_o2,
        out,
        mask=offs_d < head_dim,
    )


@triton.jit
def causal_mask_kernel(
    scores_ptr,
    seq_k,
    offset,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
):
    """
    Apply causal mask to attention scores.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        mask=mask,
        other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        scores,
        mask=mask,
    )


# ============================================================================
# Attention Classes
# ============================================================================


class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
    ):
        for name, value in {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads if num_kv_heads is not None else num_heads,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.head_dim != hidden_size // num_heads:
            raise ValueError("head_dim must equal hidden_size // num_heads")
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            q: Query (batch, num_heads, seq_q, head_dim)
            k: Key (batch, num_kv_heads, seq_k, head_dim)
            v: Value (batch, num_kv_heads, seq_k, head_dim)
            attention_mask: Optional mask (batch, 1, seq_q, seq_k)
            is_causal: Whether to apply causal masking

        Returns:
            Output (batch, num_heads, seq_q, head_dim)
        """
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k, and v must have shape [batch, heads, sequence, dim]")
        if k.shape != v.shape:
            raise ValueError("k and v must have identical shapes")
        if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
            raise ValueError("q, k, and v must share batch and head dimensions")
        if q.shape[1] != self.num_heads or k.shape[1] != self.num_kv_heads:
            raise ValueError("q/k head counts do not match this attention configuration")
        if q.shape[-1] != self.head_dim:
            raise ValueError("q/k/v head dimension does not match this configuration")
        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(q, k, v, attention_mask, is_causal, self.scale)

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Logically expand KV heads for GQA.

        ``expand`` itself is a view, but the following reshape may materialize
        repeated storage. This is a correctness fallback, not a claim of a
        GQA-native or allocation-free kernel.
        """
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(batch, num_kv_heads, num_repeats, seq_len, head_dim)
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ATTENTION_DIM = 256


def _canonicalize_attention_mask(
    attention_mask: torch.Tensor,
    *,
    batch: int,
    num_heads: int,
    seq_q: int,
    seq_k: int,
    device: torch.device,
) -> torch.Tensor:
    """Return an additive mask broadcast to ``[batch, heads, query, key]``.

    The public attention API documents ``[B, 1, Q, K]`` masks, while the
    dense Triton path flattens batch and heads.  Expanding the singleton head
    dimension before that flatten prevents an invalid ``reshape`` when
    ``num_heads > 1``.  Two- and three-dimensional forms are accepted for
    conventional key-padding and per-query masks.
    """
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a torch.Tensor")
    if attention_mask.device != device:
        raise ValueError("attention_mask and q/k/v must share a device")

    if attention_mask.ndim == 2:
        if tuple(attention_mask.shape) != (batch, seq_k):
            raise ValueError("2D attention_mask must have shape [batch, key_sequence]")
        mask = attention_mask[:, None, None, :]
    elif attention_mask.ndim == 3:
        if tuple(attention_mask.shape) == (batch, seq_q, seq_k):
            mask = attention_mask[:, None, :, :]
        elif tuple(attention_mask.shape) == (batch * num_heads, seq_q, seq_k):
            mask = attention_mask.reshape(batch, num_heads, seq_q, seq_k)
        else:
            raise ValueError(
                "3D attention_mask must have shape [batch, query, key] or "
                "[batch * heads, query, key]"
            )
    elif attention_mask.ndim == 4:
        mask = attention_mask
    else:
        raise ValueError("attention_mask must have 2, 3, or 4 dimensions")

    target = (batch, num_heads, seq_q, seq_k)
    if any(
        actual not in (1, expected) for actual, expected in zip(mask.shape, target, strict=True)
    ):
        raise ValueError("attention_mask is not broadcastable to [batch, heads, query, key]")
    mask = torch.broadcast_to(mask, target)

    if mask.dtype == torch.bool:
        return mask
    if not torch.is_floating_point(mask):
        raise TypeError("attention_mask must have floating-point or bool dtype")
    return mask.to(torch.float32)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention using Triton kernels.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, sequence, dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1]:
        raise ValueError("direct attention requires matching q/k batch and head counts")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q, k, and v must share head_dim")
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    canonical_attention_mask = None
    if attention_mask is not None:
        canonical_attention_mask = _canonicalize_attention_mask(
            attention_mask,
            batch=batch,
            num_heads=num_heads,
            seq_q=seq_q,
            seq_k=seq_k,
            device=q.device,
        )

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    # Masked attention takes the framework correctness path.  In particular,
    # bool masks preserve the meaning of a fully blocked row: PyTorch SDPA
    # returns zeros, whereas replacing False with a finite -1e9 value would
    # incorrectly produce a uniform average of V.  The unmasked path below is
    # the only path attributed to the prototype dense Triton kernel.
    if canonical_attention_mask is not None:
        torch_mask = canonical_attention_mask
        if is_causal:
            causal_allowed = torch.ones((seq_q, seq_k), dtype=torch.bool, device=q.device).tril()
            if torch_mask.dtype == torch.bool:
                torch_mask = torch_mask & causal_allowed[None, None, :, :]
            else:
                causal_additive = torch.where(
                    causal_allowed,
                    torch.zeros((), dtype=torch.float32, device=q.device),
                    torch.full((), -torch.inf, dtype=torch.float32, device=q.device),
                )
                torch_mask = torch_mask + causal_additive[None, None, :, :]
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=torch_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=float(scale),
        )

    seq_k_padded = next_power_of_two(seq_k)
    head_dim_padded = next_power_of_two(head_dim)

    use_triton = (
        TRITON_AVAILABLE
        and q.is_cuda
        and seq_k_padded <= MAX_ATTENTION_DIM
        and head_dim_padded <= MAX_ATTENTION_DIM
    )

    if use_triton:
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32)
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)

        if seq_k_padded != seq_k or head_dim_padded != head_dim:
            k_padded = torch.zeros(
                (batch * num_heads, seq_k_padded, head_dim_padded),
                dtype=torch.float32,
                device=q.device,
            )
            v_padded = torch.zeros_like(k_padded)
            q_padded = torch.zeros(
                (batch * num_heads, seq_q, head_dim_padded),
                dtype=torch.float32,
                device=q.device,
            )
            k_padded[:, :seq_k, :head_dim] = k_flat
            v_padded[:, :seq_k, :head_dim] = v_flat
            q_padded[:, :, :head_dim] = q_flat
            k_flat = k_padded
            v_flat = v_padded
            q_flat = q_padded

        scores = torch.empty(
            (batch * num_heads, seq_q, seq_k_padded),
            dtype=torch.float32,
            device=q.device,
        )
        output = torch.empty(
            (batch * num_heads, seq_q, head_dim_padded),
            dtype=torch.float32,
            device=q.device,
        )

        grid = (batch * num_heads, seq_q)
        attention_scores_kernel[grid](
            q_flat,
            k_flat,
            scores,
            float(scale),
            seq_k_padded,
            head_dim_padded,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            BLOCK_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
        )

        if seq_k_padded != seq_k:
            scores[:, :, seq_k:] = -1e9

        if is_causal:
            mask = (
                torch.triu(
                    torch.ones((seq_q, seq_k_padded), dtype=torch.float32, device=q.device),
                    diagonal=1,
                )
                * -1e9
            )
            scores = scores + mask[None, :, :]

        if canonical_attention_mask is not None:
            flattened_attention_mask = canonical_attention_mask.reshape(
                batch * num_heads, seq_q, seq_k
            )
            if seq_k_padded != seq_k:
                mask_padded = torch.zeros(
                    (batch * num_heads, seq_q, seq_k_padded),
                    dtype=torch.float32,
                    device=q.device,
                )
                mask_padded[:, :, :seq_k] = flattened_attention_mask
                mask_padded[:, :, seq_k:] = -1e9
                flattened_attention_mask = mask_padded
            scores = scores + flattened_attention_mask

        scores_2d = scores.reshape(batch * num_heads * seq_q, seq_k_padded)
        block = seq_k_padded
        softmax_inplace_kernel[(scores_2d.shape[0],)](
            scores_2d, scores_2d.stride(0), seq_k_padded, BLOCK_SIZE=block
        )
        scores = scores_2d.reshape(batch * num_heads, seq_q, seq_k_padded)

        attention_output_kernel[grid](
            scores,
            v_flat,
            output,
            seq_k_padded,
            head_dim_padded,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BLOCK_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
        )

        if head_dim_padded != head_dim:
            output = output[:, :, :head_dim]

        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale

    if is_causal:
        mask = (
            torch.triu(
                torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device),
                diagonal=1,
            )
            * -1e9
        )
        scores = scores + mask[None, None, :, :]

    if canonical_attention_mask is not None:
        scores = scores + canonical_attention_mask

    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)

    return output.to(q.dtype)
