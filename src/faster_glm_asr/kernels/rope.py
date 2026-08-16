# Provenance: adapted from the ed-aisys/edin-mls-26-spring GLM-ASR Triton
# reference at commit 525c3c4c3c584ab4c9e0ec7ff8d8ee202933d374. Upstream
# contributors include Yangshen Deng and Yeqi Huang (Chivier Humber); preserve
# that contribution boundary when describing or redistributing this module.

"""Rotary position embeddings with Torch and optional Triton cache setup."""

import torch

from ._triton import TRITON_AVAILABLE, tl, triton


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for RoPE
# ============================================================================


@triton.jit
def compute_freqs_kernel(
    positions_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    seq_len,
    half_dim,
    stride_pos,
    stride_inv,
    stride_cos0,
    stride_cos1,
    stride_sin0,
    stride_sin1,
    BLOCK: tl.constexpr,
):
    """Compute cos and sin for rotary embeddings."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < half_dim

    pos = tl.load(positions_ptr + pid * stride_pos)
    inv = tl.load(inv_freq_ptr + offs * stride_inv, mask=mask, other=0.0)
    freqs = pos * inv

    cos_half = tl.cos(freqs)
    sin_half = tl.sin(freqs)

    tl.store(cos_ptr + pid * stride_cos0 + offs * stride_cos1, cos_half, mask=mask)
    tl.store(
        cos_ptr + pid * stride_cos0 + (offs + half_dim) * stride_cos1,
        cos_half,
        mask=mask,
    )
    tl.store(sin_ptr + pid * stride_sin0 + offs * stride_sin1, sin_half, mask=mask)
    tl.store(
        sin_ptr + pid * stride_sin0 + (offs + half_dim) * stride_sin1,
        sin_half,
        mask=mask,
    )


# ============================================================================
# RoPE Classes
# ============================================================================


class RotaryEmbedding:
    """Rotary Position Embedding using Triton."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        partial_rotary_factor: float = 1.0,
    ):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if (
            not isinstance(max_position_embeddings, int)
            or isinstance(max_position_embeddings, bool)
            or max_position_embeddings <= 0
        ):
            raise ValueError("max_position_embeddings must be a positive integer")
        if not isinstance(base, (int, float)) or isinstance(base, bool) or base <= 0:
            raise ValueError("base must be positive")
        if (
            not isinstance(partial_rotary_factor, (int, float))
            or isinstance(partial_rotary_factor, bool)
            or not 0 < partial_rotary_factor <= 1
        ):
            raise ValueError("partial_rotary_factor must be in (0, 1]")
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        self.rotary_dim = int(dim * partial_rotary_factor)
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)
        if self.rotary_dim < 2:
            raise ValueError("rotary dimension must be an even integer >= 2")

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.inv_freq = inv_freq

        self._update_cache(max_position_embeddings)

    def _update_cache(self, seq_len: int, device: torch.device | None = None):
        """Pre-compute cos and sin using Triton kernel."""
        self.max_seq_len_cached = seq_len
        half_dim = self.rotary_dim // 2
        if device is None:
            device = self.inv_freq.device

        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        cos_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)
        sin_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)

        if TRITON_AVAILABLE and device.type == "cuda":
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)

            block = triton.next_power_of_2(half_dim)
            compute_freqs_kernel[(seq_len,)](
                positions,
                self.inv_freq,
                cos_cache,
                sin_cache,
                seq_len,
                half_dim,
                positions.stride(0),
                self.inv_freq.stride(0),
                cos_cache.stride(0),
                cos_cache.stride(1),
                sin_cache.stride(0),
                sin_cache.stride(1),
                BLOCK=block,
            )
        else:
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            freqs = positions[:, None] * self.inv_freq[None, :]
            cos_half = torch.cos(freqs)
            sin_half = torch.sin(freqs)
            cos_cache[:, :half_dim] = cos_half
            cos_cache[:, half_dim : half_dim * 2] = cos_half
            sin_cache[:, :half_dim] = sin_half
            sin_cache[:, half_dim : half_dim * 2] = sin_half

        self.cos_cached = cos_cache
        self.sin_cached = sin_cache

    def __call__(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get batched cos and sin tensors for the requested positions."""
        if not isinstance(x, torch.Tensor) or x.ndim not in (3, 4):
            raise ValueError(
                "x must have shape [batch, sequence, dim] or [batch, heads, sequence, dim]"
            )
        batch = x.shape[0]
        seq_len = x.shape[-2]
        if seq_len > self.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")

        if position_ids is not None:
            if not isinstance(position_ids, torch.Tensor):
                raise TypeError("position_ids must be a torch.Tensor")
            if position_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("position_ids must use int32 or int64 dtype")
            if position_ids.device != x.device:
                raise ValueError("position_ids and x must share a device")
            if position_ids.ndim != 2 or position_ids.shape[1] != seq_len:
                raise ValueError("position_ids must have shape [batch_or_one, sequence]")
            if position_ids.shape[0] not in (1, batch):
                raise ValueError("position_ids batch must be one or match x")
            in_bounds = torch.all(
                (position_ids >= 0) & (position_ids < self.max_position_embeddings)
            )
            if position_ids.is_cuda:
                torch._assert_async(
                    in_bounds,
                    "position_ids are outside max_position_embeddings",
                )
            elif not bool(in_bounds.item()):
                raise ValueError("position_ids are outside max_position_embeddings")

        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len, device=x.device)
        elif self.cos_cached.device != x.device:
            self._update_cache(self.max_seq_len_cached, device=x.device)

        if position_ids is not None:
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
        else:
            cos = self.cos_cached[:seq_len].to(x.dtype)[None, :, :]
            sin = self.sin_cached[:seq_len].to(x.dtype)[None, :, :]

        return cos, sin


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ROPE_DIM = 256


def _apply_rope_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    half_dim: int,
) -> torch.Tensor:
    """Apply RoPE to a single tensor (Q or K) using Torch."""
    batch, _, seq_len, _ = x.shape
    if cos.ndim == 2:
        cos = cos[None, :, :]
        sin = sin[None, :, :]
    if cos.ndim != 3 or sin.ndim != 3 or cos.shape != sin.shape:
        raise ValueError("cos and sin must have identical [batch_or_one, sequence, rotary] shapes")
    if cos.shape[0] not in (1, batch) or cos.shape[1] != seq_len:
        raise ValueError("RoPE batch/sequence dimensions do not match q/k")

    cos = cos[:, :, :half_dim]
    sin = sin[:, :, :half_dim]

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim : half_dim * 2]

    cos_expanded = cos[:, None, :, :]
    sin_expanded = sin[:, None, :, :]

    x1_rot = x1 * cos_expanded - x2 * sin_expanded
    x2_rot = x2 * cos_expanded + x1 * sin_expanded

    if x.shape[-1] > half_dim * 2:
        x_pass = x[..., half_dim * 2 :]
        return torch.cat([x1_rot, x2_rot, x_pass], dim=-1)
    return torch.cat([x1_rot, x2_rot], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings.
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [batch, heads, sequence, dim]")
    if q.shape[0] != k.shape[0] or q.shape[-2] != k.shape[-2]:
        raise ValueError("q and k must share batch and sequence dimensions")
    if cos.shape != sin.shape:
        raise ValueError("cos and sin must have identical shapes")
    if rotary_dim is None:
        rotary_dim = q.shape[-1]

    if (
        not isinstance(rotary_dim, int)
        or isinstance(rotary_dim, bool)
        or rotary_dim <= 0
        or rotary_dim % 2
        or rotary_dim > min(q.shape[-1], k.shape[-1])
    ):
        raise ValueError("rotary_dim must be a positive even value within q/k head_dim")
    half_dim = rotary_dim // 2

    cos = cos.to(torch.float32).contiguous()
    sin = sin.to(torch.float32).contiguous()

    q_out = _apply_rope_single(q, cos, sin, half_dim)
    k_out = _apply_rope_single(k, cos, sin, half_dim)

    return q_out.to(q.dtype), k_out.to(k.dtype)


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to partial dimensions."""
    return apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)
