"""Torch fallbacks and optional Triton kernels used by the GLM-ASR model.

The implementation is derived from the pinned ed-aisys GLM-ASR Triton
reference identified in each module's provenance notice. Defaults deliberately
select PyTorch/cuBLAS linear projections and unfused MLPs; Triton paths remain
opt-in until validated on the target GPU.
"""

from __future__ import annotations

from ._triton import TRITON_AVAILABLE
from .layers import MLP, EncoderMLP, Linear


def configure_kernels(
    *,
    linear_backend: str = "cublas",
    fused_decoder_mlp: bool = False,
    fused_encoder_mlp: bool = False,
) -> None:
    """Configure process-wide experimental kernel switches.

    ``linear_backend`` accepts ``"torch"``, ``"cublas"`` (an alias of the
    Torch matmul path), or ``"triton"``.  The Triton setting is rejected when
    Triton is not importable so a CPU fallback cannot be mistaken for a custom
    kernel run.
    """

    if linear_backend not in {"torch", "cublas", "triton"}:
        raise ValueError("linear_backend must be torch, cublas, or triton")
    if (
        linear_backend == "triton" or fused_decoder_mlp or fused_encoder_mlp
    ) and not TRITON_AVAILABLE:
        raise RuntimeError("requested Triton kernels require the triton package")
    if not isinstance(fused_decoder_mlp, bool) or not isinstance(fused_encoder_mlp, bool):
        raise TypeError("fused MLP switches must be booleans")

    Linear.BACKEND = linear_backend
    MLP.FUSED = fused_decoder_mlp
    EncoderMLP.FUSED = fused_encoder_mlp


configure_kernels()

__all__ = ["TRITON_AVAILABLE", "configure_kernels"]
