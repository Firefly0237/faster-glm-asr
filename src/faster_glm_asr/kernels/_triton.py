# Provenance: this compatibility shim came from later local development around
# the ed-aisys/edin-mls-26-spring GLM-ASR reference. Its authorship is not
# established by the pinned upstream commit and must be audited separately.

"""Import real Triton when available, or expose an import-only CPU stub.

The custom model has explicit PyTorch fallbacks for CPU execution, but the
kernel modules historically imported Triton unconditionally.  That made the
advertised CPU correctness validator fail before reaching any model code on a
machine without Triton.  The stub is intentionally not an executor: attempting
to launch a Triton kernel without the package raises a clear error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    import triton as triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ModuleNotFoundError:
    TRITON_AVAILABLE = False

    class _UnavailableKernel:
        def __init__(self, function: Callable[..., Any]) -> None:
            self.function = function
            self.__name__ = getattr(function, "__name__", "triton_kernel")
            self.__doc__ = getattr(function, "__doc__", None)

        def __getitem__(self, _grid: Any) -> Callable[..., Any]:
            def unavailable(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError(
                    f"Triton kernel {self.__name__} was requested, but the "
                    "triton package is not installed"
                )

            return unavailable

    class _TritonStub:
        @staticmethod
        def jit(function: Callable[..., Any]) -> _UnavailableKernel:
            return _UnavailableKernel(function)

        @staticmethod
        def cdiv(value: int, divisor: int) -> int:
            return (value + divisor - 1) // divisor

        @staticmethod
        def next_power_of_2(value: int) -> int:
            return 1 << (value - 1).bit_length() if value > 0 else 1

    class _LanguageStub:
        # Only needed while Python evaluates kernel annotations. Kernel bodies
        # are not executed by CPU fallback paths.
        constexpr = object()

    triton = _TritonStub()
    tl = _LanguageStub()


__all__ = ["TRITON_AVAILABLE", "tl", "triton"]
