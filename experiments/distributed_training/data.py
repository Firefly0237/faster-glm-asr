"""Deterministic byte-token streams for decoder training experiments.

Byte tokenization is intentionally simple and fully inspectable.  It is not a
replacement for a production tokenizer or a claim about foundation-model data
quality. Production use would require normalization, deduplication, filtering,
tokenizer training, and held-out-set governance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import torch


def load_byte_stream(paths: Iterable[Path]) -> torch.Tensor:
    payload = bytearray()
    file_count = 0
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        payload.extend(resolved.read_bytes())
        file_count += 1
    if file_count == 0:
        raise ValueError("at least one text file is required")
    if not payload:
        raise ValueError("combined text files must contain at least one byte")
    # bytearray provides writable buffer storage, so frombuffer is zero-copy
    # without a read-only warning. The dtype conversion creates the independent
    # token tensor before the local buffer is released.
    return torch.frombuffer(payload, dtype=torch.uint8).to(dtype=torch.long)


def synthetic_stream(token_count: int, seed: int) -> torch.Tensor:
    """Create a deterministic, learnable byte stream for smoke tests only."""
    if token_count < 2:
        raise ValueError("token_count must be >= 2")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    motif = torch.randint(0, 128, (257,), generator=generator, dtype=torch.long)
    repetitions = (token_count + motif.numel() - 1) // motif.numel()
    return motif.repeat(repetitions)[:token_count].clone()


class RandomWindowBatcher:
    """Sample next-token windows from a fixed 1-D token stream."""

    def __init__(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        seed: int,
    ) -> None:
        if tokens.ndim != 1 or tokens.dtype != torch.long:
            raise ValueError("tokens must be a 1-D torch.long tensor")
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("batch_size and sequence_length must be positive")
        if tokens.numel() < sequence_length + 1:
            raise ValueError("token stream must contain at least sequence_length + 1 tokens")
        self.tokens = tokens.cpu()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self._offsets = torch.arange(sequence_length + 1)

    def next(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start_exclusive = self.tokens.numel() - self.sequence_length
        starts = torch.randint(
            0,
            max_start_exclusive,
            (self.batch_size,),
            generator=self.generator,
        )
        windows = self.tokens[starts[:, None] + self._offsets[None, :]]
        return (
            windows[:, :-1].to(device=device, non_blocking=True),
            windows[:, 1:].to(device=device, non_blocking=True),
        )

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        self.generator.set_state(state["generator_state"].cpu())
