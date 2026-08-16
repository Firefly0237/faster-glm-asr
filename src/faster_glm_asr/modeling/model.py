# Provenance: the model topology and full-prefix path are adapted from the
# ed-aisys/edin-mls-26-spring GLM-ASR Triton reference at commit
# 525c3c4c3c584ab4c9e0ec7ff8d8ee202933d374. Upstream contributors include
# Yangshen Deng and Yeqi Huang (Chivier Humber). Tuple/static-cache extensions
# came from later local development; authorship remains unverified and must not
# be attributed until confirmed.

"""Custom GLM-ASR inference graph with three explicit decoding strategies.

The graph uses lightweight tensor containers instead of ``torch.nn.Module``.
It retains full-prefix recomputation for measurement, a growing tuple KV cache,
and a pre-allocated static KV cache. GPU kernels are optional and have Torch
fallbacks; no performance claim follows from selecting either path.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..kernels.attention import MultiHeadAttention, scaled_dot_product_attention
from ..kernels.conv import Conv1d

# Import Triton components
from ..kernels.layers import (
    MLP,
    Embedding,
    LayerNorm,
    Linear,
    RMSNorm,
    gelu,
)
from ..kernels.rope import RotaryEmbedding, apply_rotary_pos_emb

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class GlmAsrConfig:
    """Configuration for GLM-ASR model."""

    # Audio encoder
    audio_hidden_size: int = 1280
    audio_num_heads: int = 20
    audio_num_layers: int = 32
    audio_intermediate_size: int = 5120
    audio_max_position_embeddings: int = 1500

    # Text decoder
    text_hidden_size: int = 2048
    text_num_heads: int = 16
    text_num_kv_heads: int = 4
    text_num_layers: int = 28
    text_intermediate_size: int = 6144
    text_vocab_size: int = 59264
    text_max_position_embeddings: int = 8192
    text_rope_base: float = 10000.0
    text_rms_norm_eps: float = 1e-5

    # Projector
    projector_hidden_size: int = 4096  # Intermediate size in projector
    projector_pool_factor: int = 4  # Concatenate 4 audio frames

    # Generation
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int | list[int] | None = None

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "audio_hidden_size",
            "audio_num_heads",
            "audio_num_layers",
            "audio_intermediate_size",
            "audio_max_position_embeddings",
            "text_hidden_size",
            "text_num_heads",
            "text_num_kv_heads",
            "text_num_layers",
            "text_intermediate_size",
            "text_vocab_size",
            "text_max_position_embeddings",
            "projector_hidden_size",
            "projector_pool_factor",
        )
        for name in positive_integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.audio_hidden_size % self.audio_num_heads:
            raise ValueError("audio_hidden_size must be divisible by audio_num_heads")
        if self.text_hidden_size % self.text_num_heads:
            raise ValueError("text_hidden_size must be divisible by text_num_heads")
        if self.text_num_heads % self.text_num_kv_heads:
            raise ValueError("text_num_heads must be divisible by text_num_kv_heads")
        audio_head_dim = self.audio_hidden_size // self.audio_num_heads
        text_head_dim = self.text_hidden_size // self.text_num_heads
        if int(audio_head_dim * 0.5) < 2 or int(audio_head_dim * 0.5) % 2:
            raise ValueError("half of audio attention head_dim must be even and >= 2")
        if text_head_dim < 2 or text_head_dim % 2:
            raise ValueError("text attention head_dim must be even and >= 2")
        for name in ("text_rope_base", "text_rms_norm_eps"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("pad_token_id", "bos_token_id"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value < self.text_vocab_size
            ):
                raise ValueError(f"{name} must be in [0, text_vocab_size)")

        # Match the official GLM-ASR generation config without sharing a mutable
        # list between dataclass instances.
        if self.eos_token_id is None:
            self.eos_token_id = [59246, 59253, 59255]
        eos_ids = [self.eos_token_id] if isinstance(self.eos_token_id, int) else self.eos_token_id
        if (
            not isinstance(eos_ids, list)
            or not eos_ids
            or any(
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or not 0 <= token_id < self.text_vocab_size
                for token_id in eos_ids
            )
        ):
            raise ValueError("eos_token_id must contain IDs in [0, text_vocab_size)")


# ============================================================================
# Audio Encoder Components
# ============================================================================


class AudioEncoderLayer:
    """Single transformer layer for audio encoder (pre-norm with LayerNorm + RoPE)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        rotary_dim: int = 32,  # Partial RoPE: only rotate first rotary_dim dimensions
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_dim = rotary_dim

        # Layer norms
        self.self_attn_layer_norm = LayerNorm(hidden_size)
        self.final_layer_norm = LayerNorm(hidden_size)

        # Attention projections
        self.q_proj = Linear(hidden_size, hidden_size, bias=True)
        # The pinned GLM-ASR checkpoint has no audio K-projection bias.  Keeping
        # a synthetic zero bias would be numerically redundant and would add a
        # spurious operation to the custom-path benchmark.
        self.k_proj = Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = Linear(hidden_size, hidden_size, bias=True)

        # MLP
        self.fc1 = Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=True)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply RoPE if provided (partial rotation for audio encoder)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin, rotary_dim=self.rotary_dim)

        # Attention
        attn_output = scaled_dot_product_attention(q, k, v, attention_mask)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)

        # Output projection + residual
        hidden_states = self.out_proj(attn_output)
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class AudioEncoder:
    """Whisper-style audio encoder using Torch + Triton with RoPE."""

    def __init__(self, config: GlmAsrConfig):
        self.config = config
        self.head_dim = config.audio_hidden_size // config.audio_num_heads

        # Convolutional subsampling
        self.conv1 = Conv1d(128, config.audio_hidden_size, kernel_size=3, padding=1)
        self.conv2 = Conv1d(
            config.audio_hidden_size, config.audio_hidden_size, kernel_size=3, stride=2, padding=1
        )

        # Rotary position embeddings (same as HF GLM-ASR)
        # Note: audio encoder uses partial_rotary_factor=0.5
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.audio_max_position_embeddings,
            base=10000.0,
            partial_rotary_factor=0.5,  # Only apply RoPE to half the dimensions
        )

        # Transformer layers (with partial RoPE)
        self.layers = [
            AudioEncoderLayer(
                config.audio_hidden_size,
                config.audio_num_heads,
                config.audio_intermediate_size,
                rotary_dim=self.rotary_emb.rotary_dim,  # Pass rotary_dim for partial RoPE
            )
            for _ in range(config.audio_num_layers)
        ]

        # Final layer norm
        self.layer_norm = LayerNorm(config.audio_hidden_size)

    def __call__(
        self,
        input_features: torch.Tensor,  # (batch, features, time)
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Convolutional feature extraction
        # input_features: (batch, mel_channels, time)
        hidden_states = gelu(self.conv1(input_features))
        hidden_states = gelu(self.conv2(hidden_states))

        # (batch, hidden, time) -> (batch, time, hidden)
        hidden_states = hidden_states.permute(0, 2, 1)

        _, seq_len, _ = hidden_states.shape

        # Compute RoPE position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.int64, device=hidden_states.device)[
            None, :
        ]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Transformer layers with RoPE
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_embeddings)

        # Final norm
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


# ============================================================================
# Text Decoder Components
# ============================================================================


class DecoderLayer:
    """Single transformer layer for text decoder (pre-norm with RMSNorm)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        # Layer norms
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Attention projections (no bias for Llama-style)
        self.q_proj = Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # MLP (SwiGLU)
        self.mlp = MLP(hidden_size, intermediate_size, activation="silu", use_gating=True)

        # Attention handler
        self.attention = MultiHeadAttention(hidden_size, num_heads, num_kv_heads, self.head_dim)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        is_causal: bool = True,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with optional KV cache support.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_embeddings: Shared RoPE cos/sin for this decoder forward
            is_causal: Whether to use causal attention
            past_key_value: Optional (past_key, past_value) cache from previous step
            use_cache: Whether to return updated KV cache

        Returns:
            If use_cache=False: hidden_states
            If use_cache=True: (hidden_states, (key, value)) tuple
        """
        batch, seq_len, _ = hidden_states.shape
        if past_key_value is not None and seq_len != 1:
            raise NotImplementedError(
                "tuple-cache continuation currently requires seq_len=1; "
                "multi-token verification needs an offset causal mask"
            )

        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        if position_embeddings is None:
            raise ValueError("position_embeddings are required")
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Growing tuple-cache handling.
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # Concatenate past K/V with current K/V
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

        # Store current K/V for cache if needed
        if use_cache:
            present_key_value = (k, v)

        # Attention with GQA support
        # When using KV cache, only query the new positions but attend to all K/V
        attn_output = self.attention(q, k, v, attention_mask, is_causal and past_key_value is None)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)

        # Output projection + residual
        hidden_states = self.o_proj(attn_output)
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if use_cache:
            return hidden_states, present_key_value
        return hidden_states

    def forward_with_kv_buffer(
        self,
        hidden_states: torch.Tensor,
        kv_buffer: tuple[torch.Tensor, torch.Tensor],
        cache_pos: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        """Forward with a pre-allocated KV buffer and no cache concatenation.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            kv_buffer: Pre-allocated (key_buffer, value_buffer) each shape
                       (batch, num_kv_heads, max_seq_len, head_dim)
            cache_pos: Current position in the cache (start of new data)
            position_embeddings: Shared RoPE cos/sin for this decoder forward

        Returns:
            (hidden_states, new_cache_pos)
        """
        batch, seq_len, _ = hidden_states.shape
        if not isinstance(cache_pos, int) or isinstance(cache_pos, bool) or cache_pos < 0:
            raise ValueError("cache_pos must be a non-negative integer")
        if cache_pos > 0 and seq_len != 1:
            raise NotImplementedError(
                "static-cache continuation currently requires seq_len=1; "
                "multi-token verification needs an offset causal mask"
            )
        key_buffer, value_buffer = kv_buffer
        expected_prefix = (batch, self.num_kv_heads)
        if (
            key_buffer.ndim != 4
            or value_buffer.ndim != 4
            or key_buffer.shape != value_buffer.shape
            or key_buffer.shape[:2] != expected_prefix
            or key_buffer.shape[3] != self.head_dim
        ):
            raise ValueError(
                "KV buffers must share shape [batch, num_kv_heads, capacity, head_dim]"
            )
        if key_buffer.device != hidden_states.device or value_buffer.device != hidden_states.device:
            raise ValueError("KV buffers and hidden_states must share a device")
        if key_buffer.dtype != hidden_states.dtype or value_buffer.dtype != hidden_states.dtype:
            raise ValueError("KV buffers and hidden_states must share a dtype")
        if cache_pos + seq_len > key_buffer.shape[2]:
            raise ValueError("KV buffer capacity is smaller than cache_pos + seq_len")

        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # Make V contiguous now (it will not be processed by RoPE).
        v = v.contiguous()

        # Apply RoPE (creates contiguous Q/K outputs via concatenation)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Write to the pre-allocated buffer; both K and V are contiguous.
        new_cache_pos = cache_pos + seq_len
        key_buffer[:, :, cache_pos:new_cache_pos, :] = k
        value_buffer[:, :, cache_pos:new_cache_pos, :] = v

        # Use buffer slice for attention
        k_for_attn = key_buffer[:, :, :new_cache_pos, :]
        v_for_attn = value_buffer[:, :, :new_cache_pos, :]

        # Attention with GQA support
        attn_output = self.attention(q, k_for_attn, v_for_attn, None, cache_pos == 0)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)

        # Output projection + residual
        hidden_states = self.o_proj(attn_output)
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_cache_pos


class TextDecoder:
    """Llama-style text decoder using Torch + Triton."""

    def __init__(self, config: GlmAsrConfig):
        self.config = config
        self.num_layers = config.text_num_layers

        # Token embeddings
        self.embed_tokens = Embedding(config.text_vocab_size, config.text_hidden_size)

        # RoPE
        self.rope = RotaryEmbedding(
            dim=config.text_hidden_size // config.text_num_heads,
            max_position_embeddings=config.text_max_position_embeddings,
            base=config.text_rope_base,
        )

        # Transformer layers
        self.layers = [
            DecoderLayer(
                config.text_hidden_size,
                config.text_num_heads,
                config.text_num_kv_heads,
                config.text_intermediate_size,
                config.text_rms_norm_eps,
            )
            for _ in range(config.text_num_layers)
        ]

        # Final norm
        self.norm = RMSNorm(config.text_hidden_size, eps=config.text_rms_norm_eps)

    def __call__(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV cache support.

        Args:
            input_ids: Token IDs
            inputs_embeds: Pre-computed embeddings
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            past_key_values: List of (key, value) tuples for each layer
            use_cache: Whether to return KV cache

        Returns:
            If use_cache=False: hidden_states
            If use_cache=True: (hidden_states, past_key_values)
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if input_ids is not None and input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if inputs_embeds is not None and (
            inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.text_hidden_size
        ):
            raise ValueError("inputs_embeds must have shape [batch, sequence, text_hidden_size]")
        if past_key_values is not None and len(past_key_values) != self.num_layers:
            raise ValueError("past_key_values must contain one entry per decoder layer")
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds

        batch, seq_len, _ = hidden_states.shape
        past_length = 0
        if past_key_values is not None:
            expected_prefix = (batch, self.config.text_num_kv_heads)
            expected_head_dim = self.config.text_hidden_size // self.config.text_num_heads
            observed_past_length: int | None = None
            storage_addresses: set[tuple[torch.device, int]] = set()
            for layer_index, entry in enumerate(past_key_values):
                if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                    raise ValueError(f"past_key_values[{layer_index}] must be a (key, value) pair")
                past_key, past_value = entry
                if not isinstance(past_key, torch.Tensor) or not isinstance(
                    past_value, torch.Tensor
                ):
                    raise TypeError("past key/value entries must be torch.Tensor objects")
                if (
                    past_key.ndim != 4
                    or past_value.shape != past_key.shape
                    or past_key.shape[:2] != expected_prefix
                    or past_key.shape[3] != expected_head_dim
                ):
                    raise ValueError(
                        "past key/value tensors must share "
                        "[batch, num_kv_heads, sequence, head_dim] shape"
                    )
                if (
                    past_key.device != hidden_states.device
                    or past_value.device != hidden_states.device
                ):
                    raise ValueError("past key/value tensors and hidden_states must share a device")
                if past_key.dtype != hidden_states.dtype or past_value.dtype != hidden_states.dtype:
                    raise ValueError("past key/value tensors and hidden_states must share a dtype")
                for tensor in (past_key, past_value):
                    address = (tensor.device, tensor.untyped_storage().data_ptr())
                    if address in storage_addresses:
                        raise ValueError(
                            "past key/value tensors must not alias across keys, values, or layers"
                        )
                    storage_addresses.add(address)
                layer_past_length = past_key.shape[2]
                if observed_past_length is None:
                    observed_past_length = layer_past_length
                elif layer_past_length != observed_past_length:
                    raise ValueError("all decoder layers must have the same past sequence length")
            past_length = observed_past_length or 0
        if past_length + seq_len > self.config.text_max_position_embeddings:
            raise ValueError("decoder sequence exceeds text_max_position_embeddings")

        # Generate position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + seq_len,
                dtype=torch.int64,
                device=hidden_states.device,
            )[None, :].repeat(batch, 1)
        position_embeddings = self.rope(hidden_states, position_ids)

        # Transformer layers with KV cache
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if use_cache:
                hidden_states, present_kv = layer(
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                    is_causal=True,
                    past_key_value=past_kv,
                    use_cache=True,
                )
                present_key_values.append(present_kv)
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                    is_causal=True,
                    past_key_value=past_kv,
                    use_cache=False,
                )

        # Final norm
        hidden_states = self.norm(hidden_states)

        if use_cache:
            return hidden_states, present_key_values
        return hidden_states

    def forward_with_kv_buffers(
        self,
        inputs_embeds: torch.Tensor,
        kv_buffers: list[tuple[torch.Tensor, torch.Tensor]],
        cache_pos: int,
    ) -> tuple[torch.Tensor, int]:
        """Forward all decoder layers with pre-allocated KV buffers.

        Args:
            inputs_embeds: Input embeddings
            kv_buffers: List of (key_buffer, value_buffer) for each layer
            cache_pos: Current position in the cache

        Returns:
            (hidden_states, new_cache_pos)
        """
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.text_hidden_size:
            raise ValueError("inputs_embeds must have shape [batch, sequence, text_hidden_size]")
        if len(kv_buffers) != self.num_layers:
            raise ValueError("kv_buffers must contain one entry per decoder layer")
        if not isinstance(cache_pos, int) or isinstance(cache_pos, bool) or cache_pos < 0:
            raise ValueError("cache_pos must be a non-negative integer")
        hidden_states = inputs_embeds
        batch, seq_len, _ = hidden_states.shape
        if cache_pos + seq_len > self.config.text_max_position_embeddings:
            raise ValueError("decoder sequence exceeds text_max_position_embeddings")
        expected_prefix = (batch, self.config.text_num_kv_heads)
        expected_head_dim = self.config.text_hidden_size // self.config.text_num_heads
        observed_capacity: int | None = None
        storage_addresses: set[tuple[torch.device, int]] = set()
        for layer_index, entry in enumerate(kv_buffers):
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError(f"kv_buffers[{layer_index}] must be a (key, value) pair")
            key_buffer, value_buffer = entry
            for tensor in (key_buffer, value_buffer):
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError("KV buffers must be torch.Tensor objects")
            if (
                key_buffer.ndim != 4
                or value_buffer.shape != key_buffer.shape
                or key_buffer.shape[:2] != expected_prefix
                or key_buffer.shape[3] != expected_head_dim
            ):
                raise ValueError(
                    "KV buffers must share shape [batch, num_kv_heads, capacity, head_dim]"
                )
            if (
                key_buffer.device != hidden_states.device
                or value_buffer.device != hidden_states.device
            ):
                raise ValueError("KV buffers and hidden_states must share a device")
            if key_buffer.dtype != hidden_states.dtype or value_buffer.dtype != hidden_states.dtype:
                raise ValueError("KV buffers and hidden_states must share a dtype")
            capacity = key_buffer.shape[2]
            if cache_pos + seq_len > capacity:
                raise ValueError("KV buffer capacity is smaller than cache_pos + seq_len")
            if observed_capacity is None:
                observed_capacity = capacity
            elif capacity != observed_capacity:
                raise ValueError("all decoder layers must have the same KV buffer capacity")
            for tensor in (key_buffer, value_buffer):
                address = (tensor.device, tensor.untyped_storage().data_ptr())
                if address in storage_addresses:
                    raise ValueError("KV buffers must not alias across keys, values, or layers")
                storage_addresses.add(address)

        # Generate position ids
        position_ids = torch.arange(
            cache_pos,
            cache_pos + seq_len,
            dtype=torch.int64,
            device=hidden_states.device,
        )[None, :].repeat(batch, 1)
        position_embeddings = self.rope(hidden_states, position_ids)

        # Process through layers
        new_cache_pos = cache_pos
        for i, layer in enumerate(self.layers):
            hidden_states, new_cache_pos = layer.forward_with_kv_buffer(
                hidden_states,
                kv_buffers[i],
                cache_pos,
                position_embeddings,
            )

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states, new_cache_pos

    def allocate_kv_buffers(
        self,
        batch_size: int,
        max_seq_len: int,
        dtype=torch.float32,
        device: torch.device | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Allocate KV buffers for all layers.

        Returns:
            List of (key_buffer, value_buffer) for each layer.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(max_seq_len, int) or isinstance(max_seq_len, bool) or max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer")
        if max_seq_len > self.config.text_max_position_embeddings:
            raise ValueError("max_seq_len exceeds text_max_position_embeddings")
        head_dim = self.config.text_hidden_size // self.config.text_num_heads
        num_kv_heads = self.config.text_num_kv_heads

        kv_buffers = []
        if device is None:
            embedding_weight = getattr(self.embed_tokens, "weight", None)
            device = embedding_weight.device if embedding_weight is not None else None
        for _ in range(self.num_layers):
            key_buffer = torch.zeros(
                (batch_size, num_kv_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=device,
            )
            value_buffer = torch.zeros(
                (batch_size, num_kv_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=device,
            )
            kv_buffers.append((key_buffer, value_buffer))

        return kv_buffers


# ============================================================================
# Multi-Modal Projector
# ============================================================================


class MultiModalProjector:
    """Projects audio features to text embedding space with frame pooling."""

    def __init__(self, config: GlmAsrConfig):
        self.pool_factor = config.projector_pool_factor
        # After pooling: audio_hidden_size * pool_factor -> projector_hidden_size
        pooled_dim = config.audio_hidden_size * self.pool_factor
        self.linear_1 = Linear(pooled_dim, config.projector_hidden_size, bias=True)
        self.act = gelu
        self.linear_2 = Linear(config.projector_hidden_size, config.text_hidden_size, bias=True)

    def _pool_frames(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Pool audio frames by concatenating consecutive frames.

        Args:
            audio_features: (batch, seq_len, hidden_size) or (seq_len, hidden_size)

        Returns:
            Pooled features: (batch, seq_len // pool_factor, hidden_size * pool_factor)
        """
        if audio_features.ndim == 2:
            # (seq_len, hidden_size) -> add batch dimension
            audio_features = audio_features[None, :, :]
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch, seq_len, hidden_size = audio_features.shape

        # Truncate to multiple of pool_factor
        new_seq_len = (seq_len // self.pool_factor) * self.pool_factor
        audio_features = audio_features[:, :new_seq_len, :]

        # Reshape to concatenate frames
        # (batch, seq_len, hidden) -> (batch, seq_len // pool, pool * hidden)
        audio_features = audio_features.reshape(
            batch, new_seq_len // self.pool_factor, self.pool_factor * hidden_size
        )

        if squeeze_batch:
            audio_features = audio_features[0]

        return audio_features

    def __call__(self, audio_features: torch.Tensor) -> torch.Tensor:
        # Pool consecutive frames
        pooled = self._pool_frames(audio_features)
        # Project through MLP
        hidden_states = self.linear_1(pooled)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# ============================================================================
# Full Model
# ============================================================================


class GlmAsrModel:
    """GLM-ASR model using Torch + Triton kernels."""

    def __init__(self, config: GlmAsrConfig):
        self.config = config

        # Components
        self.audio_encoder = AudioEncoder(config)
        self.multi_modal_projector = MultiModalProjector(config)
        self.text_decoder = TextDecoder(config)

        # LM head (tied with embedding)
        self.lm_head = Linear(config.text_hidden_size, config.text_vocab_size, bias=False)

    def tie_lm_head_weights(self, device: torch.device | None = None) -> None:
        """Make the output projection share the token-embedding storage.

        These lightweight layers are not ``nn.Module`` instances and lazily
        move their weights.  Re-establishing the reference here prevents the
        embedding and LM head from silently becoming two resident copies.
        """
        embedding_weight = self.text_decoder.embed_tokens.weight
        if device is not None and embedding_weight.device != device:
            embedding_weight = embedding_weight.to(device)
            self.text_decoder.embed_tokens.weight = embedding_weight
        if self.lm_head.weight is not embedding_weight:
            self.lm_head.weight = embedding_weight
            self.lm_head._weight_t_padded = None

    def project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states with the tied token-embedding matrix."""
        self.tie_lm_head_weights(device=hidden_states.device)
        return self.lm_head(hidden_states)

    def encode_audio(
        self, input_features: torch.Tensor, input_features_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode audio features.

        Args:
            input_features: (batch, mel_bins, time) mel spectrogram
            input_features_mask: (batch, time) mask for valid audio frames

        Returns:
            Audio embeddings (num_valid_frames, hidden_size) or (batch, seq, hidden) if no mask
        """
        audio_features = self.audio_encoder(input_features)
        projected = self.multi_modal_projector(audio_features)

        if input_features_mask is not None:
            # Compute audio lengths after conv layers (matching HF implementation)
            audio_lengths = torch.sum(input_features_mask, dim=-1)
            for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
                audio_lengths = (audio_lengths + 2 * padding - (kernel_size - 1) - 1) // stride + 1
            merge_factor = 4
            post_lengths = (audio_lengths - merge_factor) // merge_factor + 1

            # Create mask and extract valid embeddings
            seq_len = projected.shape[1]
            valid_mask = (
                torch.arange(seq_len, device=projected.device)[None, :] < post_lengths[:, None]
            )
            # Flatten valid frames (removes batch dimension for now)
            projected = projected[valid_mask]

        return projected

    def decode(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Decode to logits with optional KV cache support."""
        result = self.text_decoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if use_cache:
            hidden_states, present_key_values = result
            logits = self.project_logits(hidden_states)
            return logits, present_key_values
        else:
            hidden_states = result
            logits = self.project_logits(hidden_states)
            return logits

    def generate(
        self,
        input_features: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        audio_pad_token_id: int = 59260,  # <|pad|> token for audio
        phase_callback: Callable[[str, str], None] | None = None,
    ) -> torch.Tensor:
        """Generate tokens from audio with proper chat template format.

        Args:
            input_features: (batch, mel_bins, time) mel spectrogram
            input_ids: (batch, seq_len) token IDs with audio placeholders (<|pad|>)
            input_features_mask: (batch, time) mask for valid audio frames
            attention_mask: (batch, seq_len) attention mask
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            audio_pad_token_id: Token ID for audio placeholders

        Returns:
            Generated token IDs
        """
        if input_ids is None:
            raise ValueError("input_ids produced by the pinned GLM-ASR processor are required")
        if phase_callback is not None and max_new_tokens > 0:
            phase_callback("time_to_first_token", "start")
        if input_ids.shape[0] != 1:
            raise NotImplementedError("the full-prefix baseline only supports batch_size=1")
        if input_features.ndim != 3:
            raise ValueError("input_features must have shape [windows, mel, frames]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if input_features_mask is not None:
            if input_features_mask.ndim != 2:
                raise ValueError("input_features_mask must have shape [windows, frames]")
            if (
                input_features_mask.shape[0] != input_features.shape[0]
                or input_features_mask.shape[1] != input_features.shape[2]
            ):
                raise ValueError("input_features_mask must match input_features windows/frames")
            if input_features_mask.device != input_features.device:
                raise ValueError("input_features_mask and input_features must share device")
            if not bool(torch.all((input_features_mask == 0) | (input_features_mask == 1)).item()):
                raise ValueError("input_features_mask must be binary")
            valid = input_features_mask != 0
            if not bool(torch.all(valid[:, 0]).item()):
                raise ValueError("every feature window must start with at least one valid frame")
            if valid.shape[1] > 1 and bool(torch.any((~valid[:, :-1]) & valid[:, 1:]).item()):
                raise ValueError(
                    "each input_features_mask row must be prefix-contiguous "
                    "(ones followed only by zeros)"
                )
        if input_features.shape[0] > 1 and input_features_mask is None:
            raise ValueError(
                "multi-window audio requires input_features_mask so no window is silently discarded"
            )
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            if attention_mask.device != input_ids.device:
                raise ValueError("attention_mask and input_ids must share device")
            if not bool(torch.all(attention_mask == 1).item()):
                raise NotImplementedError(
                    "the full-prefix baseline only supports a dense B=1 prefix"
                )

        # Encode audio
        if phase_callback is not None:
            phase_callback("audio_encoder", "start")
        audio_embeds = self.encode_audio(input_features, input_features_mask)
        if phase_callback is not None:
            phase_callback("audio_encoder", "end")

        batch_size = input_ids.shape[0]
        if audio_embeds.ndim == 3:
            audio_embeds = audio_embeds[0]
        text_embeds = self.text_decoder.embed_tokens(input_ids)
        audio_positions = torch.where(input_ids[0] == audio_pad_token_id)[0]
        if len(audio_positions) == 0:
            raise ValueError("input_ids must contain the processor-expanded audio placeholders")
        expected_positions = torch.arange(
            int(audio_positions[0].item()),
            int(audio_positions[-1].item()) + 1,
            dtype=audio_positions.dtype,
            device=audio_positions.device,
        )
        if not torch.equal(audio_positions, expected_positions):
            raise ValueError("audio placeholder tokens must form one contiguous span")
        if len(audio_positions) != audio_embeds.shape[0]:
            raise ValueError(
                "audio placeholder count must equal the number of valid "
                f"projected audio tokens: {len(audio_positions)} != {audio_embeds.shape[0]}"
            )
        first_pad_pos = int(audio_positions[0].item())
        last_pad_pos = int(audio_positions[-1].item())
        inputs_embeds = torch.cat(
            [
                text_embeds[0, :first_pad_pos, :][None, :, :],
                audio_embeds[None, :, :],
                text_embeds[0, last_pad_pos + 1 :, :][None, :, :],
            ],
            dim=1,
        )
        generated = input_ids.clone()

        # Track which sequences have finished (hit EOS)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)

        # Handle single or multiple EOS token IDs
        eos_token_ids = self.config.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        eos_token_ids_cp = torch.tensor(eos_token_ids, dtype=torch.int64, device=generated.device)

        # Autoregressive generation
        for step in range(max_new_tokens):
            # Get logits for next token
            if phase_callback is not None:
                phase_callback("full_prefix_decoder_forward", "start")
            logits = self.decode(inputs_embeds=inputs_embeds)
            next_token_logits = logits[:, -1, :] / temperature
            if phase_callback is not None:
                phase_callback("full_prefix_decoder_forward", "end")

            # Top-k sampling
            if phase_callback is not None:
                phase_callback("token_selection", "start")
            if top_k > 0 and top_k < next_token_logits.shape[-1]:
                top_k_indices = torch.argsort(next_token_logits, dim=-1)[:, -top_k:]
                top_k_logits = torch.gather(next_token_logits, dim=-1, index=top_k_indices)

                # Softmax
                top_k_logits_shifted = (
                    top_k_logits - torch.max(top_k_logits, dim=-1, keepdim=True).values
                )
                exp_logits = torch.exp(top_k_logits_shifted)
                probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)

                # Sample
                cumprobs = torch.cumsum(probs, dim=-1)
                samples = torch.rand((batch_size, 1), device=next_token_logits.device)
                next_token_idx = torch.argmax((cumprobs >= samples).to(torch.float32), dim=-1)
                next_token = torch.gather(
                    top_k_indices,
                    dim=-1,
                    index=next_token_idx[:, None],
                )
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if phase_callback is not None:
                phase_callback("token_selection", "end")

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            if phase_callback is not None:
                if step == 0:
                    phase_callback("time_to_first_token", "end")
                else:
                    phase_callback("inter_token_interval", "end")

            # Check for EOS - mark sequences that generated any EOS token
            next_token_flat = next_token.flatten()
            is_eos = torch.any(next_token_flat[:, None] == eos_token_ids_cp[None, :], dim=1)
            finished = finished | is_eos

            # Stop if all sequences have finished
            if torch.all(finished):
                break

            # Update inputs_embeds with new token
            if phase_callback is not None:
                phase_callback("inter_token_interval", "start")
            new_embeds = self.text_decoder.embed_tokens(next_token)
            inputs_embeds = torch.cat([inputs_embeds, new_embeds], dim=1)

        return generated

    def _prepare_cached_generation_inputs(
        self,
        input_features: torch.Tensor,
        input_ids: torch.Tensor | None,
        input_features_mask: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        audio_pad_token_id: int,
        phase_callback: Callable[[str, str], None] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Validate and build the shared dense B=1 cached-generation prefix.

        Tuple-cache and static-cache benchmarks must share this exact input
        contract so their primary difference is cache growth (`torch.cat`
        versus in-place writes), not placeholder handling or mask semantics.
        """
        if input_ids is None:
            raise ValueError("input_ids produced by the pinned GLM-ASR processor are required")
        if input_features.ndim != 3:
            raise ValueError("input_features must have shape [windows, mel, frames]")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise NotImplementedError("cached generation currently supports batch_size=1")
        if input_ids.device != input_features.device:
            raise ValueError("input_ids and input_features must share device")
        if input_features_mask is not None:
            if input_features_mask.ndim != 2:
                raise ValueError("input_features_mask must have shape [windows, frames]")
            if (
                input_features_mask.shape[0] != input_features.shape[0]
                or input_features_mask.shape[1] != input_features.shape[2]
            ):
                raise ValueError("input_features_mask must match input_features windows/frames")
            if input_features_mask.device != input_features.device:
                raise ValueError("input_features_mask and input_features must share device")
            if not bool(torch.all((input_features_mask == 0) | (input_features_mask == 1)).item()):
                raise ValueError("input_features_mask must be binary")
            valid = input_features_mask != 0
            if not bool(torch.all(valid[:, 0]).item()):
                raise ValueError("every feature window must start with at least one valid frame")
            if valid.shape[1] > 1 and bool(torch.any((~valid[:, :-1]) & valid[:, 1:]).item()):
                raise ValueError(
                    "each input_features_mask row must be prefix-contiguous "
                    "(ones followed only by zeros)"
                )
        if input_features.shape[0] > 1 and input_features_mask is None:
            raise ValueError(
                "multi-window audio requires input_features_mask so valid projected "
                "tokens can be flattened without window padding"
            )
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            if attention_mask.device != input_ids.device:
                raise ValueError("attention_mask and input_ids must share device")
            if not bool(torch.all(attention_mask == 1).item()):
                raise NotImplementedError("cached generation currently supports a dense B=1 prefix")

        if phase_callback is not None:
            phase_callback("audio_encoder", "start")
        audio_embeds = self.encode_audio(input_features, input_features_mask)
        if phase_callback is not None:
            phase_callback("audio_encoder", "end")
        if audio_embeds.ndim == 3:
            audio_embeds = audio_embeds[0]

        text_embeds = self.text_decoder.embed_tokens(input_ids)
        audio_positions = torch.where(input_ids[0] == audio_pad_token_id)[0]
        if len(audio_positions) == 0:
            raise ValueError("input_ids must contain the processor-expanded audio placeholders")
        first_pad_pos = int(audio_positions[0].item())
        last_pad_pos = int(audio_positions[-1].item())
        expected_positions = torch.arange(
            first_pad_pos,
            last_pad_pos + 1,
            dtype=audio_positions.dtype,
            device=audio_positions.device,
        )
        if not torch.equal(audio_positions, expected_positions):
            raise ValueError("audio placeholder tokens must form one contiguous span")
        if len(audio_positions) != audio_embeds.shape[0]:
            raise ValueError(
                "audio placeholder count must equal the number of valid "
                f"projected audio tokens: {len(audio_positions)} != "
                f"{audio_embeds.shape[0]}"
            )
        inputs_embeds = torch.cat(
            [
                text_embeds[:, :first_pad_pos, :],
                audio_embeds[None, :, :],
                text_embeds[:, last_pad_pos + 1 :, :],
            ],
            dim=1,
        )
        return inputs_embeds, input_ids, input_ids.shape[1]

    @torch.no_grad()
    def generate_tuple_cache(
        self,
        input_features: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        audio_pad_token_id: int = 59260,
        phase_callback: Callable[[str, str], None] | None = None,
    ) -> torch.Tensor:
        """Generate with the existing tuple cache that grows via ``torch.cat``.

        This path is a controlled bridge between full-prefix recomputation and
        pre-allocated static buffers.  It shares prefix preparation, greedy
        selection, last-hidden LM-head projection, and token-output
        preallocation with :meth:`generate_static_cache`; the principal cache
        variable is therefore per-layer tuple concatenation versus in-place
        writes.  It remains a B=1 benchmark prototype, not a paged cache.
        """
        if phase_callback is not None and max_new_tokens > 0:
            phase_callback("time_to_first_token", "start")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if do_sample and temperature <= 0:
            raise ValueError("temperature must be positive when sampling")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        inputs_embeds, generated_prefix, prefix_tokens = self._prepare_cached_generation_inputs(
            input_features=input_features,
            input_ids=input_ids,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            audio_pad_token_id=audio_pad_token_id,
            phase_callback=phase_callback,
        )
        if max_new_tokens == 0:
            return generated_prefix.clone()

        prefill_len = inputs_embeds.shape[1]
        requested_capacity = prefill_len + max_new_tokens
        if requested_capacity > self.config.text_max_position_embeddings:
            raise ValueError(
                f"requested cache capacity {requested_capacity} exceeds "
                f"text_max_position_embeddings="
                f"{self.config.text_max_position_embeddings}"
            )

        # Call the decoder directly: GlmAsrModel.decode(use_cache=True) would
        # project every prefill position to vocabulary logits and confound the
        # tuple-vs-static cache comparison.
        if phase_callback is not None:
            phase_callback("tuple_prefill_forward", "start")
        hidden_states, past_key_values = self.text_decoder(
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )
        next_token_logits = self.project_logits(hidden_states[:, -1:, :])[:, -1, :]
        if phase_callback is not None:
            phase_callback("tuple_prefill_forward", "end")

        generated = torch.empty(
            (1, prefix_tokens + max_new_tokens),
            dtype=generated_prefix.dtype,
            device=generated_prefix.device,
        )
        generated[:, :prefix_tokens] = generated_prefix
        output_pos = prefix_tokens

        eos_token_ids = self.config.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        eos_token_ids_tensor = torch.tensor(
            eos_token_ids, dtype=torch.int64, device=generated.device
        )

        for step in range(max_new_tokens):
            if phase_callback is not None:
                phase_callback("token_selection", "start")
            if do_sample:
                scaled_logits = next_token_logits / temperature
                if 0 < top_k < scaled_logits.shape[-1]:
                    top_values, top_indices = torch.topk(scaled_logits, k=top_k, dim=-1)
                    probabilities = torch.softmax(top_values, dim=-1)
                    sampled_index = torch.multinomial(probabilities, num_samples=1)
                    next_token = torch.gather(top_indices, dim=-1, index=sampled_index)
                else:
                    probabilities = torch.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if phase_callback is not None:
                phase_callback("token_selection", "end")

            generated[:, output_pos : output_pos + 1] = next_token
            output_pos += 1
            if phase_callback is not None:
                if step == 0:
                    phase_callback("time_to_first_token", "end")
                else:
                    phase_callback("inter_token_interval", "end")

            is_eos = torch.any(next_token[:, :, None] == eos_token_ids_tensor[None, None, :])
            if bool(is_eos.item()) or step + 1 == max_new_tokens:
                break

            if phase_callback is not None:
                phase_callback("inter_token_interval", "start")
            new_embed = self.text_decoder.embed_tokens(next_token)
            if phase_callback is not None:
                phase_callback("tuple_decode_forward", "start")
            hidden_states, past_key_values = self.text_decoder(
                inputs_embeds=new_embed,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits = self.project_logits(hidden_states[:, -1:, :])[:, -1, :]
            if phase_callback is not None:
                phase_callback("tuple_decode_forward", "end")

        return generated[:, :output_pos]

    @torch.no_grad()
    def generate_static_cache(
        self,
        input_features: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        audio_pad_token_id: int = 59260,
        phase_callback: Callable[[str, str], None] | None = None,
    ) -> torch.Tensor:
        """Generate with pre-allocated per-layer KV buffers.

        This is intentionally a separate path from :meth:`generate` so that the
        original full-prefix recomputation remains available as a benchmark
        baseline.  The implementation currently supports one request at a time;
        variable-length batching needs per-request cache positions and masks.

        The prefill produces the first next-token logits.  Every later decoder
        call receives only the newly generated token, writes its K/V at
        ``cache_pos``, and attends to the valid prefix of the static buffers.
        ``do_sample=False`` is the benchmark-safe greedy path and avoids sorting,
        softmax, RNG, and cumulative-probability work.

        Args are equivalent to :meth:`generate`, except that ``top_k=0`` means
        sampling from the full vocabulary when ``do_sample=True``.  The incoming
        attention mask is accepted for processor API parity; after the audio-pad
        span is replaced, this single-request implementation uses a dense prefix.

        Status: implemented but not yet accepted as a performance result.  It
        must pass token/logit parity and the frozen target-GPU benchmark protocol.
        """
        if phase_callback is not None and max_new_tokens > 0:
            phase_callback("time_to_first_token", "start")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if do_sample and temperature <= 0:
            raise ValueError("temperature must be positive when sampling")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        inputs_embeds, generated_prefix, prefix_tokens = self._prepare_cached_generation_inputs(
            input_features=input_features,
            input_ids=input_ids,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            audio_pad_token_id=audio_pad_token_id,
            phase_callback=phase_callback,
        )

        if max_new_tokens == 0:
            return generated_prefix.clone()

        prefill_len = inputs_embeds.shape[1]
        cache_capacity = prefill_len + max_new_tokens
        if cache_capacity > self.config.text_max_position_embeddings:
            raise ValueError(
                f"requested cache capacity {cache_capacity} exceeds "
                f"text_max_position_embeddings="
                f"{self.config.text_max_position_embeddings}"
            )

        if phase_callback is not None:
            phase_callback("static_cache_allocation", "start")
        kv_buffers = self.text_decoder.allocate_kv_buffers(
            batch_size=1,
            max_seq_len=cache_capacity,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        if phase_callback is not None:
            phase_callback("static_cache_allocation", "end")

        # Avoid materializing [batch, prefill_len, vocab] logits: only the final
        # prefill state predicts the first generated token.
        if phase_callback is not None:
            phase_callback("static_prefill_forward", "start")
        hidden_states, cache_pos = self.text_decoder.forward_with_kv_buffers(
            inputs_embeds, kv_buffers, cache_pos=0
        )
        next_token_logits = self.project_logits(hidden_states[:, -1:, :])[:, -1, :]
        if phase_callback is not None:
            phase_callback("static_prefill_forward", "end")

        # Pre-allocate token output to avoid one torch.cat allocation per step.
        generated = torch.empty(
            (1, prefix_tokens + max_new_tokens),
            dtype=generated_prefix.dtype,
            device=generated_prefix.device,
        )
        generated[:, :prefix_tokens] = generated_prefix
        output_pos = prefix_tokens

        eos_token_ids = self.config.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        eos_token_ids_tensor = torch.tensor(
            eos_token_ids, dtype=torch.int64, device=generated.device
        )

        for step in range(max_new_tokens):
            if phase_callback is not None:
                phase_callback("token_selection", "start")
            if do_sample:
                scaled_logits = next_token_logits / temperature
                if 0 < top_k < scaled_logits.shape[-1]:
                    top_values, top_indices = torch.topk(scaled_logits, k=top_k, dim=-1)
                    probabilities = torch.softmax(top_values, dim=-1)
                    sampled_index = torch.multinomial(probabilities, num_samples=1)
                    next_token = torch.gather(top_indices, dim=-1, index=sampled_index)
                else:
                    probabilities = torch.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if phase_callback is not None:
                phase_callback("token_selection", "end")

            generated[:, output_pos : output_pos + 1] = next_token
            output_pos += 1
            if phase_callback is not None:
                if step == 0:
                    phase_callback("time_to_first_token", "end")
                else:
                    phase_callback("inter_token_interval", "end")

            is_eos = torch.any(next_token[:, :, None] == eos_token_ids_tensor[None, None, :])
            if bool(is_eos.item()) or step + 1 == max_new_tokens:
                break

            # Decode exactly one new token.  Its hidden state predicts the token
            # for the next loop iteration.
            if phase_callback is not None:
                phase_callback("inter_token_interval", "start")
            new_embed = self.text_decoder.embed_tokens(next_token)
            if phase_callback is not None:
                phase_callback("static_decode_forward", "start")
            hidden_states, cache_pos = self.text_decoder.forward_with_kv_buffers(
                new_embed, kv_buffers, cache_pos=cache_pos
            )
            next_token_logits = self.project_logits(hidden_states[:, -1:, :])[:, -1, :]
            if phase_callback is not None:
                phase_callback("static_decode_forward", "end")

        return generated[:, :output_pos]
