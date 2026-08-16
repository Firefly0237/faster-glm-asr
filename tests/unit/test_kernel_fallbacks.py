from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from faster_glm_asr.kernels import TRITON_AVAILABLE, configure_kernels
from faster_glm_asr.kernels.attention import (
    MultiHeadAttention,
    scaled_dot_product_attention,
)
from faster_glm_asr.kernels.conv import Conv1d
from faster_glm_asr.kernels.layers import Embedding, LayerNorm, Linear, RMSNorm
from faster_glm_asr.kernels.rope import RotaryEmbedding, apply_rotary_pos_emb


class KernelFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        configure_kernels(
            linear_backend="cublas",
            fused_decoder_mlp=False,
            fused_encoder_mlp=False,
        )

    def test_configuration_rejects_unknown_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "linear_backend"):
            configure_kernels(linear_backend="unknown")
        if not TRITON_AVAILABLE:
            with self.assertRaisesRegex(RuntimeError, "triton package"):
                configure_kernels(linear_backend="triton")
        with self.assertRaises(ValueError):
            MultiHeadAttention(hidden_size=30, num_heads=4)
        with self.assertRaises(ValueError):
            Conv1d(3, 4, kernel_size=0)
        with self.assertRaises(ValueError):
            RotaryEmbedding(dim=2, partial_rotary_factor=0.5)

    def test_rmsnorm_and_layernorm_match_torch_definitions(self) -> None:
        x = torch.randn(3, 5, 8)
        rms = RMSNorm(8, eps=1e-5)
        rms.weight.copy_(torch.randn(8))
        expected_rms = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-5)
        expected_rms *= rms.weight
        torch.testing.assert_close(rms(x), expected_rms, rtol=1e-6, atol=1e-6)

        layer_norm = LayerNorm(8, eps=1e-5)
        layer_norm.weight.copy_(torch.randn(8))
        layer_norm.bias.copy_(torch.randn(8))
        expected_layer = F.layer_norm(
            x,
            (8,),
            weight=layer_norm.weight,
            bias=layer_norm.bias,
            eps=1e-5,
        )
        torch.testing.assert_close(layer_norm(x), expected_layer, rtol=1e-6, atol=1e-6)

    def test_linear_and_conv1d_torch_fallbacks(self) -> None:
        linear = Linear(7, 5, bias=True)
        linear.weight.copy_(torch.randn_like(linear.weight))
        linear.bias_param.copy_(torch.randn_like(linear.bias_param))
        x_linear = torch.randn(2, 3, 7)
        torch.testing.assert_close(
            linear(x_linear),
            F.linear(x_linear, linear.weight, linear.bias_param),
            rtol=1e-6,
            atol=1e-6,
        )

        convolution = Conv1d(3, 5, kernel_size=3, stride=2, padding=1)
        x_conv = torch.randn(2, 3, 17)
        expected_conv = F.conv1d(
            x_conv,
            convolution.weight.reshape(5, 3, 3),
            convolution.bias,
            stride=2,
            padding=1,
        )
        torch.testing.assert_close(convolution(x_conv), expected_conv, rtol=1e-5, atol=1e-6)

        embedding = Embedding(7, 3)
        with self.assertRaisesRegex(TypeError, "int32 or int64"):
            embedding(torch.tensor([1.0]))
        with self.assertRaisesRegex(RuntimeError, "embedding index"):
            embedding(torch.tensor([-1, 2], dtype=torch.int64))

    def test_gqa_attention_matches_explicit_kv_repeat(self) -> None:
        q = torch.randn(2, 4, 7, 8)
        k = torch.randn(2, 2, 7, 8)
        v = torch.randn(2, 2, 7, 8)
        attention = MultiHeadAttention(
            hidden_size=32,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
        )
        actual = attention(q, k, v, is_causal=True)
        repeated_k = k.repeat_interleave(2, dim=1)
        repeated_v = v.repeat_interleave(2, dim=1)
        expected = F.scaled_dot_product_attention(
            q,
            repeated_k,
            repeated_v,
            is_causal=True,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        direct = scaled_dot_product_attention(q, repeated_k, repeated_v, is_causal=True)
        torch.testing.assert_close(direct, expected, rtol=1e-5, atol=1e-6)

    def test_attention_broadcast_mask_matches_torch_definition(self) -> None:
        q = torch.randn(2, 4, 3, 8)
        k = torch.randn(2, 4, 5, 8)
        v = torch.randn(2, 4, 5, 8)
        allowed = torch.tensor(
            [
                [[[True, True, True, False, False]]],
                [[[True, True, True, True, False]]],
            ]
        ).expand(2, 1, 3, 5)

        actual = scaled_dot_product_attention(q, k, v, attention_mask=allowed)
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        additive = torch.where(
            allowed,
            torch.zeros((), dtype=torch.float32),
            torch.full((), -1e9, dtype=torch.float32),
        )
        actual_additive = scaled_dot_product_attention(q, k, v, attention_mask=additive)
        torch.testing.assert_close(actual_additive, expected, rtol=1e-5, atol=1e-6)

        with self.assertRaisesRegex(ValueError, "broadcastable"):
            scaled_dot_product_attention(q, k, v, attention_mask=torch.zeros(2, 3, 4, 5))

        fully_blocked = torch.zeros((2, 1, 3, 5), dtype=torch.bool)
        blocked_output = scaled_dot_product_attention(
            q,
            k,
            v,
            attention_mask=fully_blocked,
        )
        torch.testing.assert_close(blocked_output, torch.zeros_like(blocked_output))

    def test_partial_rope_preserves_tail_and_vector_norm(self) -> None:
        q = torch.randn(2, 4, 6, 8)
        k = torch.randn(2, 2, 6, 8)
        rope = RotaryEmbedding(
            dim=8,
            max_position_embeddings=16,
            partial_rotary_factor=0.5,
        )
        cos, sin = rope(q)
        q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin, rotary_dim=rope.rotary_dim)
        torch.testing.assert_close(q_rotated[..., 4:], q[..., 4:])
        torch.testing.assert_close(k_rotated[..., 4:], k[..., 4:])
        torch.testing.assert_close(
            q_rotated[..., :4].square().sum(dim=-1),
            q[..., :4].square().sum(dim=-1),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_rope_supports_distinct_position_ids_per_batch(self) -> None:
        q = torch.randn(2, 4, 3, 8)
        k = torch.randn(2, 2, 3, 8)
        rope = RotaryEmbedding(dim=8, max_position_embeddings=16)
        position_ids = torch.tensor([[0, 1, 2], [4, 5, 6]], dtype=torch.int64)
        cos, sin = rope(q, position_ids)
        self.assertEqual(tuple(cos.shape), (2, 3, 8))
        q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin)
        self.assertEqual(q_rotated.shape, q.shape)
        self.assertEqual(k_rotated.shape, k.shape)
        self.assertFalse(torch.equal(q_rotated[0], q_rotated[1]))

        with self.assertRaisesRegex(TypeError, "int32 or int64"):
            rope(q, position_ids.to(torch.float32))
        with self.assertRaisesRegex(ValueError, "batch"):
            rope(q, torch.zeros((3, 3), dtype=torch.int64))

    def test_rope_preserves_tail_when_q_and_k_head_dims_differ(self) -> None:
        q = torch.randn(2, 3, 4, 8)
        k = torch.randn(2, 1, 4, 12)
        rope = RotaryEmbedding(dim=8, max_position_embeddings=16)
        position_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        cos, sin = rope(q, position_ids)
        q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin, rotary_dim=8)

        self.assertEqual(q_rotated.shape, q.shape)
        self.assertEqual(k_rotated.shape, k.shape)
        torch.testing.assert_close(k_rotated[..., 8:], k[..., 8:])

        half = 4
        expected_k_first = (
            k[..., :half] * cos[:, None, :, :half] - k[..., half:8] * sin[:, None, :, :half]
        )
        expected_k_second = (
            k[..., half:8] * cos[:, None, :, :half] + k[..., :half] * sin[:, None, :, :half]
        )
        torch.testing.assert_close(k_rotated[..., :half], expected_k_first)
        torch.testing.assert_close(k_rotated[..., half:8], expected_k_second)


if __name__ == "__main__":
    unittest.main()
