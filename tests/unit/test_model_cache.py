from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from faster_glm_asr.kernels import configure_kernels
from faster_glm_asr.kernels.layers import Linear
from faster_glm_asr.modeling import (
    GlmAsrConfig,
    GlmAsrModel,
    create_config_from_hf,
    load_model_from_hf,
    load_weights_from_hf_model,
)
from faster_glm_asr.modeling.model import TextDecoder


def small_config(**overrides: object) -> GlmAsrConfig:
    values: dict[str, object] = {
        "audio_hidden_size": 16,
        "audio_num_heads": 2,
        "audio_num_layers": 1,
        "audio_intermediate_size": 32,
        "audio_max_position_embeddings": 64,
        "text_hidden_size": 16,
        "text_num_heads": 4,
        "text_num_kv_heads": 2,
        "text_num_layers": 2,
        "text_intermediate_size": 32,
        "text_vocab_size": 257,
        "text_max_position_embeddings": 64,
        "projector_hidden_size": 32,
        "projector_pool_factor": 4,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": [256],
    }
    values.update(overrides)
    return GlmAsrConfig(**values)


def randomize_linear(layer: Linear, std: float = 0.02) -> None:
    layer.weight.normal_(mean=0.0, std=std)
    if layer.bias_param is not None:
        layer.bias_param.normal_(mean=0.0, std=std)
    layer._weight_t_padded = None


def randomize_decoder(decoder: TextDecoder, std: float = 0.02) -> None:
    decoder.embed_tokens.weight.normal_(mean=0.0, std=std)
    for layer in decoder.layers:
        for projection in (
            layer.q_proj,
            layer.k_proj,
            layer.v_proj,
            layer.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ):
            randomize_linear(projection, std)


class ModelCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        configure_kernels(
            linear_backend="cublas",
            fused_decoder_mlp=False,
            fused_encoder_mlp=False,
        )

    def test_public_model_api_and_tied_lm_head(self) -> None:
        model = GlmAsrModel(small_config())
        for method_name in ("generate", "generate_tuple_cache", "generate_static_cache"):
            self.assertTrue(callable(getattr(model, method_name)))
        model.tie_lm_head_weights()
        self.assertIs(model.lm_head.weight, model.text_decoder.embed_tokens.weight)

        features = torch.randn(1, 128, 32)
        for method_name in ("generate", "generate_tuple_cache", "generate_static_cache"):
            with (
                self.subTest(method=method_name),
                self.assertRaisesRegex(ValueError, "processor are required"),
            ):
                getattr(model, method_name)(features, input_ids=None, max_new_tokens=0)

    def test_config_rejects_incompatible_shapes_and_token_ids(self) -> None:
        invalid = (
            {"text_hidden_size": 15},
            {"text_num_heads": 3, "text_num_kv_heads": 2},
            {"audio_hidden_size": 15},
            {"eos_token_id": [257]},
            {"projector_pool_factor": 0},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                small_config(**overrides)

    def test_hf_config_mapping_and_loader_fail_fast(self) -> None:
        audio = SimpleNamespace(
            model_type="glmasr_encoder",
            hidden_act="gelu",
            attention_dropout=0.0,
            hidden_size=16,
            head_dim=8,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=1,
            intermediate_size=32,
            max_position_embeddings=64,
            num_mel_bins=128,
            partial_rotary_factor=0.5,
            rope_parameters={
                "partial_rotary_factor": 0.5,
                "rope_theta": 10000.0,
                "rope_type": "default",
            },
        )
        text = SimpleNamespace(
            model_type="llama",
            hidden_act="silu",
            attention_dropout=0.0,
            attention_bias=False,
            mlp_bias=False,
            hidden_size=16,
            head_dim=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            intermediate_size=32,
            vocab_size=257,
            max_position_embeddings=64,
            rope_parameters={"rope_theta": 20_000.0, "rope_type": "default"},
            rms_norm_eps=1e-6,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=[256],
        )
        root_config = SimpleNamespace(
            model_type="glmasr",
            projector_hidden_act="gelu",
            audio_token_id=59260,
            audio_config=audio,
            text_config=text,
        )
        mapped = create_config_from_hf(
            root_config,
            projector_hidden_size=32,
            projector_pool_factor=2,
        )
        self.assertEqual(mapped.text_num_kv_heads, 2)
        self.assertEqual(mapped.text_rope_base, 20_000.0)
        text.hidden_act = "relu"
        with self.assertRaisesRegex(ValueError, "semantic config"):
            create_config_from_hf(
                root_config,
                projector_hidden_size=32,
                projector_pool_factor=2,
            )
        text.hidden_act = "silu"
        audio.rope_parameters["partial_rotary_factor"] = 0.25
        with self.assertRaisesRegex(ValueError, "semantic config"):
            create_config_from_hf(
                root_config,
                projector_hidden_size=32,
                projector_pool_factor=2,
            )
        audio.rope_parameters["partial_rotary_factor"] = 0.5
        with self.assertRaisesRegex(ValueError, "model_name"):
            load_model_from_hf(model_name="")
        with self.assertRaisesRegex(ValueError, "revision"):
            load_model_from_hf(revision="")
        with self.assertRaisesRegex(ValueError, "immutable commit SHA"):
            load_model_from_hf(revision="main")
        with self.assertRaisesRegex(ValueError, "immutable commit SHA"):
            load_model_from_hf(revision="A" * 40)

    def test_weight_mapping_covers_audio_projector_decoder_and_tied_head(self) -> None:
        model = GlmAsrModel(small_config(text_num_layers=1))
        logical_state: dict[str, torch.Tensor] = {}

        for name, convolution in (
            ("audio_tower.conv1", model.audio_encoder.conv1),
            ("audio_tower.conv2", model.audio_encoder.conv2),
        ):
            logical_state[f"{name}.weight"] = torch.randn(
                convolution.out_channels,
                convolution.in_channels,
                convolution.kernel_size,
            )
            logical_state[f"{name}.bias"] = torch.randn(convolution.out_channels)

        audio_layer = model.audio_encoder.layers[0]
        audio_prefix = "audio_tower.layers.0"
        logical_state[f"{audio_prefix}.input_layernorm.weight"] = torch.randn(16)
        logical_state[f"{audio_prefix}.input_layernorm.bias"] = torch.randn(16)
        logical_state[f"{audio_prefix}.post_attention_layernorm.weight"] = torch.randn(16)
        logical_state[f"{audio_prefix}.post_attention_layernorm.bias"] = torch.randn(16)
        for hf_name, projection in (
            ("self_attn.q_proj", audio_layer.q_proj),
            ("self_attn.k_proj", audio_layer.k_proj),
            ("self_attn.v_proj", audio_layer.v_proj),
            ("self_attn.o_proj", audio_layer.out_proj),
            ("mlp.fc1", audio_layer.fc1),
            ("mlp.fc2", audio_layer.fc2),
        ):
            logical_state[f"{audio_prefix}.{hf_name}.weight"] = torch.randn_like(projection.weight)
            if hf_name != "self_attn.k_proj":
                logical_state[f"{audio_prefix}.{hf_name}.bias"] = torch.randn(
                    projection.out_features
                )
        logical_state["audio_tower.norm.weight"] = torch.randn(16)
        logical_state["audio_tower.norm.bias"] = torch.randn(16)

        for hf_name, projection in (
            ("linear_1", model.multi_modal_projector.linear_1),
            ("linear_2", model.multi_modal_projector.linear_2),
        ):
            logical_state[f"multi_modal_projector.{hf_name}.weight"] = torch.randn_like(
                projection.weight
            )
            logical_state[f"multi_modal_projector.{hf_name}.bias"] = torch.randn(
                projection.out_features
            )

        embedding = torch.randn_like(model.text_decoder.embed_tokens.weight)
        logical_state["language_model.model.embed_tokens.weight"] = embedding
        text_layer = model.text_decoder.layers[0]
        text_prefix = "language_model.model.layers.0"
        logical_state[f"{text_prefix}.input_layernorm.weight"] = torch.randn(16)
        logical_state[f"{text_prefix}.post_attention_layernorm.weight"] = torch.randn(16)
        for hf_name, projection in (
            ("self_attn.q_proj", text_layer.q_proj),
            ("self_attn.k_proj", text_layer.k_proj),
            ("self_attn.v_proj", text_layer.v_proj),
            ("self_attn.o_proj", text_layer.o_proj),
            ("mlp.gate_proj", text_layer.mlp.gate_proj),
            ("mlp.up_proj", text_layer.mlp.up_proj),
            ("mlp.down_proj", text_layer.mlp.down_proj),
        ):
            logical_state[f"{text_prefix}.{hf_name}.weight"] = torch.randn_like(projection.weight)
        logical_state["language_model.model.norm.weight"] = torch.randn(16)
        logical_state["language_model.lm_head.weight"] = embedding.clone()

        state: dict[str, torch.Tensor] = {}
        for key, value in logical_state.items():
            if key.startswith("audio_tower.") or key.startswith("multi_modal_projector."):
                transformers_key = f"model.{key}"
            elif key.startswith("language_model.model."):
                transformers_key = "model.language_model." + key.removeprefix(
                    "language_model.model."
                )
            elif key == "language_model.lm_head.weight":
                transformers_key = "lm_head.weight"
            else:
                self.fail(f"unhandled test state key: {key}")
            state[transformers_key] = value

        load_weights_from_hf_model(model, SimpleNamespace(state_dict=lambda: state))
        self.assertIs(model.lm_head.weight, model.text_decoder.embed_tokens.weight)
        torch.testing.assert_close(
            model.audio_encoder.conv1.weight,
            logical_state["audio_tower.conv1.weight"].reshape(
                model.audio_encoder.conv1.out_channels, -1
            ),
        )
        torch.testing.assert_close(model.text_decoder.embed_tokens.weight, embedding)

        missing = dict(state)
        del missing["model.audio_tower.conv1.weight"]
        with self.assertRaisesRegex(ValueError, "missing_count=1"):
            load_weights_from_hf_model(model, SimpleNamespace(state_dict=lambda: missing))

        unexpected = dict(state)
        unexpected["model.future_private_weight"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "unexpected_count=1"):
            load_weights_from_hf_model(model, SimpleNamespace(state_dict=lambda: unexpected))

        wrong_shape = dict(state)
        wrong_shape["model.audio_tower.conv1.weight"] = torch.zeros(1, 1, 1)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            load_weights_from_hf_model(model, SimpleNamespace(state_dict=lambda: wrong_shape))

    def test_transformers_514_tiny_model_state_dict_loads_without_legacy_keys(self) -> None:
        from transformers import (
            GlmAsrConfig as TransformersGlmAsrConfig,
        )
        from transformers import (
            GlmAsrForConditionalGeneration,
        )

        hf_config = TransformersGlmAsrConfig(
            audio_config={
                "hidden_size": 16,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "max_position_embeddings": 64,
                "num_mel_bins": 128,
            },
            text_config={
                "hidden_size": 16,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "max_position_embeddings": 64,
                "vocab_size": 257,
                "rms_norm_eps": 1e-5,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": [256],
            },
        )
        hf_model = GlmAsrForConditionalGeneration(hf_config)
        self.assertEqual(len(hf_model.state_dict()), 37)
        self.assertNotIn(
            "model.audio_tower.layers.0.self_attn.k_proj.bias",
            hf_model.state_dict(),
        )
        custom_config = create_config_from_hf(
            hf_config,
            projector_hidden_size=32,
            projector_pool_factor=2,
        )
        custom_model = GlmAsrModel(custom_config)
        load_weights_from_hf_model(custom_model, hf_model)
        torch.testing.assert_close(
            custom_model.audio_encoder.layers[0].q_proj.weight,
            hf_model.state_dict()["model.audio_tower.layers.0.self_attn.q_proj.weight"],
        )
        self.assertFalse(custom_model.audio_encoder.layers[0].k_proj.has_bias)
        self.assertIs(
            custom_model.lm_head.weight,
            custom_model.text_decoder.embed_tokens.weight,
        )

    @torch.no_grad()
    def test_tuple_and_static_cache_match_full_prefix(self) -> None:
        config = small_config()
        decoder = TextDecoder(config)
        randomize_decoder(decoder)
        token_ids = torch.randint(0, config.text_vocab_size, (1, 11))
        embeddings = decoder.embed_tokens(token_ids)
        prefill_len = 6

        full_prefill = decoder(inputs_embeds=embeddings[:, :prefill_len, :])
        tuple_hidden, tuple_cache = decoder(
            inputs_embeds=embeddings[:, :prefill_len, :], use_cache=True
        )
        torch.testing.assert_close(tuple_hidden, full_prefill, rtol=0.0, atol=1e-6)

        static_buffers = decoder.allocate_kv_buffers(
            batch_size=1,
            max_seq_len=embeddings.shape[1],
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        static_hidden, cache_pos = decoder.forward_with_kv_buffers(
            embeddings[:, :prefill_len, :], static_buffers, cache_pos=0
        )
        torch.testing.assert_close(static_hidden, full_prefill, rtol=0.0, atol=1e-6)

        for end in range(prefill_len + 1, embeddings.shape[1] + 1):
            newest = embeddings[:, end - 1 : end, :]
            full_hidden = decoder(inputs_embeds=embeddings[:, :end, :])[:, -1:, :]
            tuple_hidden, tuple_cache = decoder(
                inputs_embeds=newest,
                past_key_values=tuple_cache,
                use_cache=True,
            )
            static_hidden, cache_pos = decoder.forward_with_kv_buffers(
                newest, static_buffers, cache_pos=cache_pos
            )
            torch.testing.assert_close(tuple_hidden, full_hidden, rtol=0.0, atol=2e-6)
            torch.testing.assert_close(static_hidden, full_hidden, rtol=0.0, atol=2e-6)
            for (tuple_key, tuple_value), (static_key, static_value) in zip(
                tuple_cache, static_buffers, strict=True
            ):
                torch.testing.assert_close(
                    tuple_key,
                    static_key[:, :, :cache_pos, :],
                    rtol=0.0,
                    atol=0.0,
                )
                torch.testing.assert_close(
                    tuple_value,
                    static_value[:, :, :cache_pos, :],
                    rtol=0.0,
                    atol=0.0,
                )

        with self.assertRaisesRegex(ValueError, "capacity"):
            decoder.forward_with_kv_buffers(
                embeddings[:, :1, :], static_buffers, cache_pos=embeddings.shape[1]
            )

        aliased_buffers = [static_buffers[0]] * decoder.num_layers
        with self.assertRaisesRegex(ValueError, "alias"):
            decoder.forward_with_kv_buffers(embeddings[:, :1, :], aliased_buffers, cache_pos=0)

        corrupted_buffers = list(static_buffers)
        corrupted_buffers[1] = (
            corrupted_buffers[1][0].to(torch.float64),
            corrupted_buffers[1][1].to(torch.float64),
        )
        static_buffers[0][0].fill_(123.0)
        static_buffers[0][1].fill_(456.0)
        with self.assertRaisesRegex(ValueError, "dtype"):
            decoder.forward_with_kv_buffers(embeddings[:, :1, :], corrupted_buffers, cache_pos=0)
        torch.testing.assert_close(
            static_buffers[0][0], torch.full_like(static_buffers[0][0], 123.0)
        )
        torch.testing.assert_close(
            static_buffers[0][1], torch.full_like(static_buffers[0][1], 456.0)
        )

    @torch.no_grad()
    def test_decoder_supports_batched_position_ids_and_validates_tuple_cache(self) -> None:
        config = small_config(text_num_layers=1)
        decoder = TextDecoder(config)
        randomize_decoder(decoder)
        token_ids = torch.randint(0, config.text_vocab_size, (2, 5))
        position_ids = torch.tensor([[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]])
        hidden = decoder(input_ids=token_ids, position_ids=position_ids)
        self.assertEqual(tuple(hidden.shape), (2, 5, config.text_hidden_size))

        _, cache = decoder(input_ids=token_ids[:, :2], use_cache=True)
        bad_cache = [(cache[0][0].to(torch.float64), cache[0][1])]
        with self.assertRaisesRegex(ValueError, "dtype"):
            decoder(input_ids=token_ids[:, 2:3], past_key_values=bad_cache, use_cache=True)

        multi_layer_decoder = TextDecoder(small_config(text_num_layers=2))
        _, multi_layer_cache = multi_layer_decoder(input_ids=token_ids[:1, :2], use_cache=True)
        with self.assertRaisesRegex(ValueError, "alias"):
            multi_layer_decoder(
                input_ids=token_ids[:1, 2:3],
                past_key_values=[multi_layer_cache[0], multi_layer_cache[0]],
                use_cache=True,
            )

    @torch.no_grad()
    def test_multimodal_generation_paths_produce_identical_greedy_tokens(self) -> None:
        model = GlmAsrModel(small_config(text_num_layers=1))
        randomize_decoder(model.text_decoder, std=0.01)
        for encoder_layer in model.audio_encoder.layers:
            for projection in (
                encoder_layer.q_proj,
                encoder_layer.k_proj,
                encoder_layer.v_proj,
                encoder_layer.out_proj,
                encoder_layer.fc1,
                encoder_layer.fc2,
            ):
                randomize_linear(projection, std=0.01)
        randomize_linear(model.multi_modal_projector.linear_1, std=0.01)
        randomize_linear(model.multi_modal_projector.linear_2, std=0.01)
        model.tie_lm_head_weights()

        input_features = torch.randn(2, 128, 32)
        input_features_mask = torch.cat(
            [torch.ones(1, 32), torch.cat([torch.ones(1, 24), torch.zeros(1, 8)], dim=1)],
            dim=0,
        )
        audio_tokens = model.encode_audio(input_features, input_features_mask)
        self.assertEqual(audio_tokens.shape[0], 7)
        audio_pad_token_id = 250
        input_ids = torch.tensor(
            [[1] + [audio_pad_token_id] * audio_tokens.shape[0] + [2]],
            dtype=torch.int64,
        )
        common = {
            "input_ids": input_ids,
            "input_features_mask": input_features_mask,
            "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": 3,
            "audio_pad_token_id": audio_pad_token_id,
        }
        full_prefix = model.generate(input_features, top_k=0, **common)
        tuple_cache = model.generate_tuple_cache(input_features, do_sample=False, **common)
        static_cache = model.generate_static_cache(input_features, do_sample=False, **common)
        self.assertTrue(torch.equal(full_prefix, tuple_cache))
        self.assertTrue(torch.equal(full_prefix, static_cache))


if __name__ == "__main__":
    unittest.main()
