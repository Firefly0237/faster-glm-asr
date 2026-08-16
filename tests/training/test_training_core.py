from __future__ import annotations

import copy
import math
import random
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    import torch.nn.functional as F
    from experiments.distributed_training.data import (
        RandomWindowBatcher,
        load_byte_stream,
        synthetic_stream,
    )
    from experiments.distributed_training.model import (
        CausalLM,
        GroupedQueryAttention,
        ModelConfig,
        RMSNorm,
        RotaryEmbedding,
    )
    from experiments.distributed_training.train import (
        TrainingConfig,
        _data_fingerprint,
        _load_checkpoint,
        _lr_for_step,
        _optimizer,
        _save_checkpoint,
        _streams,
        _validate_stream_token_domain,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DecoderTrainingCoreTest(unittest.TestCase):
    def test_configs_reject_nonfinite_values(self) -> None:
        for field, value in (
            ("learning_rate", float("nan")),
            ("adam_eps", float("inf")),
            ("grad_clip", float("nan")),
            ("weight_decay", float("inf")),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                TrainingConfig(**{field: value})
        for field, value in (
            ("rope_theta", float("nan")),
            ("rms_norm_eps", float("inf")),
            ("dropout", float("nan")),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                ModelConfig(**{field: value})

    def test_rms_norm_matches_definition(self) -> None:
        module = RMSNorm(4, eps=1e-5)
        with torch.no_grad():
            module.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        value = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
        expected = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-5)
        expected *= module.weight
        torch.testing.assert_close(module(value), expected)

    def test_partial_rope_preserves_norm_and_passthrough(self) -> None:
        rope = RotaryEmbedding(rotary_dim=4, max_seq_len=8, theta=10_000.0)
        query = torch.randn(2, 4, 6, 8)
        key = torch.randn(2, 2, 6, 8)
        rotated_query, rotated_key = rope(query, key)
        torch.testing.assert_close(rotated_query[..., 4:], query[..., 4:])
        torch.testing.assert_close(rotated_key[..., 4:], key[..., 4:])
        torch.testing.assert_close(
            rotated_query[..., :4].square().sum(-1),
            query[..., :4].square().sum(-1),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_rope_matches_known_half_split_layout(self) -> None:
        rope = RotaryEmbedding(rotary_dim=4, max_seq_len=4, theta=10_000.0)
        query = torch.tensor([[[[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]]]])
        rotated_query, rotated_key = rope(query, query.clone())
        angles = torch.tensor([1.0, 0.01, 1.0, 0.01])
        half = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        expected = query[0, 0, 1] * angles.cos() + half * angles.sin()
        torch.testing.assert_close(rotated_query[0, 0, 0], query[0, 0, 0])
        torch.testing.assert_close(rotated_query[0, 0, 1], expected)
        torch.testing.assert_close(rotated_key, rotated_query)

    def test_gqa_cpu_matches_explicit_kv_repeat_reference(self) -> None:
        config = ModelConfig(
            vocab_size=128,
            dim=32,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            hidden_dim=64,
            max_seq_len=16,
            rope_fraction=0.5,
        )
        attention = GroupedQueryAttention(config).eval()
        value = torch.randn(2, 7, config.dim)
        actual = attention(value)
        batch, sequence, _ = value.shape
        query = (
            attention.q_proj(value)
            .view(batch, sequence, config.n_heads, config.head_dim)
            .transpose(1, 2)
        )
        key = (
            attention.k_proj(value)
            .view(batch, sequence, config.n_kv_heads, config.head_dim)
            .transpose(1, 2)
        )
        projected_value = (
            attention.v_proj(value)
            .view(batch, sequence, config.n_kv_heads, config.head_dim)
            .transpose(1, 2)
        )
        query, key = attention.rope(query, key)
        groups = config.n_heads // config.n_kv_heads
        expected = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(groups, dim=1),
            projected_value.repeat_interleave(groups, dim=1),
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=False,
        )
        expected = attention.o_proj(
            expected.transpose(1, 2).contiguous().view(batch, sequence, config.dim)
        )
        torch.testing.assert_close(actual, expected)

    def test_model_forward_backward_and_tied_head(self) -> None:
        config = self._model_config()
        model = CausalLM(config)
        inputs = torch.randint(0, config.vocab_size, (2, 12))
        labels = torch.randint(0, config.vocab_size, (2, 12))
        logits, loss = model(inputs, labels)
        self.assertEqual(tuple(logits.shape), (2, 12, config.vocab_size))
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertEqual(model.parameter_count()["stored_lm_head_extra"], 0)

    def test_causal_prefix_is_invariant_to_future_tokens(self) -> None:
        config = self._model_config()
        model = CausalLM(config).eval()
        first = torch.randint(0, config.vocab_size, (1, 10))
        second = first.clone()
        second[:, 6:] = (second[:, 6:] + 17) % config.vocab_size
        with torch.no_grad():
            first_logits, _ = model(first)
            second_logits, _ = model(second)
        torch.testing.assert_close(first_logits[:, :6], second_logits[:, :6])

    def test_zero_logits_cross_entropy_is_log_vocab(self) -> None:
        config = ModelConfig(
            vocab_size=128,
            dim=16,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            hidden_dim=32,
            max_seq_len=8,
        )
        model = CausalLM(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        inputs = torch.randint(0, config.vocab_size, (2, 8))
        labels = torch.randint(0, config.vocab_size, (2, 8))
        logits, loss = model(inputs, labels)
        torch.testing.assert_close(logits, torch.zeros_like(logits))
        self.assertIsNotNone(loss)
        self.assertAlmostEqual(float(loss.item()), math.log(config.vocab_size), places=5)

    def test_learning_rate_hits_warmup_and_cosine_endpoints(self) -> None:
        config = TrainingConfig(
            steps=5,
            warmup_steps=2,
            learning_rate=0.01,
            min_lr_ratio=0.1,
        )
        self.assertAlmostEqual(_lr_for_step(0, config), 0.005)
        self.assertAlmostEqual(_lr_for_step(1, config), 0.01)
        self.assertAlmostEqual(_lr_for_step(2, config), 0.01)
        self.assertAlmostEqual(_lr_for_step(4, config), 0.001)

    def test_byte_loader_and_data_fingerprint_change_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.txt"
            validation = root / "validation.txt"
            train.write_bytes(b"abc" * 100)
            validation.write_bytes(b"xyz" * 100)
            loaded = load_byte_stream((train, validation))
            self.assertEqual(loaded.dtype, torch.long)
            self.assertEqual(loaded.numel(), 600)
            config = TrainingConfig(sequence_length=8)
            _, _, first = _streams(
                {
                    "mode": "byte_files",
                    "train_files": [str(train)],
                    "validation_files": [str(validation)],
                },
                config,
            )
            train.write_bytes(b"abd" * 100)
            _, _, second = _streams(
                {
                    "mode": "byte_files",
                    "train_files": [str(train)],
                    "validation_files": [str(validation)],
                },
                config,
            )
            self.assertNotEqual(_data_fingerprint(first), _data_fingerprint(second))
            contract = _validate_stream_token_domain(
                torch.tensor([0, 1, 255], dtype=torch.long),
                torch.tensor([32, 127], dtype=torch.long),
                260,
                "byte_files",
            )
            self.assertEqual(contract["observed"]["train"], {"minimum": 0, "maximum": 255})
            self.assertEqual(contract["reserved_unused_output_id_range"], [256, 259])
            self.assertEqual(contract["label_semantics"], "raw next-byte IDs 0..255")
            with self.assertRaisesRegex(ValueError, "raw byte IDs"):
                _validate_stream_token_domain(
                    torch.tensor([0, 256], dtype=torch.long),
                    torch.tensor([32, 127], dtype=torch.long),
                    260,
                    "byte_files",
                )
            with self.assertRaisesRegex(ValueError, "outside model vocabulary"):
                _validate_stream_token_domain(
                    torch.tensor([0, 128], dtype=torch.long),
                    torch.tensor([32, 127], dtype=torch.long),
                    128,
                    "synthetic",
                )

    def test_byte_data_rejects_same_content_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.txt"
            validation = root / "validation.txt"
            train.write_bytes(b"same-content" * 20)
            validation.write_bytes(train.read_bytes())
            with self.assertRaisesRegex(ValueError, "identical file content"):
                _streams(
                    {
                        "mode": "byte_files",
                        "train_files": [str(train)],
                        "validation_files": [str(validation)],
                    },
                    TrainingConfig(sequence_length=8),
                )

    def test_next_token_label_shift_oracle_and_batcher_replay(self) -> None:
        ordered = torch.arange(128, dtype=torch.long)
        batcher = RandomWindowBatcher(ordered, 4, 16, seed=6)
        inputs, labels = batcher.next(torch.device("cpu"))
        self.assertTrue(torch.equal(labels[:, :-1], inputs[:, 1:]))
        self.assertTrue(torch.equal(labels, inputs + 1))

        tokens = synthetic_stream(1024, seed=7)
        batcher = RandomWindowBatcher(tokens, 3, 16, seed=8)
        batcher.next(torch.device("cpu"))
        state = batcher.state_dict()
        expected = batcher.next(torch.device("cpu"))
        replay = RandomWindowBatcher(tokens, 3, 16, seed=999)
        replay.load_state_dict(state)
        actual = replay.next(torch.device("cpu"))
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_cpu_checkpoint_restores_exact_update_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            config = ModelConfig(
                vocab_size=128,
                dim=16,
                n_layers=1,
                n_heads=2,
                n_kv_heads=1,
                hidden_dim=32,
                max_seq_len=8,
                dropout=0.1,
            )
            model = CausalLM(config)
            optimizer = _optimizer(model, TrainingConfig(sequence_length=8))
            tokens = synthetic_stream(256, seed=3)
            train_batcher = RandomWindowBatcher(tokens, 2, 8, seed=4)
            validation_batcher = RandomWindowBatcher(tokens, 2, 8, seed=5)
            inputs, labels = train_batcher.next(torch.device("cpu"))
            _, loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            saved_optimizer = copy.deepcopy(optimizer.state_dict())
            random.seed(6)
            torch.manual_seed(7)
            _save_checkpoint(
                checkpoint,
                model,
                optimizer,
                1,
                0,
                1,
                "a" * 64,
                "b" * 64,
                train_batcher,
                validation_batcher,
                torch.device("cpu"),
            )
            expected_python = random.random()
            expected_torch = torch.rand(4)
            expected_batch = train_batcher.next(torch.device("cpu"))
            _, expected_loss = model(*expected_batch)
            expected_loss.backward()
            optimizer.step()
            expected_model = copy.deepcopy(model.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())

            restored = CausalLM(config)
            restored_optimizer = _optimizer(restored, TrainingConfig(sequence_length=8))
            replay_train = RandomWindowBatcher(tokens, 2, 8, seed=99)
            replay_validation = RandomWindowBatcher(tokens, 2, 8, seed=100)
            next_step = _load_checkpoint(
                checkpoint,
                restored,
                restored_optimizer,
                0,
                1,
                torch.device("cpu"),
                "a" * 64,
                "b" * 64,
                replay_train,
                replay_validation,
            )
            self.assertEqual(next_step, 1)
            self._assert_optimizer_equal(restored_optimizer.state_dict(), saved_optimizer)
            self.assertEqual(random.random(), expected_python)
            torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)
            replay_batch = replay_train.next(torch.device("cpu"))
            torch.testing.assert_close(replay_batch[0], expected_batch[0])
            torch.testing.assert_close(replay_batch[1], expected_batch[1])
            _, replay_loss = restored(*replay_batch)
            replay_loss.backward()
            restored_optimizer.step()
            torch.testing.assert_close(replay_loss, expected_loss, rtol=0, atol=0)
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected_model[name], rtol=0, atol=0)
            self._assert_optimizer_equal(restored_optimizer.state_dict(), expected_optimizer)

    @staticmethod
    def _model_config() -> ModelConfig:
        return ModelConfig(
            vocab_size=128,
            dim=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            hidden_dim=64,
            max_seq_len=16,
            rope_fraction=0.5,
        )

    def _assert_optimizer_equal(self, actual: dict, expected: dict) -> None:
        self.assertEqual(actual["param_groups"], expected["param_groups"])
        self.assertEqual(actual["state"].keys(), expected["state"].keys())
        for parameter_id, expected_state in expected["state"].items():
            actual_state = actual["state"][parameter_id]
            self.assertEqual(actual_state.keys(), expected_state.keys())
            for key, expected_value in expected_state.items():
                actual_value = actual_state[key]
                if torch.is_tensor(expected_value):
                    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
                else:
                    self.assertEqual(actual_value, expected_value)


if __name__ == "__main__":
    unittest.main()
