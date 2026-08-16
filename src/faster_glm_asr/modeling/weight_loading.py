# Provenance: adapted from the ed-aisys/edin-mls-26-spring GLM-ASR Triton
# reference at commit 525c3c4c3c584ab4c9e0ec7ff8d8ee202933d374. Upstream
# contributors include Yangshen Deng and Yeqi Huang (Chivier Humber); the
# state-dict mapping must not be represented as independently authored here.

"""Map a pinned Hugging Face GLM-ASR checkpoint into the custom model."""

import gc
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from faster_glm_asr import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION

from .model import GlmAsrConfig, GlmAsrModel

logger = logging.getLogger(__name__)


DEFAULT_MODEL_NAME = DEFAULT_MODEL_ID
IMMUTABLE_REVISION_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def _expected_transformers_state_keys(model: GlmAsrModel) -> set[str]:
    """Return the exact Transformers 5.14 GLM-ASR state-dict schema."""
    expected = {
        "model.audio_tower.conv1.weight",
        "model.audio_tower.conv1.bias",
        "model.audio_tower.conv2.weight",
        "model.audio_tower.conv2.bias",
        "model.audio_tower.norm.weight",
        "model.audio_tower.norm.bias",
        "model.multi_modal_projector.linear_1.weight",
        "model.multi_modal_projector.linear_1.bias",
        "model.multi_modal_projector.linear_2.weight",
        "model.multi_modal_projector.linear_2.bias",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    }
    for index in range(model.config.audio_num_layers):
        prefix = f"model.audio_tower.layers.{index}"
        expected.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.input_layernorm.bias",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.post_attention_layernorm.bias",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.v_proj.bias",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.o_proj.bias",
                f"{prefix}.mlp.fc1.weight",
                f"{prefix}.mlp.fc1.bias",
                f"{prefix}.mlp.fc2.weight",
                f"{prefix}.mlp.fc2.bias",
            }
        )
    for index in range(model.config.text_num_layers):
        prefix = f"model.language_model.layers.{index}"
        expected.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            }
        )
    return expected


def _normalize_transformers_state_dict(
    model: GlmAsrModel, state: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Validate and map the pinned Transformers namespace to logical keys.

    The safetensors archive uses base-model keys such as ``audio_tower.*``,
    while ``GlmAsrForConditionalGeneration.state_dict()`` in Transformers
    5.14 exposes ``model.audio_tower.*`` and a top-level ``lm_head``.  The
    custom mapper consumes the former logical namespace.  Requiring the exact
    current schema prevents a mixed or silently shifted upstream layout from
    producing a partially initialized model.
    """
    expected = _expected_transformers_state_keys(model)
    observed = set(state)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(
            "Transformers 5.14 GLM-ASR state-dict schema mismatch; "
            f"missing_count={len(missing)}, unexpected_count={len(unexpected)}, "
            f"missing_sample={missing[:3]}, unexpected_sample={unexpected[:3]}"
        )

    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state-dict value for {key!r} must be a torch.Tensor")
        if key.startswith("model.audio_tower.") or key.startswith("model.multi_modal_projector."):
            logical_key = key.removeprefix("model.")
        elif key.startswith("model.language_model."):
            logical_key = "language_model.model." + key.removeprefix("model.language_model.")
        elif key == "lm_head.weight":
            logical_key = "language_model.lm_head.weight"
        else:  # The exact-schema check above makes this a defensive assertion.
            raise AssertionError(f"unhandled validated state key: {key}")
        if logical_key in normalized:
            raise ValueError(f"duplicate normalized state key: {logical_key}")
        normalized[logical_key] = value
    return normalized


def create_config_from_hf(
    hf_config: Any,
    *,
    projector_hidden_size: int,
    projector_pool_factor: int,
) -> GlmAsrConfig:
    """Create :class:`GlmAsrConfig` from a Transformers config object."""
    ac = hf_config.audio_config
    tc = hf_config.text_config
    _validate_supported_hf_semantics(
        hf_config,
        projector_pool_factor=projector_pool_factor,
    )
    text_rope_parameters = getattr(tc, "rope_parameters", {}) or {}

    return GlmAsrConfig(
        audio_hidden_size=ac.hidden_size,
        audio_num_heads=ac.num_attention_heads,
        audio_num_layers=ac.num_hidden_layers,
        audio_intermediate_size=ac.intermediate_size,
        audio_max_position_embeddings=getattr(ac, "max_position_embeddings", 1500),
        text_hidden_size=tc.hidden_size,
        text_num_heads=tc.num_attention_heads,
        text_num_kv_heads=tc.num_key_value_heads,
        text_num_layers=tc.num_hidden_layers,
        text_intermediate_size=tc.intermediate_size,
        text_vocab_size=tc.vocab_size,
        text_max_position_embeddings=tc.max_position_embeddings,
        text_rope_base=text_rope_parameters.get("rope_theta", 10000.0),
        text_rms_norm_eps=getattr(tc, "rms_norm_eps", 1e-5),
        projector_hidden_size=projector_hidden_size,
        projector_pool_factor=projector_pool_factor,
        pad_token_id=getattr(tc, "pad_token_id", 0)
        if getattr(tc, "pad_token_id", None) is not None
        else 0,
        bos_token_id=getattr(tc, "bos_token_id", 1)
        if getattr(tc, "bos_token_id", None) is not None
        else 1,
        eos_token_id=getattr(tc, "eos_token_id", 2),
    )


def _validate_supported_hf_semantics(
    hf_config: Any,
    *,
    projector_pool_factor: int,
) -> None:
    """Reject checkpoints whose operators differ from the custom graph."""
    ac = hf_config.audio_config
    tc = hf_config.text_config
    audio_rope = getattr(ac, "rope_parameters", {}) or {}
    text_rope = getattr(tc, "rope_parameters", {}) or {}
    audio_head_dim = getattr(ac, "head_dim", None) or (ac.hidden_size // ac.num_attention_heads)
    text_head_dim = getattr(tc, "head_dim", None) or (tc.hidden_size // tc.num_attention_heads)
    audio_kv_heads = getattr(ac, "num_key_value_heads", None) or ac.num_attention_heads
    checks = {
        "model_type": (getattr(hf_config, "model_type", None), "glmasr"),
        "projector_hidden_act": (getattr(hf_config, "projector_hidden_act", None), "gelu"),
        "audio_token_id": (getattr(hf_config, "audio_token_id", None), 59260),
        "audio.model_type": (getattr(ac, "model_type", None), "glmasr_encoder"),
        "audio.hidden_act": (getattr(ac, "hidden_act", None), "gelu"),
        "audio.attention_dropout": (getattr(ac, "attention_dropout", None), 0.0),
        "audio.num_mel_bins": (getattr(ac, "num_mel_bins", None), 128),
        "audio.num_key_value_heads": (
            audio_kv_heads,
            ac.num_attention_heads,
        ),
        "audio.head_dim": (
            audio_head_dim,
            ac.hidden_size // ac.num_attention_heads,
        ),
        "audio.partial_rotary_factor": (
            getattr(ac, "partial_rotary_factor", None),
            0.5,
        ),
        "audio.rope_type": (audio_rope.get("rope_type"), "default"),
        "audio.rope_theta": (audio_rope.get("rope_theta"), 10000.0),
        "audio.rope_partial_rotary_factor": (
            audio_rope.get("partial_rotary_factor"),
            0.5,
        ),
        "text.model_type": (getattr(tc, "model_type", None), "llama"),
        "text.hidden_act": (getattr(tc, "hidden_act", None), "silu"),
        "text.attention_dropout": (getattr(tc, "attention_dropout", None), 0.0),
        "text.attention_bias": (getattr(tc, "attention_bias", None), False),
        "text.mlp_bias": (getattr(tc, "mlp_bias", None), False),
        "text.head_dim": (
            text_head_dim,
            tc.hidden_size // tc.num_attention_heads,
        ),
        "text.rope_type": (text_rope.get("rope_type"), "default"),
    }
    mismatches = [
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    expected_projector_input = ac.hidden_size * projector_pool_factor
    if ac.intermediate_size != expected_projector_input:
        mismatches.append(
            "audio.intermediate_size="
            f"{ac.intermediate_size!r} (expected {expected_projector_input!r} from projector pool)"
        )
    text_rope_theta = text_rope.get("rope_theta", 10000.0)
    if not isinstance(text_rope_theta, (int, float)) or text_rope_theta <= 0:
        mismatches.append("text.rope_theta must be positive")
    if mismatches:
        raise ValueError(
            "checkpoint semantic config is incompatible with the custom GLM-ASR graph: "
            + "; ".join(mismatches)
        )


def _projector_dimensions_from_transformers_state(
    hf_config: Any,
    state: Mapping[str, torch.Tensor],
) -> tuple[int, int]:
    """Derive projector dimensions from the pinned safe state dict."""
    first_key = "model.multi_modal_projector.linear_1.weight"
    second_key = "model.multi_modal_projector.linear_2.weight"
    try:
        first = state[first_key]
        second = state[second_key]
    except KeyError as exc:
        raise ValueError(f"Transformers state dict is missing {exc.args[0]!r}") from exc
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError("projector weights must be torch.Tensor objects")
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("projector weights must be rank-two matrices")
    audio_hidden_size = int(hf_config.audio_config.hidden_size)
    text_hidden_size = int(hf_config.text_config.hidden_size)
    projector_hidden_size, pooled_audio_size = first.shape
    text_output_size, second_input_size = second.shape
    if (
        pooled_audio_size % audio_hidden_size
        or second_input_size != projector_hidden_size
        or text_output_size != text_hidden_size
    ):
        raise ValueError("projector weight shapes are incompatible with audio/text config")
    projector_pool_factor = pooled_audio_size // audio_hidden_size
    if projector_pool_factor <= 0:
        raise ValueError("derived projector_pool_factor must be positive")
    return projector_hidden_size, projector_pool_factor


def load_linear_weight(triton_linear, hf_weight, hf_bias=None):
    """Load weight (and optional bias) into Triton Linear layer."""
    if tuple(hf_weight.shape) != tuple(triton_linear.weight.shape):
        raise ValueError(
            "linear weight shape mismatch: "
            f"checkpoint={tuple(hf_weight.shape)}, expected={tuple(triton_linear.weight.shape)}"
        )
    if triton_linear.has_bias:
        if hf_bias is None:
            raise ValueError("linear bias is missing from the checkpoint")
        if tuple(hf_bias.shape) != tuple(triton_linear.bias_param.shape):
            raise ValueError(
                "linear bias shape mismatch: "
                f"checkpoint={tuple(hf_bias.shape)}, expected={tuple(triton_linear.bias_param.shape)}"
            )
    elif hf_bias is not None:
        raise ValueError("checkpoint contains a bias for a bias-free linear layer")
    triton_linear.weight = hf_weight.detach().to(torch.float32).clone()
    if hasattr(triton_linear, "_weight_t_padded"):
        triton_linear._weight_t_padded = None
    if hf_bias is not None and triton_linear.has_bias:
        triton_linear.bias_param = hf_bias.detach().to(torch.float32).clone()


def load_conv1d_weight_from_hf(triton_conv, hf_weight, hf_bias=None):
    """Load weight into Triton Conv1d layer from HF format."""
    expected_weight_shape = (
        triton_conv.out_channels,
        triton_conv.in_channels,
        triton_conv.kernel_size,
    )
    if tuple(hf_weight.shape) != expected_weight_shape:
        raise ValueError(
            "conv1d weight shape mismatch: "
            f"checkpoint={tuple(hf_weight.shape)}, expected={expected_weight_shape}"
        )
    if triton_conv.has_bias:
        if hf_bias is None:
            raise ValueError("conv1d bias is missing from the checkpoint")
        if tuple(hf_bias.shape) != (triton_conv.out_channels,):
            raise ValueError("conv1d bias shape does not match out_channels")
    elif hf_bias is not None:
        raise ValueError("checkpoint contains a bias for a bias-free conv1d layer")
    weight = hf_weight.detach().to(torch.float32)
    out_channels, in_channels, kernel_size = weight.shape
    triton_conv.weight = weight.reshape(out_channels, in_channels * kernel_size).clone()

    if triton_conv.use_triton and (
        triton_conv.col_size_padded != triton_conv.col_size
        or triton_conv.out_channels_padded != out_channels
    ):
        triton_conv.weight_padded = torch.zeros(
            (triton_conv.out_channels_padded, triton_conv.col_size_padded),
            dtype=torch.float32,
        )
        triton_conv.weight_padded[:out_channels, : triton_conv.col_size] = triton_conv.weight
    else:
        triton_conv.weight_padded = triton_conv.weight

    if hf_bias is not None and triton_conv.has_bias:
        triton_conv.bias = hf_bias.detach().to(torch.float32).clone()


def load_layernorm_weight_from_hf(triton_ln, hf_weight, hf_bias):
    """Load LayerNorm weights."""
    if tuple(hf_weight.shape) != tuple(triton_ln.weight.shape):
        raise ValueError("LayerNorm weight shape mismatch")
    if tuple(hf_bias.shape) != tuple(triton_ln.bias.shape):
        raise ValueError("LayerNorm bias shape mismatch")
    triton_ln.weight = hf_weight.detach().to(torch.float32).clone()
    triton_ln.bias = hf_bias.detach().to(torch.float32).clone()


def load_rmsnorm_weight_from_hf(triton_rms, hf_weight):
    """Load RMSNorm weight."""
    if tuple(hf_weight.shape) != tuple(triton_rms.weight.shape):
        raise ValueError("RMSNorm weight shape mismatch")
    triton_rms.weight = hf_weight.detach().to(torch.float32).clone()


def load_embedding_weight_from_hf(triton_emb, hf_weight):
    """Load Embedding weight."""
    if tuple(hf_weight.shape) != tuple(triton_emb.weight.shape):
        raise ValueError("embedding weight shape mismatch")
    triton_emb.weight = hf_weight.detach().to(torch.float32).clone()


def load_weights_from_hf_model(model: GlmAsrModel, hf_model: Any) -> None:
    """
    Load weights from a Hugging Face GLM-ASR model into the custom graph.
    """
    raw_state = hf_model.state_dict()
    if not isinstance(raw_state, Mapping):
        raise TypeError("Hugging Face model state_dict() must return a mapping")
    hf_state = _normalize_transformers_state_dict(model, raw_state)

    logger.info("Loading audio encoder weights")

    load_conv1d_weight_from_hf(
        model.audio_encoder.conv1,
        hf_state["audio_tower.conv1.weight"],
        hf_state["audio_tower.conv1.bias"],
    )
    load_conv1d_weight_from_hf(
        model.audio_encoder.conv2,
        hf_state["audio_tower.conv2.weight"],
        hf_state["audio_tower.conv2.bias"],
    )

    if "audio_tower.embed_positions.weight" in hf_state:
        model.audio_encoder.embed_positions = (
            hf_state["audio_tower.embed_positions.weight"].detach().to(torch.float32).clone()
        )

    for i, layer in enumerate(model.audio_encoder.layers):
        prefix = f"audio_tower.layers.{i}"

        load_layernorm_weight_from_hf(
            layer.self_attn_layer_norm,
            hf_state[f"{prefix}.input_layernorm.weight"],
            hf_state[f"{prefix}.input_layernorm.bias"],
        )

        load_linear_weight(
            layer.q_proj,
            hf_state[f"{prefix}.self_attn.q_proj.weight"],
            hf_state.get(f"{prefix}.self_attn.q_proj.bias"),
        )
        load_linear_weight(
            layer.k_proj,
            hf_state[f"{prefix}.self_attn.k_proj.weight"],
            hf_state.get(f"{prefix}.self_attn.k_proj.bias"),
        )
        load_linear_weight(
            layer.v_proj,
            hf_state[f"{prefix}.self_attn.v_proj.weight"],
            hf_state.get(f"{prefix}.self_attn.v_proj.bias"),
        )
        load_linear_weight(
            layer.out_proj,
            hf_state[f"{prefix}.self_attn.o_proj.weight"],
            hf_state.get(f"{prefix}.self_attn.o_proj.bias"),
        )

        load_layernorm_weight_from_hf(
            layer.final_layer_norm,
            hf_state[f"{prefix}.post_attention_layernorm.weight"],
            hf_state[f"{prefix}.post_attention_layernorm.bias"],
        )

        load_linear_weight(
            layer.fc1,
            hf_state[f"{prefix}.mlp.fc1.weight"],
            hf_state[f"{prefix}.mlp.fc1.bias"],
        )
        load_linear_weight(
            layer.fc2,
            hf_state[f"{prefix}.mlp.fc2.weight"],
            hf_state[f"{prefix}.mlp.fc2.bias"],
        )

    load_layernorm_weight_from_hf(
        model.audio_encoder.layer_norm,
        hf_state["audio_tower.norm.weight"],
        hf_state["audio_tower.norm.bias"],
    )

    logger.info("Loading multimodal projector weights")

    load_linear_weight(
        model.multi_modal_projector.linear_1,
        hf_state["multi_modal_projector.linear_1.weight"],
        hf_state["multi_modal_projector.linear_1.bias"],
    )
    load_linear_weight(
        model.multi_modal_projector.linear_2,
        hf_state["multi_modal_projector.linear_2.weight"],
        hf_state["multi_modal_projector.linear_2.bias"],
    )

    logger.info("Loading text decoder weights")

    load_embedding_weight_from_hf(
        model.text_decoder.embed_tokens,
        hf_state["language_model.model.embed_tokens.weight"],
    )

    for i, layer in enumerate(model.text_decoder.layers):
        prefix = f"language_model.model.layers.{i}"

        load_rmsnorm_weight_from_hf(
            layer.input_layernorm,
            hf_state[f"{prefix}.input_layernorm.weight"],
        )

        load_linear_weight(
            layer.q_proj,
            hf_state[f"{prefix}.self_attn.q_proj.weight"],
        )
        load_linear_weight(
            layer.k_proj,
            hf_state[f"{prefix}.self_attn.k_proj.weight"],
        )
        load_linear_weight(
            layer.v_proj,
            hf_state[f"{prefix}.self_attn.v_proj.weight"],
        )
        load_linear_weight(
            layer.o_proj,
            hf_state[f"{prefix}.self_attn.o_proj.weight"],
        )

        load_rmsnorm_weight_from_hf(
            layer.post_attention_layernorm,
            hf_state[f"{prefix}.post_attention_layernorm.weight"],
        )

        load_linear_weight(
            layer.mlp.gate_proj,
            hf_state[f"{prefix}.mlp.gate_proj.weight"],
        )
        load_linear_weight(
            layer.mlp.up_proj,
            hf_state[f"{prefix}.mlp.up_proj.weight"],
        )
        load_linear_weight(
            layer.mlp.down_proj,
            hf_state[f"{prefix}.mlp.down_proj.weight"],
        )

    load_rmsnorm_weight_from_hf(
        model.text_decoder.norm,
        hf_state["language_model.model.norm.weight"],
    )

    embedding_weight = hf_state["language_model.model.embed_tokens.weight"]
    lm_head_weight = hf_state["language_model.lm_head.weight"]
    if embedding_weight.shape != lm_head_weight.shape:
        raise ValueError(
            "tied embedding/lm-head shapes differ: "
            f"{tuple(embedding_weight.shape)} != {tuple(lm_head_weight.shape)}"
        )
    if embedding_weight.data_ptr() != lm_head_weight.data_ptr() and not torch.equal(
        embedding_weight, lm_head_weight
    ):
        raise ValueError("official embedding and lm-head weights are not equal")
    model.tie_lm_head_weights()

    logger.info("Weight loading complete")


def load_model_from_hf(
    model_name: str = DEFAULT_MODEL_NAME,
    revision: str = DEFAULT_MODEL_REVISION,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> tuple[GlmAsrModel, Any]:
    """Load one pinned checkpoint and return ``(custom_model, processor)``.

    Model, config, and processor receive the same revision and download
    controls.  The official model is materialized on CPU only long enough to
    validate and copy its state dict; custom tensors move lazily to the input
    device during inference.
    """

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(revision, str) or IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
        raise ValueError("revision must be a lowercase 40- or 64-hex immutable commit SHA")
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a boolean")

    from transformers import AutoConfig, AutoProcessor, GlmAsrForConditionalGeneration

    common_kwargs = {
        "revision": revision,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "token": token,
        "trust_remote_code": False,
    }
    logger.info("Loading pinned Hugging Face model %s@%s", model_name, revision)

    hf_config = AutoConfig.from_pretrained(model_name, **common_kwargs)
    hf_model = GlmAsrForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map="cpu",
        use_safetensors=True,
        **common_kwargs,
    )
    raw_state = hf_model.state_dict()
    if not isinstance(raw_state, Mapping):
        raise TypeError("Hugging Face model state_dict() must return a mapping")
    projector_hidden_size, projector_pool_factor = _projector_dimensions_from_transformers_state(
        hf_config, raw_state
    )
    custom_config = create_config_from_hf(
        hf_config,
        projector_hidden_size=projector_hidden_size,
        projector_pool_factor=projector_pool_factor,
    )
    custom_model = GlmAsrModel(custom_config)
    processor = AutoProcessor.from_pretrained(model_name, **common_kwargs)

    load_weights_from_hf_model(custom_model, hf_model)

    del hf_model
    gc.collect()

    return custom_model, processor
