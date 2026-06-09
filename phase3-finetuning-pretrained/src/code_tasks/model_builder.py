"""Full architectural modification of CodeT5+ hidden dimension.

This module builds models where d_model is changed throughout the entire
architecture: embeddings, attention Q/K/V/O, FFN, LayerNorms, and LM head.
No projection layers — the reduced dimension IS the model dimension.

Two initialization modes:
  - random:  Fresh weights at the target d_model.
  - pca:     Pre-trained weights compressed via PCA (Principal Component Analysis).
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
from sklearn.decomposition import PCA
from transformers import AutoConfig, AutoModelForSeq2SeqLM, T5Config, T5ForConditionalGeneration

logger = logging.getLogger(__name__)


class VarianceAnalysisResult(NamedTuple):
    recommended_hidden_dim: int
    original_hidden_dim: int
    variance_threshold: float
    cumulative_variance_at_recommended: float
    explained_variance_ratios: list[float]


def find_hidden_dim_from_variance(
    model_name_or_path: str,
    variance_threshold: float = 0.95,
    *,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
) -> VarianceAnalysisResult:
    """Run PCA on the pretrained embedding matrix and find the minimum hidden_dim
    that captures at least *variance_threshold* of the total variance.

    This is used purely as an analysis tool to choose a justified hidden_dim
    for building a smaller model from scratch. PCA is NOT applied to the model weights.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        variance_threshold: Fraction of variance to retain (default 0.95 = 95%).
        cache_dir: Optional HuggingFace cache directory.
        trust_remote_code: Whether to trust remote code in the model repo.

    Returns:
        VarianceAnalysisResult with the recommended hidden_dim and full variance info.
    """
    logger.info("Loading model for PCA variance analysis: %s", model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    embedding_weight = model.get_input_embeddings().weight.data.float()  # [vocab_size, d_model]
    original_hidden_dim = embedding_weight.shape[1]
    logger.info(
        "Embedding matrix shape: %s (vocab_size=%d, d_model=%d)",
        tuple(embedding_weight.shape),
        embedding_weight.shape[0],
        original_hidden_dim,
    )

    # Run full PCA to get all explained variance ratios
    pca = PCA(n_components=original_hidden_dim)
    pca.fit(embedding_weight.numpy())
    ratios = pca.explained_variance_ratio_.tolist()

    # Find minimum dim where cumulative variance >= threshold
    cumulative = 0.0
    recommended_dim = original_hidden_dim
    for dim, ratio in enumerate(ratios, start=1):
        cumulative += ratio
        if cumulative >= variance_threshold:
            recommended_dim = dim
            break

    cumulative_at_recommended = float(sum(ratios[:recommended_dim]))

    del model
    torch.cuda.empty_cache()

    logger.info(
        "PCA variance analysis: threshold=%.0f%%, recommended_hidden_dim=%d (captures %.2f%% variance), original_d_model=%d",
        variance_threshold * 100,
        recommended_dim,
        cumulative_at_recommended * 100,
        original_hidden_dim,
    )

    return VarianceAnalysisResult(
        recommended_hidden_dim=recommended_dim,
        original_hidden_dim=original_hidden_dim,
        variance_threshold=variance_threshold,
        cumulative_variance_at_recommended=cumulative_at_recommended,
        explained_variance_ratios=ratios,
    )


def build_model(
    base_model_name: str,
    hidden_dim: int | None,
    init_method: str = "random",
    *,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
) -> T5ForConditionalGeneration:
    """Build a model, optionally with a modified hidden dimension.

    If *hidden_dim* is ``None`` or matches the base config's ``d_model``,
    returns the standard pre-trained model (baseline).

    Otherwise builds a model whose ``d_model`` equals *hidden_dim*
    throughout the entire architecture.
    """
    base_config = AutoConfig.from_pretrained(
        base_model_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code,
    )
    original_d_model = getattr(base_config, "d_model", None) or getattr(base_config, "hidden_size", 768)

    # Baseline: no hidden dim change requested and not forced to random init
    if (hidden_dim is None or hidden_dim == original_d_model) and init_method != "random":
        logger.info("Loading baseline model with d_model=%d", original_d_model)
        return AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code,
        )

    # Use original d_model if hidden_dim not specified
    target_dim = hidden_dim if hidden_dim is not None else original_d_model

    if init_method == "random":
        return _build_random_init(base_config, target_dim)
    if init_method == "pca":
        return _build_svd_compressed(
            base_model_name, base_config, target_dim,
            cache_dir=cache_dir, trust_remote_code=trust_remote_code,
        )
    if init_method == "pretrained":
        logger.info("Loading pre-trained weights from: %s (d_model=%d)", base_model_name, target_dim)
        return AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code,
        )

    raise ValueError(f"Unknown init_method '{init_method}'. Choose 'random', 'pca', or 'pretrained'.")


# ---------------------------------------------------------------------------
# Setting 1: Random initialization
# ---------------------------------------------------------------------------

def _build_random_init(base_config: T5Config, hidden_dim: int) -> T5ForConditionalGeneration:
    """Create a fresh T5 model with d_model = hidden_dim (all weights random)."""
    config = _derive_config(base_config, hidden_dim)
    model = T5ForConditionalGeneration(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Built randomly-initialized model: d_model=%d, d_ff=%d, d_kv=%d, num_heads=%d, params=%s",
        config.d_model, config.d_ff, config.d_kv, config.num_heads, f"{n_params:,}",
    )
    return model


# ---------------------------------------------------------------------------
# Setting 2: SVD-compressed pre-trained weights
# ---------------------------------------------------------------------------

def _build_svd_compressed(
    base_model_name: str,
    base_config: T5Config,
    hidden_dim: int,
    *,
    cache_dir: str | None,
    trust_remote_code: bool,
) -> T5ForConditionalGeneration:
    """Load pre-trained weights and compress them via truncated SVD."""

    logger.info("Loading full pre-trained model for SVD compression …")
    full_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code,
    )
    full_model.eval()

    target_config = _derive_config(base_config, hidden_dim)
    small_model = T5ForConditionalGeneration(target_config)

    original_d_model = base_config.d_model
    original_d_ff = base_config.d_ff
    target_d_ff = target_config.d_ff
    original_inner = base_config.d_kv * base_config.num_heads
    target_inner = target_config.d_kv * target_config.num_heads

    # Compute a global projection matrix P from the shared embedding
    logger.info("Computing SVD projection matrix from shared embeddings …")
    P = _embedding_svd_projection(full_model, hidden_dim)  # [original_d_model, hidden_dim]

    full_sd = full_model.state_dict()
    small_sd = small_model.state_dict()

    transferred = 0
    for name, target_param in small_sd.items():
        if name not in full_sd:
            continue

        source = full_sd[name].float()
        shape = target_param.shape

        try:
            compressed = _compress_parameter(
                name, source, shape,
                P=P,
                original_d_model=original_d_model,
                hidden_dim=hidden_dim,
                original_d_ff=original_d_ff,
                target_d_ff=target_d_ff,
                original_inner=original_inner,
                target_inner=target_inner,
            )
            small_sd[name] = compressed.to(dtype=target_param.dtype)
            transferred += 1
        except Exception as exc:
            logger.warning("Could not compress '%s': %s — using random init", name, exc)

    small_model.load_state_dict(small_sd, strict=True)

    n_params = sum(p.numel() for p in small_model.parameters())
    logger.info(
        "SVD-compressed model: d_model=%d, transferred %d/%d params, total=%s",
        hidden_dim, transferred, len(small_sd), f"{n_params:,}",
    )

    del full_model
    torch.cuda.empty_cache()

    return small_model


def _embedding_svd_projection(model: T5ForConditionalGeneration, hidden_dim: int) -> torch.Tensor:
    """Return projection matrix P of shape [original_d_model, hidden_dim] using PCA."""
    E = model.shared.weight.data.float()  # [vocab_size, d_model]
    pca = PCA(n_components=hidden_dim)
    pca.fit(E.numpy())
    P = torch.tensor(pca.components_.T, dtype=torch.float32)  # [original_d_model, hidden_dim]
    return P


def _compress_parameter(
    name: str,
    source: torch.Tensor,
    target_shape: torch.Size,
    *,
    P: torch.Tensor,
    original_d_model: int,
    hidden_dim: int,
    original_d_ff: int,
    target_d_ff: int,
    original_inner: int,
    target_inner: int,
) -> torch.Tensor:
    """Compress a single parameter tensor to match target_shape."""

    # 1D tensors: LayerNorm weights/biases, or other biases
    if source.dim() == 1:
        if source.shape[0] == original_d_model and target_shape[0] == hidden_dim:
            # Project via P: for LayerNorm, take the norms of P columns as scaling
            return source[:hidden_dim]
        if source.shape[0] == original_d_ff and target_shape[0] == target_d_ff:
            return source[:target_d_ff]
        if source.shape[0] == original_inner and target_shape[0] == target_inner:
            return source[:target_inner]
        if source.shape == target_shape:
            return source
        return source[:target_shape[0]]

    # 2D tensors: weight matrices
    if source.dim() == 2:
        src_r, src_c = source.shape
        tgt_r, tgt_c = target_shape

        # Embedding / LM head: [vocab, d_model] -> [vocab, hidden_dim]
        if src_r == tgt_r and src_c == original_d_model and tgt_c == hidden_dim:
            return source @ P

        # Q/K/V projection: [d_model, inner] -> [hidden_dim, target_inner]
        if src_r == original_d_model and tgt_r == hidden_dim:
            projected = P.T @ source  # [hidden_dim, src_c]
            return projected[:, :tgt_c]

        # Output projection: [inner, d_model] -> [target_inner, hidden_dim]
        if src_c == original_d_model and tgt_c == hidden_dim:
            projected = source @ P  # [src_r, hidden_dim]
            return projected[:tgt_r, :]

        # FFN wi: [d_model, d_ff] -> [hidden_dim, target_d_ff]
        # FFN wo: [d_ff, d_model] -> [target_d_ff, hidden_dim]
        # These are handled by the cases above when d_model is on one side

        # Relative position bias: [n_buckets, num_heads] — may need head slicing
        if source.shape != target_shape:
            return source[:tgt_r, :tgt_c]

        return source

    # Higher-dimensional tensors: just slice
    return source[tuple(slice(0, s) for s in target_shape)]


# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------

def _derive_config(base_config: T5Config, hidden_dim: int) -> T5Config:
    """Create a new T5Config with d_model = hidden_dim and consistent dimensions."""
    config = T5Config(**base_config.to_dict())

    original_d_model = base_config.d_model
    original_num_heads = base_config.num_heads

    config.d_model = hidden_dim

    # Scale d_ff proportionally (preserve original ratio)
    ratio = base_config.d_ff / original_d_model
    config.d_ff = int(round(ratio * hidden_dim))
    # Make d_ff divisible by 64 for efficiency
    config.d_ff = max(64, (config.d_ff // 64) * 64)

    # Determine num_heads and d_kv
    # Try to keep original num_heads; fall back to fewer heads
    num_heads = original_num_heads
    while num_heads > 1 and hidden_dim % num_heads != 0:
        num_heads -= 1

    config.num_heads = num_heads
    config.d_kv = hidden_dim // num_heads
    config.num_decoder_heads = num_heads

    # Store the original model name for reference
    config.base_model_for_compression = getattr(base_config, "_name_or_path", "unknown")
    config.compression_hidden_dim = hidden_dim
    config.compression_original_d_model = original_d_model

    return config
