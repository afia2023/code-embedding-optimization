from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModelForSeq2SeqLM
try:
    from transformers.modeling_utils import load_sharded_checkpoint
except ImportError:
    from transformers.trainer_utils import load_sharded_checkpoint


VAE_DEFAULTS = {
    "max_samples": 20000,
    "batch_size": 512,
    "epochs": 8,
    "learning_rate": 1e-3,
    "beta": 1e-4,
}


class ReducedInputEmbedding(nn.Module):
    """Input embedding adapter with a learned low-dimensional token table and decoder."""

    def __init__(
        self,
        num_embeddings: int,
        reduced_dim: int,
        output_dim: int,
        *,
        padding_idx: int | None = None,
        decoder_hidden_dims: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = reduced_dim
        self.output_dim = output_dim
        self.padding_idx = padding_idx
        self.decoder_hidden_dims = [int(value) for value in decoder_hidden_dims or [] if int(value) > 0]

        self.embedding = nn.Embedding(num_embeddings, reduced_dim, padding_idx=padding_idx)

        layers: list[nn.Module] = []
        current_dim = reduced_dim
        for hidden_dim in self.decoder_hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim, bias=True))
        self.decoder = nn.Sequential(*layers)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.decode(self.embedding(input_ids))

    def decode(self, latent_vectors: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_vectors)

    @property
    def weight(self) -> torch.Tensor:
        return self.decode(self.embedding.weight)

    def initialize_from_pretrained(
        self,
        pretrained_weight: torch.Tensor,
        *,
        reduction_method: str,
        vae_hidden_dim: int | None = None,
        vae_defaults: dict[str, float | int] | None = None,
    ) -> dict[str, int | float | str | list[int]]:
        if reduction_method == "pca":
            self._initialize_with_pca(pretrained_weight)
            return {
                "reduction_method": "pca",
                "decoder_hidden_dims": [],
            }
        if reduction_method == "vae":
            hidden_dim = vae_hidden_dim or _default_vae_hidden_dim(self.embedding_dim, self.output_dim)
            settings = dict(VAE_DEFAULTS)
            if vae_defaults is not None:
                settings.update(vae_defaults)
            self._initialize_with_vae(pretrained_weight, hidden_dim=hidden_dim, vae_settings=settings)
            return {
                "reduction_method": "vae",
                "decoder_hidden_dims": [hidden_dim],
                "vae_epochs": int(settings["epochs"]),
                "vae_batch_size": int(settings["batch_size"]),
                "vae_learning_rate": float(settings["learning_rate"]),
                "vae_beta": float(settings["beta"]),
                "vae_max_samples": int(settings["max_samples"]),
            }
        raise ValueError("Unsupported input embedding reduction. Choose from ['pca', 'vae'].")

    def _initialize_with_pca(self, pretrained_weight: torch.Tensor) -> None:
        weight = pretrained_weight.detach().to(dtype=torch.float32, device="cpu")
        mean = weight.mean(dim=0, keepdim=True)
        centered = weight - mean
        rank = min(self.embedding_dim, centered.shape[0], centered.shape[1])

        with torch.no_grad():
            self.embedding.weight.zero_()
            for layer in self._linear_layers():
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()

            if rank == 0:
                return

            _, _, right = torch.linalg.svd(centered, full_matrices=False)
            components = right[:rank]
            latents = centered @ components.transpose(0, 1)

            self.embedding.weight[:, :rank].copy_(
                latents.to(dtype=self.embedding.weight.dtype, device=self.embedding.weight.device)
            )

            linear_layers = self._linear_layers()
            output_layer = linear_layers[-1]
            output_layer.weight[:, :rank].copy_(
                components.to(dtype=output_layer.weight.dtype, device=output_layer.weight.device).transpose(0, 1)
            )
            if output_layer.bias is not None:
                output_layer.bias.copy_(
                    mean.squeeze(0).to(dtype=output_layer.bias.dtype, device=output_layer.bias.device)
                )

            if self.padding_idx is not None:
                self.embedding.weight[self.padding_idx].zero_()

    def _initialize_with_vae(
        self,
        pretrained_weight: torch.Tensor,
        *,
        hidden_dim: int,
        vae_settings: dict[str, float | int],
    ) -> None:
        if self.decoder_hidden_dims != [hidden_dim]:
            raise ValueError(
                "VAE initialization expects the runtime decoder to have exactly one hidden layer "
                f"matching the VAE hidden size. Decoder layers: {self.decoder_hidden_dims}, hidden_dim: {hidden_dim}."
            )

        fit_device = torch.device("cpu")
        weight = pretrained_weight.detach().to(dtype=torch.float32, device=fit_device)
        vae = EmbeddingVAE(input_dim=weight.shape[1], latent_dim=self.embedding_dim, hidden_dim=hidden_dim).to(fit_device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=float(vae_settings["learning_rate"]))

        sampled_embeddings = _sample_rows(weight, max_samples=int(vae_settings["max_samples"]))
        data_loader = DataLoader(
            TensorDataset(sampled_embeddings),
            batch_size=int(vae_settings["batch_size"]),
            shuffle=True,
        )

        vae.train()
        for _ in range(int(vae_settings["epochs"])):
            for (batch,) in data_loader:
                optimizer.zero_grad(set_to_none=True)
                reconstruction, mu, logvar = vae(batch)
                reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, batch)
                kl_divergence = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + float(vae_settings["beta"]) * kl_divergence
                loss.backward()
                optimizer.step()

        vae.eval()
        encoded_latents = []
        with torch.no_grad():
            for chunk in weight.split(int(vae_settings["batch_size"])):
                mu, _ = vae.encode(chunk)
                encoded_latents.append(mu)
        latent_table = torch.cat(encoded_latents, dim=0)
        with torch.no_grad():
            self.embedding.weight.copy_(
                latent_table.to(dtype=self.embedding.weight.dtype, device=self.embedding.weight.device)
            )
            _copy_decoder_weights(source_decoder=vae.decoder, target_decoder=self.decoder)

            if self.padding_idx is not None:
                self.embedding.weight[self.padding_idx].zero_()

    def _linear_layers(self) -> list[nn.Linear]:
        return [layer for layer in self.decoder if isinstance(layer, nn.Linear)]


class EmbeddingVAE(nn.Module):
    """Small VAE trained over the pretrained token embedding matrix."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(inputs)
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mu + noise * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(inputs)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent), mu, logvar


def load_model_with_optional_input_embedding_adapter(
    model_name_or_path: str,
    *,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
    tokenizer_length: int | None = None,
    requested_input_embedding_dim: int | None = None,
    requested_reduction_method: str = "pca",
):
    model_config = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )

    saved_input_embedding_dim = getattr(model_config, "input_embedding_dim", None)
    saved_reduction_method = getattr(model_config, "input_embedding_reduction", None)
    saved_decoder_hidden_dims = getattr(model_config, "input_embedding_decoder_hidden_dims", None)
    effective_input_embedding_dim = requested_input_embedding_dim or saved_input_embedding_dim

    if (
        requested_input_embedding_dim is not None
        and saved_input_embedding_dim is not None
        and requested_input_embedding_dim != saved_input_embedding_dim
    ):
        raise ValueError(
            "The requested --input-embedding-dim does not match the checkpoint configuration. "
            f"Checkpoint dimension: {saved_input_embedding_dim}, requested: {requested_input_embedding_dim}."
        )

    if (
        saved_reduction_method is not None
        and requested_input_embedding_dim is not None
        and requested_reduction_method != saved_reduction_method
    ):
        raise ValueError(
            "The requested --input-embedding-reduction does not match the checkpoint configuration. "
            f"Checkpoint method: {saved_reduction_method}, requested: {requested_reduction_method}."
        )

    checkpoint_path = Path(model_name_or_path)
    should_manually_load = checkpoint_path.is_dir() and saved_input_embedding_dim is not None

    if should_manually_load:
        model = AutoModelForSeq2SeqLM.from_config(model_config, trust_remote_code=trust_remote_code)
        if tokenizer_length is not None:
            checkpoint_vocab_size = model.get_input_embeddings().weight.shape[0]
            if tokenizer_length != checkpoint_vocab_size:
                raise ValueError(
                    "The tokenizer vocabulary size does not match the saved checkpoint vocabulary size. "
                    f"Tokenizer size: {tokenizer_length}, checkpoint size: {checkpoint_vocab_size}."
                )
        apply_input_embedding_adapter(
            model,
            saved_input_embedding_dim,
            reduction_method=saved_reduction_method or requested_reduction_method,
            decoder_hidden_dims=saved_decoder_hidden_dims,
            initialize_from_current=False,
        )
        load_sharded_checkpoint(model, str(checkpoint_path), strict=True, prefer_safe=True)
        return model, saved_input_embedding_dim

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer_length is not None:
        current_vocab_size = model.get_input_embeddings().weight.shape[0]
        if tokenizer_length > current_vocab_size:
            model.resize_token_embeddings(tokenizer_length)

    if effective_input_embedding_dim is not None:
        adapter_metadata = apply_input_embedding_adapter(
            model,
            effective_input_embedding_dim,
            reduction_method=requested_reduction_method,
            initialize_from_current=True,
        )
        _persist_adapter_metadata(
            model.config,
            input_embedding_dim=effective_input_embedding_dim,
            reduction_method=requested_reduction_method,
            base_model_name_or_path=getattr(model_config, "input_embedding_base_model_name_or_path", None)
            or model_name_or_path,
            decoder_hidden_dims=adapter_metadata.get("decoder_hidden_dims", []),
            reduction_metadata=adapter_metadata,
        )

    return model, effective_input_embedding_dim


def apply_input_embedding_adapter(
    model,
    input_embedding_dim: int,
    *,
    reduction_method: str,
    decoder_hidden_dims: list[int] | None = None,
    initialize_from_current: bool = True,
) -> dict[str, int | float | str | list[int]]:
    if input_embedding_dim <= 0:
        raise ValueError("--input-embedding-dim must be a positive integer.")
    if reduction_method not in {"pca", "vae"}:
        raise ValueError("--input-embedding-reduction must be either 'pca' or 'vae'.")

    active_embedding = _get_active_input_embedding(model)
    if not isinstance(active_embedding, nn.Embedding):
        raise ValueError(
            "The current model input embedding is not an nn.Embedding instance. "
            "This adapter currently supports standard Hugging Face token embeddings."
        )

    hidden_size = _infer_hidden_size(model)
    resolved_decoder_hidden_dims = decoder_hidden_dims
    if resolved_decoder_hidden_dims is None:
        if reduction_method == "pca":
            resolved_decoder_hidden_dims = []
        else:
            resolved_decoder_hidden_dims = [_default_vae_hidden_dim(input_embedding_dim, hidden_size)]

    adapter = ReducedInputEmbedding(
        num_embeddings=active_embedding.num_embeddings,
        reduced_dim=input_embedding_dim,
        output_dim=hidden_size,
        padding_idx=active_embedding.padding_idx,
        decoder_hidden_dims=resolved_decoder_hidden_dims,
    )
    adapter = adapter.to(device=active_embedding.weight.device, dtype=active_embedding.weight.dtype)

    metadata: dict[str, int | float | str | list[int]] = {
        "reduction_method": reduction_method,
        "decoder_hidden_dims": list(resolved_decoder_hidden_dims),
    }
    if initialize_from_current:
        metadata.update(
            adapter.initialize_from_pretrained(
                active_embedding.weight,
                reduction_method=reduction_method,
                vae_hidden_dim=resolved_decoder_hidden_dims[0] if reduction_method == "vae" else None,
            )
        )

    _set_input_embeddings(model, adapter)
    return metadata


def _get_active_input_embedding(model) -> nn.Module:
    for host, attr_name in _find_embedding_hosts(model):
        module = getattr(host, attr_name, None)
        if module is not None:
            return module

    if hasattr(model, "get_input_embeddings"):
        module = model.get_input_embeddings()
        if module is not None:
            return module

    raise ValueError("Unable to find encoder/decoder input embeddings for this model.")


def _set_input_embeddings(model, adapter: nn.Module) -> None:
    if hasattr(model, "set_input_embeddings"):
        model.set_input_embeddings(adapter)
        return

    hosts = _find_embedding_hosts(model)
    if not hosts:
        raise ValueError(
            "Unable to replace model input embeddings. Expected encoder/decoder embed_tokens attributes."
        )

    for host, attr_name in hosts:
        setattr(host, attr_name, adapter)


def _find_embedding_hosts(model) -> list[tuple[object, str]]:
    hosts: list[tuple[object, str]] = []
    seen: set[tuple[int, str]] = set()

    candidates: list[object] = [model]
    inner_model = getattr(model, "model", None)
    if inner_model is not None:
        candidates.append(inner_model)

    encoder = getattr(model, "get_encoder", None)
    decoder = getattr(model, "get_decoder", None)
    if callable(encoder):
        candidates.append(encoder())
    if callable(decoder):
        candidates.append(decoder())

    for candidate in candidates:
        if candidate is None:
            continue
        nested_encoder = getattr(candidate, "encoder", None)
        nested_decoder = getattr(candidate, "decoder", None)
        if nested_encoder is not None:
            key = (id(nested_encoder), "embed_tokens")
            if hasattr(nested_encoder, "embed_tokens") and key not in seen:
                hosts.append((nested_encoder, "embed_tokens"))
                seen.add(key)
        if nested_decoder is not None:
            key = (id(nested_decoder), "embed_tokens")
            if hasattr(nested_decoder, "embed_tokens") and key not in seen:
                hosts.append((nested_decoder, "embed_tokens"))
                seen.add(key)
        if hasattr(candidate, "embed_tokens"):
            key = (id(candidate), "embed_tokens")
            if key not in seen:
                hosts.append((candidate, "embed_tokens"))
                seen.add(key)

    return hosts


def _infer_hidden_size(model) -> int:
    for attribute_name in ("d_model", "hidden_size", "dim"):
        value = getattr(model.config, attribute_name, None)
        if isinstance(value, int) and value > 0:
            return value
    active_embedding = _get_active_input_embedding(model)
    if hasattr(active_embedding, "embedding_dim"):
        return int(active_embedding.embedding_dim)
    raise ValueError("Unable to infer the model hidden size from the model configuration.")


def _sample_rows(weight: torch.Tensor, max_samples: int) -> torch.Tensor:
    if max_samples <= 0 or max_samples >= weight.shape[0]:
        return weight
    indices = torch.randperm(weight.shape[0])[:max_samples]
    return weight[indices]


def _copy_decoder_weights(source_decoder: nn.Sequential, target_decoder: nn.Sequential) -> None:
    source_linear_layers = [layer for layer in source_decoder if isinstance(layer, nn.Linear)]
    target_linear_layers = [layer for layer in target_decoder if isinstance(layer, nn.Linear)]
    if len(source_linear_layers) != len(target_linear_layers):
        raise ValueError("Source and target decoder architectures do not match for VAE initialization.")

    for source_layer, target_layer in zip(source_linear_layers, target_linear_layers):
        target_layer.weight.copy_(source_layer.weight.to(dtype=target_layer.weight.dtype, device=target_layer.weight.device))
        if target_layer.bias is not None and source_layer.bias is not None:
            target_layer.bias.copy_(source_layer.bias.to(dtype=target_layer.bias.dtype, device=target_layer.bias.device))


def _default_vae_hidden_dim(latent_dim: int, output_dim: int) -> int:
    return max(latent_dim, min(output_dim, max(256, latent_dim * 2)))


def _persist_adapter_metadata(
    model_config,
    *,
    input_embedding_dim: int,
    reduction_method: str,
    base_model_name_or_path: str,
    decoder_hidden_dims: list[int],
    reduction_metadata: dict[str, int | float | str | list[int]],
) -> None:
    model_config.input_embedding_dim = input_embedding_dim
    model_config.input_embedding_reduction = reduction_method
    model_config.input_embedding_adapter = "reduced_embedding_decoder"
    model_config.input_embedding_base_model_name_or_path = base_model_name_or_path
    model_config.input_embedding_decoder_hidden_dims = decoder_hidden_dims

    for key, value in reduction_metadata.items():
        setattr(model_config, f"input_embedding_{key}", value)
