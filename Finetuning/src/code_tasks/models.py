from __future__ import annotations

import inspect

import torch
from transformers import AutoTokenizer, Seq2SeqTrainingArguments

from .config import PipelineConfig
from .embedding_adapters import load_model_with_optional_input_embedding_adapter
from .model_builder import build_model as build_architectural_model


def configure_runtime_device(config: PipelineConfig) -> dict[str, bool]:
    device = config.device
    training_arg_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    runtime_kwargs: dict[str, bool] = {}

    if device == "auto":
        return runtime_kwargs
    if device == "cpu":
        if "use_cpu" in training_arg_params:
            runtime_kwargs["use_cpu"] = True
        else:
            runtime_kwargs["no_cuda"] = True
        return runtime_kwargs
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available on this machine.")
        return runtime_kwargs
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but no CUDA device is available.")
        if ":" in device:
            _, index_text = device.split(":", maxsplit=1)
            if not index_text.isdigit():
                raise ValueError(f"Invalid CUDA device '{device}'. Expected 'cuda' or 'cuda:N'.")
            index = int(index_text)
            if index < 0 or index >= torch.cuda.device_count():
                raise ValueError(
                    f"Requested CUDA device index {index} is out of range for {torch.cuda.device_count()} visible devices."
                )
            torch.cuda.set_device(index)
        return runtime_kwargs

    raise ValueError("Unsupported device. Choose from auto, cpu, mps, cuda, or cuda:N.")


def resolve_inference_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer_and_model(config: PipelineConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.cache_dir,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Full architectural modification path (--hidden-dim)
    if config.hidden_dim is not None:
        model = build_architectural_model(
            base_model_name=config.model_name_or_path,
            hidden_dim=config.hidden_dim,
            init_method=config.init_method,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
        )
        if tokenizer.pad_token_id is not None:
            current_vocab = model.get_input_embeddings().weight.shape[0]
            if len(tokenizer) > current_vocab:
                model.resize_token_embeddings(len(tokenizer))
        if model.config.decoder_start_token_id is None and tokenizer.pad_token_id is not None:
            model.config.decoder_start_token_id = tokenizer.pad_token_id
        return tokenizer, model

    # Legacy projection-based path (--input-embedding-dim)
    model, adapted_input_embedding_dim = load_model_with_optional_input_embedding_adapter(
        config.model_name_or_path,
        cache_dir=config.cache_dir,
        trust_remote_code=config.trust_remote_code,
        tokenizer_length=len(tokenizer),
        requested_input_embedding_dim=config.input_embedding_dim,
        requested_reduction_method=config.input_embedding_reduction,
    )

    if model.config.decoder_start_token_id is None and tokenizer.pad_token_id is not None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id
    if adapted_input_embedding_dim is not None and getattr(model.config, "input_embedding_dim", None) is None:
        model.config.input_embedding_dim = adapted_input_embedding_dim

    return tokenizer, model
