from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ALLOWED_TASKS = {"summarization", "generation"}
ALLOWED_LANGUAGES = {"java", "python"}
ALLOWED_STRATEGIES = {"no", "steps", "epoch"}


@dataclass
class PipelineConfig:
    task: str
    language: str
    model_name_or_path: str
    dataset: str
    output_dir: str
    dataset_config: str | None = None
    source_column: str | None = None
    target_column: str | None = None
    train_split: str | None = None
    validation_split: str | None = None
    test_split: str | None = None
    max_input_length: int = 512
    max_target_length: int = 128
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    num_train_epochs: float = 3.0
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    early_stopping_patience: int | None = None
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    eval_steps: int | None = None
    save_steps: int | None = None
    logging_steps: int = 50
    save_total_limit: int = 2
    num_beams: int = 4
    generation_max_length: int | None = None
    do_train: bool = False
    do_eval: bool = False
    do_test: bool = False
    predict_text: str | None = None
    seed: int = 42
    device: str = "auto"
    overwrite_output_dir: bool = False
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    trust_remote_code: bool = False
    cache_dir: str | None = None
    preprocessing_num_workers: int | None = None
    dataloader_num_workers: int = 0
    input_embedding_dim: int | None = None
    input_embedding_reduction: str = "pca"
    hidden_dim: int | None = None
    init_method: str = "random"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_test_samples: int | None = None
    resume_from_checkpoint: str | None = None
    metric_for_best_model: str = "bleu"
    report_to: str = "none"
    source_prefix: str | None = None

    @classmethod
    def from_namespace(cls, namespace: Any) -> "PipelineConfig":
        raw_values = dict(vars(namespace))
        raw_values.pop("list_datasets", None)
        config = cls(**raw_values)
        config.finalize()
        config.validate()
        return config

    def finalize(self) -> None:
        self.task = self.task.lower().strip()
        self.language = self.language.lower().strip()
        self.dataset = self.dataset.strip()
        self.model_name_or_path = self.model_name_or_path.strip()
        self.output_dir = str(Path(self.output_dir))
        self.device = self.device.lower().strip()
        if self.resume_from_checkpoint is not None:
            self.resume_from_checkpoint = self.resume_from_checkpoint.strip() or None

        if self.generation_max_length is None:
            self.generation_max_length = self.max_target_length
        if self.eval_steps is None and self.evaluation_strategy == "steps":
            self.eval_steps = 500
        if self.save_steps is None and self.save_strategy == "steps":
            self.save_steps = self.eval_steps or 500

    def validate(self) -> None:
        if self.task not in ALLOWED_TASKS:
            raise ValueError(f"Unsupported task '{self.task}'. Choose from {sorted(ALLOWED_TASKS)}.")
        if self.language not in ALLOWED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{self.language}'. Choose from {sorted(ALLOWED_LANGUAGES)}."
            )
        if self.evaluation_strategy not in ALLOWED_STRATEGIES:
            raise ValueError(
                f"Unsupported evaluation strategy '{self.evaluation_strategy}'. "
                f"Choose from {sorted(ALLOWED_STRATEGIES)}."
            )
        if self.save_strategy not in ALLOWED_STRATEGIES:
            raise ValueError(
                f"Unsupported save strategy '{self.save_strategy}'. Choose from {sorted(ALLOWED_STRATEGIES)}."
            )
        if not any([self.do_train, self.do_eval, self.do_test, self.predict_text]):
            raise ValueError("Nothing to do. Enable at least one of --do-train, --do-eval, --do-test, or --predict-text.")
        if self.max_input_length <= 0 or self.max_target_length <= 0:
            raise ValueError("Maximum input and target lengths must be positive integers.")
        if self.per_device_train_batch_size <= 0 or self.per_device_eval_batch_size <= 0:
            raise ValueError("Batch sizes must be positive integers.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if self.num_train_epochs <= 0:
            raise ValueError("Number of epochs must be positive.")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("Warmup ratio must be between 0.0 and 1.0.")
        if self.eval_steps is not None and self.eval_steps <= 0:
            raise ValueError("Evaluation steps must be positive when provided.")
        if self.save_steps is not None and self.save_steps <= 0:
            raise ValueError("Save steps must be positive when provided.")
        if self.save_total_limit is not None and self.save_total_limit <= 0:
            raise ValueError("Save total limit must be positive when provided.")
        if self.num_beams <= 0:
            raise ValueError("Number of beams must be positive.")
        if self.generation_max_length is not None and self.generation_max_length <= 0:
            raise ValueError("Generation max length must be positive when provided.")
        if self.preprocessing_num_workers is not None and self.preprocessing_num_workers <= 0:
            raise ValueError("Preprocessing worker count must be positive when provided.")
        if self.input_embedding_dim is not None and self.input_embedding_dim <= 0:
            raise ValueError("Input embedding dimension must be positive when provided.")
        if self.input_embedding_reduction not in {"pca", "vae"}:
            raise ValueError("Input embedding reduction must be either 'pca' or 'vae'.")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("Hidden dimension must be positive when provided.")
        if self.init_method not in {"random", "pca", "pretrained"}:
            raise ValueError("Init method must be 'random', 'pca', or 'pretrained'.")
        for split_name, value in {
            "max_train_samples": self.max_train_samples,
            "max_eval_samples": self.max_eval_samples,
            "max_test_samples": self.max_test_samples,
        }.items():
            if value is not None and value <= 0:
                raise ValueError(f"{split_name} must be positive when provided.")
        if self.do_train and self.evaluation_strategy != "no" and not self.do_eval:
            raise ValueError(
                "Training with evaluation enabled requires --do-eval so the trainer has a validation dataset."
            )
        if self.do_train and self.fp16 and self.bf16:
            raise ValueError("Choose at most one of --fp16 and --bf16.")
        if self.do_train and self.save_strategy == "steps" and self.save_steps is None:
            raise ValueError("Save strategy 'steps' requires --save-steps.")
        if self.do_train and self.evaluation_strategy == "steps" and self.eval_steps is None:
            raise ValueError("Evaluation strategy 'steps' requires --eval-steps.")
        if self.do_train and self.do_eval and self.metric_for_best_model.strip() == "":
            raise ValueError("Metric for best model cannot be empty.")

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
