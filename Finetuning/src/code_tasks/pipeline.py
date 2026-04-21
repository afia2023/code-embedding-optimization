from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed

from .config import PipelineConfig
from .datasets import PreparedDatasets, load_text_datasets
from .metrics import build_compute_metrics, decode_predictions_and_labels
from .models import configure_runtime_device, load_tokenizer_and_model, resolve_inference_device
from .timing import PhaseTimer, TimingCallback, TimingRecord

logger = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig) -> None:
    set_seed(config.seed)
    config.output_path.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_path / "resolved_config.json", config.to_dict())

    timing = TimingRecord()

    # --- Model loading (timed) ---
    with PhaseTimer() as t:
        tokenizer, model = load_tokenizer_and_model(config)
    timing.model_load_seconds = t.elapsed
    logger.info("Model loaded in %.2fs", t.elapsed)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: total=%s, trainable=%s", f"{n_params:,}", f"{n_trainable:,}")

    trainer = None
    prepared: PreparedDatasets | None = None
    tokenized: dict[str, Dataset] = {}

    if any([config.do_train, config.do_eval, config.do_test]):
        # --- Data loading (timed) ---
        with PhaseTimer() as t:
            prepared = load_text_datasets(config)
        timing.data_load_seconds = t.elapsed
        logger.info("Data loaded in %.2fs", t.elapsed)

        # --- Tokenization (timed) ---
        with PhaseTimer() as t:
            tokenized = _tokenize_datasets(prepared, tokenizer, config)
        timing.tokenization_seconds = t.elapsed
        logger.info("Tokenization done in %.2fs", t.elapsed)

        training_args = _build_training_args(config)
        compute_metrics = build_compute_metrics(tokenizer)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

        timing_callback = TimingCallback(timing)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized.get("train"),
            eval_dataset=tokenized.get("validation"),
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if (config.do_eval or config.do_test) else None,
            callbacks=[timing_callback],
        )

    # --- Training (timed via callback + wall clock) ---
    if config.do_train and trainer is not None:
        train_start = time.perf_counter()
        train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
        train_wall = time.perf_counter() - train_start

        # Use callback timing if available, otherwise wall clock
        if timing.total_train_wall_clock_seconds == 0.0:
            timing.total_train_wall_clock_seconds = train_wall

        trainer.save_model()
        trainer.save_state()
        train_result.metrics["train_wall_clock_seconds"] = round(train_wall, 3)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        logger.info("Training completed in %.2fs", train_wall)

    # --- Evaluation (timed) ---
    if config.do_eval and trainer is not None:
        with PhaseTimer() as t:
            eval_metrics = trainer.evaluate(
                eval_dataset=tokenized["validation"],
                metric_key_prefix="validation",
                max_length=config.generation_max_length,
                num_beams=config.num_beams,
            )
        eval_metrics["validation_wall_clock_seconds"] = round(t.elapsed, 3)
        # NOTE: do not add to timing.total_eval_seconds here — the TimingCallback
        # already captures this via on_evaluate, which fires for trainer.evaluate() too.
        trainer.log_metrics("validation", eval_metrics)
        trainer.save_metrics("validation", eval_metrics)
        logger.info("Evaluation completed in %.2fs", t.elapsed)

    # --- Testing (timed) ---
    if config.do_test and trainer is not None and prepared is not None:
        with PhaseTimer() as t:
            test_output = trainer.predict(
                test_dataset=tokenized["test"],
                metric_key_prefix="test",
                max_length=config.generation_max_length,
                num_beams=config.num_beams,
            )
        timing.total_test_seconds = t.elapsed
        test_output.metrics["test_wall_clock_seconds"] = round(t.elapsed, 3)
        trainer.log_metrics("test", test_output.metrics)
        trainer.save_metrics("test", test_output.metrics)
        _save_predictions(
            config.output_path / "test_predictions.jsonl",
            prepared.text_splits["test"],
            test_output.predictions,
            test_output.label_ids,
            tokenizer,
        )
        logger.info("Testing completed in %.2fs", t.elapsed)

    if config.predict_text:
        generated_text = _run_single_prediction(model, tokenizer, config)
        _write_json(config.output_path / "single_prediction.json", {"prediction": generated_text})
        print(generated_text)

    # --- Save timing report ---
    timing.save(config.output_path / "timing.json")
    print(timing.summary())


def _tokenize_datasets(
    prepared: PreparedDatasets,
    tokenizer,
    config: PipelineConfig,
) -> dict[str, Dataset]:
    tokenized: dict[str, Dataset] = {}

    def tokenize_batch(batch: dict) -> dict:
        model_inputs = tokenizer(
            batch["source_text"],
            max_length=config.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=config.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    for split_name, dataset in prepared.text_splits.items():
        tokenized[split_name] = dataset.map(
            tokenize_batch,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split_name}",
        )

    return tokenized


def _build_training_args(config: PipelineConfig) -> Seq2SeqTrainingArguments:
    runtime_device_kwargs = configure_runtime_device(config)
    load_best_model = (
        config.do_train
        and config.do_eval
        and config.evaluation_strategy != "no"
        and config.save_strategy == config.evaluation_strategy
    )
    if config.do_train and config.do_eval and config.evaluation_strategy != "no" and config.save_strategy != config.evaluation_strategy:
        raise ValueError(
            "When load-best-model behavior is enabled, save and evaluation strategies must match. "
            "Set both to 'epoch' or both to 'steps'."
        )

    return Seq2SeqTrainingArguments(
        output_dir=str(config.output_path),
        overwrite_output_dir=config.overwrite_output_dir,
        do_train=config.do_train,
        do_eval=config.do_eval,
        do_predict=config.do_test,
        eval_strategy=config.evaluation_strategy if config.do_train else "no",
        save_strategy=config.save_strategy if config.do_train else "no",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        predict_with_generate=True,
        generation_max_length=config.generation_max_length,
        generation_num_beams=config.num_beams,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        load_best_model_at_end=load_best_model,
        metric_for_best_model=config.metric_for_best_model if load_best_model else None,
        report_to=config.report_to,
        seed=config.seed,
        save_safetensors=False,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,
        group_by_length=True,
        **runtime_device_kwargs,
    )


def _run_single_prediction(model, tokenizer, config: PipelineConfig) -> str:
    device = resolve_inference_device(config.device)
    model.to(device)
    model.eval()
    prompt = _build_single_prediction_prompt(config)
    model_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_input_length,
    )
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
    with torch.no_grad():
        generated_tokens = model.generate(
            **model_inputs,
            max_length=config.generation_max_length,
            num_beams=config.num_beams,
        )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()


def _build_single_prediction_prompt(config: PipelineConfig) -> str:
    text = config.predict_text or ""
    if config.task == "summarization":
        prompt = f"Summarize the following {config.language} method:\n{text}"
    else:
        prompt = f"Generate a {config.language} method from the following description:\n{text}"
    if config.source_prefix:
        prompt = f"{config.source_prefix}{prompt}"
    return prompt


def _save_predictions(
    output_path: Path,
    text_dataset: Dataset,
    predictions,
    labels,
    tokenizer,
) -> None:
    predictions_text, labels_text = decode_predictions_and_labels(predictions, labels, tokenizer)
    rows = []
    for source_text, prediction, reference in zip(
        text_dataset["source_text"],
        predictions_text,
        labels_text,
    ):
        rows.append(
            {
                "source_text": source_text,
                "prediction": prediction,
                "reference": reference,
            }
        )
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
