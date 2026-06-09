from __future__ import annotations

from typing import Iterable

import numpy as np
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from sacrebleu.metrics import CHRF


def _normalize_decoded_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def decode_predictions_and_labels(predictions, labels, tokenizer) -> tuple[list[str], list[str]]:
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1).astype(np.int32)
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    predictions_text = [_normalize_decoded_text(text) for text in decoded_predictions]
    labels_text = [_normalize_decoded_text(text) for text in decoded_labels]
    return predictions_text, labels_text


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def build_compute_metrics(tokenizer):
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    chrf = CHRF(word_order=2)

    def compute_metrics(eval_preds) -> dict[str, float]:
        predictions, labels = eval_preds
        predictions_text, labels_text = decode_predictions_and_labels(predictions, labels, tokenizer)

        if not predictions_text:
            return {
                "bleu": 0.0,
                "chrf": 0.0,
                "exact_match": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "gen_len": 0.0,
            }

        bleu = corpus_bleu(predictions_text, [labels_text]).score
        chrf_score = chrf.corpus_score(predictions_text, [labels_text]).score

        rouge_scores = [rouge.score(reference, prediction) for prediction, reference in zip(predictions_text, labels_text)]
        exact_match = 100.0 * _mean(
            float(prediction.strip() == reference.strip())
            for prediction, reference in zip(predictions_text, labels_text)
        )
        generation_lengths = [
            len(tokenizer.encode(prediction, add_special_tokens=False)) for prediction in predictions_text
        ]

        return {
            "bleu": round(float(bleu), 4),
            "chrf": round(float(chrf_score), 4),
            "exact_match": round(float(exact_match), 4),
            "rouge1": round(_mean(score["rouge1"].fmeasure for score in rouge_scores) * 100.0, 4),
            "rouge2": round(_mean(score["rouge2"].fmeasure for score in rouge_scores) * 100.0, 4),
            "rougeL": round(_mean(score["rougeL"].fmeasure for score in rouge_scores) * 100.0, 4),
            "gen_len": round(_mean(generation_lengths), 4),
        }

    return compute_metrics
