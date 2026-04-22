"""Training time tracking utilities.

Provides a timer context manager and a callback for the HuggingFace Trainer
that records wall-clock time at each training phase.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


@dataclass
class TimingRecord:
    """Accumulated wall-clock timing for a training run."""

    total_train_wall_clock_seconds: float = 0.0  # wall-clock: train + eval combined
    total_eval_seconds: float = 0.0
    total_test_seconds: float = 0.0
    epoch_seconds: list[float] = field(default_factory=list)
    eval_seconds: list[float] = field(default_factory=list)
    data_load_seconds: float = 0.0
    model_load_seconds: float = 0.0
    tokenization_seconds: float = 0.0

    # Internal bookkeeping (not serialised)
    _train_start: float = field(default=0.0, repr=False)
    _epoch_start: float = field(default=0.0, repr=False)
    _eval_start: float = field(default=0.0, repr=False)

    def to_dict(self) -> dict:
        return {
            "total_train_wall_clock_seconds": round(self.total_train_wall_clock_seconds, 3),
            "total_eval_seconds": round(self.total_eval_seconds, 3),
            "pure_training_seconds": round(self.pure_training_seconds, 3),
            "total_test_seconds": round(self.total_test_seconds, 3),
            "epoch_seconds": [round(e, 3) for e in self.epoch_seconds],
            "eval_seconds": [round(e, 3) for e in self.eval_seconds],
            "data_load_seconds": round(self.data_load_seconds, 3),
            "model_load_seconds": round(self.model_load_seconds, 3),
            "tokenization_seconds": round(self.tokenization_seconds, 3),
            "total_pipeline_seconds": round(self.total_pipeline_seconds, 3),
        }

    @property
    def pure_training_seconds(self) -> float:
        """Wall-clock training time minus evaluation time = pure training steps only."""
        return self.total_train_wall_clock_seconds - self.total_eval_seconds

    @property
    def total_pipeline_seconds(self) -> float:
        return (
            self.model_load_seconds
            + self.data_load_seconds
            + self.tokenization_seconds
            + self.pure_training_seconds
            + self.total_eval_seconds
            + self.total_test_seconds
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def summary(self) -> str:
        lines = [
            "=== Timing Summary ===",
            f"  Model loading:    {self.model_load_seconds:>10.2f}s",
            f"  Data loading:     {self.data_load_seconds:>10.2f}s",
            f"  Tokenization:     {self.tokenization_seconds:>10.2f}s",
            f"  Train wall-clock: {self.total_train_wall_clock_seconds:>10.2f}s",
            f"  Pure training:    {self.pure_training_seconds:>10.2f}s",
        ]
        for i, ep in enumerate(self.epoch_seconds, 1):
            lines.append(f"    Epoch {i}:        {ep:>10.2f}s")
        lines.append(f"  Evaluation total: {self.total_eval_seconds:>10.2f}s")
        for i, ev in enumerate(self.eval_seconds, 1):
            lines.append(f"    Eval {i}:         {ev:>10.2f}s")
        if self.total_test_seconds > 0:
            lines.append(f"  Testing:          {self.total_test_seconds:>10.2f}s")
        lines.append(f"  Pipeline total:   {self.total_pipeline_seconds:>10.2f}s")
        lines.append("======================")
        return "\n".join(lines)


class PhaseTimer:
    """Simple context manager that records elapsed time."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0

    def __enter__(self) -> "PhaseTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        self.elapsed = time.perf_counter() - self._start


class TimingCallback(TrainerCallback):
    """HuggingFace Trainer callback that populates a TimingRecord."""

    def __init__(self, record: TimingRecord) -> None:
        self.record = record
        self._in_eval = False

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.record._train_start = time.perf_counter()
        self.record._epoch_start = time.perf_counter()

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.record._epoch_start = time.perf_counter()

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        now = time.perf_counter()
        self.record.epoch_seconds.append(now - self.record._epoch_start)
        self.record._epoch_start = now

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # on_evaluate fires after evaluation completes
        if self.record._eval_start > 0.0:
            elapsed = time.perf_counter() - self.record._eval_start
            self.record.eval_seconds.append(elapsed)
            self.record.total_eval_seconds = sum(self.record.eval_seconds)
            self.record._eval_start = 0.0

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Mark eval start on the first prediction step
        if self.record._eval_start == 0.0:
            self.record._eval_start = time.perf_counter()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.record.total_train_wall_clock_seconds = time.perf_counter() - self.record._train_start
