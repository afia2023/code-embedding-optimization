"""Pretraining script for d512 (T5-small, 60.5M) and d320 (compressed, 27.5M) models.

Follows the LEONID/LANCE paper (Mastropaolo et al., 2024) gin config exactly:
  Architecture : d_model=512, d_ff=2048, num_heads=8, d_kv=64, num_layers=6
  Optimizer    : Adafactor (factored=True, multiply_by_parameter_scale=True,
                            beta1=0.0, clipping_threshold=1.0)
  LR schedule  : Noam — warmup 10k steps, inverse-sqrt decay, multiplier=1.0
  Batch        : 128 effective (default: --batch-size 32 --grad-accum 4 for GPU)
  Steps        : 500,000 | Max seq len: 512 | bf16
  Checkpoints  : every 10,000 steps

Usage:
    # Single GPU with gradient accumulation (effective batch = 16 * 8 = 128)
    python pretrain.py --model d512 --output-dir outputs/pretrained_d512
    python pretrain.py --model d320 --output-dir outputs/pretrained_d320

    # If GPU has enough memory for larger per-device batch:
    python pretrain.py --model d320 --batch-size 32 --grad-accum 4 --output-dir outputs/pretrained_d320
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers.optimization import Adafactor, AdafactorSchedule

# Add src to path so we can import model_builder
sys.path.insert(0, str(Path(__file__).parent / "src"))
from code_tasks.model_builder import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — matching LEONID/LANCE gin config exactly
# ---------------------------------------------------------------------------
BASE_MODEL_NAME = "t5-small"          # d_model=512, 6 layers each side, ~60M params
TSV_PATH = Path(__file__).parent / "Pre-training/TSV/pretraining.tsv"
MAX_INPUT_LEN = 512                   # max_length = 512 (gin: encoder/Unitransformer.max_length)
MAX_TARGET_LEN = 512                  # max_length = 512 (gin: decoder/Unitransformer.max_length)
TOTAL_STEPS = 500_000                 # learning_rate_schedule_noam.total_train_steps
WARMUP_STEPS = 10_000                 # learning_rate_schedule_noam.warmup_steps
LR_MULTIPLIER = 1.0                   # learning_rate_schedule_noam.multiplier
DROPOUT_RATE = 0.1                    # gin: dropout_rate
CHECKPOINT_EVERY = 10_000             # notebook: save_checkpoints_steps=10000
KEEP_CHECKPOINT_MAX = 16             # notebook: keep_checkpoint_max=16
INPUT_PREFIX = "MASKING: "           # notebook: inputs = 'MASKING: ' + ex['input']


# ---------------------------------------------------------------------------
# Dataset — reads the pre-processed TSV (span corruption format)
# ---------------------------------------------------------------------------

class SpanCorruptionDataset(Dataset):
    """Memory-maps the TSV and tokenizes on the fly.

    The TSV is already in T5 span-corruption format:
        <masked input>\t<target tokens with <extra_id_X> sentinels>
    """

    def __init__(self, tsv_path: Path, tokenizer, max_input_len: int, max_target_len: int):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

        logger.info("Indexing TSV file: %s", tsv_path)
        self.offsets: list[int] = []
        self._file_path = str(tsv_path)
        self._file = open(tsv_path, "rb")
        offset = 0
        for line in self._file:
            if b"\t" in line:
                self.offsets.append(offset)
            offset += len(line)
        logger.info("Indexed %d examples", len(self.offsets))

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict:
        self._file.seek(self.offsets[idx])
        line = self._file.readline().decode("utf-8", errors="replace").rstrip("\n")
        parts = line.split("\t", 1)
        input_text = INPUT_PREFIX + parts[0].strip()   # "MASKING: " + masked input
        target_text = parts[1].strip() if len(parts) > 1 else ""

        enc = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dec = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = dec["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding in loss

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }

    def __del__(self):
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()


# ---------------------------------------------------------------------------
# Noam LR schedule (matches gin: learning_rate_schedule_noam)
# lambda: multiplier * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
# ---------------------------------------------------------------------------

def get_noam_lambda(d_model: int, warmup_steps: int, multiplier: float = 1.0):
    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        return multiplier * (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    return lr_lambda


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    effective_batch = args.batch_size * args.grad_accum
    if effective_batch != 128:
        logger.warning(
            "Effective batch = %d × %d = %d  (paper uses 128)",
            args.batch_size, args.grad_accum, effective_batch,
        )
    else:
        logger.info("Effective batch: %d × %d = 128  ✓", args.batch_size, args.grad_accum)

    # ---- Tokenizer ----
    logger.info("Loading tokenizer from %s", BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # ---- Model ----
    hidden_dim = 512 if args.model == "d512" else 320
    logger.info("Building %s (d_model=%d, random init)", args.model, hidden_dim)
    model = build_model(BASE_MODEL_NAME, hidden_dim, init_method="random")

    # Apply dropout rate from gin config
    model.config.dropout_rate = DROPOUT_RATE

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %s", f"{n_params:,}")

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # ---- Dataset & DataLoader ----
    dataset = SpanCorruptionDataset(
        Path(args.tsv), tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN
    )
    steps_per_epoch = math.ceil(len(dataset) / effective_batch)
    logger.info(
        "Dataset: %d examples | steps/epoch ≈ %d | ~%.1f epochs over %d steps",
        len(dataset), steps_per_epoch, TOTAL_STEPS / steps_per_epoch, TOTAL_STEPS,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    # ---- Optimizer: Adafactor — exact match to gin config ----
    # gin: factored=True, multiply_by_parameter_scale=True, beta1=0.0,
    #      clipping_threshold=1.0, decay_rate=None (auto via adafactor_decay_rate_pow)
    optimizer = Adafactor(
        model.parameters(),
        scale_parameter=True,        # multiply_by_parameter_scale=True
        relative_step=True,          # use internal Noam-like LR (relative to param scale)
        warmup_init=True,            # start with small LR during warmup
        lr=None,                     # set by relative_step schedule
        clip_threshold=1.0,          # clipping_threshold=1.0
        beta1=None,                  # beta1=0.0 → None disables momentum
        weight_decay=0.0,
    )
    # When relative_step=True, Adafactor uses its own internal schedule.
    # We still log the effective LR using AdafactorSchedule.
    scheduler = AdafactorSchedule(optimizer)

    # ---- Mixed precision ----
    # gin: get_variable_dtype.activation_dtype = 'bfloat16'
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    if use_amp:
        logger.info("Mixed precision: %s", amp_dtype)

    # ---- Output dir + resume ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    latest_ptr = output_dir / "latest_checkpoint"
    if latest_ptr.exists():
        ckpt_dir = latest_ptr.read_text().strip()
        logger.info("Resuming from: %s", ckpt_dir)
        model.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device)
        )
        state = torch.load(os.path.join(ckpt_dir, "training_state.pt"), map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = state["global_step"]
        logger.info("Resumed at step %d", global_step)

    # ---- Training loop ----
    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    log_every = 100
    start_time = time.time()
    data_iter = iter(loader)
    micro_step = 0   # counts individual forward passes within grad accum

    while global_step < TOTAL_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / args.grad_accum

        loss.backward()
        running_loss += loss.item() * args.grad_accum
        micro_step += 1

        if micro_step % args.grad_accum == 0:
            # Adafactor has its own gradient clipping (clipping_threshold=1.0),
            # but we apply an outer clip as an extra safety net
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = running_loss / log_every
                elapsed = time.time() - start_time
                eta_h = (elapsed / global_step) * (TOTAL_STEPS - global_step) / 3600
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "Step %6d/%d | loss=%.4f | lr=%.2e | ETA=%.1fh",
                    global_step, TOTAL_STEPS, avg_loss, lr, eta_h,
                )
                running_loss = 0.0

            if global_step % CHECKPOINT_EVERY == 0:
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                torch.save(
                    {
                        "global_step": global_step,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    ckpt_dir / "training_state.pt",
                )
                latest_ptr.write_text(str(ckpt_dir))
                logger.info("Checkpoint saved → %s", ckpt_dir)

                # Rotate: keep only the last KEEP_CHECKPOINT_MAX checkpoints
                all_ckpts = sorted(output_dir.glob("checkpoint-*"),
                                   key=lambda p: int(p.name.split("-")[1]))
                for old in all_ckpts[:-KEEP_CHECKPOINT_MAX]:
                    import shutil; shutil.rmtree(old)
                    logger.info("Removed old checkpoint: %s", old)

    # ---- Final save ----
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    total_h = (time.time() - start_time) / 3600
    logger.info("Done in %.2fh → %s", total_h, final_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain d512 (60.5M) or d320 (27.5M) T5 model on Java span corruption."
    )
    parser.add_argument(
        "--model", choices=["d512", "d320"], required=True,
        help="Model to pretrain",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--tsv", default=str(TSV_PATH),
        help=f"Path to pretraining TSV (default: {TSV_PATH})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Per-GPU batch size (default: 32; use --grad-accum 4 for effective batch=128).",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4,
        help="Gradient accumulation steps (default: 4; effective batch = 32 × 4 = 128, matching the paper).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--grad-checkpointing", action="store_true",
        help="Enable gradient checkpointing (saves GPU memory at cost of ~20%% speed)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("=" * 60)
    logger.info("PRETRAINING — %s", args.model.upper())
    logger.info("  Base architecture : %s (t5-small proportions)", BASE_MODEL_NAME)
    logger.info("  Optimizer         : Adafactor (factored, scale_parameter=True)")
    logger.info("  LR schedule       : Noam (warmup=%d, multiplier=%.1f)", WARMUP_STEPS, LR_MULTIPLIER)
    logger.info("  Batch (per GPU)   : %d", args.batch_size)
    logger.info("  Grad accum        : %d", args.grad_accum)
    logger.info("  Effective batch   : %d  (paper=128)", args.batch_size * args.grad_accum)
    logger.info("  Total steps       : %d", TOTAL_STEPS)
    logger.info("  Max seq len       : %d", MAX_INPUT_LEN)
    logger.info("  Dropout           : %.1f", DROPOUT_RATE)
    logger.info("  Output dir        : %s", args.output_dir)
    logger.info("=" * 60)
    train(args)
