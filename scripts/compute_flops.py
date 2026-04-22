"""
compute_flops.py
================
Estimates FLOPs for T5-style encoder-decoder models (code summarization).

Supports any saved experiment directory — reads batch size, epochs, and
total training steps dynamically from resolved_config.json and
trainer_state.json. Works for both batch=32 and batch=64 experiments.

──────────────────────────────────────────────────────────────────────
FORMULA (plain language)
──────────────────────────────────────────────────────────────────────

  per_sample_FLOPs
    Measured by calflops with batch=1 and the max sequence lengths used
    in training (enc=512, dec=128). This is a WORST-CASE estimate because
    real batches are padded to the longest sequence in that batch, not
    always 512. Real average cost is lower.

  per_step_FLOPs  (forward only)
    = per_sample_FLOPs × batch_size
    Each training step processes batch_size samples simultaneously.

  training_step_FLOPs  (forward + backward)
    = per_step_FLOPs × 3
    Backward pass ≈ 2× forward (standard ML approximation used in
    Chinchilla, PaLM, GPT-4 papers). So total per step = 3× forward.

  total_training_FLOPs
    = training_step_FLOPs × total_steps
    Equivalently = (3 × per_sample_FLOPs) × (total_steps × batch_size)
    = (3 × per_sample_FLOPs) × total_examples_seen_across_all_epochs

  inference_FLOPs  (1 sample, no gradient)
    = per_sample_FLOPs  (forward only, batch=1)

──────────────────────────────────────────────────────────────────────
USAGE
──────────────────────────────────────────────────────────────────────

  # Default: compares batch-64 experiments
  python scripts/compute_flops.py

  # Custom: pass any output directories
  python scripts/compute_flops.py \\
      --models "d512 Batch32=outputs/summarization_baseline_t5small_d512_random" \\
               "d320 Batch32=outputs/summarization_compressed_t5small_d320_random" \\
               "d512 Batch64=outputs/summarization_baseline_t5small_d512_batch64_lr3e4" \\
               "d320 Batch64=outputs/summarization_compressed_t5small_d320_batch64_lr3e4"
"""

import argparse
import json
import os
import sys
import torch
from calflops import calculate_flops
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ─────────────────────────────────────────────
# DEFAULT MODELS (batch-64 experiments)
# ─────────────────────────────────────────────

DEFAULT_MODELS = [
    ("d512 Baseline  (batch=64)", "outputs/summarization_baseline_t5small_d512_batch64_lr3e4"),
    ("d320 Compressed (batch=64)", "outputs/summarization_compressed_t5small_d320_batch64_lr3e4"),
]

# Test set size (CodeSearchNet Java)
TEST_SET_SIZE = 10954

# Backward pass multiplier (industry standard: backward ≈ 2× forward)
BACKWARD_MULT = 3

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

def gflops_to_float(s):
    """Parse calflops string '36.649 GFLOPS' → float GFLOPs."""
    val, unit = s.strip().split()
    val = float(val)
    unit = unit.upper()
    if "TFLOP" in unit:
        return val * 1000
    elif "GFLOP" in unit:
        return val
    elif "MFLOP" in unit:
        return val / 1000
    return val

def fmt(gflops):
    """Human-readable FLOPs."""
    if gflops >= 1e6:
        return f"{gflops/1e6:.3f} EFLOPs"
    elif gflops >= 1000:
        return f"{gflops/1000:.3f} TFLOPs"
    elif gflops >= 1:
        return f"{gflops:.3f} GFLOPs"
    else:
        return f"{gflops*1000:.3f} MFLOPs"

def read_run_config(model_path):
    """
    Dynamically read batch size, epochs, and total steps from saved files.
    Falls back gracefully if a file is missing.
    """
    info = {}

    # resolved_config.json → batch size, epochs, sequence lengths
    rc_path = os.path.join(model_path, "resolved_config.json")
    if os.path.exists(rc_path):
        rc = load_json(rc_path)
        info["batch_size"]     = rc.get("per_device_train_batch_size", None)
        info["epochs"]         = rc.get("num_train_epochs", None)
        info["enc_seq_len"]    = rc.get("max_input_length", 512)
        info["dec_seq_len"]    = rc.get("max_target_length", 128)
        info["learning_rate"]  = rc.get("learning_rate", None)
    else:
        print(f"  [WARN] resolved_config.json not found — using defaults")
        info["batch_size"] = None
        info["epochs"] = None
        info["enc_seq_len"] = 512
        info["dec_seq_len"] = 128
        info["learning_rate"] = None

    # trainer_state.json → total steps actually completed
    ts_path = os.path.join(model_path, "trainer_state.json")
    if os.path.exists(ts_path):
        ts = load_json(ts_path)
        info["total_steps"] = ts.get("global_step", None)
    else:
        print(f"  [WARN] trainer_state.json not found — total steps unknown")
        info["total_steps"] = None

    return info

def read_arch_config(model_path):
    """Read model architecture from config.json."""
    cfg = load_json(os.path.join(model_path, "config.json"))
    return {
        "d_model":    cfg["d_model"],
        "d_ff":       cfg["d_ff"],
        "d_kv":       cfg["d_kv"],
        "num_heads":  cfg["num_heads"],
        "num_layers": cfg["num_layers"],
        "vocab_size": cfg["vocab_size"],
    }

def pct_reduction(a, b):
    return (a - b) / a * 100

def compute_avg_seq_lens(model_path):
    """
    Tokenize source_text and reference from test_predictions.jsonl
    to get the real average encoder and decoder token lengths.
    These are used for the REALISTIC FLOPs estimate.
    Returns (avg_enc_len, avg_dec_len) rounded to nearest integer.
    """
    pred_path = os.path.join(model_path, "test_predictions.jsonl")
    if not os.path.exists(pred_path):
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    src_lens, tgt_lens = [], []
    with open(pred_path) as f:
        for line in f:
            item = json.loads(line)
            src_lens.append(len(tokenizer(item["source_text"], truncation=False)["input_ids"]))
            tgt_lens.append(len(tokenizer(item["reference"],   truncation=False)["input_ids"]))

    avg_enc = round(sum(src_lens) / len(src_lens))
    avg_dec = round(sum(tgt_lens) / len(tgt_lens))
    print(f"  Avg token lengths from test_predictions.jsonl ({len(src_lens)} samples):")
    print(f"    encoder (source) — mean={avg_enc}, median={sorted(src_lens)[len(src_lens)//2]}, max={max(src_lens)}")
    print(f"    decoder (target) — mean={avg_dec}, median={sorted(tgt_lens)[len(tgt_lens)//2]}, max={max(tgt_lens)}")
    return avg_enc, avg_dec

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run(models):
    all_results = {}

    for model_name, model_path in models:
        print(f"\n{'='*62}")
        print(f"  {model_name}")
        print(f"  Path: {model_path}")
        print(f"{'='*62}")

        # 1. Read configs from disk
        run_cfg  = read_run_config(model_path)
        arch_cfg = read_arch_config(model_path)

        enc_len    = run_cfg["enc_seq_len"]
        dec_len    = run_cfg["dec_seq_len"]
        batch_size = run_cfg["batch_size"]
        total_steps = run_cfg["total_steps"]
        epochs     = run_cfg["epochs"]
        lr         = run_cfg["learning_rate"]

        print(f"\n  Architecture (from config.json):")
        for k, v in arch_cfg.items():
            print(f"    {k:12s} = {v}")

        print(f"\n  Training config (from resolved_config.json + trainer_state.json):")
        print(f"    batch_size   = {batch_size}")
        print(f"    epochs       = {epochs}")
        print(f"    total_steps  = {total_steps}  (steps actually completed)")
        print(f"    learning_rate= {lr}")
        print(f"    enc_seq_len  = {enc_len}  (max_input_length — WORST CASE)")
        print(f"    dec_seq_len  = {dec_len}  (max_target_length — WORST CASE)")

        # 2. Load model
        print(f"\n  Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval()
        param_count = sum(p.numel() for p in model.parameters())

        # 3. Profile ONE SAMPLE forward pass with max sequence lengths
        #    Sequence lengths = worst case (real batches may be shorter due to padding)
        dummy = {
            "input_ids":         torch.ones((1, enc_len),  dtype=torch.long),
            "decoder_input_ids": torch.ones((1, dec_len), dtype=torch.long),
        }
        print(f"  Profiling forward pass (batch=1, enc={enc_len}, dec={dec_len})...")
        flops_str, macs_str, _ = calculate_flops(
            model=model,
            kwargs=dummy,
            output_as_string=True,
            output_precision=4,
            print_results=False,
            print_detailed=False,
        )
        per_sample_fwd_gflops = gflops_to_float(flops_str)
        per_sample_macs       = gflops_to_float(macs_str)

        # 3b. REALISTIC FLOPs — profile again using actual average sequence lengths
        #     measured from tokenizing real test data (not the padded max)
        print(f"\n  Computing average sequence lengths from real test data...")
        avg_enc, avg_dec = compute_avg_seq_lens(model_path)
        if avg_enc and avg_dec:
            dummy_avg = {
                "input_ids":         torch.ones((1, avg_enc), dtype=torch.long),
                "decoder_input_ids": torch.ones((1, avg_dec), dtype=torch.long),
            }
            print(f"  Profiling forward pass (batch=1, enc={avg_enc}, dec={avg_dec}) [REALISTIC]...")
            flops_avg_str, macs_avg_str, _ = calculate_flops(
                model=model,
                kwargs=dummy_avg,
                output_as_string=True,
                output_precision=4,
                print_results=False,
                print_detailed=False,
            )
            per_sample_fwd_avg_gflops = gflops_to_float(flops_avg_str)
        else:
            print(f"  [WARN] test_predictions.jsonl not found — skipping realistic estimate")
            per_sample_fwd_avg_gflops = None
            avg_enc, avg_dec = None, None

        # 4. Scale to per-step (multiply by real batch size)
        per_step_fwd_gflops = per_sample_fwd_gflops * batch_size

        # 5. Training step = forward + backward = 3× forward
        training_step_gflops = BACKWARD_MULT * per_step_fwd_gflops

        # 6. Total training = training_step × total_steps
        total_training_gflops = training_step_gflops * total_steps
        # Equivalently: 3 × per_sample × (total_steps × batch_size) = 3 × per_sample × examples_seen
        total_examples_seen = total_steps * batch_size

        # 7. Inference = per_sample forward only
        inference_per_sample_gflops = per_sample_fwd_gflops
        inference_test_set_gflops   = per_sample_fwd_gflops * TEST_SET_SIZE

        # Realistic totals (if available)
        if per_sample_fwd_avg_gflops:
            per_step_fwd_avg_gflops       = per_sample_fwd_avg_gflops * batch_size
            training_step_avg_gflops      = BACKWARD_MULT * per_step_fwd_avg_gflops
            total_training_avg_gflops     = training_step_avg_gflops * total_steps
            inference_test_set_avg_gflops = per_sample_fwd_avg_gflops * TEST_SET_SIZE
        else:
            per_step_fwd_avg_gflops = training_step_avg_gflops = None
            total_training_avg_gflops = inference_test_set_avg_gflops = None

        print(f"\n  ── FLOPs breakdown ──")
        print(f"  [WORST CASE — enc={enc_len}, dec={dec_len}]")
        print(f"  Per-sample FLOPs (fwd)                   : {fmt(per_sample_fwd_gflops)}")
        print(f"  Per-sample MACs                          : {fmt(per_sample_macs)}")
        print(f"  Per-step FLOPs (fwd, batch={batch_size})       : {fmt(per_step_fwd_gflops)}")
        print(f"  Training-step FLOPs (fwd+bwd, batch={batch_size}) : {fmt(training_step_gflops)}")
        print(f"  Total training FLOPs                     : {fmt(total_training_gflops)}")
        print(f"  Inference FLOPs (1 sample)               : {fmt(inference_per_sample_gflops)}")
        print(f"  Inference FLOPs (full test set)          : {fmt(inference_test_set_gflops)}")
        if per_sample_fwd_avg_gflops:
            print(f"\n  [REALISTIC — enc={avg_enc}, dec={avg_dec} (actual avg from test data)]")
            print(f"  Per-sample FLOPs (fwd)                   : {fmt(per_sample_fwd_avg_gflops)}")
            print(f"  Per-step FLOPs (fwd, batch={batch_size})       : {fmt(per_step_fwd_avg_gflops)}")
            print(f"  Training-step FLOPs (fwd+bwd, batch={batch_size}) : {fmt(training_step_avg_gflops)}")
            print(f"  Total training FLOPs                     : {fmt(total_training_avg_gflops)}")
            print(f"  Inference FLOPs (full test set)          : {fmt(inference_test_set_avg_gflops)}")
        print(f"  Total examples seen ({total_steps} steps × {batch_size})   : {total_examples_seen:,}")

        all_results[model_name] = {
            "params_M":                      param_count / 1e6,
            "per_sample_fwd_gflops":         per_sample_fwd_gflops,
            "per_sample_macs":               per_sample_macs,
            "per_step_fwd_gflops":           per_step_fwd_gflops,
            "training_step_gflops":          training_step_gflops,
            "total_training_gflops":         total_training_gflops,
            "total_examples_seen":           total_examples_seen,
            "inference_per_sample":          inference_per_sample_gflops,
            "inference_test_set":            inference_test_set_gflops,
            "per_sample_fwd_avg_gflops":     per_sample_fwd_avg_gflops,
            "training_step_avg_gflops":      training_step_avg_gflops,
            "total_training_avg_gflops":     total_training_avg_gflops,
            "inference_test_set_avg_gflops": inference_test_set_avg_gflops,
            "avg_enc": avg_enc,
            "avg_dec": avg_dec,
            "batch_size":                    batch_size,
            "total_steps":                   total_steps,
        }

    # ─────────────────────────────────────────────
    # COMPARISON TABLE
    # ─────────────────────────────────────────────

    if len(all_results) == 2:
        names = list(all_results.keys())
        r_a   = all_results[names[0]]
        r_b   = all_results[names[1]]

        print(f"\n\n{'='*72}")
        print("  COMPARISON TABLE")
        print(f"{'='*72}")
        print(f"  Sequence lengths : encoder={enc_len} tokens, decoder={dec_len} tokens")
        print(f"  Note: lengths are MAX (worst-case). Real FLOPs may be lower due to")
        print(f"        variable-length padding within batches.")
        print(f"  Backward approx  : {BACKWARD_MULT}× forward  (backward ≈ 2× forward)")
        print(f"{'='*72}\n")

        w = [36, 18, 18, 12]
        header = (f"  {'Metric':<{w[0]}} {names[0]:<{w[1]}} {names[1]:<{w[2]}} {'Reduction':<{w[3]}}")
        print(header)
        print("  " + "-" * (sum(w) + 2))

        rows = [
            ("── WORST CASE (enc=512, dec=128) ──", "", "", ""),

            ("Parameters",
             f"{r_a['params_M']:.2f}M",
             f"{r_b['params_M']:.2f}M",
             f"-{pct_reduction(r_a['params_M'], r_b['params_M']):.1f}%"),

            ("Per-sample FLOPs (fwd only)",
             fmt(r_a['per_sample_fwd_gflops']),
             fmt(r_b['per_sample_fwd_gflops']),
             f"-{pct_reduction(r_a['per_sample_fwd_gflops'], r_b['per_sample_fwd_gflops']):.1f}%"),

            ("Per-sample MACs",
             fmt(r_a['per_sample_macs']),
             fmt(r_b['per_sample_macs']),
             f"-{pct_reduction(r_a['per_sample_macs'], r_b['per_sample_macs']):.1f}%"),

            (f"Training-step FLOPs (fwd+bwd, batch={r_a['batch_size']})",
             fmt(r_a['training_step_gflops']),
             fmt(r_b['training_step_gflops']),
             f"-{pct_reduction(r_a['training_step_gflops'], r_b['training_step_gflops']):.1f}%"),

            ("Total training FLOPs",
             fmt(r_a['total_training_gflops']),
             fmt(r_b['total_training_gflops']),
             f"-{pct_reduction(r_a['total_training_gflops'], r_b['total_training_gflops']):.1f}%"),

            (f"Inference FLOPs (test set {TEST_SET_SIZE})",
             fmt(r_a['inference_test_set']),
             fmt(r_b['inference_test_set']),
             f"-{pct_reduction(r_a['inference_test_set'], r_b['inference_test_set']):.1f}%"),
        ]

        if r_a['per_sample_fwd_avg_gflops'] and r_b['per_sample_fwd_avg_gflops']:
            rows += [
                (f"── REALISTIC (enc={r_a['avg_enc']}, dec={r_a['avg_dec']}) ──", "", "", ""),

                ("Per-sample FLOPs (fwd only)",
                 fmt(r_a['per_sample_fwd_avg_gflops']),
                 fmt(r_b['per_sample_fwd_avg_gflops']),
                 f"-{pct_reduction(r_a['per_sample_fwd_avg_gflops'], r_b['per_sample_fwd_avg_gflops']):.1f}%"),

                (f"Training-step FLOPs (fwd+bwd, batch={r_a['batch_size']})",
                 fmt(r_a['training_step_avg_gflops']),
                 fmt(r_b['training_step_avg_gflops']),
                 f"-{pct_reduction(r_a['training_step_avg_gflops'], r_b['training_step_avg_gflops']):.1f}%"),

                ("Total training FLOPs",
                 fmt(r_a['total_training_avg_gflops']),
                 fmt(r_b['total_training_avg_gflops']),
                 f"-{pct_reduction(r_a['total_training_avg_gflops'], r_b['total_training_avg_gflops']):.1f}%"),

                (f"Inference FLOPs (test set {TEST_SET_SIZE})",
                 fmt(r_a['inference_test_set_avg_gflops']),
                 fmt(r_b['inference_test_set_avg_gflops']),
                 f"-{pct_reduction(r_a['inference_test_set_avg_gflops'], r_b['inference_test_set_avg_gflops']):.1f}%"),
            ]

        for row in rows:
            print(f"  {row[0]:<{w[0]}} {row[1]:<{w[1]}} {row[2]:<{w[2]}} {row[3]:<{w[3]}}")

        print(f"\n{'='*72}")
        print("  TOTAL TRAINING — EQUIVALENCE CHECK")
        print(f"{'='*72}")
        for name, r in [(names[0], r_a), (names[1], r_b)]:
            way1 = BACKWARD_MULT * r['per_sample_fwd_gflops'] * r['total_examples_seen']
            way2 = r['training_step_gflops'] * r['total_steps']
            print(f"  {name}")
            print(f"    Way 1 (3 × per_sample × examples_seen): {fmt(way1)}")
            print(f"    Way 2 (training_step × total_steps)   : {fmt(way2)}")
            print(f"    Match: {'YES' if abs(way1-way2) < 1e-6 else 'NO'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FLOPs for T5 experiments.")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help='List of "Name=path" pairs, e.g. "d512=outputs/summarization_baseline_t5small_d512_batch64_lr3e4"'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.models:
        models = []
        for item in args.models:
            if "=" in item:
                name, path = item.split("=", 1)
                models.append((name.strip(), path.strip()))
            else:
                print(f"[ERROR] Expected 'Name=path' format, got: {item}")
                sys.exit(1)
    else:
        models = DEFAULT_MODELS

    run(models)
