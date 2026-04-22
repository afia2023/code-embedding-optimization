"""Compare baseline vs compressed experiment results.

Usage:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py \
        --baseline  outputs/summarization_baseline_d768 \
        --compressed outputs/summarization_compressed_d397_random
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def count_parameters(output_dir: Path) -> int | None:
    """Count total parameters from saved model (deduplicates tied weights)."""
    try:
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(str(output_dir))
        return sum(p.numel() for p in model.parameters())
    except Exception:
        pass
    return None


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def format_params(n: int | None) -> str:
    if n is None:
        return "N/A"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def summarize(output_dir: Path, label: str) -> dict:
    test = load_json(output_dir / "test_results.json")
    timing = load_json(output_dir / "timing.json")
    train = load_json(output_dir / "train_results.json")
    config = load_json(output_dir / "resolved_config.json")

    n_params = count_parameters(output_dir)

    return {
        "label": label,
        "hidden_dim": config.get("hidden_dim") or 768,
        "init_method": config.get("init_method", "pretrained"),
        "epochs": train.get("epoch", "?"),
        "n_params": n_params,
        # Metrics
        "bleu": test.get("test_bleu"),
        "chrf": test.get("test_chrf"),
        "rouge1": test.get("test_rouge1"),
        "rouge2": test.get("test_rouge2"),
        "rougeL": test.get("test_rougeL"),
        "exact_match": test.get("test_exact_match"),
        "gen_len": test.get("test_gen_len"),
        # Timing
        "total_train_seconds": timing.get("total_train_seconds"),
        "total_eval_seconds": timing.get("total_eval_seconds"),
        "total_test_seconds": timing.get("total_test_seconds"),
        "total_pipeline_seconds": timing.get("total_pipeline_seconds"),
        "model_load_seconds": timing.get("model_load_seconds"),
    }


def print_comparison(baseline: dict, compressed: dict) -> None:
    def delta(b, c, higher_is_better=True):
        if b is None or c is None:
            return ""
        diff = c - b
        pct = (diff / b * 100) if b != 0 else 0
        sign = "+" if diff > 0 else ""
        arrow = "↑" if diff > 0 else "↓"
        color = ""
        if higher_is_better:
            color = "✅" if diff >= 0 else "❌"
        else:
            color = "✅" if diff <= 0 else "❌"
        return f"{color} {sign}{pct:.1f}%"

    w = 28
    print("\n" + "=" * 75)
    print(f"{'EXPERIMENT COMPARISON':^75}")
    print("=" * 75)
    print(f"{'Metric':<{w}} {'Baseline':>18} {'Compressed':>18} {'Change':>10}")
    print("-" * 75)

    # Architecture
    print(f"\n{'--- Architecture ---':<{w}}")
    print(f"{'d_model':<{w}} {baseline['hidden_dim']:>18} {compressed['hidden_dim']:>18}")
    print(f"{'Parameters':<{w}} {format_params(baseline['n_params']):>18} {format_params(compressed['n_params']):>18}")
    print(f"{'Epochs trained':<{w}} {str(baseline['epochs']):>18} {str(compressed['epochs']):>18}")

    # Compute param reduction
    if baseline['n_params'] and compressed['n_params']:
        reduction = (1 - compressed['n_params'] / baseline['n_params']) * 100
        print(f"{'Param reduction':<{w}} {'':>18} {f'{reduction:.1f}% smaller':>18}")

    # Performance metrics
    print(f"\n{'--- Test Metrics ---':<{w}}")
    metrics = [
        ("BLEU", "bleu", True),
        ("ChrF", "chrf", True),
        ("ROUGE-1", "rouge1", True),
        ("ROUGE-2", "rouge2", True),
        ("ROUGE-L", "rougeL", True),
        ("Exact Match", "exact_match", True),
        ("Gen Length", "gen_len", False),
    ]
    for name, key, higher_better in metrics:
        b_val = baseline.get(key)
        c_val = compressed.get(key)
        b_str = f"{b_val:.4f}" if b_val is not None else "N/A"
        c_str = f"{c_val:.4f}" if c_val is not None else "N/A"
        d_str = delta(b_val, c_val, higher_better)
        print(f"{name:<{w}} {b_str:>18} {c_str:>18} {d_str:>10}")

    # Timing
    print(f"\n{'--- Training Time ---':<{w}}")
    timing_metrics = [
        ("Model Load", "model_load_seconds", False),
        ("Training", "total_train_seconds", False),
        ("Evaluation", "total_eval_seconds", False),
        ("Testing", "total_test_seconds", False),
        ("Total Pipeline", "total_pipeline_seconds", False),
    ]
    for name, key, higher_better in timing_metrics:
        b_val = baseline.get(key)
        c_val = compressed.get(key)
        b_str = format_seconds(b_val) if b_val else "N/A"
        c_str = format_seconds(c_val) if c_val else "N/A"
        d_str = delta(b_val, c_val, higher_better)
        print(f"{name:<{w}} {b_str:>18} {c_str:>18} {d_str:>10}")

    print("\n" + "=" * 75)

    # Summary verdict
    bleu_b = baseline.get("bleu") or 0
    bleu_c = compressed.get("bleu") or 0
    train_b = baseline.get("total_train_seconds") or 0
    train_c = compressed.get("total_train_seconds") or 0

    print("\nSUMMARY")
    print("-" * 40)
    if bleu_c > 0 and bleu_b > 0:
        bleu_retention = bleu_c / bleu_b * 100
        print(f"BLEU retention    : {bleu_retention:.1f}% of baseline")
    if train_b > 0 and train_c > 0:
        speedup = train_b / train_c
        print(f"Training speedup  : {speedup:.2f}x faster")
    if baseline['n_params'] and compressed['n_params']:
        print(f"Parameter savings : {reduction:.1f}% fewer parameters")
    print("=" * 75 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs compressed experiment results.")
    parser.add_argument(
        "--baseline",
        default="outputs/summarization_baseline_d768",
        help="Path to baseline experiment output directory",
    )
    parser.add_argument(
        "--compressed",
        default="outputs/summarization_compressed_d397_random",
        help="Path to compressed experiment output directory",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save comparison as JSON",
    )
    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    compressed_dir = Path(args.compressed)

    if not baseline_dir.exists():
        print(f"Baseline directory not found: {baseline_dir}")
        print("Run the baseline experiment first.")
        return
    if not compressed_dir.exists():
        print(f"Compressed directory not found: {compressed_dir}")
        print("Run the compressed experiment first.")
        return

    print(f"Loading baseline   : {baseline_dir}")
    print(f"Loading compressed : {compressed_dir}")

    baseline = summarize(baseline_dir, "Baseline (d768)")
    compressed = summarize(compressed_dir, "Compressed (d397)")

    print_comparison(baseline, compressed)

    if args.save:
        result = {"baseline": baseline, "compressed": compressed}
        Path(args.save).write_text(json.dumps(result, indent=2))
        print(f"Results saved to {args.save}")


if __name__ == "__main__":
    main()
