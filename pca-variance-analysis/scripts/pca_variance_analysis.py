"""PCA Variance Analysis Script

Loads a pretrained model, runs PCA on its embedding matrix, and finds the
minimum hidden_dim that captures a target % of variance.

Usage:
    python scripts/pca_variance_analysis.py
    python scripts/pca_variance_analysis.py --model Salesforce/codet5p-220m --threshold 0.95
    python scripts/pca_variance_analysis.py --threshold 0.90 0.95 0.99 --plot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from code_tasks.model_builder import find_hidden_dim_from_variance


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA variance analysis to determine ideal hidden_dim.")
    parser.add_argument(
        "--model",
        default="Salesforce/codet5p-220m",
        help="HuggingFace model name or local path (default: Salesforce/codet5p-220m)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        default=[0.90, 0.95, 0.99],
        help="Variance threshold(s) to analyse (default: 0.90 0.95 0.99)",
    )
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    parser.add_argument("--output", default=None, help="Optional path to save results as JSON")
    parser.add_argument("--plot", action="store_true", help="Plot cumulative explained variance curve")
    args = parser.parse_args()

    # Run analysis once using the highest threshold (reuse ratios for all thresholds)
    max_threshold = max(args.threshold)
    result = find_hidden_dim_from_variance(
        args.model,
        variance_threshold=max_threshold,
        cache_dir=args.cache_dir,
    )

    ratios = result.explained_variance_ratios
    original_dim = result.original_hidden_dim

    print("\n" + "=" * 60)
    print(f"Model           : {args.model}")
    print(f"Original d_model: {original_dim}")
    print("=" * 60)
    print(f"{'Threshold':<12} {'Recommended hidden_dim':<25} {'Variance captured'}")
    print("-" * 60)

    summary = []
    for threshold in sorted(args.threshold):
        cumulative = 0.0
        recommended = original_dim
        for dim, ratio in enumerate(ratios, start=1):
            cumulative += ratio
            if cumulative >= threshold:
                recommended = dim
                break
        variance_captured = sum(ratios[:recommended]) * 100
        reduction_pct = (1 - recommended / original_dim) * 100
        print(
            f"{threshold*100:.0f}%{'':<9} {recommended:<25} {variance_captured:.2f}%  "
            f"(reduction: {reduction_pct:.1f}%)"
        )
        summary.append({
            "threshold": threshold,
            "recommended_hidden_dim": recommended,
            "variance_captured": round(variance_captured / 100, 6),
            "reduction_pct": round(reduction_pct, 2),
        })

    print("=" * 60)
    print(f"\nRecommendation: use --hidden-dim {summary[1]['recommended_hidden_dim']} --init-method random")
    print("(based on 95% variance threshold)\n")

    if args.output:
        output_data = {
            "model": args.model,
            "original_hidden_dim": original_dim,
            "analysis": summary,
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"Results saved to {args.output}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            cumulative_variance = np.cumsum(ratios)
            dims = list(range(1, len(ratios) + 1))

            plt.figure(figsize=(10, 5))
            plt.plot(dims, cumulative_variance * 100, linewidth=2)

            colors = ["orange", "red", "purple"]
            for (entry, color) in zip(summary, colors):
                plt.axhline(
                    y=entry["threshold"] * 100,
                    color=color,
                    linestyle="--",
                    alpha=0.7,
                    label=f"{entry['threshold']*100:.0f}% → dim={entry['recommended_hidden_dim']}",
                )
                plt.axvline(x=entry["recommended_hidden_dim"], color=color, linestyle=":", alpha=0.5)

            plt.xlabel("Number of PCA components (hidden_dim)")
            plt.ylabel("Cumulative explained variance (%)")
            plt.title(f"PCA Variance Analysis — {args.model}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = "pca_variance_analysis.png"
            plt.savefig(plot_path, dpi=150)
            print(f"Plot saved to {plot_path}")
            plt.show()
        except ImportError:
            print("matplotlib not installed — skipping plot. Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
