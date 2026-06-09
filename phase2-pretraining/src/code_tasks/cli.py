from __future__ import annotations

import argparse

from .config import PipelineConfig
from .datasets import dataset_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, test, and run inference for method-level code summarization and generation.",
    )

    parser.add_argument("--task", choices=["summarization", "generation"], default="summarization")
    parser.add_argument("--language", choices=["java", "python"], default="java")
    parser.add_argument("--model-name-or-path", required=False, default="Salesforce/codet5p-220m")
    parser.add_argument("--dataset", required=False, default="codexglue_code_to_text")
    parser.add_argument("--output-dir", required=False, default="outputs/run")

    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--source-column", default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--validation-split", default=None)
    parser.add_argument("--test-split", default=None)

    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--evaluation-strategy", choices=["no", "steps", "epoch"], default="epoch")
    parser.add_argument("--save-strategy", choices=["no", "steps", "epoch"], default="epoch")
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--generation-max-length", type=int, default=None)

    parser.add_argument("--do-train", action="store_true")
    parser.add_argument("--do-eval", action="store_true")
    parser.add_argument("--do-test", action="store_true")
    parser.add_argument("--predict-text", default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--preprocessing-num-workers", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--input-embedding-dim", type=int, default=None)
    parser.add_argument("--input-embedding-reduction", choices=["pca", "vae"], default="pca")
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Target d_model for full architectural modification (not projection-based).")
    parser.add_argument("--init-method", choices=["random", "pca"], default="random",
                        help="Weight initialization: 'random' (Setting 1) or 'pca' (Setting 2, from pre-trained).")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--metric-for-best-model", default="bleu")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--source-prefix", default=None)

    parser.add_argument("--list-datasets", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_datasets:
        _print_dataset_table()
        return

    try:
        config = PipelineConfig.from_namespace(args)
        from .pipeline import run_pipeline

        run_pipeline(config)
    except ValueError as exc:
        parser.error(str(exc))


def _print_dataset_table() -> None:
    rows = dataset_rows()
    print("alias\ttask\tlanguages\tdescription")
    for alias, task, languages, description in rows:
        print(f"{alias}\t{task}\t{languages}\t{description}")


if __name__ == "__main__":
    main()
