# Implementation Plan

## 1. Dataset strategy

### Method-level code summarization

- `google/code_x_glue_ct_code_to_text`
  - Why: this is the standard CodeXGLUE code-to-text benchmark derived from CodeSearchNet, with well-known train/validation/test splits and direct support for both `java` and `python`.
  - Fit: each example contains a method/function body plus a paired docstring or summary, which matches method-level summarization directly.

### Method-level code generation

- `google/code_x_glue_ct_code_to_text` used in reverse (`docstring -> code`)
  - Why: it gives the cleanest first-version method-level generation setup for both `java` and `python`, with the same schema and splits as the summarization task.
  - Fit: although the original benchmark is code-to-text, reversing it creates a practical docstring-to-method generation task with method-level alignment.

- `semeru/Text-Code-concode-Java`
  - Why: Concode is a widely used Java text-to-code benchmark and includes natural language plus class context, which makes it realistic for Java method generation.
  - Fit: the target is a Java member function, so it is strongly aligned with the requested task.

- `codeparrot/xlcost-text-to-code`
  - Why: XLCoST is a common multilingual text-to-code benchmark with `Java` and `Python` subsets already hosted in Hugging Face Datasets.
  - Fit: it is not strictly method-only in every subset, but it is a practical extension target for multilingual code generation experiments.

## 2. Pipeline design

### Data loading

- Use a dataset registry keyed by CLI dataset aliases.
- Each dataset adapter defines:
  - Hugging Face dataset name
  - optional dataset config per language
  - split names
  - source and target columns
  - dataset-specific prompt formatting
  - optional text normalization
- Allow a fallback generic dataset mode for arbitrary Hugging Face datasets when the user supplies source/target columns manually.

### Preprocessing

- Normalize raw fields conservatively.
- Build task-specific source prompts:
  - summarization: code -> summary/docstring
  - generation: natural language/docstring -> code
- Keep dataset-specific prompt builders isolated so new datasets can be added without changing the trainer.

### Tokenization

- Use `AutoTokenizer`.
- Use source truncation with `max_input_length`.
- Use target truncation with `max_target_length`.
- Perform dynamic padding with `DataCollatorForSeq2Seq`.

### Training

- Use `AutoModelForSeq2SeqLM` and `Seq2SeqTrainer`.
- Expose core hyperparameters via CLI:
  - model/checkpoint
  - learning rate
  - batch size
  - epochs
  - seed
  - evaluation strategy
  - generation beam size
  - output directory
- Save trainer state, metrics, tokenizer, and model checkpoints.

### Validation and testing

- Validation: `trainer.evaluate(...)` on the validation split.
- Testing: `trainer.predict(...)` on the test split with generation enabled.
- Save metrics and generated predictions for downstream analysis.

### Metrics

- BLEU using `sacrebleu`
- ROUGE-1, ROUGE-2, and ROUGE-L using `rouge-score`
- chrF as an additional text-generation metric
- Exact match as a strict code-generation metric

### Checkpoints and inference

- Resume training via `--resume-from-checkpoint`.
- Load a saved checkpoint by pointing `--model-name-or-path` at a checkpoint directory.
- Support single-example generation through `--predict-text`.

## 3. Research-friendly project structure

- `src/code_tasks/config.py`
  - CLI-backed dataclass config and validation
- `src/code_tasks/datasets.py`
  - dataset registry, dataset resolution, prompt building, and text normalization
- `src/code_tasks/models.py`
  - tokenizer/model loading and runtime device handling
- `src/code_tasks/metrics.py`
  - decoding and evaluation metrics
- `src/code_tasks/pipeline.py`
  - orchestration for train/eval/test/predict
- `src/code_tasks/cli.py`
  - argparse entrypoint

## 4. Implementation sequence

1. Create the package skeleton and dependency manifest.
2. Implement config parsing and validation so invalid task/dataset/language combinations fail early.
3. Implement built-in dataset adapters for CodeXGLUE, Concode, and XLCoST.
4. Implement tokenizer/model loading for Hugging Face seq2seq checkpoints.
5. Implement trainer setup, evaluation metrics, checkpointing, and prediction export.
6. Add CLI examples and extension notes in the documentation.
7. Run syntax and import-level validation.
