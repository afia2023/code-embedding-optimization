# Code Embedding Optimization

Flexible Hugging Face training and evaluation pipeline for method-level code summarization and code generation.

The first version is centered on seq2seq checkpoints that expose a standard `AutoModelForSeq2SeqLM` interface, which makes models such as CodeT5 and many CodeT5+ variants straightforward to use. Model loading is isolated in one module so additional architectures or custom wrappers can be added cleanly.

## Configurable input embedding reduction

You can compress the model input embedding path to a custom dimension with `--input-embedding-dim` and choose the dimensionality reduction method with `--input-embedding-reduction`.

- Example values: `562`, `323`, `128`, `64`, `32`
- Supported reduction methods:
  - `pca`
  - `vae`
- The runtime embedding uses a reduced token table plus a decoder back to the model hidden size:
  - a smaller token table of size `vocab_size x input_embedding_dim`
  - followed by a decoder back to the model hidden size
- `pca` initializes the reduced embedding with a principal-component projection of the pretrained embedding matrix
- `vae` fits a small variational autoencoder over the pretrained embedding matrix, uses the encoder mean as the token latent table, and copies the learned decoder into the runtime embedding adapter

## Built-in datasets

### Summarization

- `codexglue_code_to_text`
  - Backed by `google/code_x_glue_ct_code_to_text`
  - Supports `java` and `python`
  - Source: method/function code
  - Target: docstring/summary

### Generation

- `codexglue_docstring_to_code`
  - Backed by `google/code_x_glue_ct_code_to_text`
  - Supports `java` and `python`
  - Source: docstring/summary
  - Target: method/function code

- `concode_java`
  - Backed by `semeru/Text-Code-concode-Java`
  - Supports `java`
  - Source: NL description plus class context
  - Target: Java member function

- `xlcost_text_to_code`
  - Backed by `codeparrot/xlcost-text-to-code`
  - Supports `java` and `python`
  - Defaults to snippet-level subsets
  - Source: text description
  - Target: code snippet or method-like code unit

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

## CLI examples

### List built-in datasets

```bash
python -m code_tasks.cli --list-datasets
```

### Train Java method summarization with CodeT5+

```bash
python -m code_tasks.cli \
  --task summarization \
  --language java \
  --dataset codexglue_code_to_text \
  --model-name-or-path Salesforce/codet5-base \
  --input-embedding-dim 128 \
  --input-embedding-reduction pca \
  --max-input-length 512 \
  --max-target-length 128 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 8 \
  --learning-rate 5e-5 \
  --num-train-epochs 3 \
  --evaluation-strategy epoch \
  --output-dir outputs/codet5-sum-java \
  --do-train \
  --do-eval \
  --do-test
```

### Train Python docstring-to-code generation with CodeT5+

```bash
python -m code_tasks.cli \
  --task generation \
  --language python \
  --dataset codexglue_docstring_to_code \
  --model-name-or-path Salesforce/codet5-base \
  --input-embedding-dim 64 \
  --input-embedding-reduction vae \
  --max-input-length 256 \
  --max-target-length 256 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 8 \
  --learning-rate 5e-5 \
  --num-train-epochs 5 \
  --num-beams 5 \
  --evaluation-strategy epoch \
  --output-dir outputs/codet5-gen-python \
  --do-train \
  --do-eval \
  --do-test
```

### Train Java generation on Concode

```bash
python -m code_tasks.cli \
  --task generation \
  --language java \
  --dataset concode_java \
  --model-name-or-path Salesforce/codet5-base \
  --input-embedding-dim 323 \
  --input-embedding-reduction pca \
  --max-input-length 320 \
  --max-target-length 160 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 8 \
  --learning-rate 3e-5 \
  --num-train-epochs 5 \
  --evaluation-strategy epoch \
  --output-dir outputs/concode-java \
  --do-train \
  --do-eval \
  --do-test
```

### Evaluate or test from a saved checkpoint

```bash
python -m code_tasks.cli \
  --task summarization \
  --language java \
  --dataset codexglue_code_to_text \
  --model-name-or-path outputs/codet5-sum-java/checkpoint-5000 \
  --output-dir outputs/codet5-sum-java-eval \
  --do-eval \
  --do-test
```

### Run single-example inference

```bash
python -m code_tasks.cli \
  --task generation \
  --language python \
  --dataset codexglue_docstring_to_code \
  --model-name-or-path outputs/codet5-gen-python/checkpoint-4000 \
  --output-dir outputs/inference \
  --predict-text "Return the maximum element in a list of integers."
```

## Extension points

- Add a new dataset by registering a new `DatasetSpec` in [src/code_tasks/datasets.py](/Users/amastro/Temp/src/code_tasks/datasets.py).
- Add support for another model family by extending [src/code_tasks/models.py](/Users/amastro/Temp/src/code_tasks/models.py).
- If a checkpoint requires custom code from the Hugging Face Hub, pass `--trust-remote-code`.

## Notes

- The implementation uses generation-oriented metrics: BLEU, ROUGE, chrF, and exact match.
- Checkpoint loading for evaluation or inference is handled by reusing `--model-name-or-path` with a saved checkpoint directory.
- If a checkpoint was trained with `--input-embedding-dim`, the reduction method and adapter configuration are stored in the checkpoint config and reloaded automatically.
- If you want to use an arbitrary Hugging Face dataset instead of a built-in alias, pass the dataset name in `--dataset` together with `--source-column` and `--target-column`.
