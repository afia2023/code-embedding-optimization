# Code Embedding Optimization

Research project benchmarking T5-small model compression for Java method-level code summarization (CodeXGLUE / CodeSearchNet). Two model configurations are compared: a **d512 baseline** (standard T5-small, 60.5M parameters) and a **d320 compressed variant** (27.5M parameters), where the hidden dimension was chosen via PCA variance analysis.

---

## Research Overview

| Model | Hidden dim | Parameters | Init method |
|-------|-----------|------------|-------------|
| d512  | 512       | 60.5M      | Random      |
| d320  | 320       | 27.5M      | Random / Pre-trained |

**Task:** Java code summarization — given a Java method, generate a natural language summary.  
**Dataset:** CodeXGLUE (`google/code_x_glue_ct_code_to_text`, Java split) — 164,923 train / 5,183 val / 10,955 test.  
**Metrics:** BLEU, ROUGE-1/2/L, chrF, Exact Match.

---

## Project Structure

### PCA Variance Analysis (`pca-variance-analysis/`)

Before choosing the compressed dimension, PCA was applied to the T5-small shared embedding matrix (shape `32,128 × 512`) to find the minimum number of dimensions that preserve a given percentage of variance.

**Results on T5-small (original d_model = 512):**

| Variance threshold | Recommended hidden_dim | Variance captured | Reduction |
|--------------------|----------------------|-------------------|-----------|
| 90%                | 322                  | 90.07%            | 37.1%     |
| 95%                | 379                  | 95.03%            | 26.0%     |
| 99%                | 451                  | 99.01%            | 11.9%     |

**Why 320 and not 322?** T5-small uses 8 attention heads, so `hidden_dim` must be evenly divisible by 8 to distribute dimensions equally across heads. 322 is not divisible by 8, so it is rounded down to **320** (320 ÷ 8 = 40 exactly). This is the nearest valid dimension below the 90% variance threshold result.

**Conclusion:** `hidden_dim = 320` captures ~90% of the embedding variance while reducing the model from 60.5M to 27.5M parameters (37% reduction). PCA is used only as an analysis tool to justify the dimension choice — it does not initialize model weights.


What your project does:
1. Can d320 learn from scratch?
2. Does domain pre-training help d320?
3. By compressing the original d512 weights reduce the model performance? 



**Run the analysis:**

```bash
cd pca-variance-analysis
python scripts/pca_variance_analysis.py --model t5-small --threshold 0.90 0.95 0.99
```

---

### LR Sweep (`lr-sweep/`)

Learning rate sweep across all batch sizes to determine the best learning rate per batch size, following the **linear scaling rule** (Goyal et al., 2017): `LR_new = LR_ref × (BS_new / BS_ref)`.

| Batch Size | LR candidates |
|------------|--------------|
| 8          | 2.5e-5, 5e-5 |
| 16         | 5e-5, 1e-4   |
| 32         | 1e-4, 2e-4   |
| 64         | 2e-4, 4e-4   |

Each sweep run uses 40,000 training samples, 5 epochs, and early stopping (patience = 2). Both d512 and d320 models are tested.

---

### Phase 1 — Fine-tuning with Random Init (`phase1-finetuning-random-init/`)

Full fine-tuning of d512 and d320 models from random initialization using the best learning rate identified in the LR sweep.

| Config | Batch Sizes   | Epochs | Dataset              |
|--------|--------------|--------|----------------------|
| d512   | 8, 16, 32, 64 | 15     | Full CodeXGLUE Java (164,923 examples) |
| d320   | 8, 16, 32, 64 | 15     | Full CodeXGLUE Java (164,923 examples) |

---

### Phase 2 — Pre-training (`phase2-pretraining/`)

Span-corruption pre-training of d512 and d320 models on a domain-specific Java corpus (12.6M Java methods, ~4.4GB).

| Parameter           | Value                        |
|---------------------|------------------------------|
| Pre-training task   | Span corruption (T5-style)   |
| Corpus size         | 12,671,475 Java methods      |
| Max sequence length | 512 / 512 tokens             |
| Batch size          | 32 per GPU (effective 128 with grad accum × 4) |
| Total steps         | 500,000                      |
| Warmup steps        | 10,000                       |
| Optimizer           | Adafactor                    |
| Hardware            | 2× L40S GPUs                 |

```bash
cd phase2-pretraining
python pretrain.py --model d320 --output-dir outputs/pretrained_d320
python pretrain.py --model d512 --output-dir outputs/pretrained_d512
```

---

### Phase 3 — Fine-tuning with Pre-trained Weights (`phase3-finetuning-pretrained/`)

Fine-tuning d512 and d320 models initialized from the pre-trained checkpoints (Phase 2). Same hyperparameter settings as Phase 1, enabling a direct comparison: **random init vs. pre-trained init**.

---

## Model Downloads (Google Drive)

Pre-trained and fine-tuned model weights are hosted on Google Drive (too large for GitHub).

| File | Contents | Size |
|------|----------|------|
| `phase2_pretrained_models.zip` | d320 + d512 pre-trained weights (500k steps) | 312MB |
| `lr_sweep_models.zip` | 16 LR sweep fine-tuned models (all batch sizes) | 1.5GB |
| `phase1_finetuning_models.zip` | 8 random-init fine-tuned models | 1.3GB |
| `phase3_finetuning_models.zip` | 8 pretrained-init fine-tuned models | 1.3GB |

> **Links:** *(add Google Drive links here after upload)*

---

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

---

## CLI Usage

### Fine-tune d320 on Java code summarization

```bash
python -m code_tasks.cli \
  --task summarization \
  --language java \
  --dataset codexglue_code_to_text \
  --model-name-or-path t5-small \
  --hidden-dim 320 \
  --max-input-length 512 \
  --max-target-length 128 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 16 \
  --learning-rate 5e-5 \
  --num-train-epochs 15 \
  --warmup-ratio 0.1 \
  --weight-decay 0.01 \
  --num-beams 4 \
  --output-dir outputs/d320_bs16 \
  --do-train --do-eval --do-test
```

### Fine-tune d512 baseline

```bash
python -m code_tasks.cli \
  --task summarization \
  --language java \
  --dataset codexglue_code_to_text \
  --model-name-or-path t5-small \
  --hidden-dim 512 \
  --max-input-length 512 \
  --max-target-length 128 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 16 \
  --learning-rate 5e-5 \
  --num-train-epochs 15 \
  --warmup-ratio 0.1 \
  --weight-decay 0.01 \
  --num-beams 4 \
  --output-dir outputs/d512_bs16 \
  --do-train --do-eval --do-test
```

### Evaluate from a saved checkpoint

```bash
python -m code_tasks.cli \
  --task summarization \
  --language java \
  --dataset codexglue_code_to_text \
  --model-name-or-path outputs/d320_bs16 \
  --output-dir outputs/d320_bs16_eval \
  --do-eval --do-test
```

---

## Built-in Datasets

### Summarization

- `codexglue_code_to_text` — backed by `google/code_x_glue_ct_code_to_text`, supports `java` and `python`. Source: method code → Target: docstring summary.

### Generation

- `codexglue_docstring_to_code` — backed by `google/code_x_glue_ct_code_to_text`, supports `java` and `python`. Source: docstring → Target: method code.
- `concode_java` — backed by `semeru/Text-Code-concode-Java`. Source: NL description + class context → Target: Java member function.
- `xlcost_text_to_code` — backed by `codeparrot/xlcost-text-to-code`, supports `java` and `python`. Source: text description → Target: code snippet.

---

## Extension Points

- Add a new dataset by registering a `DatasetSpec` in `src/code_tasks/datasets.py`.
- Add support for another model family by extending `src/code_tasks/models.py`.
- Pass `--trust-remote-code` for checkpoints that require custom code from the Hugging Face Hub.

---

## Notes

- Metrics used: BLEU, ROUGE-1/2/L, chrF, and Exact Match.
- The `--hidden-dim` flag modifies the entire model architecture (embeddings, attention, FFN, LM head), not just the input embedding.
- Checkpoint configs store the `hidden_dim` so evaluation reloads the correct architecture automatically.
- Pre-training corpus (Java methods TSV) is available on request — not included in this repo due to size (4.4GB).
