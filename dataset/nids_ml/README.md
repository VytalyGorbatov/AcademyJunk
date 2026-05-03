# nids_ml: Network Intrusion Detection ML Framework

A PyTorch-based toolkit for training network intrusion detection models on raw packet payload (byte-level) data. It supports multiple deep learning architectures and specializes in **Positive-Unlabeled (PU)** learning and **Two-Way Contrastive** pretraining — techniques designed for real-world IDS scenarios where labeled attack samples are scarce and unlabeled traffic is abundant.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Package Structure](#package-structure)
- [Models](#models)
- [Training Pipelines](#training-pipelines)
- [Data Pipeline](#data-pipeline)
- [Losses and Metrics](#losses-and-metrics)
- [Configuration Reference](#configuration-reference)
- [Usage](#usage)
- [Artifacts](#artifacts)

---

## Overview

`nids_ml` processes network packet payloads stored as byte arrays (the `buffers` field in JSON dataset files). Each sample is a variable-length sequence of bytes representing packet content. The framework handles the full ML lifecycle: data loading and preprocessing, model construction, training with early stopping, evaluation on a held-out test set, and saving checkpoints and metrics.

Two distinct training regimes are supported:

1. **Standard / nnPU** — Single-stage training with supervised cross-entropy or non-negative Positive-Unlabeled (nnPU) loss. Suitable for `cnn`, `lstm`, `byte_cnn`, and `tcn` models.
2. **Two-Way** — Two-stage training exclusive to `tcn_2way`. Stage 1 uses SimCLR-style NT-Xent contrastive pretraining over unlabeled traffic. Stage 2 fine-tunes all heads using nnPU multitask loss.

---

## Features

- **Multiple Architectures**: Five built-in model types registered via a `@register_model` decorator factory. Custom models can be loaded dynamically via `class_path` in the config.
- **Positive-Unlabeled Learning**: Both a class-based `PULoss` (for mixed batches) and a function-based `nnpu_loss` (for pre-split P/U batches) implementing the non-negative nnPU risk estimator.
- **Contrastive SSL Pretraining**: `contrastive_nt_xent` (SimCLR / NT-Xent) loss for self-supervised pretraining before the nnPU fine-tuning stage.
- **Config-Driven**: All hyperparameters, dataset paths, model architecture, and training settings are defined in JSON `.conf` files with CLI overrides.
- **Robust Data Handling**: Supports `list[int]`, `list[float]` (legacy normalized), and `str` (latin-1) buffer encodings. Header/body splitting at a configurable separator byte. Fixed-length padding and truncation.
- **Reproducibility**: Global seed control across `random`, `numpy`, and `torch` via `set_global_seed`.
- **Graceful Interruption**: SIGINT/SIGTERM handlers allow cleanly stopping training after the current step.
- **Automatic Device Selection**: Resolves `cuda` → `mps` → `cpu` automatically, or accepts an explicit `--device` override.

---

## Package Structure

```
nids_ml/
├── classifier_creator.py   # CLI entry point — parses config and dispatches pipelines
├── local_types.py          # Shared type aliases (TensorPair, TensorTriple, Metrics)
├── utils.py                # set_global_seed(), DataUtils (device resolution, path helpers)
│
├── models/
│   ├── base.py             # BaseClassifier, _MODEL_REGISTRY, @register_model
│   ├── factory.py          # build_model() — resolves type key or class_path
│   ├── blocks.py           # Shared building blocks: DepthwiseSeparableConv1d,
│   │                       #   ResTCNBlock, PrefixPooling, masked_* helpers
│   ├── cnn.py              # CNN classifier (multi-scale conv → global pool → MLP)
│   ├── lstm.py             # Bidirectional LSTM classifier
│   ├── byte_cnn.py         # Byte-level CNN with configurable MLP head
│   ├── tcn.py              # ByteTCNClassifier — single-stream TCN with nnPU support
│   └── tcn_2way.py         # ByteTCN2WayClassifier — dual-encoder (header + body)
│                           #   with contrastive + nnPU multitask heads
│
├── data/
│   ├── common.py           # decode_buffers_field(), pad_or_truncate(),
│   │                       #   split_header_body(), augment_ids(), to_device()
│   ├── standard.py         # NIDSDataset, StandardDatasetBuilder — builds train/val/test
│   │                       #   DataLoaders for the standard pipeline
│   └── twoway.py           # TwoWayDataset, TwoWayDatasetBuilder — produces
│                           #   header_ids/header_mask/body_ids/body_mask batches
│
├── training/
│   ├── base.py             # BaseTrainer (optimizer/scheduler init, gradient clipping),
│   │                       #   EarlyStopper
│   ├── losses.py           # PULoss (class), nnpu_loss (function), contrastive_nt_xent
│   ├── metrics.py          # MetricUtils.compute_binary_metrics(), pr_curve_best_f1()
│   ├── standard.py         # Trainer — single-stage supervised / nnPU training loop
│   └── twoway.py           # TwoWayTrainer — Stage 1 contrastive pretrain +
│                           #   Stage 2 nnPU multitask fine-tuning
│
└── pipelines/
    ├── standard.py         # ClassifierPipeline — orchestrates standard training
    └── twoway.py           # TwoWayPipeline — orchestrates 2-stage training
```

---

## Models

All models extend `BaseClassifier` and are registered by name. They are instantiated via `build_model(config)` in `models/factory.py`. The first four models (`cnn`, `lstm`, `byte_cnn`, `tcn`) follow the same flat input interface — they receive a padded/truncated byte sequence as a `[B, L]` integer tensor and output logits directly. `tcn_2way` is the exception and is covered in detail below.

---

### `cnn` — Conv1DClassifier

**Input**: a flat `[B, L]` tensor of byte IDs (integers 0–255, padded to `fixed_len` with value 256).

Each byte is first turned into a learned embedding vector. The resulting `[B, L, embed_dim]` sequence is then passed through a stack of 1-D convolutions with progressively larger channels and decreasing kernel sizes, interleaved with ReLU activations, max-pooling (halves sequence length each layer), and optional dropout. After all conv layers, a global average pool collapses the time dimension, and a linear layer produces the class logits.

**Output**: raw class logits of shape `[B, num_classes]`. Trained with `CrossEntropyLoss` (softmax multiclass). Best used when you have roughly balanced labeled data and want a quick, interpretable baseline.

---

### `lstm` — LSTMClassifier

**Input**: same flat `[B, L]` byte-ID tensor.

Each byte is embedded, and the full sequence is fed through a multi-layer bidirectional LSTM. After processing the entire sequence, the final hidden states from the forward and backward passes are concatenated into a single fixed-size vector, which a linear classifier maps to output logits.

The bidirectional setup means the model sees each position in context of both what came before and after — useful for byte patterns that only make sense in context (e.g., attack signatures that appear mid-payload). However, LSTMs process sequences step-by-step, so they are slower than convolution-based models on long sequences.

**Output**: raw class logits of shape `[B, num_classes]`. Supports `class_weights` to handle imbalanced classes (e.g., weight attack class higher).

---

### `byte_cnn` — ByteCNNClassifier

**Input**: same flat `[B, L]` byte-ID tensor.

Architecturally similar to `cnn` but tuned for binary classification: it uses global *max* pooling instead of average pooling (more sensitive to any single strong feature anywhere in the sequence), and a deeper configurable MLP head (`mlp_dims`) instead of a single linear layer. The final layer outputs a single scalar logit.

**Output**: a single logit per sample, shape `[B]`. Trained with `BCEWithLogitsLoss` or `PULoss` (when `pu_prior` is set). A `threshold` (default `0.5`) is applied to sigmoid(logit) to produce hard predictions. Supports `class_weights` for the positive class.

---

### `tcn` — ByteTCNClassifier

**Input**: a dict batch with `header_ids`, `header_mask`, `body_ids`, `body_mask` — the packet split at `sep_byte` into a header half and a body half, both padded to their respective lengths.

This is the single-stream TCN. Each of the two streams (header, body) passes independently through:
1. Byte embedding
2. A stem 1×1 convolution to project to `tcn_channels`
3. A stack of `ResTCNBlock`s — each block uses depthwise-separable convolutions with increasing dilation (e.g., 1, 2, 4), giving the model an exponentially growing receptive field without adding many parameters. Each block is residual (output = input + conv(input)), so gradients flow cleanly even through deep stacks.
4. `PrefixPooling`: computes mean and max over the full sequence *and* over several fixed-length prefixes (e.g., first 128 and 256 bytes). This produces a multi-scale summary that captures both early-packet patterns and global structure. The header and body pools are concatenated.
5. A projection MLP turns the pooled vector into a fixed-size representation.

The header and body representations are then fused by a small MLP to produce a single scalar logit.

**Output**: a single logit per sample, shape `[B]`. Supports nnPU training via `pu_prior`. Supports `lr_scheduler` to reduce LR on plateau.

---

### `tcn_2way` — ByteTCN2WayClassifier

This is the flagship model. It uses the same header/body split and TCN architecture as `tcn`, but is specifically designed for two-stage training with very limited labeled attack data. See the [Two-Way Pipeline](#two-way-pipeline-twowaypipeline) section for the full training walkthrough.

**Architecture:**

The model has three components:

1. **Two `ByteEncoder`s** (one for header, one for body) — each is an independent copy of: embedding → 1×1 stem conv → stack of `ResTCNBlock`s → `PrefixPooling` → LayerNorm → Linear → SiLU → Dropout. Each encoder maps its input to a vector of size `proj_dim`.

2. **`ByteTCNBackbone`** — concatenates the header and body encoder outputs (`2 × proj_dim`) and passes them through a two-layer fusion MLP with LayerNorm, SiLU, and dropout. Produces a single representation vector of size `fusion_dim`.

3. **`Heads`** — three parallel linear layers on top of the backbone output:
   - `risk_logit` — the primary binary output: "is this attack traffic?" (single scalar, used at inference)
   - `alerted_logit` — auxiliary output for existing rule-based alert labels (used as weak supervision during Stage 1)
   - `proj` — a 2-layer MLP projecting to a 128-dim L2-normalized vector used exclusively during contrastive pretraining

**Input** (at inference / standard forward pass): the same dict batch as `tcn` — `header_ids`, `header_mask`, `body_ids`, `body_mask`.

**Output**: at inference, `forward()` returns only `risk_logit` (shape `[B]`). During training, the trainer accesses `backbone` and `heads` directly and uses all three head outputs.

---

### Custom Models

Set `"class_path": "my_module.MyModel"` in the config's `model` block. The class must subclass `nn.Module`. If it implements `from_config(config)`, that is used; otherwise `init_args` from the config are passed to `__init__`.

---

## Training Pipelines

### Standard Pipeline (`ClassifierPipeline`)

Used for `cnn`, `lstm`, `byte_cnn`, and `tcn` model types. This is a conventional single-stage training loop.

1. Builds train/val/test `DataLoader`s via `StandardDatasetBuilder`. The training set mixes benign samples with a configurable fraction of attack samples (`attack_percent`).
2. Instantiates the model via `build_model`.
3. Runs `Trainer` — each epoch iterates over batches, computes the loss, backpropagates, clips gradients, and optionally steps a learning rate scheduler.
4. After each epoch, evaluates on the validation set. Saves `model_best.pt` whenever the tracked metric (`best_metric`, default `f1`) improves. Stops early if the metric has not improved for `patience` epochs.
5. Loads the best checkpoint and evaluates on the held-out test set.
6. Saves `model_best.pt`, `model_last.pt`, `metrics.json`, `config_used.json` to `artifacts.out_dir`.

**Loss selection** in `Trainer`:
- `pu_prior` set + `output_mode=binary` → `PULoss` (nnPU). Use this when your training set contains labeled attacks mixed into a pool of unlabeled traffic (some of which may secretly be attacks).
- `output_mode=binary`, no prior → `BCEWithLogitsLoss` (with optional `pos_weight` from `class_weights`)
- `output_mode=multiclass` → `CrossEntropyLoss` (with optional `class_weights`)

---

### Two-Way Pipeline (`TwoWayPipeline`)

Used exclusively for `tcn_2way`. This pipeline exists to solve a common real-world problem: you have a large amount of network traffic but only a small number of *confirmed* attack samples. There are not enough labeled attacks to train a reliable classifier from scratch with standard supervised learning.

The solution is to split training into two stages: first teach the model what "normal traffic structure" looks like using all the unlabeled data, then fine-tune it to distinguish attacks from benign using the small labeled set and nnPU loss.

#### Data Loaders

`TwoWayDatasetBuilder` produces four loaders from the same source files:

- `train_p` — only confirmed attack samples (the labeled positives)
- `train_u` — the full unlabeled training corpus (all traffic, benign + potentially unlabeled attacks)
- `val` — validation set for monitoring during both stages
- `test` — held-out test set, only touched after training is fully complete

---

#### Stage 1 — Contrastive SSL Pretraining

> *Goal: learn a representation of traffic that is robust, generalizable, and meaningfully separates different traffic patterns — without using any attack labels.*

For each batch from `train_u` (unlabeled traffic):

1. **Two augmented views are created** from the same sample using `augment_ids`: random byte dropout (randomly zeroing out some bytes) and random byte insertion. This simulates the natural variability in how the same type of traffic might look (retransmissions, padding differences, etc.).
2. **Both views are passed through the backbone** independently, producing representation vectors `z1` and `z2`, then through the `ssl_proj` head to produce L2-normalized 128-dim projection vectors.
3. **NT-Xent (contrastive) loss** is computed: the two views of the same sample should have similar projections (pulled together in embedding space), while views from different samples should be dissimilar (pushed apart). This is the SimCLR approach — no labels needed.
4. **Auxiliary alert loss**: additionally, `alerted_logit` is supervised with whatever rule-based alert labels exist in the data (`alerted` field), weighted down by `w_alert`. These are imperfect labels (existing IDS rules fire, but miss novel attacks), but they provide a free weak signal to keep the representation grounded to known threat patterns.

The combined loss per batch is: `w_ssl × NT-Xent + w_alert × BCE(alerted_logit)`.

After each pretraining epoch, the full model state is checkpointed to `pretrain_epoch{N}.pt`. Validation PR-AUC is logged but does *not* drive LR decay during this stage — the contrastive objective and a classification metric are different enough that using one to gate the other would be harmful.

**Why contrastive pretraining?** The model needs to learn a feature space where structurally similar traffic clusters together before it can reliably distinguish the rare attack signal. Training directly on the tiny labeled set would overfit quickly. The contrastive stage exploits the large unlabeled corpus to build a strong initialization that Stage 2 can specialize.

---

#### Stage 2 — nnPU Multitask Fine-tuning

> *Goal: use the pretrained backbone to learn to classify attacks, despite having no confirmed negative (benign-only) labels.*

The challenge: `train_u` is not a clean negative set — some of those samples may be attacks. Using them as negatives in standard BCE would corrupt training. nnPU loss solves this by estimating the expected risk of calling unlabeled samples negative without assuming they are truly benign.

Each training step:

1. A batch of labeled positives (`train_p`) and a batch of unlabeled samples (`train_u`) are drawn in parallel.
2. Both are passed through the backbone and heads.
3. The **nnPU loss** (`nnpu_loss`) is computed using the `risk_logit` outputs from both batches and the class prior `pi_p` (your estimate of what fraction of traffic in the wild is actually an attack). It correctly penalizes misclassifying known attacks while avoiding over-penalizing the unlabeled set.
4. The **SSL regularizer** (NT-Xent on the `proj` head) continues to run on augmented views of the unlabeled batch, preserving the representational structure learned in Stage 1 and preventing the fine-tuning from catastrophically forgetting it.
5. The **alert auxiliary loss** (BCE on `alerted_logit`) continues as in Stage 1.

Total loss per step: `w_pu × nnpu_loss + w_ssl × NT-Xent + w_alert × BCE(alerted_logit)`.

The optimizer and LR scheduler are reset at the start of Stage 2 (fresh start with full LR). The LR is reduced on validation plateau. `model_best.pt` is saved whenever validation F1 / PR-AUC improves.

**Evaluation**: After Stage 2, the pipeline loads `model_best.pt` and runs `eval_on_loader` on the test set. The PR curve is swept across all thresholds and the threshold that maximizes F1 is selected. Reported metrics: `best_threshold`, `best_f1`, `pr_auc`, `precision_at_best`, `recall_at_best`.

---

## Data Pipeline

### Dataset Format

Each JSON file contains a list of records. Required fields:

| Field | Type | Description |
|-------|------|-------------|
| `buffers` | `list[int]`, `list[float]`, or `str` | Raw packet bytes (field name configurable via `buffer_field`) |
| `alerted` (or custom) | `int` / `bool` | Attack label (`1` = attack, `0` = benign). Field name set by `label_field`. |

Three encoding formats are accepted for `buffers`:
- `list[int]` — raw byte values in `[0, 255]`
- `list[float]` — legacy normalized values in `[0, 1]`, converted via `round(x * 255)`
- `str` — latin-1 encoded string, decoded to bytes

### Sequence Processing

1. Pad or truncate to `fixed_len` (default `1024`) using `PAD_IDX = 256`.
2. For `tcn_2way` and `tcn`: split at the first occurrence of `sep_byte` (default `0x1E` = 30). If not found, split at `fallback_header_len`.
3. **Augmentation** (`augment_ids`): random byte dropout and insertion applied during contrastive pretraining to create two views of the same sample.

### DataLoaders

- `StandardDatasetBuilder`: builds separate `train`, `val`, `test` loaders from `benign_paths` + `attack_paths`. Controls `attack_percent` sampling and `with_replacement`.
- `TwoWayDatasetBuilder`: produces three loaders — `train_p` (labeled positives / attacks), `train_u` (all unlabeled training data), `val`, and `test`.

---

## Losses and Metrics

### Losses (`training/losses.py`)

| Name | Type | Description |
|------|------|-------------|
| `PULoss` | `nn.Module` | nnPU loss for mixed P/U batches. Clamps negative risk to zero to enforce non-negativity. |
| `nnpu_loss` | function | nnPU risk estimator over pre-split P and U logit tensors. Returns `(loss, stats_dict)`. |
| `contrastive_nt_xent` | function | SimCLR NT-Xent loss. Takes two projection vectors `z1`, `z2` and a `temperature` parameter. |

### Metrics (`training/metrics.py`)

| Name | Description |
|------|-------------|
| `MetricUtils.compute_binary_metrics` | Computes accuracy, precision, recall, F1 from hard predictions. |
| `pr_curve_best_f1` | Sweeps thresholds over sigmoid scores to find the threshold maximizing F1. Returns `best_threshold`, `best_f1`, `pr_auc`, `precision_at_best`, `recall_at_best`. |

---

## Configuration Reference

All configs are JSON files. The path is passed via `--config`. Below is a full reference with all supported keys.

### Top-level keys

| Key | Type | Description |
|-----|------|-------------|
| `seed` | `int` | Global random seed (default `42`) |
| `attack_percent` | `float` | Fraction of attack samples to mix into the training set (standard pipeline only) |
| `sampling.with_replacement` | `bool` | Whether to sample attack records with replacement |

### `benign_paths` / `attack_paths`

```json
{
  "benign_paths": { "train": "../benign/train.json", "val": "../benign/val.json", "test": "../benign/test.json" },
  "attack_paths": { "train": "../attack/train.json", "val": "../attack/val.json", "test": "../attack/test.json" }
}
```

### `data` block

| Key | Default | Description |
|-----|---------|-------------|
| `buffer_field` | `"buffers"` | JSON field containing the byte sequence |
| `label_field` | `"alerted"` | JSON field used as the binary label |
| `fixed_len` | `1024` | Pad/truncate all sequences to this length |
| `sep_byte` | `30` | Byte value used to split header from body |
| `fallback_header_len` | `512` | Header length when `sep_byte` is absent |

### `training` block (standard)

| Key | Description |
|-----|-------------|
| `batch_size` | Samples per mini-batch |
| `epochs` | Maximum training epochs |
| `learning_rate` | Adam optimizer learning rate |
| `weight_decay` | L2 regularization coefficient |
| `patience` | Early stopping patience (0 = disabled) |
| `pu_prior` | Class prior π for nnPU loss (omit for standard cross-entropy) |
| `lr_scheduler.factor` | LR reduction factor on plateau |
| `lr_scheduler.patience` | Plateau patience for LR scheduler |
| `lr_scheduler.min_lr` | Minimum LR floor |

### `training` block (two-way)

| Key | Description |
|-----|-------------|
| `max_epochs_pretrain` | Stage 1 contrastive pretraining epochs |
| `max_epochs_pu` | Stage 2 nnPU fine-tuning epochs |
| `patience` | Early stopping patience (both stages) |
| `clip_grad` | Gradient norm clipping threshold |
| `w_ssl` | Weight for the NT-Xent SSL loss term |
| `w_alert` | Weight for the auxiliary alert head loss |
| `w_pu` | Weight for the nnPU classification loss |
| `pi_p` | Class prior π for `nnpu_loss` |
| `temp` | Temperature for `contrastive_nt_xent` |

### `model` block

Common fields:

| Key | Description |
|-----|-------------|
| `type` | `"cnn"`, `"lstm"`, `"byte_cnn"`, `"tcn"`, or `"tcn_2way"` |
| `class_path` | Dotted path to a custom `nn.Module` subclass (overrides `type`) |
| `vocab_size` | Byte vocabulary size (typically `257` to include padding index) |
| `embed_dim` | Byte embedding dimension |
| `dropout` | Dropout probability |
| `threshold` | Decision threshold for binary output (default `0.5`) |

`tcn_2way`-specific fields:

| Key | Description |
|-----|-------------|
| `channels` | TCN residual block channel width |
| `kernel` | Temporal convolution kernel size |
| `dilations` | List of dilation factors per TCN block |
| `prefix_lengths` | Prefix lengths for `PrefixPooling` (multi-scale aggregation) |
| `proj_dim` | Output dimension of each `ByteEncoder` projection |
| `fusion_dim` | Dimension of the fusion MLP backbone output |

### `artifacts` block

| Key | Default | Description |
|-----|---------|-------------|
| `out_dir` | `"./artifacts"` | Directory for checkpoints, metrics, and config snapshots |
| `best_metric` | `"f1"` | Metric to track for saving `model_best.pt` (standard pipeline) |

---

## Usage

```bash
# Standard training (cnn / lstm / byte_cnn / tcn)
python nids_ml/classifier_creator.py --config nids_ml/cnn_classifier.conf

# Two-way contrastive + nnPU training
python nids_ml/classifier_creator.py --config nids_ml/tcn_2way_classifier.conf

# Override epochs
python nids_ml/classifier_creator.py --config nids_ml/tcn_classifier.conf --epochs 20

# Dry run (single batch — validates data shapes and config without full training)
python nids_ml/classifier_creator.py --config nids_ml/cnn_classifier.conf --dry_run

# Force device
python nids_ml/classifier_creator.py --config nids_ml/tcn_classifier.conf --device cpu
```

Graceful shutdown: send `SIGINT` (Ctrl+C) or `SIGTERM` to stop after the current training step. The best checkpoint up to that point is preserved.

---

## Artifacts

After training, the following files are written to `artifacts.out_dir` (default `./artifacts`):

| File | Description |
|------|-------------|
| `model_best.pt` | Best checkpoint (by `best_metric` or best val F1) |
| `model_last.pt` | Checkpoint from the final epoch (standard pipeline) |
| `pretrain_epoch{N}.pt` | Backbone snapshot after each pretraining epoch (two-way pipeline) |
| `metrics.json` | Final train/val/test metrics |
| `config_used.json` | Snapshot of the exact config used for the run |
| `test_samples.json` | Per-sample test predictions (when `testing.log_samples` is set) |
