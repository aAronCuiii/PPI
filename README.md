# PPI Network Discovery From Patient Proteomics

This project builds protein-protein interaction (PPI) candidate networks from a
protein x patient proteomics matrix. It compares correlation, missingness-aware
kernel, autoencoder, and transformer similarity methods, then builds protein
networks, detects modules, evaluates against CORUM complexes, and annotates
selected modules with CORUM, g:Profiler, and OpenAI-assisted interpretation.

The main entry points are command-line Python scripts under `src/`, `train/`,
and `infer/`.

Large data and output artifacts are stored externally on Google Drive:
[PPI data and outputs](https://drive.google.com/drive/folders/1OuwYYEU3DOsLKCA65QM21ZxXyyPITRYp?usp=sharing).
For running similarity calculation and training, please download and unzip the data and place the folder in ROOT directory.

## Repository Layout

```text
data/
  raw/                         Raw proteomics and CORUM input files
  processed/                   Processed protein x patient matrix
src/
  similarity/                  Correlation and kernel similarity methods
  model/                       Autoencoder and transformer model definitions
  rank/                        Matrix-to-edge ranking utilities
  graph/                       Network construction and visualization
  modules/                     Leiden and MCODE module detection
  evaluation/                  CORUM pair and complex-level evaluation
  annotation/                  CORUM/g:Profiler/OpenAI module annotation
train/                         Self-supervised model training scripts
infer/                         Embedding and cosine-similarity inference scripts
notebooks/                     Exploratory analysis notebooks
checkpoints/                   Trained SSL model checkpoints and logs
outputs/                       Similarity matrices, graphs, modules, results
```

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For OpenAI-based interpretation, create a local `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
```

The OpenAI step is optional. All non-LLM analysis can run without an API key.

## Input Data

The primary processed input is:

```text
data/processed/proteomics_data_processed.csv
```

Expected shape is proteins as rows and patients/samples as columns. Missing
protein measurements should be represented as `NaN`.

Current processed dataset:

| Item | Value |
| --- | ---: |
| Proteins | 12,142 |
| Patients | 118 |
| Proteomics-CORUM overlap genes | 3,713 |
| CORUM-positive protein pairs in proteomics | 40,776 |
| CORUM complexes with proteomics coverage | 3,627 |

## End-to-End Usage

The pipeline can be run in stages. Existing outputs in this repository follow
the folder conventions shown below.

### 1. Compute Similarity Matrices

NaN-aware baseline correlations:

```bash
python -m src.similarity.baseline
```

Reliability-adjusted correlations and combined abundance/missingness scores:

```bash
python -m src.similarity.adjusted_correlation
```

The correlation scripts are currently script-style modules. Before running them
from scratch, confirm the input path, output path, and minimum-overlap settings
inside the script match the dataset you want to process. In the current result
layout, correlation-family matrices are expected under
`outputs/similarity_correlation/`.

Kernel similarities:

```bash
python -m src.similarity.kernel \
  --input data/processed/proteomics_data_processed.csv \
  --output_dir outputs/similarity_correlation \
  --min_observed 10 \
  --min_pair_overlap 10 \
  --gamma 0.5 \
  --alpha 0.8 \
  --adjust_by_overlap
```

Note: some similarity scripts are exploratory and contain hard-coded default
input/output paths. Check the script header before regenerating outputs from
scratch.

Before ranking a directory of `.npy` matrices, make sure that directory contains
a `protein_index.csv` file with the same protein order as the matrix rows and
columns. The autoencoder and transformer inference scripts write this file
automatically.

### 2. Train Self-Supervised Models

Mask-aware autoencoder:

```bash
python -m train.train_ssl_autoencoder \
  --input data/processed/proteomics_data_processed.csv \
  --checkpoint_dir checkpoints/ssl_autoencoder \
  --normalization_mode protein \
  --mask_prob 0.3 \
  --hidden_dim 256 \
  --latent_dim 64 \
  --dropout 0.1 \
  --batch_size 512 \
  --epochs 500 \
  --lr 1e-4 \
  --weight_decay 1e-3 \
  --patience 50
```

Patient-token transformer:

```bash
python -m train.train_ssl_transformer \
  --input data/processed/proteomics_data_processed.csv \
  --checkpoint_dir checkpoints/ssl_transformer \
  --normalization_mode protein \
  --mask_prob 0.3 \
  --d_model 64 \
  --n_heads 4 \
  --n_layers 2 \
  --dim_feedforward 128 \
  --latent_dim 64 \
  --dropout 0.1 \
  --batch_size 256 \
  --epochs 500 \
  --lr 5e-4 \
  --weight_decay 1e-3 \
  --patience 50
```

Current trained checkpoints:

| Model | Normalization | Mask probability | Latent dim | Best validation loss |
| --- | --- | ---: | ---: | ---: |
| Autoencoder | protein | 0.3 | 64 | 0.5180 |
| Transformer | protein | 0.3 | 64 | 0.5679 |

### 3. Infer Embeddings and Cosine Similarities

Autoencoder embeddings:

```bash
python -m infer.infer_ssl_autoencoder \
  --input data/processed/proteomics_data_processed.csv \
  --checkpoint checkpoints/ssl_autoencoder/ssl_autoencoder_best.pt \
  --output_dir outputs/similarity_autoencoder \
  --normalization_mode checkpoint \
  --stats_mode checkpoint \
  --batch_size 512 \
  --top_k 100000
```

Transformer embeddings:

```bash
python -m infer.infer_ssl_transformer \
  --input data/processed/proteomics_data_processed.csv \
  --checkpoint checkpoints/ssl_transformer/ssl_transformer_best.pt \
  --output_dir outputs/similarity_transformer \
  --normalization_mode checkpoint \
  --stats_mode checkpoint \
  --batch_size 512 \
  --top_k 100000
```

These commands write embeddings, `protein_index.csv`, full cosine matrices, and
top-ranked edge tables.

### 4. Rank Similarity Matrices Into Edge Tables

```bash
python -m src.rank.rank_similarity \
  --similarity-dirs outputs/similarity_correlation outputs/similarity_autoencoder outputs/similarity_transformer \
  --output-dir outputs/network/rank \
  --top-k 100000 \
  --thresholds 0.85
```

This creates two edge-selection views per similarity method:

- `top100000`: the 100,000 highest-scoring protein pairs.
- `threshold_0p85`: all protein pairs with score >= 0.85.

### 5. Build Graphs

Graphs from ranked edge tables:

```bash
python -m src.graph.build_graph \
  --rank-dir outputs/network/rank \
  --graph-root outputs/network/graph
```

Top-k-per-node graphs:

```bash
python -m src.graph.build_graph \
  --build-top-per-node \
  --per-node-k 10 \
  --similarity-dirs outputs/similarity_correlation outputs/similarity_autoencoder outputs/similarity_transformer \
  --graph-root outputs/network/graph
```

Optional interactive Plotly graph:

```bash
python -m src.graph.plotly_network \
  --edges outputs/network/graph/autoencoder_cosine/top10perNodes/edges.csv \
  --output-html outputs/network/graph/autoencoder_cosine/top10perNodes/network.html \
  --top-n 5000
```

### 6. Detect Network Modules

```bash
python -m src.modules.detect_modules \
  --graph-root outputs/network/graph \
  --output-root outputs/network/modules
```

The module step runs:

- Leiden community detection for broad graph modules.
- MCODE-style dense-core detection for tighter protein subcomplex candidates.

### 7. Evaluate Against CORUM

```bash
python -m src.evaluation.evaluate_corum \
  --overlap_genes outputs/dataset_overlap/overlap_proteomics_CORUM_genes.csv \
  --positive_pairs outputs/dataset_overlap/CORUM_positive_pairs_in_proteomics.csv \
  --complex_coverage outputs/dataset_overlap/CORUM_complex_coverage_in_proteomics.csv \
  --output_dir outputs/evaluation_CORUM \
  --k_values 100,500,1000,5000,10000 \
  --min_complex_size 5 \
  --n_random 500
```

Evaluation treats CORUM co-complex pairs as positives and all other pairs in the
evaluation universe as background, not guaranteed true negatives.

### 8. Annotate Selected Modules

```bash
python -m src.annotation.run_annotation
```

This selects the top 5 Leiden modules and top 5 MCODE cores for these methods:

- `autoencoder_cosine`
- `composite_rbf_jaccard`
- `combined_pearson_jaccard`

It writes CORUM enrichment, g:Profiler enrichment, and summary comparison tables
under `outputs/network/annotation/`.

Optional OpenAI interpretation:

```bash
python -m src.annotation.llm_interpret
```

## Methods Used

| Method | Output name | Description |
| --- | --- | --- |
| Pearson correlation | `pearson` | NaN-aware pairwise Pearson over jointly observed patients. |
| Spearman correlation | `spearman` | Per-protein rank transform followed by NaN-aware Pearson. |
| Reliability-adjusted Pearson | `pearson_adjusted` | Pearson multiplied by `sqrt(shared_patients / total_patients)`. |
| Reliability-adjusted Spearman | `spearman_adjusted` | Spearman multiplied by the same overlap reliability weight. |
| Combined Pearson + Jaccard | `combined_pearson_jaccard` | Adjusted Pearson plus a weighted detection-pattern Jaccard term. |
| Combined Spearman + Jaccard | `combined_spearman_jaccard` | Adjusted Spearman plus a weighted detection-pattern Jaccard term. |
| Intensity RBF | `intensity_rbf` | NaN-aware RBF kernel over standardized abundance values. |
| Composite RBF + Jaccard | `composite_rbf_jaccard` | Weighted combination of intensity RBF and detection-pattern Jaccard. |
| SSL autoencoder cosine | `autoencoder_cosine` | Cosine similarity between learned autoencoder protein embeddings. |
| SSL transformer cosine | `transformer_cosine` | Cosine similarity between learned patient-token transformer embeddings. |

## Current Results

### CORUM Pair-Level Evaluation

Sorted by AUPRC:

| Method | AUROC | AUPRC | Precision@1k | Enrichment@1k | Precision@10k | Enrichment@10k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `composite_rbf_jaccard` | 0.8012 | 0.1383 | 0.890 | 143.56 | 0.4120 | 66.46 |
| `intensity_rbf` | 0.8022 | 0.1382 | 0.890 | 143.56 | 0.4120 | 66.46 |
| `combined_pearson_jaccard` | 0.8321 | 0.1336 | 0.661 | 106.62 | 0.3832 | 61.81 |
| `pearson_adjusted` | 0.8309 | 0.1333 | 0.661 | 106.62 | 0.3826 | 61.71 |
| `combined_spearman_jaccard` | 0.8330 | 0.1261 | 0.587 | 94.68 | 0.3626 | 58.49 |
| `spearman_adjusted` | 0.8320 | 0.1259 | 0.587 | 94.68 | 0.3620 | 58.39 |
| `autoencoder_cosine` | 0.8236 | 0.1250 | 0.584 | 98.45 | 0.3978 | 67.06 |
| `pearson` | 0.8294 | 0.1227 | 0.598 | 96.46 | 0.3588 | 57.87 |
| `spearman` | 0.8308 | 0.1192 | 0.560 | 90.33 | 0.3474 | 56.04 |
| `transformer_cosine` | 0.7958 | 0.0744 | 0.404 | 68.11 | 0.2584 | 43.56 |

The best pair-level AUPRC was achieved by `composite_rbf_jaccard` and
`intensity_rbf`. The highest Precision@10k among the listed methods was
`composite_rbf_jaccard` / `intensity_rbf` at 0.4120, while
`autoencoder_cosine` also performed strongly at 0.3978.

### Top-10-Per-Node Graphs

| Method | Nodes | Edges | Leiden modules | Top module size | MCODE cores | Top MCODE size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `autoencoder_cosine` | 12,142 | 93,250 | 47 | 664 | 976 | 15 |
| `combined_pearson_jaccard` | 11,356 | 101,991 | 39 | 780 | 935 | 23 |
| `combined_spearman_jaccard` | 11,356 | 101,788 | 41 | 746 | 839 | 18 |
| `composite_rbf_jaccard` | 11,356 | 107,520 | 36 | 1,102 | 1,126 | 25 |
| `intensity_rbf` | 11,356 | 106,705 | 37 | 785 | 953 | 24 |
| `pearson` | 11,356 | 99,052 | 47 | 598 | 590 | 14 |
| `pearson_adjusted` | 11,356 | 102,008 | 38 | 837 | 959 | 23 |
| `spearman` | 11,356 | 98,471 | 56 | 520 | 484 | 18 |
| `spearman_adjusted` | 11,356 | 101,832 | 39 | 776 | 840 | 26 |
| `transformer_cosine` | 12,142 | 84,630 | 43 | 490 | 808 | 24 |

### Functional Annotation Highlights

The selected annotation set contains 30 targets:

- 3 methods
- 5 Leiden modules per method
- 5 MCODE cores per method

Method-level annotation summary:

| Method | Leiden CORUM hits | Leiden clusters with CORUM | MCODE CORUM hits | MCODE cores with CORUM | Total significant terms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `autoencoder_cosine` | 633 | 100% | 41 | 80% | 6,305 |
| `composite_rbf_jaccard` | 492 | 100% | 0 | 0% | 4,740 |
| `combined_pearson_jaccard` | 532 | 100% | 67 | 80% | 3,722 |

Representative annotated modules:

| Method | Cluster | Size | Top CORUM match | Top compartment/pathway signal |
| --- | --- | ---: | --- | --- |
| `autoencoder_cosine` | Leiden `L0001` | 664 | TNFR1 signaling complex, TNF-induced | cytosol / Signal Transduction |
| `autoencoder_cosine` | MCODE `MC0001` | 15 | PA700 complex | proteasome regulatory particle / Proteasome |
| `composite_rbf_jaccard` | Leiden `L0002` | 694 | cBAF complex | nucleus / Gene expression |
| `combined_pearson_jaccard` | Leiden `L0001` | 780 | DNAJB11-SDF2L1 complex | endoplasmic reticulum / N-linked glycosylation |
| `combined_pearson_jaccard` | Leiden `L0002` | 582 | F1F0-ATPase, mitochondrial | mitochondrion / aerobic respiration |

## Key Output Files

| Path | Contents |
| --- | --- |
| `outputs/similarity_correlation/` | Correlation, adjusted correlation, and kernel similarity matrices. |
| `outputs/similarity_autoencoder/` | Autoencoder embeddings, protein index, cosine matrix, and top edges. |
| `outputs/similarity_transformer/` | Transformer embeddings, protein index, cosine matrix, and top edges. |
| `outputs/network/rank/` | Ranked and thresholded edge tables for all methods. |
| `outputs/network/graph/` | Network summaries, node tables, previews, and top-per-node edge sets. |
| `outputs/network/modules/` | Leiden assignments, module summaries, MCODE cores, and overview tables. |
| `outputs/evaluation_CORUM/` | CORUM pair-level and complex-level evaluation results. |
| `outputs/network/annotation/` | CORUM enrichment, g:Profiler enrichment, and LLM interpretations. |
| `checkpoints/ssl_autoencoder/` | Autoencoder checkpoints, config, loss plot, and training log. |
| `checkpoints/ssl_transformer/` | Transformer checkpoints, config, loss plot, and training log. |

## Notes

- Outputs can be large. The `.gitignore` excludes `data/`, `outputs/`, and
  `checkpoints/` by default.
- `MPLCONFIGDIR` may need to point to a writable directory on locked-down
  systems when generating Matplotlib previews.
- CORUM background pairs are not true negatives; interpret AUROC/AUPRC and
  enrichment as benchmark signals, not definitive PPI labels.
- The OpenAI interpretation step summarizes enrichment evidence. It should be
  reviewed as biological hypothesis generation rather than ground truth.
