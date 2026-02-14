# ARG Family Classifier

Classify antibiotic resistance gene (ARG) sequences into carbapenemase families using machine learning.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start — Predict on Your Own FASTA

```bash
python -m src.arg_classifier.predict \
    --fasta my_sequences.fasta \
    --output predictions.csv
```

Output CSV columns: `sequence_id, predicted_label, confidence, prob_KPC, prob_NDM, prob_VIM, prob_IMP`

## Reproducing the Full Pipeline

Run each step in order from the project root:

```bash
# 1. Parse CARD FASTA → metadata.csv (489 sequences)
python -m src.arg_classifier.data_acquisition

# 2. Validate dataset (class distribution, duplicates, length stats)
python -m src.arg_classifier.data_validation

# 3. Train / val / test split at accession level (70/15/15)
python -m src.arg_classifier.data_split

# 4. K-mer TF-IDF featurization (k=5, 1,023 features)
python -m src.arg_classifier.featurize_kmer

# 5. Jaccard nearest-neighbour baseline
python -m src.arg_classifier.baseline_similarity

# 6. Train Logistic Regression
python -m src.arg_classifier.train

# 7. Evaluate and compare to baseline
python -m src.arg_classifier.evaluate
```

## Results

### Model Comparison

|                     | Accuracy | Macro F1 | Inference Time |
|---------------------|----------|----------|----------------|
| Baseline (kNN)      | 1.0000   | 1.0000   | 0.7 s          |
| Logistic Regression | 1.0000   | 1.0000   | < 0.01 s       |

### Per-Class F1 (test set, n=74)

| Class | Precision | Recall | F1    | Support |
|-------|-----------|--------|-------|---------|
| KPC   | 1.000     | 1.000  | 1.000 | 35      |
| NDM   | 1.000     | 1.000  | 1.000 | 10      |
| VIM   | 1.000     | 1.000  | 1.000 | 14      |
| IMP   | 1.000     | 1.000  | 1.000 | 15      |

## Why Does the Baseline Achieve Perfect Accuracy?

Short answer: allelic variants within each carbapenemase family differ by only 1–2 SNPs
(>99% nucleotide identity). A nearest-neighbour search over k-mer sets will always find
an almost-identical training sequence and assign the correct label.

This is not a data leakage bug — accessions are fully disjoint between splits. It reflects
the **biological reality** of curated ARG databases such as CARD: each family forms a tight
sequence cluster that is clearly separated from the others.

**Value of the ML approach over nearest-neighbour:**

| Property                  | kNN Baseline                  | Logistic Regression          |
|---------------------------|-------------------------------|------------------------------|
| Inference time            | O(n_train) — 0.7 s            | O(1) — <0.01 s               |
| Memory at runtime         | 342 sequences (270 KB)        | 1,023 coefficients (40 KB)   |
| Outputs probability       | No                            | Yes (calibrated softmax)     |
| Deployable to edge        | No                            | Yes                          |

Even when accuracy is tied, the ML model is **~70× faster**, requires 7× less memory,
and natively provides per-class probability scores for downstream decision-making.

## Future Improvements

1. **Harder evaluation split** — cluster sequences at 90% identity (cd-hit) and split
   by cluster, so the test set contains novel alleles with no close training neighbour.

2. **More ARG families** — expand from 4 carbapenemase families to 15+ families
   (OXA, CTX-M, TEM, SHV, …) to test multi-class scalability.

3. **Deep learning** — replace TF-IDF with a 1-D CNN or Transformer encoder
   trained end-to-end on raw nucleotide sequences.

4. **Protein-space features** — translate to amino acid sequences and use ESM-2
   embeddings for richer representation.

## Project Layout

```
arg-classifier/
├── configs/mvp.yaml           # Hyperparameters and paths
├── data/
│   ├── raw/                   # card_sequences.fasta, metadata.csv
│   └── processed/             # train / val / test CSVs
├── artifacts/                 # model.pkl, vectorizer.pkl, *.npy
├── reports/                   # JSON metrics, PNG plots, text summaries
└── src/arg_classifier/
    ├── io_fasta.py            # FASTA I/O
    ├── utils.py               # Config loader, seed setter
    ├── data_acquisition.py    # Parse CARD → metadata.csv
    ├── data_validation.py     # Dataset QC
    ├── data_split.py          # Accession-level train/val/test split
    ├── featurize_kmer.py      # K-mer TF-IDF featurization
    ├── baseline_similarity.py # Jaccard kNN baseline
    ├── train.py               # Logistic Regression training
    ├── evaluate.py            # Metrics, confusion matrix, comparison
    └── predict.py             # CLI inference tool
```

## Project Status

- [x] Milestone 1: Repository scaffolding
- [x] Milestone 2: Data acquisition
- [x] Milestone 3: Train/val/test split
- [x] Milestone 4: K-mer featurization
- [x] Milestone 5: Baseline implementation
- [x] Milestone 6: ML model training
- [x] Milestone 7: CLI inference tool
- [x] Milestone 8: Documentation & polish
