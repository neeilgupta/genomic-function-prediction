"""Convert DNA sequences to k-mer TF-IDF feature vectors."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from .utils import load_config, set_seed

SPLITS = ("train", "val", "test")


def seq_to_kmers(sequence: str, k: int) -> str:
    """Return space-separated k-mers for a DNA sequence."""
    return " ".join(sequence[i:i + k] for i in range(len(sequence) - k + 1))


def run(config_path: str = "configs/mvp.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    proc_dir     = Path(cfg["data"]["processed_dir"])
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    k         = cfg["features"]["kmer_size"]
    use_tfidf = cfg["features"]["use_tfidf"]

    # ── Load splits ──────────────────────────────────────────────────────────
    dfs = {split: pd.read_csv(proc_dir / f"{split}.csv") for split in SPLITS}
    print(f"Loaded splits — train: {len(dfs['train'])}, "
          f"val: {len(dfs['val'])}, test: {len(dfs['test'])}")

    # ── Build k-mer strings ──────────────────────────────────────────────────
    kmer_docs = {
        split: df["sequence"].apply(lambda s: seq_to_kmers(s, k))
        for split, df in dfs.items()
    }

    # ── Fit vectorizer on train only ─────────────────────────────────────────
    VectClass = TfidfVectorizer if use_tfidf else CountVectorizer
    vectorizer = VectClass(analyzer="word", token_pattern=r"(?u)\b\w+\b")
    vectorizer.fit(kmer_docs["train"])

    vocab_size = len(vectorizer.vocabulary_)
    vect_name  = "TF-IDF" if use_tfidf else "Count"
    print(f"\nVectorizer : {vect_name}Vectorizer")
    print(f"k-mer size : {k}")
    print(f"Vocab size : {vocab_size:,}")

    # ── Transform all splits → dense arrays ─────────────────────────────────
    print("\nFeature matrix shapes:")
    for split in SPLITS:
        X_sparse = vectorizer.transform(kmer_docs[split])
        X_dense  = X_sparse.toarray().astype(np.float32)
        y        = dfs[split]["label"].values

        np.save(artifacts_dir / f"X_{split}.npy", X_dense)
        np.save(artifacts_dir / f"y_{split}.npy", y)

        zeros    = (X_dense == 0).sum()
        sparsity = 100.0 * zeros / X_dense.size
        print(f"  X_{split:<5}: {str(X_dense.shape):<20}  sparsity: {sparsity:.1f}%")

    # ── Save vectorizer ──────────────────────────────────────────────────────
    vec_path = artifacts_dir / "vectorizer.pkl"
    with open(vec_path, "wb") as fh:
        pickle.dump(vectorizer, fh)

    print(f"\n✓ Saved vectorizer  → {vec_path}")
    print(f"✓ Saved X/y arrays  → {artifacts_dir}/X_{{train,val,test}}.npy, "
          f"y_{{train,val,test}}.npy")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/mvp.yaml"
    run(config)
