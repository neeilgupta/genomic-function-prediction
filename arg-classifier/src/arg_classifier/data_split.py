"""Cluster-aware train/val/test split.

Sequences whose k-mer Jaccard similarity exceeds `split.cluster_identity`
are grouped into the same cluster.  Whole clusters are assigned to a single
split, so no near-identical sequences leak across train/val/test.

If a family collapses to a single cluster (all variants are extremely
similar), sequences are distributed randomly within that cluster and a
warning is printed.
"""
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

from .utils import load_config, set_seed

MIN_TEST_CLASS = 10


# ── Clustering helpers ────────────────────────────────────────────────────────

def _kmer_set(sequence: str, k: int) -> frozenset:
    return frozenset(sequence[i : i + k] for i in range(len(sequence) - k + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def cluster_by_similarity(sequences: list, k: int, threshold: float) -> list:
    """Greedy single-linkage clustering on k-mer Jaccard similarity.

    Each cluster is represented by its first member.  A new sequence joins
    the first cluster whose representative has Jaccard >= threshold with it;
    otherwise a new cluster is created.

    Returns a list of integer cluster IDs (one per sequence).
    """
    ksets = [_kmer_set(s, k) for s in sequences]
    cluster_ids = []
    # representative index for each cluster
    representatives: list[int] = []

    for i, kset in enumerate(ksets):
        assigned = -1
        for cid, rep_idx in enumerate(representatives):
            if _jaccard(kset, ksets[rep_idx]) >= threshold:
                assigned = cid
                break
        if assigned == -1:
            assigned = len(representatives)
            representatives.append(i)
        cluster_ids.append(assigned)

    return cluster_ids


# ── Split helpers ─────────────────────────────────────────────────────────────

def _assign_splits(cluster_ids: list, train_frac: float, val_frac: float,
                   rng: np.random.Generator) -> list:
    """Map each cluster to train/val/test. Returns per-row split label."""
    unique = list(set(cluster_ids))
    rng.shuffle(unique)
    n = len(unique)
    n_train = max(1, round(n * train_frac))
    n_val   = max(1, round(n * val_frac))

    train_set = set(unique[:n_train])
    val_set   = set(unique[n_train : n_train + n_val])

    return [
        "train" if cid in train_set else ("val" if cid in val_set else "test")
        for cid in cluster_ids
    ]


def _random_splits(n: int, train_frac: float, val_frac: float,
                   rng: np.random.Generator) -> list:
    """Fallback: randomly assign n items to splits (single-cluster case)."""
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = max(1, round(n * train_frac))
    n_val   = max(1, round(n * val_frac))
    assignments = [""] * n
    for i in idx[:n_train]:
        assignments[i] = "train"
    for i in idx[n_train : n_train + n_val]:
        assignments[i] = "val"
    for i in idx[n_train + n_val :]:
        assignments[i] = "test"
    return assignments


# ── Main ──────────────────────────────────────────────────────────────────────

def run(config_path: str = "configs/mvp.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    raw_dir  = Path(cfg["data"]["raw_dir"])
    proc_dir = Path(cfg["data"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    k         = cfg["features"]["kmer_size"]
    threshold = cfg["split"].get("cluster_identity", 0.98)
    train_frac = cfg["split"]["train"]
    val_frac   = cfg["split"]["val"]
    families  = tuple(cfg["data"]["families"])

    df = pd.read_csv(raw_dir / "metadata.csv")
    print(f"Loaded {len(df)} sequences from metadata.csv")
    print(f"Clustering at Jaccard threshold={threshold}  (k={k})\n")

    split_col = [""] * len(df)
    cluster_report = {}

    for fam in families:
        mask = df["label"] == fam
        fam_idx = df.index[mask].tolist()
        seqs    = df.loc[fam_idx, "sequence"].tolist()

        cids = cluster_by_similarity(seqs, k, threshold)
        n_clusters = len(set(cids))
        cluster_report[fam] = n_clusters

        if n_clusters == 1:
            print(f"  {fam}: {len(seqs)} sequences → 1 cluster "
                  f"(all variants are >{threshold:.0%} similar; "
                  f"splitting sequences randomly within cluster)")
            assignments = _random_splits(len(seqs), train_frac, val_frac, rng)
        else:
            print(f"  {fam}: {len(seqs)} sequences → {n_clusters} clusters")
            assignments = _assign_splits(cids, train_frac, val_frac, rng)

        for row_pos, global_idx in enumerate(fam_idx):
            split_col[df.index.get_loc(global_idx)] = assignments[row_pos]

    df["_split"] = split_col

    train_df = df[df["_split"] == "train"].drop(columns="_split").reset_index(drop=True)
    val_df   = df[df["_split"] == "val"  ].drop(columns="_split").reset_index(drop=True)
    test_df  = df[df["_split"] == "test" ].drop(columns="_split").reset_index(drop=True)

    train_df.to_csv(proc_dir / "train.csv", index=False)
    val_df.to_csv(proc_dir / "val.csv",     index=False)
    test_df.to_csv(proc_dir / "test.csv",   index=False)

    # ── Verify no overlap ────────────────────────────────────────────────────
    train_acc = set(train_df["accession"])
    val_acc   = set(val_df["accession"])
    test_acc  = set(test_df["accession"])
    overlap_tv = train_acc & val_acc
    overlap_tt = train_acc & test_acc
    overlap_vt = val_acc   & test_acc
    assert not overlap_tv, f"Train/Val overlap: {overlap_tv}"
    assert not overlap_tt, f"Train/Test overlap: {overlap_tt}"
    assert not overlap_vt, f"Val/Test overlap: {overlap_vt}"

    # ── Summary ──────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("CLUSTER-AWARE TRAIN / VAL / TEST SPLIT SUMMARY")
    lines.append("=" * 60)
    lines.append(f"\nClustering: k={k}, Jaccard threshold={threshold}")
    lines.append(f"\nClusters per family:")
    for fam in families:
        n_c = cluster_report[fam]
        note = " ← single cluster, split randomly" if n_c == 1 else ""
        lines.append(f"  {fam}: {n_c} cluster(s){note}")

    lines.append(f"\n{'Split':<8} {'Total':>6}  " + "  ".join(f"{f:>5}" for f in families))
    lines.append("-" * 60)

    warnings = []
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist   = split_df["label"].value_counts()
        counts = [dist.get(f, 0) for f in families]
        lines.append(
            f"{split_name:<8} {len(split_df):>6}  " +
            "  ".join(f"{c:>5}" for c in counts)
        )
        if split_name == "test":
            for fam, c in zip(families, counts):
                if c < MIN_TEST_CLASS:
                    warnings.append(f"WARNING: {fam} has only {c} sequences in test set")

    lines.append("-" * 60)
    lines.append(f"\nAccession overlap checks:")
    lines.append(f"  Train ∩ Val  : {len(overlap_tv)}")
    lines.append(f"  Train ∩ Test : {len(overlap_tt)}")
    lines.append(f"  Val ∩ Test   : {len(overlap_vt)}")

    if warnings:
        lines.append("\nWarnings:")
        for w in warnings:
            lines.append(f"  {w}")
    else:
        lines.append(f"\nAll classes have ≥{MIN_TEST_CLASS} sequences in test set ✓")

    lines.append("=" * 60)
    summary = "\n".join(lines)
    print("\n" + summary)

    summary_path = proc_dir / "split_summary.txt"
    with open(summary_path, "w") as fh:
        fh.write(summary + "\n")
    print(f"\n✓ Splits written to {proc_dir}/")
    print(f"✓ Summary written to {summary_path}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/mvp.yaml"
    run(config)
