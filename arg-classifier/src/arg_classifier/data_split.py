"""Stratified train/val/test split at the accession level to prevent leakage."""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from .utils import load_config, set_seed

FAMILIES = ("KPC", "NDM", "VIM", "IMP")
MIN_TEST_CLASS = 10


def split_accessions(accessions, train_frac, val_frac, rng):
    """Shuffle accessions and split into train/val/test lists."""
    acc = list(accessions)
    rng.shuffle(acc)
    n = len(acc)
    n_train = max(1, round(n * train_frac))
    n_val   = max(1, round(n * val_frac))
    train = acc[:n_train]
    val   = acc[n_train:n_train + n_val]
    test  = acc[n_train + n_val:]
    return train, val, test


def run(config_path: str = "configs/mvp.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    raw_dir  = Path(cfg["data"]["raw_dir"])
    proc_dir = Path(cfg["data"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_dir / "metadata.csv")
    print(f"Loaded {len(df)} sequences from metadata.csv")

    train_frac = cfg["split"]["train"]
    val_frac   = cfg["split"]["val"]

    train_acc, val_acc, test_acc = set(), set(), set()

    for fam in FAMILIES:
        fam_df = df[df["label"] == fam]
        unique_acc = fam_df["accession"].unique()
        tr, va, te = split_accessions(unique_acc, train_frac, val_frac, rng)
        train_acc.update(tr)
        val_acc.update(va)
        test_acc.update(te)

    # Verify no overlap
    overlap_tv = train_acc & val_acc
    overlap_tt = train_acc & test_acc
    overlap_vt = val_acc   & test_acc
    assert not overlap_tv, f"Train/Val overlap: {overlap_tv}"
    assert not overlap_tt, f"Train/Test overlap: {overlap_tt}"
    assert not overlap_vt, f"Val/Test overlap: {overlap_vt}"

    train_df = df[df["accession"].isin(train_acc)].reset_index(drop=True)
    val_df   = df[df["accession"].isin(val_acc)].reset_index(drop=True)
    test_df  = df[df["accession"].isin(test_acc)].reset_index(drop=True)

    train_df.to_csv(proc_dir / "train.csv", index=False)
    val_df.to_csv(proc_dir / "val.csv",   index=False)
    test_df.to_csv(proc_dir / "test.csv", index=False)

    # ── Build summary ────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 55)
    lines.append("TRAIN / VAL / TEST SPLIT SUMMARY")
    lines.append("=" * 55)
    lines.append(f"\n{'Split':<8} {'Total':>6}  " + "  ".join(f"{f:>5}" for f in FAMILIES))
    lines.append("-" * 55)

    warnings = []
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df["label"].value_counts()
        counts = [dist.get(f, 0) for f in FAMILIES]
        lines.append(
            f"{split_name:<8} {len(split_df):>6}  " +
            "  ".join(f"{c:>5}" for c in counts)
        )
        if split_name == "test":
            for fam, c in zip(FAMILIES, counts):
                if c < MIN_TEST_CLASS:
                    warnings.append(f"WARNING: {fam} has only {c} sequences in test set")

    lines.append("-" * 55)
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

    lines.append("=" * 55)
    summary = "\n".join(lines)
    print(summary)

    summary_path = proc_dir / "split_summary.txt"
    with open(summary_path, "w") as fh:
        fh.write(summary + "\n")
    print(f"\n✓ Splits written to {proc_dir}/")
    print(f"✓ Summary written to {summary_path}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/mvp.yaml"
    run(config)
