"""K-mer Jaccard nearest-neighbour baseline classifier."""
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from .utils import load_config, set_seed

FAMILIES = ("KPC", "NDM", "VIM", "IMP")


def kmer_set(sequence: str, k: int) -> frozenset:
    return frozenset(sequence[i:i + k] for i in range(len(sequence) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def run(config_path: str = "configs/mvp.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    k        = cfg["features"]["kmer_size"]
    proc_dir = Path(cfg["data"]["processed_dir"])
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(proc_dir / "train.csv")
    test_df  = pd.read_csv(proc_dir / "test.csv")

    print(f"Train: {len(train_df)} sequences  |  Test: {len(test_df)} sequences")
    print(f"k-mer size: {k}")
    print(f"\nBuilding train k-mer sets …")
    train_sets  = [kmer_set(s, k) for s in train_df["sequence"]]
    train_labels = train_df["label"].tolist()

    print("Running nearest-neighbour classification …\n")
    t0 = time.time()

    y_true, y_pred, confidences = [], [], []

    for idx, row in test_df.iterrows():
        test_kset = kmer_set(row["sequence"], k)
        best_sim, best_label = -1.0, None

        for tr_set, tr_label in zip(train_sets, train_labels):
            sim = jaccard(test_kset, tr_set)
            if sim > best_sim:
                best_sim   = sim
                best_label = tr_label

        y_true.append(row["label"])
        y_pred.append(best_label)
        confidences.append(best_sim)

        pos = test_df.index.get_loc(idx) + 1
        if pos % 10 == 0 or pos == len(test_df):
            elapsed = time.time() - t0
            print(f"  [{pos:3d}/{len(test_df)}]  elapsed: {elapsed:.1f}s  "
                  f"last sim: {best_sim:.3f}  predicted: {best_label}")

    elapsed_total = time.time() - t0

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=list(FAMILIES))
    report   = classification_report(
        y_true, y_pred, labels=list(FAMILIES), output_dict=True
    )
    cm       = confusion_matrix(y_true, y_pred, labels=list(FAMILIES)).tolist()

    print(f"\n{'='*50}")
    print("BASELINE RESULTS  (Jaccard Nearest Neighbour)")
    print(f"{'='*50}")
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Macro F1   : {macro_f1:.4f}")
    print(f"  Mean conf. : {sum(confidences)/len(confidences):.4f}")
    print(f"  Runtime    : {elapsed_total:.1f}s")
    print(f"\nPer-class F1:")
    for fam in FAMILIES:
        r = report.get(fam, {})
        print(f"  {fam}: P={r.get('precision',0):.3f}  "
              f"R={r.get('recall',0):.3f}  F1={r.get('f1-score',0):.3f}  "
              f"n={r.get('support',0)}")
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(f"  {'':6s}" + "  ".join(f"{f:>5}" for f in FAMILIES))
    for fam, row_cm in zip(FAMILIES, cm):
        print(f"  {fam:<6s}" + "  ".join(f"{v:>5}" for v in row_cm))
    print(f"{'='*50}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results = {
        "model": "jaccard_nearest_neighbour",
        "k": k,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "mean_confidence": round(sum(confidences) / len(confidences), 4),
        "runtime_seconds": round(elapsed_total, 1),
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": list(FAMILIES),
    }
    out_path = reports_dir / "baseline_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n✓ Results saved → {out_path}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/mvp.yaml"
    run(config)
