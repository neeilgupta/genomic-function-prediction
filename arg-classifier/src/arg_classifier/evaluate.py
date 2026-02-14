"""Evaluate trained model on test set and compare to baseline."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from .utils import load_config

FAMILIES = ["KPC", "NDM", "VIM", "IMP"]


def run(config_path: str = "configs/mvp.yaml") -> None:
    cfg = load_config(config_path)
    artifacts_dir = Path("artifacts")
    reports_dir   = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model & encoder ──────────────────────────────────────────────────
    with open(artifacts_dir / "model.pkl", "rb") as fh:
        clf = pickle.load(fh)
    with open(artifacts_dir / "label_encoder.pkl", "rb") as fh:
        le = pickle.load(fh)

    # ── Load test data ────────────────────────────────────────────────────────
    X_test     = np.load(artifacts_dir / "X_test.npy")
    y_test_raw = np.load(artifacts_dir / "y_test.npy", allow_pickle=True)
    y_test     = le.transform(y_test_raw)

    # ── Predict ───────────────────────────────────────────────────────────────
    y_pred       = clf.predict(X_test)
    y_proba      = clf.predict_proba(X_test)
    y_pred_labels = le.inverse_transform(y_pred)
    y_true_labels = le.inverse_transform(y_test)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report   = classification_report(
        y_true_labels, y_pred_labels, labels=FAMILIES, output_dict=True
    )
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=FAMILIES)

    print(f"{'='*52}")
    print("ML MODEL RESULTS  (Logistic Regression + TF-IDF)")
    print(f"{'='*52}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"\nPer-class metrics:")
    print(f"  {'Class':<6} {'P':>6} {'R':>6} {'F1':>6} {'n':>4}")
    print(f"  {'-'*28}")
    for fam in FAMILIES:
        r = report.get(fam, {})
        print(f"  {fam:<6} {r.get('precision',0):>6.3f} "
              f"{r.get('recall',0):>6.3f} {r.get('f1-score',0):>6.3f} "
              f"{int(r.get('support',0)):>4}")

    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(f"  {'':6s}" + "  ".join(f"{f:>5}" for f in FAMILIES))
    for fam, row in zip(FAMILIES, cm.tolist()):
        print(f"  {fam:<6}" + "  ".join(f"{v:>5}" for v in row))

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=FAMILIES, yticklabels=FAMILIES,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Confusion Matrix — Logistic Regression", fontsize=13)
    plt.tight_layout()
    cm_path = reports_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n✓ Confusion matrix plot → {cm_path}")

    # ── Save ML results JSON ──────────────────────────────────────────────────
    ml_results = {
        "model": "logistic_regression_tfidf",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": FAMILIES,
    }
    ml_path = reports_dir / "ml_results.json"
    with open(ml_path, "w") as fh:
        json.dump(ml_results, fh, indent=2)
    print(f"✓ ML results saved      → {ml_path}")

    # ── Load baseline & compare ───────────────────────────────────────────────
    baseline_path = reports_dir / "baseline_results.json"
    if not baseline_path.exists():
        print("\nBaseline results not found — skipping comparison.")
        return

    with open(baseline_path) as fh:
        bl = json.load(fh)

    lines = []
    lines.append("=" * 58)
    lines.append("MODEL COMPARISON")
    lines.append("=" * 58)
    lines.append(f"{'Metric':<22} {'Baseline (kNN)':>16} {'LR + TF-IDF':>16}")
    lines.append("-" * 58)
    lines.append(f"{'Accuracy':<22} {bl['accuracy']:>16.4f} {acc:>16.4f}")
    lines.append(f"{'Macro F1':<22} {bl['macro_f1']:>16.4f} {macro_f1:>16.4f}")

    bl_runtime = bl.get("runtime_seconds", "N/A")
    lines.append(f"{'Runtime (s)':<22} {str(bl_runtime):>16} {'<0.01':>16}")
    lines.append("-" * 58)

    lines.append(f"\nPer-class F1 comparison:")
    lines.append(f"  {'Class':<6} {'Baseline':>10} {'LR+TF-IDF':>10}")
    lines.append(f"  {'-'*30}")
    for fam in FAMILIES:
        bl_f1 = bl["classification_report"].get(fam, {}).get("f1-score", 0)
        ml_f1 = report.get(fam, {}).get("f1-score", 0)
        delta = ml_f1 - bl_f1
        flag = "  ▲" if delta > 0 else ("  ▼" if delta < 0 else "  =")
        lines.append(f"  {fam:<6} {bl_f1:>10.3f} {ml_f1:>10.3f}{flag}")

    lines.append("=" * 58)
    comparison = "\n".join(lines)
    print(f"\n{comparison}")

    cmp_path = reports_dir / "comparison.txt"
    with open(cmp_path, "w") as fh:
        fh.write(comparison + "\n")
    print(f"\n✓ Comparison saved      → {cmp_path}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/mvp.yaml"
    run(config)
