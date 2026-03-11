"""Evaluate trained model on test set and compare to baseline."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from .utils import load_config
from .io_fasta import load_fasta
from .featurize_kmer import seq_to_kmers


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
    FAMILIES = list(le.classes_)

    # ── Load test data ────────────────────────────────────────────────────────
    X_test     = scipy.sparse.load_npz(artifacts_dir / "X_test.npz")
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
    n_classes = len(FAMILIES)
    fig_size = max(6, n_classes)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))
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

    # ── Calibration Check ─────────────────────────────────────────────────────
    max_proba = y_proba.max(axis=1)              # top-class confidence per sequence
    y_correct = (y_pred == y_test).astype(int)   # 1=correct prediction, 0=wrong

    # ECE: weighted mean |acc_in_bin - conf_in_bin| across 10 uniform bins
    n_bins = 10
    ece = 0.0
    N   = len(max_proba)
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (max_proba >= lo) & (max_proba < hi)
        if mask.sum() == 0:
            continue
        ece += abs(y_correct[mask].mean() - max_proba[mask].mean()) * mask.sum() / N

    frac_pos, mean_conf = calibration_curve(
        y_correct, max_proba, n_bins=n_bins, strategy="uniform"
    )

    print(f"\n{'='*52}")
    print("CALIBRATION CHECK")
    print(f"{'='*52}")
    print(f"  ECE (Expected Calibration Error): {ece:.4f}")
    print(f"  (0.00 = perfect; >0.05 = consider recalibrating)")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_conf, frac_pos, "s-", color="steelblue", label="LR + TF-IDF")
    ax.set_xlabel("Mean predicted confidence", fontsize=12)
    ax.set_ylabel("Fraction correct", fontsize=12)
    ax.set_title("Reliability Diagram (Top-Class Calibration)", fontsize=13)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    cal_path = reports_dir / "calibration_plot.png"
    plt.savefig(cal_path, dpi=150)
    plt.close()
    print(f"✓ Calibration plot      → {cal_path}")

    calibration_stats = {
        "ece": round(ece, 4),
        "n_bins": n_bins,
        "reliability": [
            {"mean_conf": round(float(c), 4), "fraction_correct": round(float(a), 4)}
            for c, a in zip(mean_conf, frac_pos)
        ],
    }

    # ── OOD Detection ─────────────────────────────────────────────────────────
    ood_threshold = cfg.get("inference", {}).get("ood_threshold")
    k = cfg["features"]["kmer_size"]

    with open(artifacts_dir / "vectorizer.pkl", "rb") as fh:
        vectorizer = pickle.load(fh)

    ood_stats = None
    other_fasta = Path(cfg["data"]["raw_dir"]) / "other_sequences.fasta"

    if ood_threshold is None:
        print("\nOOD threshold not configured — skipping OOD evaluation.")
    elif not other_fasta.exists():
        print(f"\nOOD evaluation skipped — {other_fasta} not found.")
        print("  Run: python3.11 scripts/generate_synthetic_data.py")
    else:
        # False-positive rate: known-family test sequences wrongly flagged UNKNOWN
        test_max_proba = y_proba.max(axis=1)
        n_fp = int((test_max_proba < ood_threshold).sum())
        fpr = n_fp / len(test_max_proba)

        # True-positive rate: "other" sequences correctly flagged UNKNOWN
        other_records = load_fasta(other_fasta)
        other_docs = [seq_to_kmers(r["sequence"], k) for r in other_records]
        X_other = vectorizer.transform(other_docs).toarray().astype(np.float32)
        other_proba = clf.predict_proba(X_other)
        other_max = other_proba.max(axis=1)
        n_tp = int((other_max < ood_threshold).sum())
        n_other = len(other_records)
        tpr = n_tp / n_other

        print(f"\n{'='*52}")
        print(f"OOD DETECTION  (threshold={ood_threshold})")
        print(f"{'='*52}")
        print(f"  Known-family sequences (test set, n={len(test_max_proba)}):")
        print(f"    Correctly retained (not flagged UNKNOWN): "
              f"{len(test_max_proba)-n_fp}/{len(test_max_proba)}  ({(1-fpr)*100:.1f}%)")
        print(f"    False positive rate (wrongly → UNKNOWN):  "
              f"{n_fp}/{len(test_max_proba)}  ({fpr*100:.1f}%)")
        print(f"  'Other' sequences (n={n_other}):")
        print(f"    Correctly flagged UNKNOWN: {n_tp}/{n_other}  ({tpr*100:.1f}%)")
        print(f"    Missed (given a family label): {n_other-n_tp}/{n_other}  ({(1-tpr)*100:.1f}%)")

        ood_stats = {
            "threshold": ood_threshold,
            "n_test": len(test_max_proba),
            "n_test_flagged_unknown": n_fp,
            "false_positive_rate": round(fpr, 4),
            "n_other": n_other,
            "n_other_flagged_unknown": n_tp,
            "true_positive_rate": round(tpr, 4),
        }

    # ── Save ML results JSON ──────────────────────────────────────────────────
    ml_results = {
        "model": "logistic_regression_tfidf",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": FAMILIES,
    }
    ml_results["calibration"] = calibration_stats
    if ood_stats is not None:
        ml_results["ood_detection"] = ood_stats
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
