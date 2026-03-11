"""Train Logistic Regression on k-mer TF-IDF features.

Uses GridSearchCV over the C values in configs/mvp.yaml to select
the best regularization strength via 5-fold stratified cross-validation.
Results (per-C CV scores, best C, val accuracy) are saved to
reports/training_report.json.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .utils import load_config, set_seed

FAMILIES = ["KPC", "NDM", "VIM", "IMP"]
CV_FOLDS = 5


def run(config_path: str = "configs/mvp.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    artifacts_dir = Path("artifacts")
    reports_dir   = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Load features & labels ────────────────────────────────────────────────
    X_train = np.load(artifacts_dir / "X_train.npy")
    X_val   = np.load(artifacts_dir / "X_val.npy")
    y_train_raw = np.load(artifacts_dir / "y_train.npy", allow_pickle=True)
    y_val_raw   = np.load(artifacts_dir / "y_val.npy",   allow_pickle=True)

    print(f"X_train: {X_train.shape}  X_val: {X_val.shape}")

    # ── Label encoding ────────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_val   = le.transform(y_val_raw)
    print(f"Classes : {list(le.classes_)}")

    # ── Hyperparameter search ─────────────────────────────────────────────────
    model_cfg  = cfg["model"]
    param_grid = model_cfg.get("param_grid", {"C": [model_cfg["C"]]})
    c_values   = param_grid["C"]

    base_clf = LogisticRegression(
        max_iter=model_cfg["max_iter"],
        solver="lbfgs",
        random_state=cfg["seed"],
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=cfg["seed"])

    print(f"\nGridSearchCV over C={c_values}  ({CV_FOLDS}-fold stratified CV) …")
    t0 = time.time()
    gs = GridSearchCV(
        base_clf,
        {"C": c_values},
        cv=cv,
        scoring="accuracy",
        refit=True,
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    elapsed = time.time() - t0

    # ── CV results table ──────────────────────────────────────────────────────
    print(f"\n{'C':>8}  {'Mean CV Acc':>12}  {'Std':>7}")
    print("-" * 32)
    cv_results = []
    for c, mean, std in zip(
        gs.cv_results_["param_C"],
        gs.cv_results_["mean_test_score"],
        gs.cv_results_["std_test_score"],
    ):
        marker = " ◄ best" if c == gs.best_params_["C"] else ""
        print(f"{c:>8}  {mean:>12.4f}  {std:>7.4f}{marker}")
        cv_results.append({"C": float(c), "mean_cv_acc": round(float(mean), 4),
                            "std_cv_acc": round(float(std), 4)})

    best_c = gs.best_params_["C"]
    best_cv_acc = gs.best_score_
    print(f"\nBest C : {best_c}  (mean CV acc = {best_cv_acc:.4f})")

    # ── Evaluate on train / val ───────────────────────────────────────────────
    clf = gs.best_estimator_
    train_acc = clf.score(X_train, y_train)
    val_acc   = clf.score(X_val,   y_val)

    print(f"  GridSearch time : {elapsed:.2f}s")
    print(f"  Train acc       : {train_acc:.4f}  ({train_acc*100:.1f}%)")
    print(f"  Val acc         : {val_acc:.4f}  ({val_acc*100:.1f}%)")
    print(f"  Iterations      : {clf.n_iter_[0]}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    model_path = artifacts_dir / "model.pkl"
    le_path    = artifacts_dir / "label_encoder.pkl"

    with open(model_path, "wb") as fh:
        pickle.dump(clf, fh)
    with open(le_path, "wb") as fh:
        pickle.dump(le, fh)

    print(f"\n✓ Model saved          → {model_path}")
    print(f"✓ Label encoder saved  → {le_path}")

    # ── Save training report ──────────────────────────────────────────────────
    report = {
        "best_C": best_c,
        "cv_folds": CV_FOLDS,
        "cv_results": cv_results,
        "best_cv_acc": round(float(best_cv_acc), 4),
        "train_acc": round(float(train_acc), 4),
        "val_acc": round(float(val_acc), 4),
        "gridsearch_time_s": round(elapsed, 2),
    }
    report_path = reports_dir / "training_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"✓ Training report      → {report_path}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/mvp.yaml"
    run(config)
