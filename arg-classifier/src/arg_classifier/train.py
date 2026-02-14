"""Train Logistic Regression on k-mer TF-IDF features."""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .utils import load_config, set_seed

FAMILIES = ["KPC", "NDM", "VIM", "IMP"]


def run(config_path: str = "configs/mvp.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    artifacts_dir = Path("artifacts")

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

    # ── Train ─────────────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    clf = LogisticRegression(
        C=model_cfg["C"],
        max_iter=model_cfg["max_iter"],
        solver="lbfgs",
        multi_class="multinomial",
        random_state=cfg["seed"],
        verbose=0,
    )

    print("\nTraining Logistic Regression …")
    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    # ── Validation accuracy ───────────────────────────────────────────────────
    train_acc = clf.score(X_train, y_train)
    val_acc   = clf.score(X_val,   y_val)

    print(f"  Train time   : {elapsed:.2f}s")
    print(f"  Train acc    : {train_acc:.4f}  ({train_acc*100:.1f}%)")
    print(f"  Val acc      : {val_acc:.4f}  ({val_acc*100:.1f}%)")
    print(f"  Converged    : {clf.n_iter_[0] < model_cfg['max_iter']}  "
          f"(iterations: {clf.n_iter_[0]})")

    # ── Save artifacts ────────────────────────────────────────────────────────
    model_path = artifacts_dir / "model.pkl"
    le_path    = artifacts_dir / "label_encoder.pkl"

    with open(model_path, "wb") as fh:
        pickle.dump(clf, fh)
    with open(le_path, "wb") as fh:
        pickle.dump(le, fh)

    print(f"\n✓ Model saved          → {model_path}")
    print(f"✓ Label encoder saved  → {le_path}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/mvp.yaml"
    run(config)
