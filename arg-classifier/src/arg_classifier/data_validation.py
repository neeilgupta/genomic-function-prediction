"""Validate metadata.csv and write data_summary.txt."""
import sys
from pathlib import Path

import pandas as pd

from .utils import load_config

MIN_CLASS_SIZE = 50
FAMILIES = ("KPC", "NDM", "VIM", "IMP")


def run(config_path: str = "configs/mvp.yaml") -> None:
    cfg = load_config(config_path)
    raw_dir = Path(cfg["data"]["raw_dir"])
    meta_path = raw_dir / "metadata.csv"
    summary_path = raw_dir / "data_summary.txt"

    df = pd.read_csv(meta_path)

    lines = []

    lines.append("=" * 50)
    lines.append("ARG DATASET VALIDATION SUMMARY")
    lines.append("=" * 50)
    lines.append(f"\nTotal sequences: {len(df)}")

    # Class distribution
    lines.append("\nClass distribution:")
    dist = df["label"].value_counts()
    for fam in FAMILIES:
        n = dist.get(fam, 0)
        lines.append(f"  {fam}: {n}")
    lines.append(f"  TOTAL: {dist.sum()}")

    # Length statistics
    lines.append("\nLength statistics:")
    lines.append(f"  min  : {df['length'].min()}")
    lines.append(f"  max  : {df['length'].max()}")
    lines.append(f"  mean : {df['length'].mean():.1f}")
    lines.append(f"  median: {df['length'].median():.1f}")

    # Duplicate sequence check
    n_dup = df.duplicated(subset="sequence").sum()
    lines.append(f"\nDuplicate sequences: {n_dup}")
    if n_dup > 0:
        lines.append("  WARNING: duplicates found")

    # Class size check
    lines.append(f"\nClass size check (≥{MIN_CLASS_SIZE}):")
    all_ok = True
    for fam in FAMILIES:
        n = dist.get(fam, 0)
        status = "✓" if n >= MIN_CLASS_SIZE else "✗ FAIL"
        lines.append(f"  {fam}: {n} {status}")
        if n < MIN_CLASS_SIZE:
            all_ok = False

    lines.append(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    lines.append("=" * 50)

    summary = "\n".join(lines)
    print(summary)

    with open(summary_path, "w") as fh:
        fh.write(summary + "\n")

    print(f"\n✓ Summary written to {summary_path}")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/mvp.yaml"
    run(config)
