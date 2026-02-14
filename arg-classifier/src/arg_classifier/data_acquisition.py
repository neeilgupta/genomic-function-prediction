"""Parse CARD FASTA, filter for 4 carbapenemase families, write metadata.csv."""
import csv
import sys
from pathlib import Path
from typing import Optional, Tuple

from .io_fasta import load_fasta
from .utils import load_config

# Families to keep (prefix match on gene_name)
FAMILIES = ("KPC", "NDM", "VIM", "IMP")

# Reasons tracked for exclusion reporting
_excluded = {"wrong_family": 0, "too_short": 0, "too_long": 0, "ambiguous": 0}


def _parse_header(header: str) -> dict:
    """
    Parse a CARD FASTA header of the form:
      gb|ACCESSION|STRAND|COORDS|ARO:ID|GENE_NAME [ORGANISM]
    Returns dict with keys: accession, aro_id, gene_name.
    """
    parts = header.split("|")
    accession = parts[1] if len(parts) > 1 else ""
    aro_id = ""
    gene_name = ""
    for part in parts:
        if part.startswith("ARO:"):
            aro_id = part
    # gene_name is in the last pipe-separated field, before optional '[ORGANISM]'
    if len(parts) >= 6:
        last = parts[5]
        gene_name = last.split("[")[0].strip()
    return {"accession": accession, "aro_id": aro_id, "gene_name": gene_name}


def _assign_label(gene_name: str) -> Optional[str]:
    """Return family label or None if not in the 4 target families."""
    for family in FAMILIES:
        if gene_name.upper().startswith(family):
            return family
    return None


def _passes_filters(seq: str, min_len: int, max_len: int) -> Tuple[bool, str]:
    """Return (passes, reason). reason is empty string when passes=True."""
    length = len(seq)
    if length < min_len:
        return False, "too_short"
    if length > max_len:
        return False, "too_long"
    n_frac = seq.upper().count("N") / length
    if n_frac > 0.05:
        return False, "ambiguous"
    return True, ""


def run(config_path: str = "configs/mvp.yaml") -> None:
    cfg = load_config(config_path)
    raw_dir = Path(cfg["data"]["raw_dir"])
    fasta_path = raw_dir / "card_sequences.fasta"
    out_path = raw_dir / "metadata.csv"
    min_len = cfg["data"]["min_length"]
    max_len = cfg["data"]["max_length"]

    print(f"Loading sequences from {fasta_path} ...")
    records = load_fasta(fasta_path)
    print(f"  Total records in file: {len(records)}")

    kept = []
    excl = {r: 0 for r in ("wrong_family", "too_short", "too_long", "ambiguous")}

    for rec in records:
        parsed = _parse_header(rec["id"])
        label = _assign_label(parsed["gene_name"])
        if label is None:
            excl["wrong_family"] += 1
            continue

        seq = rec["sequence"]
        ok, reason = _passes_filters(seq, min_len, max_len)
        if not ok:
            excl[reason] += 1
            continue

        kept.append(
            {
                "id": rec["id"],
                "label": label,
                "sequence": seq,
                "length": len(seq),
                "source": "CARD",
                "accession": parsed["accession"],
                "gene_name": parsed["gene_name"],
            }
        )

    fieldnames = ["id", "label", "sequence", "length", "source", "accession", "gene_name"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"\n✓ Wrote {len(kept)} sequences to {out_path}")
    print(f"\nExclusion summary:")
    for reason, count in excl.items():
        print(f"  {reason}: {count}")

    # Per-class counts
    from collections import Counter
    dist = Counter(r["label"] for r in kept)
    print(f"\nClass distribution:")
    for fam in FAMILIES:
        print(f"  {fam}: {dist.get(fam, 0)}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/mvp.yaml"
    run(config)
