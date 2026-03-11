"""Generate synthetic CARD-like FASTA data for pipeline testing.

Creates sequences across 8 families (KPC, NDM, VIM, IMP, OXA, CTX-M, TEM, SHV)
that mimic real CARD data: within-family sequences differ by only a few SNPs, while
cross-family sequences are biologically distinct.

Also generates data/raw/other_sequences.fasta — 50 fully random sequences with
no family affiliation, used for out-of-distribution (OOD) detection evaluation.

Usage (from arg-classifier/):
    python3 scripts/generate_synthetic_data.py
"""
import random
from pathlib import Path

random.seed(42)

FAMILIES = {
    # Original 4 carbapenemase families
    "KPC":   {"n": 230, "length": 882},
    "NDM":   {"n": 67,  "length": 813},
    "VIM":   {"n": 91,  "length": 801},
    "IMP":   {"n": 101, "length": 861},
    # Extended families (Milestone 3)
    "OXA":   {"n": 150, "length": 801},   # OXA-type beta-lactamases
    "CTX-M": {"n": 120, "length": 876},   # Extended-spectrum beta-lactamases
    "TEM":   {"n": 110, "length": 861},   # TEM-type (oldest/most common)
    "SHV":   {"n": 90,  "length": 861},   # SHV-type (common in Klebsiella)
}

ARO_COUNTER = 3000000


def random_seq(length: int) -> str:
    return "".join(random.choices("ATGC", k=length))


def mutate(seq: str, n_snps: int) -> str:
    s = list(seq)
    positions = random.sample(range(len(s)), min(n_snps, len(s)))
    for pos in positions:
        s[pos] = random.choice([c for c in "ATGC" if c != s[pos]])
    return "".join(s)


def make_header(accession: str, aro_id: str, gene_name: str, organism: str) -> str:
    # Mimics CARD format: gb|ACCESSION|+|1-LEN|ARO:ID|GENE_NAME [ORGANISM]
    return f"gb|{accession}|+|1-999|{aro_id}|{gene_name} [{organism}]"


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "card_sequences.fasta"

    records = []
    global ARO_COUNTER

    for family, cfg in FAMILIES.items():
        n_seqs = cfg["n"]
        length = cfg["length"]

        # Generate a "founder" sequence for this family
        founder = random_seq(length)

        for i in range(1, n_seqs + 1):
            # Each allele differs from the founder by 0-15 SNPs
            n_snps = random.randint(0, 15)
            seq = mutate(founder, n_snps)

            gene_name = f"{family}-{i}"
            accession = f"SYN{family}{i:04d}"
            aro_id = f"ARO:{ARO_COUNTER}"
            ARO_COUNTER += 1
            organism = "Klebsiella pneumoniae"

            header = make_header(accession, aro_id, gene_name, organism)
            records.append((header, seq))

    # Write FASTA
    with open(out_path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            # Wrap at 60 chars
            for j in range(0, len(seq), 60):
                fh.write(seq[j : j + 60] + "\n")

    print(f"Wrote {len(records)} synthetic sequences to {out_path}")
    for fam, cfg in FAMILIES.items():
        print(f"  {fam}: {cfg['n']} sequences ({cfg['length']} bp each)")

    # ── Write "other" sequences for OOD evaluation ────────────────────────────
    # These are fully random sequences with no family affiliation.
    # They are NOT included in training — only used to test OOD detection.
    other_path = raw_dir / "other_sequences.fasta"
    n_other = 50
    other_length = 861  # typical ARG length so the test is realistic
    with open(other_path, "w") as fh:
        for i in range(1, n_other + 1):
            seq = random_seq(other_length)
            fh.write(f">OTHER-{i:03d} [synthetic_unknown_organism]\n")
            for j in range(0, len(seq), 60):
                fh.write(seq[j : j + 60] + "\n")
    print(f"\nWrote {n_other} OOD 'other' sequences to {other_path}")


if __name__ == "__main__":
    main()
