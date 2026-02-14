"""CLI inference tool: predict ARG family for sequences in a FASTA file."""
import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

from .io_fasta import load_fasta
from .featurize_kmer import seq_to_kmers
from .utils import load_config

FAMILIES = ["KPC", "NDM", "VIM", "IMP"]


def load_artifacts(model_path, vectorizer_path, encoder_path):
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    with open(vectorizer_path, "rb") as fh:
        vectorizer = pickle.load(fh)
    with open(encoder_path, "rb") as fh:
        encoder = pickle.load(fh)
    return model, vectorizer, encoder


def predict(sequences, model, vectorizer, encoder, k):
    """Return list of prediction dicts for a list of sequence records."""
    kmer_docs = [seq_to_kmers(rec["sequence"], k) for rec in sequences]
    X = vectorizer.transform(kmer_docs).toarray().astype(np.float32)
    y_pred   = model.predict(X)
    y_proba  = model.predict_proba(X)

    # Map encoder classes to FAMILIES column order
    enc_classes = list(encoder.classes_)  # e.g. ['IMP','KPC','NDM','VIM']
    fam_indices = {fam: enc_classes.index(fam) for fam in FAMILIES
                   if fam in enc_classes}

    results = []
    for i, rec in enumerate(sequences):
        pred_label = encoder.inverse_transform([y_pred[i]])[0]
        confidence = float(y_proba[i].max())
        probs = {fam: round(float(y_proba[i][fam_indices[fam]]), 6)
                 for fam in FAMILIES if fam in fam_indices}
        results.append({
            "sequence_id":     rec["id"],
            "predicted_label": pred_label,
            "confidence":      round(confidence, 6),
            **{f"prob_{fam}": probs.get(fam, 0.0) for fam in FAMILIES},
        })
    return results


def run(args=None):
    parser = argparse.ArgumentParser(
        description="Predict ARG carbapenemase family from a FASTA file."
    )
    parser.add_argument("--fasta",      required=True,
                        help="Input FASTA file path")
    parser.add_argument("--output",     default="predictions.csv",
                        help="Output CSV path (default: predictions.csv)")
    parser.add_argument("--model",      default="artifacts/model.pkl")
    parser.add_argument("--vectorizer", default="artifacts/vectorizer.pkl")
    parser.add_argument("--encoder",    default="artifacts/label_encoder.pkl")
    parser.add_argument("--config",     default="configs/mvp.yaml")
    opts = parser.parse_args(args)

    cfg = load_config(opts.config)
    k   = cfg["features"]["kmer_size"]

    # ── Load artifacts ────────────────────────────────────────────────────────
    print(f"Loading model artifacts …")
    model, vectorizer, encoder = load_artifacts(
        opts.model, opts.vectorizer, opts.encoder
    )
    print(f"  model      : {opts.model}")
    print(f"  vectorizer : {opts.vectorizer}")
    print(f"  encoder    : {opts.encoder}")
    print(f"  k-mer size : {k}")

    # ── Load sequences ────────────────────────────────────────────────────────
    sequences = load_fasta(opts.fasta)
    if not sequences:
        print(f"ERROR: No sequences found in {opts.fasta}", file=sys.stderr)
        sys.exit(1)
    print(f"\nLoaded {len(sequences)} sequence(s) from {opts.fasta}")

    # ── Predict ───────────────────────────────────────────────────────────────
    results = predict(sequences, model, vectorizer, encoder, k)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = ["sequence_id", "predicted_label", "confidence",
                  "prob_KPC", "prob_NDM", "prob_VIM", "prob_IMP"]
    out_path = Path(opts.output)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ── Console summary ───────────────────────────────────────────────────────
    from collections import Counter
    dist = Counter(r["predicted_label"] for r in results)
    confs = [r["confidence"] for r in results]

    print(f"\nPrediction distribution:")
    for fam in FAMILIES:
        if dist.get(fam, 0):
            print(f"  {fam}: {dist[fam]}")

    print(f"\nConfidence scores:")
    print(f"  min : {min(confs):.4f}")
    print(f"  max : {max(confs):.4f}")
    print(f"  mean: {sum(confs)/len(confs):.4f}")

    print(f"\nFirst {min(5, len(results))} predictions:")
    print(f"  {'ID':<45} {'Label':<6} {'Conf':>6}")
    print(f"  {'-'*60}")
    for r in results[:5]:
        seq_id = r["sequence_id"][:43] + ".." if len(r["sequence_id"]) > 45 else r["sequence_id"]
        print(f"  {seq_id:<45} {r['predicted_label']:<6} {r['confidence']:>6.4f}")

    print(f"\n✓ Predictions written → {out_path}  ({len(results)} rows)")


if __name__ == "__main__":
    run()
