"""FastAPI REST endpoint for ARG family classification.

Usage (from arg-classifier/):
    python3.11 -m uvicorn src.arg_classifier.api:app --reload

Endpoints:
    GET  /health   → server status, loaded families, OOD threshold
    POST /predict  → accept raw FASTA text, return JSON predictions
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from .predict import load_artifacts, predict
from .utils import load_config

# Artifacts are loaded once at startup and stored here for the lifetime of the process
_state: dict = {}


def _parse_fasta_text(text: str) -> list:
    """Parse FASTA-formatted text into list of {id, sequence} dicts."""
    records = []
    current_id, current_seq = None, []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current_id is not None:
                records.append({"id": current_id, "sequence": "".join(current_seq)})
            current_id = line[1:].strip()
            current_seq = []
        elif current_id is not None and line:
            current_seq.append(line.upper())
    if current_id is not None:
        records.append({"id": current_id, "sequence": "".join(current_seq)})
    return records


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup; clear on shutdown."""
    config_path = os.getenv("CONFIG_PATH", "configs/mvp.yaml")
    cfg = load_config(config_path)

    model, vectorizer, encoder = load_artifacts(
        "artifacts/model.pkl",
        "artifacts/vectorizer.pkl",
        "artifacts/label_encoder.pkl",
    )
    _state.update(
        {
            "model":     model,
            "vectorizer": vectorizer,
            "encoder":   encoder,
            "k":         cfg["features"]["kmer_size"],
            "threshold": cfg.get("inference", {}).get("ood_threshold"),
            "families":  list(encoder.classes_),
        }
    )
    print(f"  Loaded {len(_state['families'])} families: {_state['families']}")
    print(f"  OOD threshold: {_state['threshold']}")
    yield
    _state.clear()


app = FastAPI(
    title="ARG Family Classifier",
    description=(
        "Classify antibiotic resistance gene sequences into 8 families "
        "(KPC, NDM, VIM, IMP, OXA, CTX-M, TEM, SHV) using k-mer TF-IDF + "
        "Logistic Regression."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Return server status and loaded model metadata."""
    return {
        "status":        "ok",
        "families":      _state["families"],
        "ood_threshold": _state["threshold"],
        "n_families":    len(_state["families"]),
    }


@app.post("/predict")
async def predict_endpoint(request: Request):
    """Predict ARG family for sequences in a FASTA-formatted request body.

    Send raw FASTA text as the request body (Content-Type: text/plain).
    Returns a JSON array — one object per sequence — with:
      - sequence_id: the FASTA header (without '>')
      - predicted_label: family name or "UNKNOWN" if below confidence threshold
      - confidence: max class probability (float)
      - prob_<FAMILY>: per-class probability for each family
    """
    body = await request.body()
    text = body.decode("utf-8").strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Request body must be non-empty FASTA text.",
        )

    sequences = _parse_fasta_text(text)
    if not sequences:
        raise HTTPException(
            status_code=422,
            detail="No valid FASTA records found. Ensure each sequence has a header line starting with '>'.",
        )

    results = predict(
        sequences,
        _state["model"],
        _state["vectorizer"],
        _state["encoder"],
        _state["k"],
        threshold=_state["threshold"],
    )
    return results
