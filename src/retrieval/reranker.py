"""Cross-encoder reranker — loads a pre-quantized ONNX build directly.
No local export/tracing through torch, avoiding the same crash category
as the embedder."""
from __future__ import annotations

import threading

import numpy as np

from config.settings import settings

_lock = threading.Lock()
_tokenizer = None
_model = None

RERANK_ONNX_REPO = "onnx-community/bge-reranker-v2-m3-ONNX"
RERANK_ONNX_FILE = "model_int8.onnx"


def _load():
    global _tokenizer, _model
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        from transformers import AutoTokenizer
        from optimum.onnxruntime import ORTModelForSequenceClassification

        print("[reranker] loading pre-quantized ONNX reranker (one-time download)...", flush=True)
        _tokenizer = AutoTokenizer.from_pretrained(
            RERANK_ONNX_REPO, cache_dir=str(settings.model_cache)
        )
        _model = ORTModelForSequenceClassification.from_pretrained(
            RERANK_ONNX_REPO,
            subfolder="onnx",
            file_name=RERANK_ONNX_FILE,
            cache_dir=str(settings.model_cache),
        )
        print("[reranker] ready.", flush=True)


def rerank(query: str, docs: list[str]) -> np.ndarray:
    """Return relevance logits, one per doc. Higher = more relevant."""
    if not docs:
        return np.array([], dtype=np.float32)
    _load()

    scores: list[float] = []
    bs = settings.rerank_batch_size
    for i in range(0, len(docs), bs):
        batch = docs[i : i + bs]
        enc = _tokenizer(
            [query] * len(batch), batch, padding=True, truncation=True,
            max_length=settings.rerank_max_length, return_tensors="pt",
        )
        out = _model(**enc)
        logits = out.logits.detach().numpy()
        scores.extend(logits[:, 0].tolist() if logits.shape[-1] == 1 else logits[:, -1].tolist())
    return np.asarray(scores, dtype=np.float32)


def warmup() -> None:
    rerank("warmup query", ["warmup document"])