"""Cross-encoder reranking via Cohere Rerank v3.5 (strong multilingual
performance, including Hindi)."""
from __future__ import annotations

import numpy as np

from config.settings import settings

_cohere_client = None


def _get_cohere():
    global _cohere_client
    if _cohere_client is None:
        import cohere
        _cohere_client = cohere.Client(settings.cohere_api_key)
    return _cohere_client


def rerank(query: str, docs: list[str]) -> np.ndarray:
    """Return relevance scores (0-1), one per doc, in the SAME order as `docs`."""
    if not docs:
        return np.array([], dtype=np.float32)
    co = _get_cohere()
    resp = co.rerank(query=query, documents=docs, model=settings.cohere_rerank_model, top_n=len(docs))
    # Cohere returns results reordered by score, each with an `index` back
    # into the original docs list — reconstruct scores in ORIGINAL order,
    # since retriever.py zips scores against candidates positionally.
    scores = np.zeros(len(docs), dtype=np.float32)
    for r in resp.results:
        scores[r.index] = float(r.relevance_score)
    return scores


def warmup() -> None:
    rerank("warmup query", ["warmup document"])