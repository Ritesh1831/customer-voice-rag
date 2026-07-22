"""Dense embeddings via Cohere Embed (multilingual — covers English and Hinglish)."""
from __future__ import annotations

from typing import Any

import numpy as np

from config.settings import settings

_cohere_client = None


def _get_cohere():
    global _cohere_client
    if _cohere_client is None:
        import cohere
        _cohere_client = cohere.Client(settings.cohere_api_key)
    return _cohere_client


def _embed(texts: list[str], input_type: str) -> np.ndarray:
    co = _get_cohere()
    all_vecs = []
    bs = 90
    for i in range(0, len(texts), bs):
        resp = co.embed(
            texts=texts[i:i + bs], model=settings.cohere_embed_model, input_type=input_type,
        )
        all_vecs.extend(resp.embeddings)
    dense = np.asarray(all_vecs, dtype=np.float32)
    return dense / (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9)


def encode_documents(texts: list[str]) -> dict[str, Any]:
    return {"dense": _embed(texts, "search_document"), "sparse": [{} for _ in texts]}


def encode_query(text: str) -> dict[str, Any]:
    dense = _embed([text], "search_query")[0]
    return {"dense": dense, "sparse": {}}


def sparse_dot(q: dict, d: dict) -> float:
    """Cohere has no lexical/sparse signal — BM25 covers exact-token
    matching instead, so this is always a harmless no-op (returns 0.0)."""
    if not q or not d:
        return 0.0
    if len(q) > len(d):
        q, d = d, q
    return float(sum(w * d.get(k, 0.0) for k, w in q.items()))


def warmup() -> None:
    encode_query("warmup")