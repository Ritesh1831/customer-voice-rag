"""Hybrid retrieval: dense (Cohere, via Chroma Cloud) + BM25 (rebuilt fresh
at startup from Chroma Cloud's stored documents — the cloud database is
the single source of truth) -> Reciprocal Rank Fusion -> Cohere rerank ->
neighbor expansion."""
from __future__ import annotations

import re
from dataclasses import dataclass

from config.settings import settings
from src.ingestion.indexer import COLLECTION, get_client
from src.retrieval import embedder, reranker

SYNONYMS = {
    "kyc": "verification identity document tier",
    "verify": "verification KYC identity",
    "topup": "add money deposit load",
    "top up": "add money deposit load",
    "top-up": "add money deposit load",
    "reload": "add money deposit load",
    "send money": "transfer P2P peer-to-peer",
    "wire": "international transfer",
    "abroad": "international transfer foreign",
    "overseas": "international transfer foreign",
    "atm": "withdrawal cash ATM",
    "cash out": "ATM withdrawal transfer out",
    "charge": "fee cost",
    "cost": "fee charge",
    "refund": "dispute chargeback reversal",
    "chargeback": "dispute refund unauthorized",
    "scam": "fraud unauthorized phishing",
    "hacked": "fraud unauthorized security",
    "stolen": "lost stolen freeze card",
    "loan": "credit line borrow",
    "borrow": "credit line loan",
    "interest": "APR interest rate",
    "cashback": "rewards cashback referral",
    "savings": "savings pocket goal APY interest",
    "apy": "savings pocket interest APY",
    "bill": "bill pay payee recurring payment",
    "statement": "statement account history export",
    "delete account": "close account closure",
    "password": "login security 2FA reset",
    "otp": "one-time passcode 2FA OTP",
    "declined": "card declined troubleshooting controls",
    "crash": "troubleshooting app issue",
    "country": "supported countries regional availability",
    "currency": "multi-currency USD EUR GBP conversion",
}

_state: dict = {}


def _load_state():
    if _state:
        return _state
    client = get_client()
    coll = client.get_collection(COLLECTION)
    _state["coll"] = coll

    all_docs = coll.get(include=["documents", "metadatas"])
    ids = all_docs["ids"]
    raw_texts = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    tokenized = []
    for text, meta in zip(raw_texts, metadatas):
        header = meta.get("header", "")
        combined = f"{header}\n{text}" if header else text
        tokenized.append(re.findall(r"[a-z0-9$%.]+", combined.lower()))

    from rank_bm25 import BM25Okapi
    _state["bm25"] = BM25Okapi(tokenized)
    _state["bm25_ids"] = ids
    return _state


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    dense_rank: int = 10_000
    bm25_rank: int = 10_000
    fused: float = 0.0
    rerank_score: float = -99.0


def expand_query(q: str) -> str:
    low = q.lower()
    extra = [v for k, v in SYNONYMS.items() if k in low]
    return f"{q} {' '.join(extra)}" if extra else q


def _rrf(rank: int, k: int) -> float:
    return 1.0 / (k + rank)


def retrieve(query: str) -> list[Hit]:
    st = _load_state()
    expanded = expand_query(query)

    qv = embedder.encode_query(expanded)
    hits: dict[str, Hit] = {}

    res = st["coll"].query(
        query_embeddings=[qv["dense"].tolist()],
        n_results=settings.dense_k,
        include=["documents", "metadatas"],
    )
    for rank, (cid, doc, meta) in enumerate(
        zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
    ):
        hits[cid] = Hit(id=cid, text=doc, metadata=meta, dense_rank=rank)

    toks = re.findall(r"[a-z0-9$%.]+", expanded.lower())
    bm_scores = st["bm25"].get_scores(toks)
    order = sorted(range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True)
    for rank, idx in enumerate(order[: settings.sparse_k]):
        if bm_scores[idx] <= 0:
            break
        cid = st["bm25_ids"][idx]
        if cid not in hits:
            got = st["coll"].get(ids=[cid], include=["documents", "metadatas"])
            if not got["ids"]:
                continue
            hits[cid] = Hit(id=cid, text=got["documents"][0], metadata=got["metadatas"][0])
        hits[cid].bm25_rank = rank

    k = settings.rrf_k
    for h in hits.values():
        h.fused = 1.00 * _rrf(h.dense_rank, k) + 0.75 * _rrf(h.bm25_rank, k)
    candidates = sorted(hits.values(), key=lambda h: h.fused, reverse=True)
    candidates = candidates[: settings.fusion_k]
    if not candidates:
        return []

    scores = reranker.rerank(query, [h.text for h in candidates])
    for h, s in zip(candidates, scores):
        h.rerank_score = float(s)
    candidates.sort(key=lambda h: h.rerank_score, reverse=True)

    top = candidates[: settings.final_k]

    if settings.neighbor_expansion and top:
        have = {h.id for h in top}
        extra: list[Hit] = []
        for h in top[:2]:
            if h.rerank_score < settings.rerank_confident_threshold:
                continue
            for nid_key in ("prev_id", "next_id"):
                nid = h.metadata.get(nid_key, "")
                if not nid or nid in have:
                    continue
                got = st["coll"].get(ids=[nid], include=["documents", "metadatas"])
                if not got["ids"]:
                    continue
                nmeta = got["metadatas"][0]
                if nmeta.get("section_no") != h.metadata.get("section_no"):
                    continue
                have.add(nid)
                extra.append(
                    Hit(id=nid, text=got["documents"][0], metadata=nmeta,
                        rerank_score=h.rerank_score - 0.01)
                )
        top.extend(extra)

    return top


def warmup() -> None:
    _load_state()
    embedder.warmup()
    reranker.warmup()