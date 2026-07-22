"""Build the Chroma collection + BM25 index."""
from __future__ import annotations

import json
import pickle

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from src.ingestion.chunker import Chunk

COLLECTION = "novapay_kb"


def get_client():
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )


def build_index(chunks: list[Chunk]) -> None:
    from src.ingestion.chunker import est_tokens  # noqa
    from src.retrieval.embedder import encode_documents

    client = get_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    coll = client.create_collection(
        name=COLLECTION,
        metadata={
            # HNSW tuned for a small corpus + high recall
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 400,
            "hnsw:search_ef": 200,
            "hnsw:M": 32,
        },
    )

    texts = [c.text for c in chunks]
    print(f"Embedding {len(texts)} chunks with BGE-M3 (CPU)...")
    emb = encode_documents(texts)

    coll.add(
        ids=[c.id for c in chunks],
        embeddings=emb["dense"].tolist(),
        documents=[c.raw_text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )

    # Persist sparse vectors + BM25 corpus alongside
    sparse_store = {
        c.id: {str(k): float(v) for k, v in sp.items()}
        for c, sp in zip(chunks, emb["sparse"])
    }
    (settings.chroma_dir / "sparse.json").write_text(
        json.dumps(sparse_store), encoding="utf-8"
    )

    from rank_bm25 import BM25Okapi
    import re
    tokenized = [re.findall(r"[a-z0-9$%.]+", c.text.lower()) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(settings.chroma_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "ids": [c.id for c in chunks]}, f)

    manifest = {
        "n_chunks": len(chunks),
        "by_kind": {},
        "sections": sorted({c.metadata["section_title"] for c in chunks}),
    }
    for c in chunks:
        manifest["by_kind"][c.metadata["kind"]] = (
            manifest["by_kind"].get(c.metadata["kind"], 0) + 1
        )
    (settings.chroma_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Indexed {len(chunks)} chunks -> {settings.chroma_dir}")
    print(json.dumps(manifest["by_kind"], indent=2))