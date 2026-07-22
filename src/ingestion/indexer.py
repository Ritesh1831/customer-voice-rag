"""Build the Chroma Cloud collection. No local persistence — the cloud
database is the only place the index lives, so there's nothing to gitignore
or keep in sync locally."""
from __future__ import annotations

from config.settings import settings
from src.ingestion.chunker import Chunk

COLLECTION = "customer_voice_rag_kb"


def get_client():
    import chromadb
    kwargs = {"api_key": settings.chroma_api_key}
    if settings.chroma_tenant:
        kwargs["tenant"] = settings.chroma_tenant
    if settings.chroma_database:
        kwargs["database"] = settings.chroma_database
    return chromadb.CloudClient(**kwargs)


def build_index(chunks: list[Chunk]) -> None:
    from src.retrieval.embedder import encode_documents

    client = get_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    coll = client.create_collection(name=COLLECTION)

    texts = [c.text for c in chunks]
    print(f"Embedding {len(texts)} chunks with Cohere...")
    emb = encode_documents(texts)

    coll.add(
        ids=[c.id for c in chunks],
        embeddings=emb["dense"].tolist(),
        documents=[c.raw_text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )

    by_kind: dict[str, int] = {}
    for c in chunks:
        by_kind[c.metadata["kind"]] = by_kind.get(c.metadata["kind"], 0) + 1
    print(f"Indexed {len(chunks)} chunks in Chroma Cloud collection '{COLLECTION}'")
    print(by_kind)