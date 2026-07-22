"""Run: python -m scripts.ingest"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.chunker import build_chunks
from src.ingestion.indexer import build_index


def main():
    if not settings.pdf_path.exists():
        raise SystemExit(f"PDF not found: {settings.pdf_path}")

    print(f"Parsing {settings.pdf_path.name} ...")
    blocks = parse_pdf(settings.pdf_path)
    print(f"  {len(blocks)} blocks "
          f"({sum(b.kind == 'table_row' for b in blocks)} table rows, "
          f"{sum(b.kind == 'faq' for b in blocks)} FAQs)")

    chunks = build_chunks(blocks)
    print(f"  {len(chunks)} chunks")

    sizes = [c.metadata["n_tokens"] for c in chunks]
    print(f"  tokens: min={min(sizes)} avg={sum(sizes)//len(sizes)} max={max(sizes)}")

    build_index(chunks)

    print("\n--- Sample chunks ---")
    for c in chunks[:3]:
        print(f"\n[{c.id}] kind={c.metadata['kind']}")
        print(c.text[:240])


if __name__ == "__main__":
    main()