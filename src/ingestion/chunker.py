"""Section-aware, structure-preserving chunker with contextual headers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.ingestion.pdf_parser import Block

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
MONEY_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?")
PCT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
DUR_RE = re.compile(
    r"\b\d+\s*(?:-\s*\d+\s*)?(?:business\s+)?(?:day|days|hour|hours|minute|minutes|month|months|year|years)\b",
    re.I,
)
TIER_RE = re.compile(r"\bTier\s*[123]\b", re.I)


@dataclass
class Chunk:
    id: str
    text: str                  # what gets embedded (includes context header)
    raw_text: str              # clean text for the LLM prompt
    metadata: dict[str, Any] = field(default_factory=dict)


def est_tokens(text: str) -> int:
    """Cheap token estimate — no tokenizer load needed at chunk time."""
    return max(1, int(len(text) / 3.7))


def _facets(text: str) -> dict[str, Any]:
    """Extract entities that matter for support queries -> metadata filters."""
    return {
        "amounts": sorted(set(MONEY_RE.findall(text)))[:8],
        "percentages": sorted(set(PCT_RE.findall(text)))[:8],
        "durations": sorted({d.lower() for d in DUR_RE.findall(text)})[:8],
        "tiers": sorted({t.title().replace("  ", " ") for t in TIER_RE.findall(text)}),
    }


def _ctx_header(b: Block) -> str:
    parts = []
    if b.section_no:
        parts.append(f"Section {b.section_no}: {b.section_title}")
    if b.subsection_no:
        parts.append(b.subsection_title)
    return " > ".join(parts) if parts else "NovaPay Customer Support"


def build_chunks(blocks: list[Block]) -> list[Chunk]:
    chunks: list[Chunk] = []
    counter = 0

    def emit(text: str, b: Block, kind: str, extra: dict | None = None):
        nonlocal counter
        text = text.strip()
        if not text or est_tokens(text) < 8:
            return
        header = _ctx_header(b)
        embedded = f"[{header}]\n{text}"
        facets = _facets(text)
        meta = {
            "chunk_id": counter,
            "kind": kind,
            "section_no": b.section_no or "0",
            "section_title": b.section_title or "General",
            "subsection_no": b.subsection_no or "",
            "subsection_title": b.subsection_title or "",
            "page": b.page,
            "header": header,
            "n_tokens": est_tokens(text),
            # Chroma metadata must be scalar -> join lists
            "amounts": " | ".join(facets["amounts"]),
            "percentages": " | ".join(facets["percentages"]),
            "durations": " | ".join(facets["durations"]),
            "tiers": " | ".join(facets["tiers"]),
            "has_numbers": bool(facets["amounts"] or facets["percentages"]),
        }
        if extra:
            meta.update(extra)
        chunks.append(
            Chunk(id=f"nova-{counter:04d}", text=embedded, raw_text=text, metadata=meta)
        )
        counter += 1

    # Group blocks by (section, subsection)
    groups: list[tuple[Block, list[Block]]] = []
    cur_key = None
    cur: list[Block] = []
    for b in blocks:
        key = (b.section_no, b.subsection_no)
        if key != cur_key:
            if cur:
                groups.append((cur[0], cur))
            cur_key, cur = key, [b]
        else:
            cur.append(b)
    if cur:
        groups.append((cur[0], cur))

    from config.settings import settings

    for anchor, group in groups:
        # --- Atomic kinds: never merge ---
        for b in group:
            if b.kind == "faq":
                emit(b.text, b, "faq", {"question": b.meta.get("question", "")})
            elif b.kind == "table_row":
                emit(b.text, b, "table_row",
                     {"row_subject": b.meta.get("row_subject", "")})

        # --- Bullets: keep whole, pack small ones together ---
        bullets = [b for b in group if b.kind == "bullet"]
        buf: list[str] = []
        buf_tok = 0
        for b in bullets:
            t = est_tokens(b.text)
            if t >= settings.chunk_target_tokens:
                if buf:
                    emit("\n".join(f"- {x}" for x in buf), anchor, "bullets")
                    buf, buf_tok = [], 0
                emit(f"- {b.text}", b, "bullets")
                continue
            if buf_tok + t > settings.chunk_target_tokens and buf:
                emit("\n".join(f"- {x}" for x in buf), anchor, "bullets")
                buf, buf_tok = [], 0
            buf.append(b.text)
            buf_tok += t
        if buf:
            emit("\n".join(f"- {x}" for x in buf), anchor, "bullets")

        # --- Prose: sentence-window with overlap ---
        prose = " ".join(b.text for b in group if b.kind == "prose").strip()
        if prose:
            sents = [s.strip() for s in SENT_SPLIT.split(prose) if s.strip()]
            win: list[str] = []
            win_tok = 0
            for s in sents:
                st = est_tokens(s)
                if win_tok + st > settings.chunk_target_tokens and win:
                    emit(" ".join(win), anchor, "prose")
                    ov = settings.chunk_overlap_sentences
                    win = win[-ov:] if ov else []
                    win_tok = sum(est_tokens(x) for x in win)
                win.append(s)
                win_tok += st
            if win:
                emit(" ".join(win), anchor, "prose")

    # Link neighbours for context expansion at retrieval time
    for i, c in enumerate(chunks):
        c.metadata["prev_id"] = chunks[i - 1].id if i > 0 else ""
        c.metadata["next_id"] = chunks[i + 1].id if i < len(chunks) - 1 else ""

    return chunks