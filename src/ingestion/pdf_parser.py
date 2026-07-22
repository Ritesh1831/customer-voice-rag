"""Structure-aware PDF parser for the NovaPay KB."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

SECTION_RE = re.compile(r"^(\d{1,2})\.\s+(.+?)\s*$")
SUBSECTION_RE = re.compile(r"^(\d{1,2}\.\d{1,2})\s+(.+?)\s*$")
FAQ_Q_RE = re.compile(r"^Q:\s*(.+)$")
FAQ_A_RE = re.compile(r"^A:\s*(.+)$")
BULLET_RE = re.compile(r"^\s*[•●○\-\u2022]\s*")


@dataclass
class Block:
    """One semantic unit of the document."""
    kind: str                     # prose | bullet | table_row | faq | heading
    text: str
    section_no: str = ""
    section_title: str = ""
    subsection_no: str = ""
    subsection_title: str = ""
    page: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def _clean(s: str) -> str:
    s = s.replace("\u00a0", " ").replace("\ufb01", "fi").replace("\ufb02", "fl")
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _table_to_rows(table: list[list[str | None]], page: int) -> list[Block]:
    """Linearise a table so each row is a standalone, retrievable sentence."""
    if not table or len(table) < 2:
        return []

    header = [_clean(c or "") for c in table[0]]
    if not any(header):
        return []

    blocks: list[Block] = []
    row_label = header[0] or "Item"

    for raw in table[1:]:
        cells = [_clean(c or "") for c in raw]
        if not any(cells):
            continue
        subject = cells[0]
        if not subject:
            continue

        parts = []
        for col_name, val in zip(header[1:], cells[1:]):
            if val and col_name:
                parts.append(f"{col_name}: {val}")
            elif val:
                parts.append(val)

        if not parts:
            continue

        sentence = f"For {row_label} '{subject}' — " + "; ".join(parts) + "."
        blocks.append(
            Block(
                kind="table_row",
                text=sentence,
                page=page,
                meta={"row_subject": subject, "columns": header},
            )
        )

    return blocks


def parse_pdf(pdf_path: Path) -> list[Block]:
    blocks: list[Block] = []
    cur_sec_no = cur_sec_title = ""
    cur_sub_no = cur_sub_title = ""
    pending_q: str | None = None

    with pdfplumber.open(str(pdf_path)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):

            # --- 1. Tables first, and record their vertical extent ---
            table_regions: list[tuple[float, float]] = []
            try:
                found = page.find_tables()
            except Exception:
                found = []

            for tbl in found:
                try:
                    extracted = tbl.extract()
                except Exception:
                    continue
                rows = _table_to_rows(extracted, pno)
                for b in rows:
                    b.section_no, b.section_title = cur_sec_no, cur_sec_title
                    b.subsection_no, b.subsection_title = cur_sub_no, cur_sub_title
                blocks.extend(rows)
                x0, top, x1, bottom = tbl.bbox
                table_regions.append((top, bottom))

            # --- 2. Text outside table regions ---
            def _outside(obj):
                mid = (obj["top"] + obj["bottom"]) / 2
                return not any(t <= mid <= b for t, b in table_regions)

            try:
                filtered = page.filter(_outside)
                text = filtered.extract_text(layout=False) or ""
            except Exception:
                text = page.extract_text() or ""

            for raw_line in text.split("\n"):
                line = _clean(raw_line)
                if not line:
                    continue

                # Section heading
                m = SUBSECTION_RE.match(line)
                if m and len(m.group(2)) < 80:
                    cur_sub_no, cur_sub_title = m.group(1), m.group(2)
                    continue

                m = SECTION_RE.match(line)
                if m and len(m.group(2)) < 80 and not line.endswith("."):
                    cur_sec_no, cur_sec_title = m.group(1), m.group(2)
                    cur_sub_no = cur_sub_title = ""
                    continue

                # FAQ pairs
                mq = FAQ_Q_RE.match(line)
                if mq:
                    pending_q = mq.group(1)
                    continue
                ma = FAQ_A_RE.match(line)
                if ma and pending_q:
                    blocks.append(
                        Block(
                            kind="faq",
                            text=f"Question: {pending_q}\nAnswer: {ma.group(1)}",
                            section_no=cur_sec_no,
                            section_title=cur_sec_title,
                            page=pno,
                            meta={"question": pending_q},
                        )
                    )
                    pending_q = None
                    continue

                # Bullet vs prose
                is_bullet = bool(BULLET_RE.match(raw_line.strip())) or raw_line.strip().startswith("●")
                body = BULLET_RE.sub("", line).strip()
                if not body:
                    continue

                kind = "bullet" if is_bullet else "prose"

                # Continuation line -> append to previous block of same kind
                if (
                    blocks
                    and blocks[-1].kind == kind
                    and blocks[-1].page == pno
                    and blocks[-1].section_no == cur_sec_no
                    and not is_bullet
                    and not blocks[-1].text.endswith((".", ":", "?"))
                ):
                    blocks[-1].text = f"{blocks[-1].text} {body}"
                    continue

                blocks.append(
                    Block(
                        kind=kind,
                        text=body,
                        section_no=cur_sec_no,
                        section_title=cur_sec_title,
                        subsection_no=cur_sub_no,
                        subsection_title=cur_sub_title,
                        page=pno,
                    )
                )

    # Drop the disclaimer/footer noise
    noise = ("This document is a fictional", "Note: this knowledge base intentionally")
    return [b for b in blocks if not b.text.startswith(noise)]