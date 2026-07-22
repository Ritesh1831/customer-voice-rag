"""Per-session conversational memory + follow-up detection heuristic."""
from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from config.settings import settings

FOLLOWUP_STARTERS = (
    "and ", "also ", "what about", "or ", "aur ", "uska kya", "vaise",
    "waise", "phir", "toh phir",
)
PRONOUNS = {
    "it", "that", "this", "those", "these", "them", "he", "she", "they",
    "its", "their", "uska", "uski", "iska", "iski", "wo", "woh", "yeh", "ye",
    "unka", "inka", "usme", "ismein", "usmein",
}


@dataclass
class Turn:
    question: str
    standalone_question: str
    answer: str
    language: str
    ts: float = field(default_factory=time.time)


class ConversationMemory:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._store: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_turns))

    def add(self, session_id: str, turn: Turn) -> None:
        self._store[session_id].append(turn)

    def get_recent(self, session_id: str, n: int = 3) -> list[Turn]:
        turns = list(self._store.get(session_id, []))
        return turns[-n:]

    def as_text(self, session_id: str, n: int = 3) -> str:
        turns = self.get_recent(session_id, n)
        if not turns:
            return ""
        lines = []
        for t in turns:
            lines.append(f"Customer: {t.question}")
            lines.append(f"Nova: {t.answer}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)


memory = ConversationMemory(max_turns=settings.memory_max_turns)


def needs_rewrite(question: str) -> bool:
    """Heuristic gate: only pay for an LLM rewrite call when the question
    actually looks like a follow-up (short + pronoun, or a connector start).
    Everything else skips straight to retrieval — no added latency."""
    q = question.strip().lower()
    if not q:
        return False
    if any(q.startswith(s) for s in FOLLOWUP_STARTERS):
        return True
    words = re.findall(r"[a-zA-Z']+", q)
    if 0 < len(words) <= 6 and any(w in PRONOUNS for w in words):
        return True
    return False