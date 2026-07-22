"""Pre- and post-generation grounding gates, language-aware."""
from __future__ import annotations

import re

from config.settings import settings
from src.nlu.language import HINGLISH_WORDS

ABSTAIN = {
    "en": ("I don't have that information in my records. "
           "Let me connect you with a human agent who can help."),
    "hinglish": ("Yeh jaankari mere records mein nahi hai. "
                 "Main aapko ek human agent se connect kar deta hoon jo madad kar sakega."),
}


def get_abstain(language: str) -> str:
    return ABSTAIN.get(language, ABSTAIN["en"])


NUM_RE = re.compile(r"\$?\d+(?:[.,]\d+)*\s*%?")
_STOP = {
    "the", "a", "an", "is", "are", "was", "to", "of", "in", "on", "for", "and",
    "or", "you", "your", "it", "that", "this", "with", "can", "will", "be",
    "have", "has", "do", "does", "i", "we", "my", "at", "as", "by", "from",
    "if", "not", "but", "so", "there", "their", "they", "me", "our",
}


def should_abstain_pre(hits) -> tuple[bool, str]:
    if not hits:
        return True, "no_hits"
    best = max(h.rerank_score for h in hits)
    if best < settings.rerank_abstain_threshold:
        return True, f"low_rerank_score={best:.2f}"
    return False, ""


def _norm_num(s: str) -> str:
    return s.replace("$", "").replace(",", "").replace(" ", "").rstrip("%").rstrip(".")


def check_grounding(answer: str, hits, abstain_sentence: str,
                     language: str = "en") -> tuple[bool, str]:
    if not settings.enable_grounding_check:
        return True, ""
    if answer.strip() == abstain_sentence.strip():
        return True, "abstained"

    context = " ".join(h.text for h in hits).lower()

    ctx_nums = {_norm_num(m) for m in NUM_RE.findall(context)}
    ans_nums = {_norm_num(m) for m in NUM_RE.findall(answer)}
    ans_nums = {n for n in ans_nums if n and n not in {"1", "2", "3", "0"}}
    missing = {n for n in ans_nums if n and n not in ctx_nums}
    if missing:
        return False, f"ungrounded_numbers={sorted(missing)}"

    extra_stop = HINGLISH_WORDS if language == "hinglish" else set()
    ctx_words = set(re.findall(r"[a-z]{3,}", context)) - _STOP
    ans_words = set(re.findall(r"[a-z]{3,}", answer.lower())) - _STOP - extra_stop
    if not ans_words:
        return True, ""
    overlap = len(ans_words & ctx_words) / len(ans_words)
    if overlap < settings.grounding_min_overlap:
        return False, f"low_overlap={overlap:.2f}"

    return True, ""