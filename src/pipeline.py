"""End-to-end orchestration: chitchat routing, language detection, conversational
memory + query rewriting, hybrid retrieval, grounded generation, semantic caching."""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from config.settings import settings
from src.generation import guardrails, llm
from src.memory.history import Turn, memory, needs_rewrite
from src.nlu import router
from src.nlu.language import detect_language
from src.retrieval import retriever
from src.retrieval.embedder import encode_query


@dataclass
class Answer:
    text: str
    abstained: bool = False
    reason: str = ""
    sources: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    cached: bool = False
    language: str = "en"
    standalone_question: str = ""


_cache: "OrderedDict[str, tuple[np.ndarray, str, Answer]]" = OrderedDict()


def _cache_lookup(vec: np.ndarray, lang: str) -> Answer | None:
    if not _cache:
        return None
    best_key, best_sim = None, -1.0
    for key, (v, l, _) in _cache.items():
        if l != lang:
            continue
        sim = float(np.dot(vec, v) /
                    ((np.linalg.norm(vec) * np.linalg.norm(v)) + 1e-9))
        if sim > best_sim:
            best_key, best_sim = key, sim
    if best_key is not None and best_sim >= settings.semantic_cache_threshold:
        _cache.move_to_end(best_key)
        stored = _cache[best_key][2]
        return Answer(text=stored.text, abstained=stored.abstained, reason=stored.reason,
                      sources=stored.sources, timings={"cache_hit": True}, cached=True,
                      language=lang)
    return None


def _cache_put(key: str, vec: np.ndarray, lang: str, ans: Answer):
    _cache[key] = (vec, lang, ans)
    _cache.move_to_end(key)
    while len(_cache) > settings.semantic_cache_size:
        _cache.popitem(last=False)


def _sources(hits) -> list[dict]:
    return [
        {
            "id": h.id,
            "section": h.metadata.get("header", ""),
            "page": h.metadata.get("page", 0),
            "score": round(h.rerank_score, 3),
            "preview": h.text[:150],
        }
        for h in hits
    ]


def answer(question: str, session_id: str = "default", use_cache: bool = True) -> Answer:
    t: dict = {}
    t0 = time.perf_counter()

    # --- 0. Chitchat router: skip retrieval + LLM entirely ---
    kind = router.classify(question)
    if kind:
        lang = detect_language(question)
        text = router.reply(kind, lang)
        t["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self_abstained = raw.strip() == abstain_sentence.strip()
        ans = Answer(text=raw, abstained=self_abstained,
                 reason="llm_self_abstained" if self_abstained else "",
                 sources=_sources(hits), timings=t, language=lang,
                 standalone_question=standalone if standalone != question else "")
        memory.add(session_id, Turn(question, standalone, raw, lang))
        if use_cache:
            _cache_put(standalone.lower().strip(), qvec, lang, ans)
        return ans

    lang = detect_language(question)
    abstain_sentence = guardrails.get_abstain(lang)

    # --- 1. Conversational memory: rewrite follow-ups into standalone questions ---
    history_text = memory.as_text(session_id, n=settings.memory_turns_for_rewrite)
    standalone = question
    if history_text and needs_rewrite(question):
        t_rw = time.perf_counter()
        standalone = llm.rewrite_query(question, history_text)
        t["rewrite_ms"] = round((time.perf_counter() - t_rw) * 1000, 1)

    t1 = time.perf_counter()
    qvec = encode_query(standalone)["dense"]
    t["embed_ms"] = round((time.perf_counter() - t1) * 1000, 1)

    if use_cache:
        cached = _cache_lookup(qvec, lang)
        if cached:
            memory.add(session_id, Turn(question, standalone, cached.text, lang))
            return cached

    t2 = time.perf_counter()
    hits = retriever.retrieve(standalone)
    t["retrieve_ms"] = round((time.perf_counter() - t2) * 1000, 1)

    ab, reason = guardrails.should_abstain_pre(hits)
    if ab:
        t["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        ans = Answer(text=abstain_sentence, abstained=True, reason=reason,
                     sources=_sources(hits), timings=t, language=lang,
                     standalone_question=standalone if standalone != question else "")
        memory.add(session_id, Turn(question, standalone, ans.text, lang))
        return ans

    t3 = time.perf_counter()
    raw = llm.generate(standalone, hits, lang, abstain_sentence)
    t["llm_ms"] = round((time.perf_counter() - t3) * 1000, 1)

    ok, why = guardrails.check_grounding(raw, hits, abstain_sentence, lang)
    if not ok:
        t["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        ans = Answer(text=abstain_sentence, abstained=True, reason=f"post_gate:{why}",
                     sources=_sources(hits), timings=t, language=lang,
                     standalone_question=standalone if standalone != question else "")
        memory.add(session_id, Turn(question, standalone, ans.text, lang))
        return ans

    t["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    ans = Answer(text=raw, sources=_sources(hits), timings=t, language=lang,
                 standalone_question=standalone if standalone != question else "")
    memory.add(session_id, Turn(question, standalone, raw, lang))
    if use_cache:
        _cache_put(standalone.lower().strip(), qvec, lang, ans)
    return ans


def answer_stream(question: str, session_id: str = "default") -> Iterator[dict]:
    """Yields dicts: {type: 'meta'|'token'|'sentence'|'done'|'abstain'}. Every event
    after the first carries 'language' so the caller knows which TTS voice to use."""
    t0 = time.perf_counter()

    # --- 0. Chitchat router ---
    kind = router.classify(question)
    if kind:
        lang = detect_language(question)
        text = router.reply(kind, lang)
        memory.add(session_id, Turn(question, question, text, lang))
        yield {"type": "meta", "sources": [], "language": lang, "retrieve_ms": 0.0}
        yield {"type": "sentence", "text": text}
        yield {"type": "done", "full_text": text, "grounded": True, "reason": "chitchat",
               "language": lang, "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
        return

    lang = detect_language(question)
    abstain_sentence = guardrails.get_abstain(lang)

    history_text = memory.as_text(session_id, n=settings.memory_turns_for_rewrite)
    standalone = question
    if history_text and needs_rewrite(question):
        standalone = llm.rewrite_query(question, history_text)

    hits = retriever.retrieve(standalone)

    abstain, reason = guardrails.should_abstain_pre(hits)
    if abstain:
        memory.add(session_id, Turn(question, standalone, abstain_sentence, lang))
        yield {"type": "abstain", "text": abstain_sentence, "reason": reason, "language": lang}
        return

    yield {"type": "meta", "sources": _sources(hits), "language": lang,
           "retrieve_ms": round((time.perf_counter() - t0) * 1000, 1)}

    buf, full = "", ""
    for tok in llm.generate_stream(standalone, hits, lang, abstain_sentence):
        buf += tok
        full += tok
        yield {"type": "token", "text": tok}
        while True:
            m = None
            for i, ch in enumerate(buf):
                if ch in ".!?" and i + 1 < len(buf) and buf[i + 1] == " ":
                    m = i + 1
                    break
            if m is None:
                break
            sent, buf = buf[:m].strip(), buf[m:].lstrip()
            if sent:
                yield {"type": "sentence", "text": sent}
    if buf.strip():
        yield {"type": "sentence", "text": buf.strip()}

    ok, why = guardrails.check_grounding(full, hits, abstain_sentence, lang)
    memory.add(session_id, Turn(question, standalone, full.strip(), lang))
    yield {
        "type": "done",
        "full_text": full.strip(),
        "grounded": ok,
        "reason": why,
        "language": lang,
        "total_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def warmup() -> None:
    print("Warming up models...")
    retriever.warmup()
    from src.voice import tts
    tts.warmup()
    try:
        llm.rewrite_query("warmup", "Customer: hi\nNova: hello")
    except Exception:
        pass
    print("Ready.")