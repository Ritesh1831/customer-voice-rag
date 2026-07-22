"""Groq LLM client: answer generation (streaming + non-streaming) and fast query rewriting."""
from __future__ import annotations

from typing import Iterator

from groq import Groq

from config.settings import settings
from src.generation.prompts import (
    ANSWER_TEMPLATE,
    LANGUAGE_INSTRUCTIONS,
    REWRITE_SYSTEM,
    REWRITE_TEMPLATE,
    SYSTEM_PROMPT,
    build_context,
)

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY missing in .env")
        _client = Groq(api_key=settings.groq_api_key, timeout=settings.llm_timeout_s)
    return _client


def _messages(question: str, hits, language: str, abstain_sentence: str) -> list[dict]:
    lang_instr = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["en"])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ANSWER_TEMPLATE.format(
            context=build_context(hits),
            question=question,
            language_instruction=lang_instr,
            abstain_sentence=abstain_sentence,
        )},
    ]


def generate(question: str, hits, language: str = "en", abstain_sentence: str = "") -> str:
    resp = get_client().chat.completions.create(
        model=settings.llm_model,
        messages=_messages(question, hits, language, abstain_sentence),
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        max_tokens=settings.llm_max_tokens,
        seed=settings.llm_seed,
        stream=False,
    )
    return resp.choices[0].message.content.strip()


def generate_stream(question: str, hits, language: str = "en",
                     abstain_sentence: str = "") -> Iterator[str]:
    stream = get_client().chat.completions.create(
        model=settings.llm_model,
        messages=_messages(question, hits, language, abstain_sentence),
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        max_tokens=settings.llm_max_tokens,
        seed=settings.llm_seed,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def rewrite_query(question: str, history_text: str) -> str:
    """Fast, cheap contextualization: resolve follow-up phrasing into a standalone
    question, using a small/fast model (not the 70B). Fails safe -> returns the
    original question on empty history or any error, so it never blocks the pipeline."""
    if not history_text:
        return question
    try:
        resp = get_client().chat.completions.create(
            model=settings.rewrite_model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": REWRITE_TEMPLATE.format(
                    history=history_text, question=question)},
            ],
            temperature=settings.rewrite_temperature,
            max_tokens=settings.rewrite_max_tokens,
            timeout=8.0,
        )
        text = (resp.choices[0].message.content or "").strip().strip('"').strip()
        return text or question
    except Exception:
        return question