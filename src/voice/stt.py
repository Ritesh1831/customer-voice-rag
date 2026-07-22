"""Groq Whisper-large-v3-turbo STT. Language auto-detect by default so
code-switched Hindi-English (Hinglish) speech isn't forced into English-only decoding."""
from __future__ import annotations

import io

from groq import Groq

from config.settings import settings

_client: Groq | None = None

# Short, natural-sentence style — long jargon-dense prompts are a documented
# cause of Whisper regurgitating the prompt text instead of transcribing the
# actual audio, especially on short/quiet clips or non-English speech.
PROMPT_HINT = "NovaPay customer support call about wallet fees, cards, and transfers."

_PROMPT_FRAGMENT = set(PROMPT_HINT.lower().replace(".", "").split())


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key, timeout=30.0)
    return _client


def _looks_like_prompt_echo(text: str) -> bool:
    """Whisper occasionally regurgitates its own biasing prompt instead of
    transcribing the audio, especially on short/unclear clips. Flag it so
    the caller can treat it as a failed transcription rather than a real
    (garbled) user question."""
    words = set(text.lower().replace(".", "").replace(",", "").split())
    if not words:
        return False
    overlap = len(words & _PROMPT_FRAGMENT) / len(words)
    return overlap > 0.6


def transcribe(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    buf = io.BytesIO(audio_bytes)
    buf.name = filename
    kwargs = dict(
        file=(filename, buf.read()),
        model=settings.stt_model,
        temperature=settings.stt_temperature,
        prompt=PROMPT_HINT,
        response_format="text",
    )
    # Empty stt_language -> let Whisper auto-detect (needed for Hinglish speech).
    if settings.stt_language:
        kwargs["language"] = settings.stt_language
    resp = get_client().audio.transcriptions.create(**kwargs)
    text = (resp if isinstance(resp, str) else resp.text).strip()

    if _looks_like_prompt_echo(text):
        print(f"[stt] suspected prompt echo, discarding: {text!r}", flush=True)
        return ""
    return text