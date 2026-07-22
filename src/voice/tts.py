"""TTS: Groq Orpheus for English, Sarvam Bulbul v3 for Hinglish.
Sarvam is purpose-trained for Hindi/English code-switching, not a general
multilingual model adapted to it."""
from __future__ import annotations

import base64
import re
from typing import Iterator

import httpx
from groq import Groq

from config.settings import settings

_groq_client: Groq | None = None
SENT_END = re.compile(r"(?<=[.!?])\s+")


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key, timeout=30.0)
    return _groq_client


def _speakable(text: str) -> str:
    text = re.sub(r"\bP2P\b", "peer to peer", text)
    text = re.sub(r"\bACH\b", "A C H", text)
    text = re.sub(r"\bATM\b", "A T M", text)
    text = re.sub(r"\bAPY\b", "A P Y", text)
    text = re.sub(r"\bAPR\b", "A P R", text)
    text = re.sub(r"\bOTP\b", "one time passcode", text)
    text = re.sub(r"\bKYC\b", "K Y C", text)
    text = re.sub(r"\bFX\b", "foreign exchange", text)
    text = re.sub(r"\b2FA\b", "two factor authentication", text)
    text = re.sub(r"\$([\d,]+)\.(\d{2})\b", r"\1 dollars and \2 cents", text)
    text = re.sub(r"\$([\d,]+)\b", r"\1 dollars", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1 percent", text)
    return re.sub(r"\s+", " ", text).strip()


def _synth_groq(text: str) -> bytes:
    response = get_groq_client().audio.speech.create(
        model=settings.groq_tts_model, voice=settings.groq_tts_voice,
        input=text, response_format="wav",
    )
    return response.read()


def _synth_sarvam(text: str) -> bytes:
    resp = httpx.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={"api-subscription-key": settings.sarvam_api_key, "Content-Type": "application/json"},
        json={
            "text": text[:2500],
            "target_language_code": settings.sarvam_tts_language,
            "model": settings.sarvam_tts_model,
            "speaker": settings.sarvam_tts_speaker,
        },
        timeout=20.0,
    )
    resp.raise_for_status()
    return base64.b64decode(resp.json()["audios"][0])


def synth(text: str, language: str = "en") -> bytes:
    """A TTS failure here is caught by server.py's _safe_send_audio, which
    degrades to a text-only reply rather than breaking the exchange —
    no local fallback engine needed."""
    clean = _speakable(text)
    if language == "hinglish":
        return _synth_sarvam(clean)
    return _synth_groq(clean)


def synth_sentences(sentences: list[str], language: str = "en") -> Iterator[bytes]:
    for s in sentences:
        s = s.strip()
        if s:
            yield synth(s, language=language)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_END.split(text) if s.strip()]


def warmup() -> None:
    synth("Ready.", language="en")