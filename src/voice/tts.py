"""TTS: Groq Orpheus (cloud) for English, Kokoro (local) for Hinglish.

Orpheus is used for English because it's more natural-sounding and has
better built-in number/symbol pronunciation than a small local model, and
it removes Kokoro from the memory budget for the common case. Kokoro stays
for Hinglish only, since Orpheus has no confirmed Hindi support.
"""
from __future__ import annotations

import io
import re
import threading
import urllib.request
from typing import Iterator

import soundfile as sf
from groq import Groq

from config.settings import settings

_lock = threading.Lock()
_tts = None
_groq_client: Groq | None = None

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

SENT_END = re.compile(r"(?<=[.!?])\s+")


def _download(url: str, dest):
    if dest.exists():
        return
    print(f"Downloading {dest.name} (one-time)...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def get_kokoro():
    global _tts
    if _tts is None:
        with _lock:
            if _tts is None:
                from kokoro_onnx import Kokoro
                mp = settings.model_cache / "kokoro-v1.0.onnx"
                vp = settings.model_cache / "voices-v1.0.bin"
                _download(MODEL_URL, mp)
                _download(VOICES_URL, vp)
                _tts = Kokoro(str(mp), str(vp))
    return _tts


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key, timeout=30.0)
    return _groq_client


def _speakable(text: str) -> str:
    """Make text sound natural when read aloud. Applied for both engines —
    Orpheus's own number handling is better, but this is a harmless safety net."""
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
        model=settings.groq_tts_model,
        voice=settings.groq_tts_voice,
        input=text,
        response_format="wav",
    )
    return response.read()


def _synth_kokoro(text: str, language: str) -> bytes:
    voice_map = {
        "en": (settings.tts_voice, "en-us"),
        "hinglish": (settings.tts_voice_hinglish, settings.tts_lang_hinglish),
    }
    voice, lang_code = voice_map.get(language, voice_map["en"])
    try:
        audio, sr = get_kokoro().create(
            text, voice=voice, speed=settings.tts_speed, lang=lang_code,
        )
    except Exception as e:
        print(f"Kokoro voice '{voice}'/'{lang_code}' failed ({e}); falling back to English voice.")
        en_voice, en_lang = voice_map["en"]
        audio, sr = get_kokoro().create(
            text, voice=en_voice, speed=settings.tts_speed, lang=en_lang,
        )
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def synth(text: str, language: str = "en") -> bytes:
    """Full-utterance synthesis -> WAV bytes.
    English -> Groq Orpheus (cloud). Hinglish -> Kokoro (local).
    Falls back to Kokoro-English if the Groq call fails for any reason
    (network issue, rate limit, etc.) so a single API hiccup doesn't kill
    the whole reply."""
    clean = _speakable(text)
    if language == "en":
        try:
            return _synth_groq(clean)
        except Exception as e:
            print(f"[tts] Groq Orpheus failed ({e}); falling back to local Kokoro.", flush=True)
            return _synth_kokoro(clean, "en")
    return _synth_kokoro(clean, language)


def synth_sentences(sentences: list[str], language: str = "en") -> Iterator[bytes]:
    for s in sentences:
        s = s.strip()
        if s:
            yield synth(s, language=language)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_END.split(text) if s.strip()]


def warmup() -> None:
    try:
        synth("Ready.", language="en")
    except Exception as e:
        print(f"[tts] warmup (Groq) failed, will retry per-request: {e}", flush=True)