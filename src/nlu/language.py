"""Lightweight, dependency-free Hinglish vs English detection.

Deliberately a heuristic, not a model call -> adds ~0ms latency.
Two signals: Devanagari script presence, and a marker-word list of common
Hindi function/verb words as typically romanized in Hinglish text.
"""
from __future__ import annotations

import re

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

HINGLISH_WORDS = {
    "hai", "hain", "ho", "hoga", "hogi", "honge", "tha", "thi",
    "kya", "kyu", "kyun", "kaise", "kaisi", "kaisa", "kab", "kahan", "kaha",
    "karo", "karna", "kar", "kiya", "kijiye", "kijiyega", "karenge",
    "chahiye", "chahiyeta", "mera", "meri", "mere", "tera", "teri", "tere",
    "uska", "uski", "uske", "iska", "iski", "iske", "yeh", "ye", "woh", "wo",
    "aur", "abhi", "nahi", "nahin", "haan", "bhai", "yaar", "paisa", "paise",
    "rupay", "rupaye", "bata", "batao", "bataiye", "dikhao", "dikha",
    "milega", "milegi", "milenge", "tak", "wala", "wali", "wale", "matlab",
    "samjha", "samajh", "acha", "accha", "theek", "thik", "zyada", "kam",
    "jaldi", "turant", "mujhe", "mujhko", "aapko", "aapka", "aapki", "aapke",
    "humko", "hume", "kuch", "sab", "sabhi", "koi", "kisi", "vaise", "waise",
    "shukriya", "dhanyavad", "dhanyawad", "namaste", "namaskar",
    "kripya", "zaroor", "bilkul", "haal", "hal",
}


def detect_language(text: str) -> str:
    """Returns 'hinglish' or 'en'."""
    if not text or not text.strip():
        return "en"
    if DEVANAGARI_RE.search(text):
        return "hinglish"

    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return "en"

    hits = sum(1 for t in tokens if t in HINGLISH_WORDS)
    if hits >= 2:
        return "hinglish"
    if hits >= 1 and (hits / len(tokens)) >= 0.12:
        return "hinglish"
    return "en"