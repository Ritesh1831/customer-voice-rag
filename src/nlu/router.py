"""Rule-based chitchat router: skip retrieval + LLM entirely for greetings,
thanks, goodbyes, and small talk. Pure regex -> near-zero latency."""
from __future__ import annotations

import re

GREETING_RE = re.compile(
    r"^(hi+|hello+|hey+|yo|namaste|namaskar|good\s?(morning|afternoon|evening))[\s!.,]*$",
    re.I,
)
THANKS_RE = re.compile(
    r"^(thanks?( you)?( so much| very much)?|thankyou|shukriya|dhanyavad|dhanyawad)[\s!.,]*$",
    re.I,
)
BYE_RE = re.compile(
    r"^(bye+|goodbye|see\s?y?a|catch\s?you\s?later|take\s?care|"
    r"ok(ay)?\s?bye|theek\s?hai\s?bye|alright\s?,?\s?thanks?)[\s!.,]*$",
    re.I,
)
SMALLTALK_RE = re.compile(
    r"^(how\s?are\s?you|what'?s\s?up|who\s?are\s?you|what\s?can\s?you\s?do|"
    r"are\s?you\s?(a\s?)?(bot|human|ai|robot)|"
    r"kaise\s?ho|kaisa\s?hai|kya\s?haal\s?hai|kya\s?hal\s?hai|"
    r"tum\s?kaun\s?ho|aap\s?kaun\s?hai?n?)[\s?!.,]*$",
    re.I,
)

TEMPLATES = {
    ("greeting", "en"): "Hi there! I'm your support assistant. "
                       "How can I help you with your account today?",
    ("greeting", "hinglish"): "Hi! Main aapka support assistant hoon. "
                             "Aaj main aapki kis tarah madad kar sakta hoon?",
    ("thanks", "en"): "You're welcome! Let me know if there's anything else I can help with.",
    ("thanks", "hinglish"): "Koi baat nahi! Agar aur kuch chahiye toh zaroor bataiye.",
    ("goodbye", "en"): "Take care! Reach out anytime you need help with your account.",
    ("goodbye", "hinglish"): "Theek hai, dhyaan rakhiyega! Jab bhi zarurat ho, message kijiye.",
    ("smalltalk", "en"): "I'm a support assistant for a digital wallet and payments app. I can "
                        "help with account verification, transfers, cards, fees, and disputes — "
                        "what would you like to know?",
    ("smalltalk", "hinglish"): "Main ek support assistant hoon digital wallet aur payments app ke "
                              "liye. Main verification, transfers, card, fees aur disputes jaise "
                              "topics mein madad kar sakta hoon — aap kya jaanna chahenge?",
}


def classify(text: str) -> str | None:
    """Returns 'greeting' | 'thanks' | 'goodbye' | 'smalltalk' | None (-> run full RAG pipeline)."""
    q = text.strip()
    if not q:
        return None
    if GREETING_RE.match(q):
        return "greeting"
    if THANKS_RE.match(q):
        return "thanks"
    if BYE_RE.match(q):
        return "goodbye"
    if SMALLTALK_RE.match(q):
        return "smalltalk"
    return None


def reply(kind: str, language: str) -> str:
    return TEMPLATES.get((kind, language), TEMPLATES[(kind, "en")])