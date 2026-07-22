"""Central configuration. Every tunable lives here."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---------- Paths ----------
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    storage_dir: Path = ROOT / "storage"
    chroma_dir: Path = ROOT / "storage" / "chroma_db"
    model_cache: Path = ROOT / "storage" / "models"
    pdf_path: Path = ROOT / "data" / "NovaPay_Customer_Support_KB.pdf"

    # ---------- API ----------
    groq_api_key: str = ""

    # ---------- STT ----------
    stt_model: str = "whisper-large-v3-turbo"
    # Empty = auto-detect. Needed so code-switched Hinglish speech isn't
    # forcibly decoded as English-only (which mangles the Hindi portions).
    stt_language: str = ""
    stt_temperature: float = 0.0

    # ---------- LLM (main answer generation) ----------
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_max_tokens: int = 400
    llm_seed: int = 42
    llm_timeout_s: float = 30.0

    # ---------- LLM (fast query rewriting for follow-up questions) ----------
    rewrite_model: str = "llama-3.1-8b-instant"
    rewrite_temperature: float = 0.0
    rewrite_max_tokens: int = 60

    # ---------- Conversational memory ----------
    memory_max_turns: int = 6          # turns kept per session
    memory_turns_for_rewrite: int = 3  # turns fed into the rewrite prompt

    # ---------- Embeddings ----------
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    embed_max_length: int = 1024
    embed_batch_size: int = 12
    embed_use_fp16: bool = False

    # ---------- Reranker ----------
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_length: int = 1024
    rerank_batch_size: int = 8

    # ---------- Chunking ----------
    chunk_target_tokens: int = 320
    chunk_max_tokens: int = 480
    chunk_min_tokens: int = 45
    chunk_overlap_sentences: int = 1

    # ---------- Retrieval ----------
    dense_k: int = 25
    sparse_k: int = 25
    rrf_k: int = 60
    fusion_k: int = 20
    final_k: int = 4
    neighbor_expansion: bool = True

    # ---------- Guardrails ----------
    rerank_abstain_threshold: float = -2.0
    rerank_confident_threshold: float = 0.5
    grounding_min_overlap: float = 0.40
    enable_grounding_check: bool = True

    # ---------- TTS ----------
    # English replies: Groq's Orpheus (cloud) — more natural, better number/
    # symbol pronunciation, and removes Kokoro's RAM footprint for the common case.
    groq_tts_model: str = "canopylabs/orpheus-v1-english"
    groq_tts_voice: str = "troy"          # try "hannah" too and pick by ear

    # Hinglish replies: Kokoro (local) — Orpheus has no confirmed Hindi support.
    tts_voice: str = "af_heart"
    tts_voice_hinglish: str = "hf_alpha"
    tts_lang_hinglish: str = "hi"
    tts_speed: float = 1.05
    tts_sample_rate: int = 24000

    # ---------- Cache ----------
    semantic_cache_threshold: float = 0.94
    semantic_cache_size: int = 256

    # ---------- Server ----------
    host: str = "127.0.0.1"
    port: int = 8000


settings = Settings()

for _d in (settings.storage_dir, settings.chroma_dir,
           settings.model_cache, settings.data_dir):
    _d.mkdir(parents=True, exist_ok=True)