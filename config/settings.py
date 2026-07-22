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
    pdf_path: Path = ROOT / "data" / "NovaPay_Customer_Support_KB.pdf"

    # ---------- Groq: STT, LLM, English TTS ----------
    groq_api_key: str = ""
    stt_model: str = "whisper-large-v3-turbo"
    stt_language: str = ""  # empty = auto-detect (needed for Hinglish speech)
    stt_temperature: float = 0.0

    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_max_tokens: int = 400
    llm_seed: int = 42
    llm_timeout_s: float = 30.0

    rewrite_model: str = "llama-3.1-8b-instant"
    rewrite_temperature: float = 0.0
    rewrite_max_tokens: int = 60

    groq_tts_model: str = "canopylabs/orpheus-v1-english"
    groq_tts_voice: str = "troy"

    # ---------- Sarvam: Hinglish TTS (Bulbul v3) ----------
    sarvam_api_key: str = ""
    sarvam_tts_model: str = "bulbul:v3"
    sarvam_tts_speaker: str = "shubh"
    sarvam_tts_language: str = "hi-IN"

    # ---------- Cohere: embeddings + reranking ----------
    cohere_api_key: str = ""
    cohere_embed_model: str = "embed-multilingual-v3.0"
    cohere_rerank_model: str = "rerank-v3.5"

    # ---------- Chroma Cloud: vector database ----------
    chroma_api_key: str = ""
    chroma_tenant: str = ""
    chroma_database: str = ""

    # ---------- Conversational memory ----------
    memory_max_turns: int = 6
    memory_turns_for_rewrite: int = 3

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
    # Cohere's relevance_score is 0-1 — thresholds are tuned to that scale.
    rerank_abstain_threshold: float = 0.15
    rerank_confident_threshold: float = 0.5
    grounding_min_overlap: float = 0.40
    enable_grounding_check: bool = True

    # ---------- Cache ----------
    semantic_cache_threshold: float = 0.94
    semantic_cache_size: int = 256

    # ---------- Server ----------
    host: str = "127.0.0.1"
    port: int = 8000


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)