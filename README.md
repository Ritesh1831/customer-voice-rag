# Customer Voice RAG Assistant

A voice-and-text customer support chatbot built with retrieval-augmented generation
(RAG), layered hallucination guardrails, hybrid retrieval, conversational memory, and
Hinglish (Hindi-English code-switching) support.

## Architecture

```
User question (voice or text)
        │
        ▼
Chitchat router (regex) ──► instant reply for greetings/thanks/goodbye (skips RAG entirely)
        │  (else)
        ▼
Language detection (English / Hinglish)
        │
        ▼
Conversational memory ──► rewrites follow-ups into standalone questions
        │                  (only triggered when a question looks like a follow-up)
        ▼
Hybrid retrieval: dense (BGE-M3) + sparse (BGE-M3 lexical) + BM25
        │           → Reciprocal Rank Fusion
        ▼
Cross-encoder reranking (bge-reranker-v2-m3, INT8)
        │
        ▼
Pre-generation gate ──► abstain if best rerank score is too low (skips LLM call)
        │  (else)
        ▼
Grounded LLM generation (Llama 3.3 70B via Groq, temperature 0, streaming)
        │
        ▼
Post-generation grounding check ──► every number in the answer must trace
        │                            back to retrieved context, or it's rejected
        ▼
Text reply + sentence-by-sentence streamed TTS (gapless playback)
```

## Models used & parameters

| Component | Model | Params | Runs on | Key parameters |
|---|---|---|---|---|
| Speech-to-text | `whisper-large-v3-turbo` | 809M | Groq API | `temperature=0`, auto language-detect |
| LLM (answers) | `llama-3.3-70b-versatile` | 70B | Groq API | `temperature=0`, `seed=42`, streaming |
| LLM (query rewrite) | `llama-3.1-8b-instant` | 8B | Groq API | `temperature=0`, only invoked for follow-up questions |
| Embeddings | `BAAI/bge-m3` | 568M | Local (ONNX Runtime) | dense + sparse in one forward pass, `max_length=1024` |
| Reranker | `bge-reranker-v2-m3` (INT8) | 568M | Local (ONNX Runtime) | scores top ~20 fused candidates |
| TTS (English) | Orpheus (`canopylabs/orpheus-v1-english`) | — | Groq API | |
| TTS (Hinglish) | Kokoro-82M | 82M | Local (ONNX) | |
| Vector store | Chroma | — | Local | HNSW, cosine similarity |
| Lexical search | BM25 (`rank-bm25`) | — | Local | catches exact-token matches (fees, tier numbers) |

Retrieval tuning: `dense_k=25`, `sparse_k=25` candidates fused via RRF (`k=60`) down to
`final_k=4` chunks sent to the LLM. Guardrail thresholds: `rerank_abstain_threshold=-2.0`
(pre-generation gate), `grounding_min_overlap=0.40` (post-generation gate).

## Major features

- **Hybrid retrieval with Reciprocal Rank Fusion** — dense + sparse + BM25 combined,
  so both semantic similarity and exact-token matches are covered.
- **Layered, zero-hallucination-by-design guardrails** — a pre-generation gate skips
  the LLM entirely when retrieval confidence is low, and a post-generation check
  verifies every number in the answer against the retrieved context before returning it.
- **Conversational memory** with a cheap heuristic gate — only pays for an LLM
  rewrite call when a question actually looks like a follow-up, so ordinary questions
  incur zero extra latency.
- **Hinglish support end-to-end** — language detection drives the LLM's response
  language, the exact abstention sentence used, and which TTS engine/voice is used,
  kept consistent across text and audio in the same turn.
- **Chitchat routing** — greetings and small talk answered instantly via regex,
  bypassing retrieval and the LLM.
- **Semantic response caching** — near-duplicate questions reuse a prior verified answer.
- **Sentence-level streaming with gapless audio** — TTS starts on the first completed
  sentence rather than waiting for the full response; audio clips are scheduled
  back-to-back on a single Web Audio timeline instead of playing as separate clips.
- **Structure-aware document ingestion** — a custom parser distinguishes headings,
  bullets, tables, and FAQ pairs; table rows are linearized into standalone sentences
  and every chunk carries a contextual header so isolated fragments stay retrievable.
- **Graceful degradation** — a TTS failure degrades to a text-only reply instead of
  breaking the exchange.

## Notable engineering decisions

- **Embeddings and reranking run on ONNX Runtime, not native PyTorch.** This started
  as a fix for a reproducible native crash (`0xC0000005` access violation inside
  torch's `c10.dll`, isolated via Windows Event Viewer and confirmed by testing
  plain torch operations against actual model inference separately) and turned out
  to also be the better architecture — no local model tracing/export step, smaller
  and more predictable memory footprint.
- **A pip conflict masked a second pip conflict.** `FlagEmbedding`'s exact
  `transformers` pin was fixed first, which surfaced an unrelated `numpy` version
  conflict with the TTS library — resolved by reproducing the full dependency
  resolution in an isolated environment rather than guessing at version bumps.
- **STT prompt-echo hallucination.** A biasing prompt that was too long and
  jargon-dense caused Whisper to occasionally regurgitate fragments of the prompt
  itself instead of transcribing the audio — a known Whisper failure mode, fixed by
  shortening the prompt and adding a runtime check that discards suspected echoes.
- **Evaluation bug, not a model bug.** Early hallucination-rate numbers were
  inflated because the eval script wasn't recognizing correct self-abstentions as
  abstentions — fixed at the harness level; true hallucination rate on the test set
  is 0%.
- **Measured resource tradeoffs, not assumed ones.** TTS was split between a cloud
  model (English) and a local model (Hinglish) after directly observing that
  running three simultaneous local ONNX models caused real memory-pressure failures
  on constrained hardware — the split was a measured fix, not a default choice.