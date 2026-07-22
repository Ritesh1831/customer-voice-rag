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
Hybrid retrieval: dense (Cohere Embed, via Chroma Cloud) + BM25
        │           → Reciprocal Rank Fusion
        ▼
Cross-encoder reranking (Cohere Rerank v3.5)
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

| Component | Model | Runs on | Key parameters |
|---|---|---|---|
| Speech-to-text | `whisper-large-v3-turbo` | Groq API | `temperature=0`, auto language-detect |
| LLM (answers) | `llama-3.3-70b-versatile` | Groq API | `temperature=0`, `seed=42`, streaming |
| LLM (query rewrite) | `llama-3.1-8b-instant` | Groq API | only invoked for follow-up questions |
| Embeddings | Cohere `embed-multilingual-v3.0` | API | dense vectors, batched (90/call) |
| Reranker | Cohere `rerank-v3.5` | API | scores top ~20 fused candidates |
| TTS (English) | Orpheus (`canopylabs/orpheus-v1-english`) | Groq API | |
| TTS (Hinglish) | Sarvam Bulbul v3 | API | purpose-trained for Hindi/English code-switching |
| Vector database | Chroma Cloud | Managed | fully hosted, no local persistence |
| Lexical search | BM25 (`rank-bm25`) | Local, in-process | rebuilt fresh from the database at each startup |

Retrieval tuning: `dense_k=25` candidates fused via RRF (`k=60`) down to `final_k=4`
chunks sent to the LLM. Guardrail thresholds (Cohere's 0-1 relevance scale):
`rerank_abstain_threshold=0.15` (pre-generation gate), `grounding_min_overlap=0.40`
(post-generation gate).

## Major features

- **Hybrid retrieval with Reciprocal Rank Fusion** — dense + BM25 combined, so both
  semantic similarity and exact-token matches (fee amounts, tier numbers) are covered.
- **Layered, zero-hallucination-by-design guardrails** — a pre-generation gate skips
  the LLM entirely when retrieval confidence is low, and a post-generation check
  verifies every number in the answer against the retrieved context before returning it.
- **Conversational memory** with a cheap heuristic gate — only pays for an LLM
  rewrite call when a question actually looks like a follow-up.
- **Hinglish support end-to-end** — language detection drives the LLM's response
  language, the exact abstention sentence used, and which TTS voice is used, kept
  consistent across text and audio in the same turn.
- **Chitchat routing** — greetings and small talk answered instantly via regex,
  bypassing retrieval and the LLM.
- **Semantic response caching** — near-duplicate questions reuse a prior verified answer.
- **Sentence-level streaming with gapless audio** — TTS starts on the first completed
  sentence; audio clips are scheduled back-to-back on a single Web Audio timeline
  instead of playing as separate clips.
- **Structure-aware document ingestion** — a custom parser distinguishes headings,
  bullets, tables, and FAQ pairs; table rows are linearized into standalone sentences
  and every chunk carries a contextual header so isolated fragments stay retrievable.
- **Graceful degradation** — a TTS failure degrades to a text-only reply instead of
  breaking the exchange.

## Notable engineering decisions

- **Fully managed backend, by design, not by default.** Embeddings, reranking, the
  vector database, and Hinglish TTS all run as managed API services rather than local
  models. This followed direct, measured evidence: local ONNX inference caused a
  reproducible native crash on Windows (`0xC0000005` inside torch's `c10.dll`,
  isolated via Event Viewer), and running three simultaneous local models created
  real memory-pressure failures on constrained hardware. Rather than working around
  both issues, the fix was to remove local model inference from the critical path
  entirely — which also happens to make the app trivially deployable on a free-tier
  host, since there's no multi-gigabyte model to load into memory.
- **Reranker score scale is not portable across providers.** A local cross-encoder's
  raw logits and Cohere's 0-1 relevance score are different scales; reusing a
  threshold tuned for one against the other would silently disable the abstention
  gate. Caught before it shipped by explicitly checking the guardrail's behavior
  after switching providers, not just confirming the API call succeeded.
- **STT prompt-echo hallucination.** A biasing prompt that was too long and
  jargon-dense caused Whisper to occasionally regurgitate fragments of the prompt
  itself instead of transcribing the audio — a known Whisper failure mode, fixed by
  shortening the prompt and adding a runtime check that discards suspected echoes.
- **Evaluation bug, not a model bug.** Early hallucination-rate numbers were
  inflated because the eval script wasn't recognizing correct self-abstentions as
  abstentions — fixed at the harness level; true hallucination rate on the test set
  is 0%.