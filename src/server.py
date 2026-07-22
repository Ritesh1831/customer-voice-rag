"""FastAPI server: REST + WebSocket streaming + voice endpoints."""
from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from config.settings import settings
from src import pipeline
from src.voice import stt, tts

app = FastAPI(title="NovaPay Voice RAG")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATIC = Path(__file__).resolve().parent.parent / "static"


@app.on_event("startup")
def _startup():
    pipeline.warmup()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


class Ask(BaseModel):
    question: str
    session_id: str = "default"


@app.post("/api/ask")
def ask(body: Ask):
    a = pipeline.answer(body.question, session_id=body.session_id)
    return {
        "answer": a.text,
        "abstained": a.abstained,
        "reason": a.reason,
        "sources": a.sources,
        "timings": a.timings,
        "cached": a.cached,
        "language": a.language,
        "standalone_question": a.standalone_question,
    }


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    return {"text": stt.transcribe(data, file.filename or "audio.webm")}


class SpeakBody(BaseModel):
    text: str
    language: str = "en"


@app.post("/api/speak")
def speak(body: SpeakBody):
    return Response(content=tts.synth(body.text, language=body.language), media_type="audio/wav")


async def _safe_send_audio(sock: WebSocket, text: str, language: str) -> None:
    """TTS can fail (e.g. under memory pressure) — degrade to text-only
    instead of breaking the whole exchange when that happens."""
    try:
        wav = tts.synth(text, language=language)
        await sock.send_json({"type": "audio", "data": base64.b64encode(wav).decode()})
    except Exception as e:
        print(f"[tts] synth failed, degrading to text-only: {e}", flush=True)
        await sock.send_json({"type": "tts_failed"})


@app.websocket("/ws")
async def ws(sock: WebSocket):
    """Full streaming loop: text or audio in -> tokens + audio out.
    session_id is client-supplied (falls back to a per-connection uuid) so
    conversational memory persists across turns within the same chat session."""
    await sock.accept()
    session_id = str(uuid.uuid4())
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            session_id = msg.get("session_id") or session_id

            if msg.get("type") == "audio":
                audio = base64.b64decode(msg["data"])
                question = stt.transcribe(audio, "audio.webm")
                if not question:
                    await sock.send_json({
                        "type": "error",
                        "text": "Sorry, I couldn't hear that clearly — could you try again?",
                    })
                    continue
                await sock.send_json({"type": "transcript", "text": question})
            else:
                question = (msg.get("text") or "").strip()

            if not question:
                await sock.send_json({"type": "error", "text": "Empty question."})
                continue

            want_audio = bool(msg.get("audio_reply", True))
            lang_for_audio = "en"

            for evt in pipeline.answer_stream(question, session_id=session_id):
                if evt["type"] == "meta":
                    lang_for_audio = evt.get("language", "en")
                    await sock.send_json(evt)
                elif evt["type"] == "sentence":
                    await sock.send_json({"type": "sentence", "text": evt["text"]})
                    if want_audio:
                        await _safe_send_audio(sock, evt["text"], lang_for_audio)
                elif evt["type"] == "abstain":
                    lang_for_audio = evt.get("language", "en")
                    await sock.send_json(evt)
                    if want_audio:
                        await _safe_send_audio(sock, evt["text"], lang_for_audio)
                else:
                    await sock.send_json(evt)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)