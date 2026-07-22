"""BGE-M3 embedder — ONNX Runtime execution (bypasses torch CPU kernels entirely).

Uses the pre-converted aapot/bge-m3-onnx weights, run through onnxruntime
directly. Produces numerically equivalent dense + sparse (lexical) outputs
to the original FlagEmbedding/torch path, but never executes a torch forward
pass — this avoids a native c10.dll crash some Windows CPUs hit specifically
during BGE-M3's compute (plain torch matmul works fine; the model's actual
kernels don't, on this machine).
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

import numpy as np

from config.settings import settings

_lock = threading.Lock()
_session = None
_tokenizer = None

ONNX_REPO = "aapot/bge-m3-onnx"


def _load():
    global _session, _tokenizer
    if _session is not None:
        return
    with _lock:
        if _session is not None:
            return
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        print("[embedder] loading BGE-M3 ONNX (one-time, ~2.3GB if not cached)...", flush=True)
        onnx_path = hf_hub_download(ONNX_REPO, "model.onnx", cache_dir=str(settings.model_cache))
        # model.onnx references model.onnx.data by relative filename, so it
        # must be downloaded into the same directory to be found at load time.
        hf_hub_download(ONNX_REPO, "model.onnx.data", cache_dir=str(settings.model_cache))

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        _tokenizer = AutoTokenizer.from_pretrained(ONNX_REPO, cache_dir=str(settings.model_cache))
        print("[embedder] BGE-M3 ONNX ready.", flush=True)


def _unused_token_ids(tok) -> set[int]:
    return {
        i for i in (tok.cls_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id)
        if i is not None
    }


def _process_token_weights(token_weights: np.ndarray, input_ids: list[int],
                            unused: set[int]) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for w, idx in zip(token_weights, input_ids):
        if idx not in unused and w > 0:
            key = str(idx)
            if w > result[key]:
                result[key] = float(w)
    return dict(result)


def _run(texts: list[str], max_length: int) -> dict[str, Any]:
    _load()
    all_dense, all_sparse = [], []
    unused = _unused_token_ids(_tokenizer)
    bs = settings.embed_batch_size

    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        inputs = _tokenizer(
            batch, padding="longest", truncation=True,
            max_length=max_length, return_tensors="np",
        )
        onnx_inputs = {k: v for k, v in inputs.items() if k in {"input_ids", "attention_mask"}}
        outputs = _session.run(None, onnx_inputs)

        dense = outputs[0]                    # (batch, 1024)
        sparse_raw = outputs[1].squeeze(-1)   # (batch, seq_len)

        all_dense.append(dense)
        for row_weights, row_ids in zip(sparse_raw, inputs["input_ids"]):
            all_sparse.append(_process_token_weights(row_weights, row_ids.tolist(), unused))

    return {
        "dense": np.concatenate(all_dense, axis=0).astype(np.float32),
        "sparse": all_sparse,
    }


def encode_documents(texts: list[str]) -> dict[str, Any]:
    print(f"[embedder] starting encode of {len(texts)} texts...", flush=True)
    out = _run(texts, settings.embed_max_length)
    print("[embedder] encode finished", flush=True)
    return out


def encode_query(text: str) -> dict[str, Any]:
    out = _run([text], 256)
    return {"dense": out["dense"][0], "sparse": out["sparse"][0]}


def sparse_dot(q: dict, d: dict) -> float:
    """Lexical similarity between two BGE-M3 sparse weight dicts."""
    if not q or not d:
        return 0.0
    if len(q) > len(d):
        q, d = d, q
    return float(sum(w * d.get(k, 0.0) for k, w in q.items()))


def warmup() -> None:
    encode_query("warmup")