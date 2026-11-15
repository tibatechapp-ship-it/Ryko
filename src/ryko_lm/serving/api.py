"""Ryko inference server (FastAPI).

Endpoints
---------
GET  /                    — the web playground (static, RTL Arabic UI)
GET  /health              — liveness + readiness
GET  /model               — architecture + training metadata
POST /generate            — JSON generation
POST /generate/stream     — Server-Sent-Events streaming generation

Configuration (environment variables):
    RYKO_MODEL_PATH      default: artifacts/training/final.pt
    RYKO_TOKENIZER_DIR   default: artifacts/tokenizer
    RYKO_DEVICE          default: cpu
    RYKO_THREADS         default: 2

Run:
    uvicorn ryko_lm.serving.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..model.ryko_model import RykoLM
from ..tokenizer.loader import RykoTokenizer

MODEL_PATH = Path(os.environ.get("RYKO_MODEL_PATH", "artifacts/training/final.pt"))
TOKENIZER_DIR = Path(os.environ.get("RYKO_TOKENIZER_DIR", "artifacts/tokenizer"))
DEVICE = os.environ.get("RYKO_DEVICE", "cpu")
THREADS = int(os.environ.get("RYKO_THREADS", "2"))

state: dict = {"model": None, "tokenizer": None, "meta": {}}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    torch.set_num_threads(max(1, THREADS))
    if MODEL_PATH.exists():
        state["model"] = RykoLM.load(MODEL_PATH, device=DEVICE)
        state["tokenizer"] = RykoTokenizer.from_dir(TOKENIZER_DIR)
        summary = Path("artifacts/training/summary.json")
        state["meta"] = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}
        print(f"[ryko] model loaded: {MODEL_PATH} ({state['model'].num_params():,} params)")
    else:
        print(f"[ryko] WARNING: model not found at {MODEL_PATH} — /generate disabled")
    yield
    state["model"] = None


app = FastAPI(title="Ryko Arabic LM", version="1.0.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=64, ge=1, le=200)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=1, le=200)
    top_p: float = Field(default=0.95, ge=0.05, le=1.0)
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=2.0)
    seed: int | None = None


def _require_ready() -> None:
    if state["model"] is None or state["tokenizer"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")


def _make_generator(req: GenerateRequest) -> torch.Generator | None:
    generator = torch.Generator().manual_seed(req.seed) if req.seed is not None else None
    return generator


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if state["model"] is not None else "degraded",
        "model_loaded": state["model"] is not None,
        "device": DEVICE,
    }


@app.get("/model")
def model_info() -> dict:
    _require_ready()
    cfg = state["model"].cfg
    return {
        "name": "ryko-tiny",
        "params": state["model"].num_params(),
        "vocab_size": cfg.vocab_size,
        "block_size": cfg.block_size,
        "layers": cfg.n_layer,
        "heads": cfg.n_head,
        "kv_heads": cfg.n_kv_head,
        "embedding_dim": cfg.n_embd,
        "tokenizer": "Byte-Level BPE (tokenizers, Rust)",
        "training": state.get("meta", {}),
    }


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    _require_ready()
    t0 = time.perf_counter()
    model, tok = state["model"], state["tokenizer"]
    ids = tok.encode(req.prompt, add_bos=True)
    if not ids:
        raise HTTPException(status_code=400, detail="prompt encoded to zero tokens")
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(
        idx,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        eos_id=tok.eos_id,
        generator=_make_generator(req),
    )
    text = tok.decode(out[0].tolist())
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "completion": text,
        "prompt_tokens": len(ids),
        "generated_tokens": int(out.size(1)),
        "latency_ms": round(latency_ms, 1),
    }


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest) -> StreamingResponse:
    """SSE stream: every event carries the full decoded text so far."""
    _require_ready()
    model, tok = state["model"], state["tokenizer"]
    ids = tok.encode(req.prompt, add_bos=True)
    if not ids:
        raise HTTPException(status_code=400, detail="prompt encoded to zero tokens")
    idx = torch.tensor([ids], dtype=torch.long)
    generator = _make_generator(req)

    async def event_source():
        started = time.perf_counter()
        buf: list[int] = []
        for token in model.generate_iter(
            idx,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            eos_id=tok.eos_id,
            generator=generator,
        ):
            buf.append(int(token[0, 0]))
            text = tok.decode(buf)
            yield f"data: {json.dumps({'text': text, 'tokens': len(buf)}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)  # yield control to the event loop
        latency = round((time.perf_counter() - started) * 1000, 1)
        yield f"data: {json.dumps({'done': True, 'latency_ms': latency, 'tokens': len(buf)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_source(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    static = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(static.read_text(encoding="utf-8"))
