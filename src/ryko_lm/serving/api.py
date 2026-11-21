"""Ryko inference server (FastAPI).

Endpoints
---------
GET  /                    — the web playground (static, RTL Arabic UI)
GET  /health              — liveness + readiness
GET  /model               — architecture + training metadata
POST /generate            — JSON generation
POST /generate/stream     — Server-Sent-Events streaming generation
GET  /knowledge/search    — hybrid retrieval over the Knowledge Core (provenance)
GET  /knowledge/ideas     — idea layer (العقل الفكري): school-attributed claims
GET  /knowledge/relations — argument-relations graph (علاقات الحجاج)
POST /knowledge/cite      — citation verification (ryko-cite/1 verdict)
POST /knowledge/ask       — grounded generation: retrieve → generate → guard
                            → per-sentence cite audit → deliver or abstain

Configuration (environment variables):
    RYKO_MODEL_PATH      default: artifacts/dpo_round2_small/final.pt — the
                         promoted round-2 serving point (52.18M params, the
                         v1.23.0 point: SFT+DPO + ling merge + DPO round 2;
                         receipt verified offline — since v1.24.0).
                         Promotion ledger (each step a conscious release act):
                         artifacts/training/final.pt (stale v1.1 archive, wrong
                         for years — what the first PalmX-IC runs measured by
                         mistake) → artifacts/training_v5c/final.pt (11.5M,
                         receipted, since v1.17.0) →
                         artifacts/dpo_round2_small/final.pt (since v1.24.0).
                         The default is pinned by
                         test_guard_provenance
                         .test_serving_default_is_the_promoted_round2_point —
                         a future promotion must re-measure the external
                         benchmarks on the new point and update the pin, this
                         ledger, conftest.FINAL_MODEL and docs/ROUTING.md.
    RYKO_TOKENIZER_DIR   default: artifacts/tokenizer
    RYKO_DEVICE          default: cpu
    RYKO_THREADS         default: 2
    RYKO_KNOWLEDGE_DB    default: data/knowledge/ryko.sqlite

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

# v1.24.0 promotion: the serving default is the measured round-2 point —
# re-uniting the serving chain with the measurement chain (every published
# number is now measured on the exact checkpoint the server serves).  The
# promotion is pinned byte-exactly by
# test_guard_provenance.test_serving_default_is_the_promoted_round2_point.
MODEL_PATH = Path(os.environ.get("RYKO_MODEL_PATH",
                                 "artifacts/dpo_round2_small/final.pt"))
TOKENIZER_DIR = Path(os.environ.get("RYKO_TOKENIZER_DIR", "artifacts/tokenizer"))
DEVICE = os.environ.get("RYKO_DEVICE", "cpu")
THREADS = int(os.environ.get("RYKO_THREADS", "2"))
KNOWLEDGE_DB = Path(os.environ.get("RYKO_KNOWLEDGE_DB", "data/knowledge/ryko.sqlite"))

state: dict = {"model": None, "tokenizer": None, "meta": {}, "store": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    torch.set_num_threads(max(1, THREADS))
    if MODEL_PATH.exists():
        state["model"] = RykoLM.load(MODEL_PATH, device=DEVICE)
        state["tokenizer"] = RykoTokenizer.from_dir(TOKENIZER_DIR)
        summary = MODEL_PATH.with_name("summary.json")
        if not summary.exists():
            summary = Path("artifacts/training/summary.json")
        state["meta"] = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}
        print(f"[ryko] model loaded: {MODEL_PATH} ({state['model'].num_params():,} params)")
    else:
        print(f"[ryko] WARNING: model not found at {MODEL_PATH} — /generate disabled")
    if KNOWLEDGE_DB.exists():
        from ..knowledge.store import KnowledgeStore

        # v1.17.0 fix: the serving app is a READ-ONLY consumer of the store —
        # persist_vectors=False keeps the derived passage_vec table empty, so
        # serving against the committed repo DB never doubles its size or
        # breaks its receipt (the old default persisted ~130 MB of vectors on
        # every session, silently invalidating the committed sha256).
        #
        # v1.28.0: the first search against the committed DB re-derived the
        # whole index per session (~41s measured on 93k passages) — a 45s
        # cold-start on /knowledge/*.  The derived sidecar cache
        # (gitignored, best-effort, fingerprint-keyed) turns that into a
        # ~2s load with bit-identical search results.  Set
        # RYKO_VECTOR_CACHE="" to disable.
        state["store"] = KnowledgeStore(
            str(KNOWLEDGE_DB), persist_vectors=False,
            vector_cache=os.environ.get("RYKO_VECTOR_CACHE",
                                        "data/knowledge/ryko.vecs.cache.pkl") or None,
        )
        print(f"[ryko] knowledge store loaded: {KNOWLEDGE_DB}")
    else:
        print(f"[ryko] WARNING: knowledge store not found at {KNOWLEDGE_DB} — /knowledge/* disabled")
    yield
    state["model"] = None
    state["store"] = None


app = FastAPI(title="Ryko Arabic LM", version="1.28.2", lifespan=lifespan)


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
        "knowledge_store_loaded": state["store"] is not None,
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
    generated = int(out.size(1))
    # QA v1.27.0-prep (FLAW_REPORT F1): immediate-EOS abstention must be
    # observable to clients, not indistinguishable from a failed call.
    eos_immediate = generated <= 1 and not text.strip()
    return {
        "completion": text,
        "prompt_tokens": len(ids),
        "generated_tokens": generated,
        "abstained": eos_immediate,
        "finish_reason": "eos_immediate" if eos_immediate else "stop",
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


# --------------------------------------------------------------- knowledge
DISCLAIMER = ("مسودة مولّدة آلياً مدعومة بمصادر مسترجعة من نواة المعرفة — "
              "ليست فتوى ولا مرجعاً شرعياً؛ راجع النصوص الأصلية بصفحاتها.")


def _require_store() -> "KnowledgeStore":  # noqa: F821
    if state["store"] is None:
        raise HTTPException(status_code=503, detail="knowledge store not loaded")
    return state["store"]


@app.get("/knowledge/search")
def knowledge_search(q: str, k: int = 5, mode: str = "hybrid",
                     book: str | None = None) -> dict:
    """Deterministic hybrid retrieval with page-level provenance."""
    store = _require_store()
    k = max(1, min(k, 20))
    hits = store.search(q, k=k, mode=mode, book_id=book)
    return {"query": q, "mode": mode, "hits": [h.as_dict() for h in hits]}


@app.get("/knowledge/ideas")
def knowledge_ideas(q: str, k: int = 8, school: str | None = None,
                    layer: str | None = None) -> dict:
    """Curated idea layers (الفكر والاقتصاد واللغة): hybrid ranking over claims.

    Returns school-attributed ideas (originator + school + layer in meta)
    seeded from the curated manifests (فكر v2 منقحة / اقتصاد وحقوق / لغة).
    ``school`` optionally restricts to one curated school/section id and
    ``layer`` to one curated layer (``fikr`` | ``econom`` | ``lugha``).
    """
    from ..knowledge.ideas import search_ideas

    store = _require_store()
    k = max(1, min(k, 20))
    return {"query": q, "protocol": "ryko-ideas/1",
            "school": school, "layer": layer,
            "ideas": search_ideas(store, q, k=k,
                                  schools={school} if school else None,
                                  layers={layer} if layer else None)}


@app.get("/knowledge/relations")
def knowledge_relations(claim_id: str | None = None, layer: str | None = None,
                        relation: str | None = None, k: int = 100) -> dict:
    """Argument-relations graph (علاقات الحجاج) — ryko-relations/1.

    Lists seeded support/opposition relations between curated claims with
    both endpoints fully resolved and a documented evidence line each.
    ``claim_id`` traverses from any node (matches either endpoint); ``layer``
    restricts to one curated layer and ``relation`` to one closed-vocabulary
    verb (supports / responds / summarizes / …).
    """
    from ..knowledge.relations import list_relations, relations_stats

    store = _require_store()
    k = max(1, min(k, 200))
    return {"protocol": "ryko-relations/1",
            "claim_id": claim_id, "layer": layer, "relation": relation,
            "graph": relations_stats(store),
            "relations": list_relations(store, layer=layer,
                                        relation=relation,
                                        claim_id=claim_id, k=k)}


class CiteRequest(BaseModel):
    quote: str = Field(min_length=10, max_length=2000)
    book_id: str | None = None


@app.post("/knowledge/cite")
def knowledge_cite(req: CiteRequest) -> dict:
    """ryko-cite/1 verdict: verified / partial / not_found + quote_sha256.

    v1.17.0: ``write_status=False`` — the serving app is a read-only
    consumer of the committed store, so verified-quote status caching must
    never dirty the repo DB (it would invalidate the receipt sha256).  A
    deployment that wants the status cache points ``RYKO_KNOWLEDGE_DB`` at
    a writable copy."""
    from ..knowledge.citation import verify_quote

    store = _require_store()
    return verify_quote(store, req.quote, book_id=req.book_id,
                        write_status=False).as_dict()


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    k: int = Field(default=3, ge=1, le=5)
    max_new_tokens: int = Field(default=80, ge=8, le=200)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=1, le=200)
    top_p: float = Field(default=0.95, ge=0.05, le=1.0)
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=2.0)
    seed: int | None = None
    strict: bool = Field(default=False, description="deliver only fully attributed answers")


ABSTAIN_MESSAGE = ("أمتنع عن تقديم إجابة موثقة لهذا السؤال: لم يُعثر في المقاطع "
                   "المسترجعة من نواة المعرفة على سند نصي كافٍ لأي جملة من الجواب. "
                   "المصادر أدناه تعرض ما وُجد فعلاً — راجعها مباشرة.")


def _lexical_grounding(answer: str, passages: list[str]) -> tuple[float, int]:
    """Deterministic lexical-grounding ratio: answer tokens covered by sources."""
    from ..knowledge.textproc import ar_tokens

    a_tokens = set(ar_tokens(answer))
    if not a_tokens:
        return 0.0, -1
    covered: set[str] = set()
    best_idx, best_ratio = 0, 0.0
    for i, p in enumerate(passages):
        p_tokens = set(ar_tokens(p))
        hit = len(a_tokens & p_tokens)
        ratio = hit / len(a_tokens)
        covered |= a_tokens & p_tokens
        if ratio > best_ratio:
            best_idx, best_ratio = i, ratio
    return len(covered) / len(a_tokens), best_idx


@app.post("/knowledge/ask")
def knowledge_ask(req: AskRequest) -> dict:
    """Grounded generation: retrieve → build a sourced prompt → generate →
    guard → cite-audit every sentence against the retrieval pool →
    deliver the answer with its anchor, or abstain (v1.15.0).

    The decision rule is structural, documented and two-sided:
    * ``attributed``  (≥1 sentence verbatim in a stored passage) — deliver.
    * ``partial``     (≥1 sentence ≥82% token-contained) — deliver unless
      ``strict=true``; flagged as unattributed-verbatim either way.
    * ``unattributed`` (nothing anchors) — **abstain**: ``answer`` keeps the
      raw draft for auditability, ``delivered_answer`` is the abstention
      message, never presented as sourced.

    The answer is returned as an *attribution-eligible draft*, never a fatwa
    (see DISCLAIMER and docs/KNOWLEDGE_CORE.md §4).
    """
    from ..knowledge.citation import audit_answer
    from ..serving.guard import guard_output

    _require_ready()
    store = _require_store()
    t0 = time.perf_counter()
    hits = store.search(req.question, k=req.k, mode="hybrid")
    if not hits:
        raise HTTPException(status_code=404, detail="no passages retrieved for question")
    clamped = [h.text[:220] for h in hits]
    context = "\n\n".join(f"[{i + 1}] {txt}" for i, txt in enumerate(clamped))
    prompt = f"المصادر:\n{context}\n\nالسؤال: {req.question}\nالجواب:"
    ids = tok_encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)
    out = state["model"].generate(
        idx,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        eos_id=state["tokenizer"].eos_id,
        generator=_make_generator(req),
    )
    answer = state["tokenizer"].decode(out[0].tolist())
    grounding, best_idx = _lexical_grounding(answer, clamped)
    audit = audit_answer(answer, hits)
    deliver = audit.status == "attributed" or \
        (audit.status == "partial" and not req.strict)
    abstain_reason = None if deliver else \
        ("strict: الإجابة جزئية السند والوضع الصارم يشترط سنداً حرفياً"
         if req.strict and audit.status == "partial"
         else "لا توجد جملة واحدة في الجواب لها سند نصي في المقاطع المسترجعة")
    sources = [{
        "passage_id": h.passage_id, "book_id": h.book_id, "title_ar": h.title_ar,
        "page_num": h.page_num, "chars_used": len(clamped[i]),
    } for i, h in enumerate(hits)]
    return {
        "question": req.question,
        "answer": answer,
        "delivered_answer": answer if deliver else ABSTAIN_MESSAGE,
        "decision": "deliver" if deliver else "abstain",
        "abstain_reason": abstain_reason,
        "citation_audit": audit.as_dict(),
        "sources": sources,
        "primary_source": sources[max(best_idx, 0)] if best_idx >= 0 else None,
        "grounding_ratio": round(grounding, 4),
        "guard": guard_output(answer, prompt=req.question),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "disclaimer": DISCLAIMER,
    }


def tok_encode(text: str) -> list[int]:
    return state["tokenizer"].encode(text, add_bos=True)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    static = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(static.read_text(encoding="utf-8"))
