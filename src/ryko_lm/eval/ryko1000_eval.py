"""RYKO-1000 evaluation runner — model gates + materialized failure probes.

Turns the finished ``ryko1000_seed.jsonl`` (10 families x 100 self-documenting
cases) into a **runnable** evaluation for any Ryko checkpoint:

1. ``mcq`` families (700 cases)  — length-normalized log-likelihood scoring
   (``benchmarks.score_option`` semantics, deterministic, no sampling).
2. ``abstain`` family (100 cases) — two-sided abstention gate at the
   calibrated operating point (default 0.65, docs/results/abstain_calibration).
3. ``cite`` cases (200) — engine-only citation audit (``ryko-cite/1``);
   independent of the model, reported as alignment.
4. **Failure probes materialized** — the seed's declarative probes
   (``splice`` / ``wrong_attribution`` / ``evidenceless``) become executable
   engine checks: a cross-book splice must return ``not_found``; the case's
   quote scoped to a WRONG book must never verify; an absence-token question
   must be provably unanswerable from the corpus.  Probes without the
   anchors needed to materialize are counted honestly as
   ``not_materializable`` (never silently skipped).
5. Two prompt modes — ``base`` (question only) and ``rag`` (question +
   top-2 hybrid-retrieved passages from the Knowledge Core, blind retrieval,
   context-budget aware via the same deterministic truncation the bench
   runner uses).  Latency (mean / p50 / p95 ms per scored case + retrieval
   time) is measured in both modes — the سرعة contract is a first-class
   metric, not a footnote.

Determinism: no sampling, fixed retrieval, fixed pairing, and the JSON
report carries NO wall-clock timestamps, so two identical runs are
byte-comparable (regression discipline like the dataset receipts).

Usage::

    python -m ryko_lm.eval.ryko1000_eval \\
        --seed docs/benchmarks/ryko1000_seed.jsonl \\
        --store data/knowledge/ryko.sqlite \\
        --checkpoint artifacts/training_v5c/final.pt \\
        --tokenizer artifacts/tokenizer/ryko-tokenizer.json \\
        --mode both \\
        --out docs/results/ryko1000_eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from ryko_lm.eval.benchmarks import _softmax, score_option
from ryko_lm.eval.verify_ryko1000 import VerificationError, load_ryko1000
from ryko_lm.knowledge.citation import verify_quote
from ryko_lm.knowledge.store import KnowledgeStore
from ryko_lm.knowledge.textproc import ar_tokens

SCHEMA = "ryko1000-eval/1"
# v1.14.0 calibrated operating point (docs/results/abstain_calibration.*):
# balanced criterion over the two-sided abstain set.
DEFAULT_ABSTAIN_THRESHOLD = 0.65
RAG_K = 2          # retrieved passages injected per case in rag mode
RAG_MODE = "hybrid"
QUOTE_HEAD = 120   # builder-mirroring word-truncated passage quote
SPLICE_HEAD, SPLICE_TAIL = 60, 60
PROBE_KINDS = ("splice", "wrong_attribution", "evidenceless")


def seed_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def to_bench_cases(cases: list[dict]) -> list:
    """ryko1000/1 dicts -> benchmarks.Case list (family carried in meta)."""
    from ryko_lm.eval.benchmarks import Case

    out: list = []
    for rec in cases:
        out.append(Case(
            case_id=rec["case_id"], kind=rec["kind"], source=rec["source"],
            domain=rec["domain"], question=rec.get("question", ""),
            options=tuple(rec.get("options") or ()),
            answer=rec.get("answer"),
            should_answer=rec.get("should_answer"),
            quote=rec.get("quote", ""), expect=rec.get("expect"),
            meta={**(rec.get("meta") or {}), "family": rec["family"]},
        ))
    return out


# ------------------------------------------------------------- model scoring
def _score_case(model, tok, case, device: str, context: str = "") -> dict:
    """benchmarks.run_mcq semantics for ONE case + wall-clock timing.

    ``context`` (rag mode) is appended AFTER the question so the prompt
    template ``السؤال: {q}\nالجواب:`` stays intact; over-budget prompts
    take the same deterministic head-truncation path as the bench runner.
    """
    question = case.question
    if context:
        question = (f"{question}\nالشواهد المُسترجعة من نواة المعرفة:\n"
                    f"{context}")
    t0 = time.perf_counter()
    scores: list[float] = []
    truncated = False
    max_len = getattr(getattr(model, "cfg", None), "block_size", None)
    for option in case.options:
        s, t = score_option(model, tok, question, option, device,
                            max_len=max_len)
        scores.append(s)
        truncated = truncated or t
    best = max(range(len(scores)), key=lambda i: (scores[i], -i))
    probs = _softmax(scores)
    ms = (time.perf_counter() - t0) * 1000.0
    return {
        "case_id": case.case_id, "family": case.meta.get("family", "?"),
        "domain": case.domain, "predicted": best, "answer": case.answer,
        "correct": best == case.answer,
        "confidence": round(probs[best], 4),
        "question_truncated": truncated, "ms": round(ms, 3),
    }


def retrieve(store: KnowledgeStore, question: str) -> tuple[str, float]:
    """Blind top-2 hybrid retrieval; returns (context, ms). Mirrors the
    serving RAG contract (no book filter — retrieval must find evidence,
    not assume the case's own book)."""
    t0 = time.perf_counter()
    hits = store.search(question, k=RAG_K, mode=RAG_MODE)
    ms = (time.perf_counter() - t0) * 1000.0
    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(f"[{i}] «{hit.title_ar}»: {hit.text}")
    return ("\n".join(parts), ms)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def _aggregate(rows: list[dict], abstain_rows: list[dict],
               mode: str, retrieval_ms: float | None = None) -> dict:
    """Aggregate per-case rows into a mode report (shared with merge)."""
    def by_family(rs: list[dict], key: str) -> dict:
        out: dict[str, dict] = {}
        for row in rs:
            b = out.setdefault(row["family"], {"n": 0, "hits": 0})
            b["n"] += 1
            b["hits"] += 1 if row[key] else 0
        return {f: {"n": v["n"], "rate": round(v["hits"] / v["n"], 4)}
                for f, v in sorted(out.items())}

    scored_ms = [r["ms"] for r in rows + abstain_rows]
    report: dict = {
        "mode": mode,
        "mcq": {
            "n": len(rows),
            "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4)
            if rows else None,
            "per_family": by_family(rows, "correct"),
            "truncated_prompts": sum(1 for r in rows
                                     if r["question_truncated"]),
        },
        "abstain": {
            "n": len(abstain_rows),
            "correct_abstain_rate": round(
                sum(r["correct_abstain"] for r in abstain_rows)
                / max(1, sum(1 for r in abstain_rows
                             if not r["should_answer"])), 4),
            "wrong_abstain_rate": round(
                sum(r["wrong_abstain"] for r in abstain_rows)
                / max(1, sum(1 for r in abstain_rows
                             if r["should_answer"])), 4),
            "per_family": by_family(abstain_rows, "correct_abstain"),
        },
        "latency_ms": {
            "mean": round(sum(scored_ms) / len(scored_ms), 2)
            if scored_ms else None,
            "p50": round(_percentile(scored_ms, 0.50), 2),
            "p95": round(_percentile(scored_ms, 0.95), 2),
        },
    }
    if retrieval_ms is not None:
        report["latency_ms"]["retrieval_mean"] = retrieval_ms
    return report


def run_mode(model, tok, store: KnowledgeStore | None, cases: list[dict],
             mode: str, device: str = "cpu") -> dict:
    """Score mcq+abstain cases in one prompt mode; emits per-case rows."""
    bench = to_bench_cases(cases)
    mcq_cases = [c for c in bench if c.kind == "mcq"]
    abstain_cases = [c for c in bench if c.kind == "abstain"]

    rows: list[dict] = []
    for case in mcq_cases:
        ctx, _ = retrieve(store, case.question) if (
            mode == "rag" and store is not None) else ("", 0.0)
        rows.append(_score_case(model, tok, case, device, context=ctx))
    abstain_rows: list[dict] = []
    for case in abstain_cases:
        ctx, _ = retrieve(store, case.question) if (
            mode == "rag" and store is not None) else ("", 0.0)
        row = _score_case(model, tok, case, device, context=ctx)
        row["should_answer"] = case.should_answer
        row["abstained"] = row["confidence"] < DEFAULT_ABSTAIN_THRESHOLD
        row["correct_abstain"] = (not case.should_answer) and row["abstained"]
        row["wrong_abstain"] = case.should_answer and row["abstained"]
        abstain_rows.append(row)

    retrieval_ms = None
    if mode == "rag" and store is not None:
        # retrieval cost measured on a deterministic sample (every 10th case)
        sample = [rec for rec in cases][::10]
        t0 = time.perf_counter()
        for rec in sample:
            store.search(rec.get("question", ""), k=RAG_K, mode=RAG_MODE)
        retrieval_ms = round(
            (time.perf_counter() - t0) * 1000.0 / max(1, len(sample)), 2)
    report = _aggregate(rows, abstain_rows, mode, retrieval_ms)
    report["rows"] = rows
    report["abstain_rows"] = abstain_rows
    return report


# --------------------------------------------------- materialized probes
def _quote_head(text: str, n: int = QUOTE_HEAD) -> str:
    t = " ".join(text.split())
    cut = t[:n]
    if len(t) > n and " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut


def _probe_quote(text: str, tries: int = 4,
                 min_tokens: int = 3) -> str | None:
    """First deterministic window of ``text`` usable as a probe quote.

    Any window cut from the whitespace-normalized passage is verbatim-
    contained in it BY CONSTRUCTION (character-level normalization preserves
    contiguous cuts), so the control half of the wrong_attribution probe
    needs no search call at all.  The window must carry a real Arabic token
    stream (≥ ``min_tokens``) — numeral-table passages (e.g. Ibn Khaldun's
    tables) yield no usable window and are reported honestly as not
    materializable.
    """
    t = " ".join(text.split())
    for k in range(tries):
        start = k * QUOTE_HEAD
        if start >= len(t):
            break
        cut = t[start:start + QUOTE_HEAD]
        if len(t) > start + QUOTE_HEAD and " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        if len(cut) < 40:
            break
        if len(list(ar_tokens(cut))) >= min_tokens:
            return cut
    return None


def materialize_probes(store: KnowledgeStore,
                       cases: list[dict]) -> dict:
    """Execute every failure probe as a deterministic engine check.

    * ``splice``            — cross-book splice of the case's own quote with
      the next same-family quote must return ``not_found``.
    * ``wrong_attribution`` — the case's passage quote verified against its
      own book (control: ``verified``) then scoped to the first WRONG book
      must NEVER verify (engine rejects the false attribution).
    * ``evidenceless``      — absence-token questions must be PROVEN absent
      from every committed passage (gate-6 semantics); term cases must show
      a foreign anchored term is NOT in the case's source passage.
    Probes without the anchors needed to materialize are reported as
    ``not_materializable`` — never counted as compliant, never silent.
    """
    books = sorted(b["book_id"] for b in store.iter_books())
    cite_quotes: dict[str, list[str]] = {}
    terms: dict[str, list[str]] = {}
    for rec in cases:
        if rec.get("quote"):
            cite_quotes.setdefault(rec["family"], []).append(rec["quote"])
        term = (rec.get("meta") or {}).get("term")
        if term:
            terms.setdefault(rec["family"], []).append(term)

    # corpus token sets for absence proofs (gate-6 semantics, built once)
    token_sets: list[set[str]] | None = None

    def _absent_from_corpus(tokens: set[str]) -> bool:
        nonlocal token_sets
        if token_sets is None:
            token_sets = [set(ar_tokens(p["text"]))
                          for p in store.iter_passages()]
        return not any(tokens and tokens <= ts for ts in token_sets)

    results: list[dict] = []
    for rec in cases:
        family = rec["family"]
        for probe in rec.get("failure_probes") or []:
            kind = probe.get("kind")
            must = probe.get("must")
            row = {"case_id": rec["case_id"], "family": family,
                   "kind": kind, "must": must, "ok": None, "detail": ""}
            if kind not in PROBE_KINDS:
                row["detail"] = f"unknown probe kind {kind!r}"
                results.append(row)
                continue
            if kind == "splice":
                quote = rec.get("quote") or ""
                pool = [q for q in cite_quotes.get(family, [])
                        if q != quote]
                if not quote or not pool:
                    row["detail"] = "no donor quote for splice"
                    results.append(row)
                    continue
                donor = pool[(pool.index(quote) + 1) % len(pool)] \
                    if quote in pool else pool[0]
                splice = (quote[:SPLICE_HEAD] + " "
                          + donor[-SPLICE_TAIL:])
                verdict = verify_quote(store, splice, write_status=False)
                row["ok"] = verdict.status == "not_found"
                row["detail"] = (f"splice verdict {verdict.status!r} "
                                 f"(must {must!r})")
            elif kind == "wrong_attribution":
                pid = None
                src = (rec.get("sources") or [{}])[0]
                if src.get("passage_id"):
                    pid = src["passage_id"]
                elif rec.get("supporting_text"):
                    pid = rec["supporting_text"][0].get("passage_id")
                else:
                    mpids = (rec.get("meta") or {}).get("passage_ids")
                    if mpids:
                        pid = mpids[0]
                if not pid:
                    row["detail"] = ("no passage anchor — "
                                     "not_materializable")
                    results.append(row)
                    continue
                passage = store.passage(pid)
                if not passage:
                    row["detail"] = f"passage {pid} missing"
                    results.append(row)
                    continue
                quote = _probe_quote(passage["text"])
                if quote is None:
                    row["detail"] = ("passage has no tokenizable "
                                     "text window — not_materializable")
                    results.append(row)
                    continue
                # control is verbatim-containment BY CONSTRUCTION (the
                # window is cut from the passage itself) — the probe is
                # the scoped search: the quote must NEVER verify against
                # a book that is not its source
                own = set((rec.get("meta") or {}).get("book_ids")
                          or [src.get("book_id")])
                wrong = next((b for b in books if b not in own), None)
                scoped = verify_quote(store, quote, book_id=wrong,
                                      write_status=False, candidates=100)
                row["ok"] = scoped.status != "verified"
                row["detail"] = (f"verbatim-by-construction, "
                                 f"wrong-book({wrong}) "
                                 f"{scoped.status!r}")
            elif kind == "evidenceless":
                meta = rec.get("meta") or {}
                if meta.get("absence_token"):
                    tokens = set(ar_tokens(meta["absence_token"]))
                    absent = _absent_from_corpus(tokens)
                    row["ok"] = absent
                    row["detail"] = ("absence token "
                                     f"{meta['absence_token']!r} "
                                     + ("absent from corpus" if absent
                                        else "OCCURS in corpus"))
                elif meta.get("term"):
                    terms_pool = [t for t in terms.get(family, [])
                                  if t != meta["term"]]
                    if not terms_pool:
                        row["detail"] = "no foreign term available"
                        results.append(row)
                        continue
                    foreign = terms_pool[0]  # deterministic first foreign
                    pid = (rec.get("sources") or [{}])[0].get(
                        "passage_id") or (
                        rec.get("supporting_text") or [{}])[0].get(
                        "passage_id")
                    if not pid:
                        row["detail"] = ("no passage anchor — "
                                         "not_materializable")
                        results.append(row)
                        continue
                    passage = store.passage(pid)
                    ptokens = set(ar_tokens(passage["text"]))
                    foreign_tokens = set(ar_tokens(foreign))
                    row["ok"] = not foreign_tokens <= ptokens
                    row["detail"] = (f"foreign term {foreign!r} "
                                     + ("absent from the case passage"
                                        if row["ok"]
                                        else "PRESENT in passage"))
            results.append(row)

    materialized = [r for r in results if r["ok"] is not None]
    by_kind: dict[str, dict] = {}
    for kind in PROBE_KINDS:
        rows_k = [r for r in materialized if r["kind"] == kind]
        by_kind[kind] = {
            "n": len(rows_k),
            "compliant": sum(1 for r in rows_k if r["ok"]),
            "rate": round(sum(1 for r in rows_k if r["ok"]) / len(rows_k), 4)
            if rows_k else None,
        }
    return {
        "probes_total": len(results),
        "materialized": len(materialized),
        "not_materializable": len(results) - len(materialized),
        "by_kind": by_kind,
        "compliance": round(
            sum(1 for r in materialized if r["ok"]) / len(materialized), 4)
        if materialized else None,
        "failures": [r for r in materialized if not r["ok"]][:20],
        "unmaterialized_sample": [r for r in results if r["ok"] is None][:10],
    }


# ------------------------------------------------------------- cite audit
def cite_alignment(store: KnowledgeStore, cases: list[dict]) -> dict:
    rows = []
    for rec in cases:
        if rec["kind"] != "cite":
            continue
        verdict = verify_quote(store, rec["quote"], write_status=False,
                               candidates=100)
        rows.append({
            "case_id": rec["case_id"], "family": rec["family"],
            "verdict": verdict.status, "expected": rec["expect"],
            "match": verdict.status == rec["expect"],
        })
    per_family: dict[str, dict] = {}
    for r in rows:
        b = per_family.setdefault(r["family"], {"n": 0, "hits": 0})
        b["n"] += 1
        b["hits"] += 1 if r["match"] else 0
    return {
        "n": len(rows),
        "match_rate": round(sum(r["match"] for r in rows) / len(rows), 4)
        if rows else None,
        "per_family": {f: {"n": v["n"],
                           "rate": round(v["hits"] / v["n"], 4)}
                       for f, v in sorted(per_family.items())},
    }


# ------------------------------------------------------------------ report
def run_eval(checkpoint: Path, tokenizer_path: Path, store_path: Path,
             seed_path: Path, modes: list[str], device: str = "cpu",
             limit: int | None = None, config: Path | None = None,
             case_slice: tuple[int, int] | None = None,
             skip_engine: bool = False) -> dict:
    """Full evaluation; deterministic JSON (no wall-clock timestamps).

    ``case_slice`` (start, end) chunks the scored worklist (mcq+abstain,
    seed order) so long runs can be executed call-by-call and merged via
    :func:`merge_eval_parts`; ``skip_engine`` drops the engine-only
    sections (cite/probes) from chunk runs — they are produced once by a
    dedicated ``engine`` mode pass over the FULL seed.
    """
    model = None
    tok = None
    if set(modes) != {"engine"}:
        from ryko_lm.model.ryko_model import RykoLM
        from ryko_lm.tokenizer.loader import RykoTokenizer

        model = RykoLM.load(checkpoint, device=device,
                            config=None if config is None else str(config))
        model.eval()
        tok = RykoTokenizer.from_file(tokenizer_path)
    store = KnowledgeStore(str(store_path), persist_vectors=False)
    cases = load_ryko1000(seed_path)
    if limit:
        by_fam: dict[str, list[dict]] = {}
        for rec in cases:
            by_fam.setdefault(rec["family"], []).append(rec)
        cases = [rec for fam in sorted(by_fam)
                 for rec in by_fam[fam][:limit]]

    report: dict = {
        "schema": SCHEMA,
        "model": {
            "checkpoint": str(checkpoint),
            "params_M": (round(sum(p.numel() for p in model.parameters())
                               / 1e6, 2) if model is not None else None),
            "block_size": (getattr(model.cfg, "block_size", None)
                           if model is not None else None),
        },
        "seed": {"path": str(seed_path),
                 "sha256": seed_sha256(seed_path),
                 "cases": len(cases)},
        "abstain_threshold": DEFAULT_ABSTAIN_THRESHOLD,
    }
    scored = [rec for rec in cases if rec["kind"] in {"mcq", "abstain"}]
    if case_slice is not None:
        scored = scored[case_slice[0]:case_slice[1]]
    if set(modes) == {"engine"}:
        # engine-only pass: citation audit + materialized probes over the
        # full seed — no model scoring, no checkpoint needed
        report["cite_alignment"] = cite_alignment(store, cases)
        report["probes"] = materialize_probes(store, cases)
        report["engine_scope"] = "full-seed"
        store.close()
        return report
    for mode in modes:
        if mode not in {"base", "rag"}:
            raise VerificationError(f"unknown mode {mode!r}")
        report[mode] = run_mode(model, tok, store, scored, mode, device)
    if not skip_engine:
        report["cite_alignment"] = cite_alignment(store, cases)
        report["probes"] = materialize_probes(store, cases)
        report["engine_scope"] = "full-seed"
    store.close()
    return report


def merge_eval_parts(parts: list[Path]) -> dict:
    """Merge sliced chunk reports (+ one engine report) into the final one.

    Engine sections (cite/probes) come from the dedicated full-seed engine
    pass; mode sections are re-aggregated from the chunks' per-case rows —
    identical arithmetic to :func:`_aggregate`, applied once to the union.
    """
    merged: dict | None = None
    mode_rows: dict[str, list[dict]] = {}
    mode_abstain: dict[str, list[dict]] = {}
    for path in parts:
        rep = json.loads(Path(path).read_text(encoding="utf-8"))
        if merged is None:
            merged = rep
        else:
            assert merged["seed"] == rep["seed"], "seed mismatch"
            # the engine pass carries a placeholder model section — the
            # merged artifact keeps the FIRST real model section
            if rep.get("model", {}).get("params_M") is not None:
                if merged.get("model", {}).get("params_M") is None:
                    merged["model"] = rep["model"]
                else:
                    assert merged["model"] == rep["model"], "model mismatch"
        for mode in ("base", "rag"):
            if mode in rep:
                mode_rows.setdefault(mode, []).extend(
                    rep[mode].get("rows") or [])
                mode_abstain.setdefault(mode, []).extend(
                    rep[mode].get("abstain_rows") or [])
        if "cite_alignment" in rep:
            merged["cite_alignment"] = rep["cite_alignment"]
            merged["probes"] = rep["probes"]
            merged["engine_scope"] = rep.get("engine_scope", "full-seed")
    assert merged is not None
    out = {k: merged[k] for k in ("schema", "model", "seed",
                                  "abstain_threshold")}
    for k in ("cite_alignment", "probes", "engine_scope"):
        if k in merged:
            out[k] = merged[k]
    for mode in ("base", "rag"):
        if mode not in mode_rows:
            continue
        agg = _aggregate(mode_rows[mode], mode_abstain[mode], mode,
                         merged.get(mode, {}).get("latency_ms", {})
                         .get("retrieval_mean"))
        # keep per-case rows in the merged artifact: provenance for every
        # measured number (case_id → what the model actually did)
        agg["rows"] = mode_rows[mode]
        agg["abstain_rows"] = mode_abstain[mode]
        out[mode] = agg
    return out


def write_eval_report(report: dict, out_base: str | Path) -> tuple[Path, Path]:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base.with_suffix(".json")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    lines = ["# قياس RYKO-1000 — نتائج", "",
             f"- النموذج: `{report['model']['checkpoint']}` "
             f"({report['model']['params_M']}M معلم، "
             f"نافذة {report['model']['block_size']})",
             f"- المجموعة: {report['seed']['cases']} حالة "
             f"(sha256 المجموعة: `{report['seed']['sha256'][:16]}…`)",
             f"- عتبة الامتناع: {report['abstain_threshold']}", ""]
    if "cite_alignment" in report:
        lines += ["## محاذاة تدقيق الاقتباس (محرك مستقل)",
                  f"- تطابق الحكم المتوقع: "
                  f"**{report['cite_alignment']['match_rate']}** على "
                  f"{report['cite_alignment']['n']} حالة", ""]
    probes = report.get("probes")
    if probes is not None:
        lines += ["## المسبارات المتعمدة المُجسَّدة (محرك)",
                  f"- المجسَّد: {probes['materialized']} / "
                  f"{probes['probes_total']} "
                  f"(غير قابل للتجسيد: {probes['not_materializable']} — "
                  "يُحصى بصدق ولا يُعدّ منجماً)",
                  f"- الالتزام الكلي: **{probes['compliance']}**", ""]
        lines += ["| المسبار | n | ملتزم | الالتزام |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['n']} | {v['compliant']} | {v['rate']} |"
                  for k, v in probes["by_kind"].items()]
    for mode in ("base", "rag"):
        if mode not in report:
            continue
        m = report[mode]
        lines += ["", f"## وضع {mode}", "",
                  f"- دقة MCQ: **{m['mcq']['accuracy']}** على "
                  f"{m['mcq']['n']} (مقاطع مقتطعة: "
                  f"{m['mcq']['truncated_prompts']})",
                  f"- امتناع صائب: {m['abstain']['correct_abstain_rate']}، "
                  f"امتناع خاطئ (رفض زائد): "
                  f"{m['abstain']['wrong_abstain_rate']}",
                  f"- زمن الاستجابة p50/p95: {m['latency_ms']['p50']}/"
                  f"{m['latency_ms']['p95']} ms"]
        if "retrieval_mean" in m["latency_ms"]:
            lines.append(f"- زمن الاسترجاع الوسطي: "
                         f"{m['latency_ms']['retrieval_mean']} ms")
        lines += ["", "| العائلة | n | الدقة |", "|---|---|---|"]
        lines += [f"| {f} | {v['n']} | {v['rate']} |"
                  for f, v in m["mcq"]["per_family"].items()]
    md_path = out_base.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ryko1000_eval",
        description="RYKO-1000 runner — base/rag/engine modes + merge.")
    parser.add_argument("--seed", type=Path,
                        default=Path("docs/benchmarks/ryko1000_seed.jsonl"))
    parser.add_argument("--store", type=Path,
                        default=Path("data/knowledge/ryko.sqlite"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--mode", default="both",
                        choices=["base", "rag", "both", "engine", "merge"])
    parser.add_argument("--parts", default=None,
                        help="glob of chunk JSONs (merge mode)")
    parser.add_argument("--slice", default=None,
                        help="scored-worklist slice A:B (chunked runs)")
    parser.add_argument("--skip-engine", action="store_true",
                        help="drop engine sections (chunked model runs)")
    parser.add_argument("--out", type=Path,
                        default=Path("docs/results/ryko1000_eval"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N cases per family (smoke runs)")
    args = parser.parse_args(argv)

    if args.mode == "merge":
        import glob as _glob
        parts = [Path(p) for p in sorted(_glob.glob(args.parts or ""))]
        if not parts:
            raise SystemExit("merge needs --parts glob matching files")
        report = merge_eval_parts(parts)
    else:
        if args.checkpoint is None or args.tokenizer is None:
            if args.mode != "engine":
                raise SystemExit("--checkpoint/--tokenizer required for "
                                 "model modes")
        if args.checkpoint is None or args.tokenizer is None:
            # engine pass: satisfy the loader with placeholder paths
            args.checkpoint = args.checkpoint or Path("-")
            args.tokenizer = args.tokenizer or Path("-")
        modes = (["base", "rag"] if args.mode == "both"
                 else [args.mode] if args.mode != "merge" else [])
        case_slice = None
        if args.slice:
            a, b = args.slice.split(":")
            case_slice = (int(a), int(b))
        report = run_eval(args.checkpoint, args.tokenizer, args.store,
                          args.seed, modes, device=args.device,
                          limit=args.limit, config=args.config,
                          case_slice=case_slice,
                          skip_engine=args.skip_engine)
    json_path, md_path = write_eval_report(report, args.out)
    print(f"[ryko1000-eval] -> {json_path}")
    print(f"[ryko1000-eval] report -> {md_path}")
    summary = {}
    if "cite_alignment" in report:
        summary["cite_match"] = report["cite_alignment"]["match_rate"]
    if "probes" in report:
        summary["probe_compliance"] = report["probes"]["compliance"]
    for mode in ("base", "rag"):
        if mode in report:
            summary[f"{mode}_mcq"] = report[mode]["mcq"]["accuracy"]
            summary[f"{mode}_wrong_abstain"] = \
                report[mode]["abstain"]["wrong_abstain_rate"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
