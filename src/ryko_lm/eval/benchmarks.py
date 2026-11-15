"""RYKO scientific evaluation gates — benchmark cases, runner, reports.

Closes ROADMAP phase 7.  Three closed case kinds (explicit schema, like the
receipt contract — anything else is a load error, not a silent fallback):

* ``mcq``    — multiple choice scored by length-normalized log-likelihood of
  each option conditioned on the question (deterministic, no sampling).
* ``abstain``— MCQ-shaped cases with ``should_answer``; the runner abstains
  when the best option confidence falls under ``--abstain-threshold``.
  Reported separately: correct-abstain vs wrong-abstain.
* ``cite``   — quote verification against the Knowledge Core
  (``ryko_lm.knowledge.citation``); reported as verified/partial/not_found.

Honesty rules (research doc §evaluation):

* every capability reports **separately** — no single aggregate number;
* a case must carry ``source`` and ``domain`` for slicing;
* RYKO-Bench seed cases are auto-generated from the committed Knowledge Core
  and self-checked at build time; they are a *seed*, not a headline metric,
  until human-reviewed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

SCHEMA = "ryko-bench/1"
CASE_KINDS = ("mcq", "abstain", "cite")


class BenchmarkError(ValueError):
    """Raised for any schema violation — closed contract, no fallbacks."""


@dataclass(frozen=True)
class Case:
    case_id: str
    kind: str
    source: str
    domain: str
    question: str = ""
    options: tuple[str, ...] = ()
    answer: int | None = None
    should_answer: bool | None = None
    quote: str = ""
    expect: str | None = None
    meta: dict = field(default_factory=dict)


def load_benchmark(path: str | Path) -> list[Case]:
    """Load and validate a benchmark JSONL — closed schema, deterministic."""
    cases: list[Case] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"line {line_no}: invalid JSON: {exc}") from exc
        kind = rec.get("kind")
        if kind not in CASE_KINDS:
            raise BenchmarkError(f"line {line_no}: kind {kind!r} not in {CASE_KINDS}")
        cid = rec.get("case_id")
        if not cid or cid in seen:
            raise BenchmarkError(f"line {line_no}: missing or duplicate case_id {cid!r}")
        seen.add(cid)
        if not rec.get("source") or not rec.get("domain"):
            raise BenchmarkError(f"line {line_no}: source/domain required ({cid})")
        if kind in {"mcq", "abstain"}:
            opts = rec.get("options")
            if not isinstance(opts, list) or len(opts) < 2:
                raise BenchmarkError(f"{cid}: options must be a list of >= 2")
            if any(not isinstance(o, str) or not o.strip() for o in opts):
                raise BenchmarkError(f"{cid}: empty option text")
            if not rec.get("question"):
                raise BenchmarkError(f"{cid}: question required for {kind}")
        if kind == "mcq":
            ans = rec.get("answer")
            if not isinstance(ans, int) or not 0 <= ans < len(rec["options"]):
                raise BenchmarkError(f"{cid}: answer index out of range")
        if kind == "abstain" and not isinstance(rec.get("should_answer"), bool):
            raise BenchmarkError(f"{cid}: should_answer (bool) required for abstain")
        if kind == "cite":
            if not rec.get("quote"):
                raise BenchmarkError(f"{cid}: quote required for cite")
            if rec.get("expect") not in {"verified", "partial", "not_found"}:
                raise BenchmarkError(f"{cid}: expect must be a citation verdict")
        cases.append(Case(
            case_id=cid, kind=kind, source=rec["source"], domain=rec["domain"],
            question=rec.get("question", ""),
            options=tuple(rec.get("options", ())),
            answer=rec.get("answer"),
            should_answer=rec.get("should_answer"),
            quote=rec.get("quote", ""),
            expect=rec.get("expect"),
            meta=rec.get("meta", {}),
        ))
    if not cases:
        raise BenchmarkError(f"{path}: no cases")
    return cases


# --------------------------------------------------------------- MCQ scoring
def _score_prompt_ids(tok, question: str, option: str,
                      max_len: int | None = None) -> tuple[list[int], list[int], bool]:
    """Deterministic prompt construction honouring the model context.

    Without ``max_len`` this is the historical behaviour exactly. With a
    context budget, an over-long question is head-truncated at the character
    level (then, as a last resort, the prompt token list is hard-cut) so
    ``prompt + option`` always fits — deterministic, and the truncation is
    reported to the caller instead of being silently swallowed.
    """
    option_ids = tok.encode(" " + option)
    prompt_ids = tok.encode(f"السؤال: {question}\nالجواب:", add_bos=True)
    if max_len is None or len(prompt_ids) + len(option_ids) <= max_len:
        return prompt_ids, option_ids, False
    q_ids = tok.encode(question)
    overhead = len(prompt_ids) - len(q_ids)
    budget = max(8, max_len - len(option_ids) - 1 - overhead)
    cut = max(1, int(len(question) * budget / max(1, len(q_ids))))
    prompt_ids = tok.encode(f"السؤال: {question[:cut]}\nالجواب:", add_bos=True)
    if len(prompt_ids) + len(option_ids) > max_len:
        prompt_ids = prompt_ids[:max(1, max_len - len(option_ids))]
    return prompt_ids, option_ids, True


@torch.no_grad()
def score_option(model, tok, question: str, option: str, device: str = "cpu",
                 max_len: int | None = None) -> tuple[float, bool]:
    """Length-normalized mean log-likelihood of ``option`` after ``question``.

    Deterministic teacher-forcing pass; the prompt is
    ``السؤال: {question}\\nالجواب: {option}`` and only the option's token
    positions are scored. When ``max_len`` (the model context) is given,
    over-long questions are deterministically truncated (see
    ``_score_prompt_ids``) and the truncation is reported with the score.
    """
    prompt_ids, option_ids, truncated = _score_prompt_ids(tok, question, option, max_len)
    if not option_ids:
        return -math.inf, truncated
    ids = torch.tensor([prompt_ids + option_ids], dtype=torch.long, device=device)
    logits, _ = model(idx=ids[:, :-1])
    n_prompt = len(prompt_ids)
    logprobs = torch.log_softmax(logits[0].float(), dim=-1)
    target = ids[0, 1:]
    picked = logprobs.gather(1, target.unsqueeze(1)).squeeze(1)
    option_logprob = picked[n_prompt - 1:].sum().item()
    return option_logprob / len(option_ids), truncated


def run_mcq(model, tok, cases: list[Case], device: str = "cpu") -> list[dict]:
    max_len = getattr(getattr(model, "cfg", None), "block_size", None)
    rows = []
    for case in cases:
        scores = []
        truncated = False
        for o in case.options:
            s, t = score_option(model, tok, case.question, o, device, max_len=max_len)
            scores.append(s)
            truncated = truncated or t
        best = max(range(len(scores)), key=lambda i: (scores[i], -i))
        probs = _softmax(scores)
        rows.append({
            "case_id": case.case_id, "domain": case.domain, "source": case.source,
            "predicted": best, "answer": case.answer,
            "correct": best == case.answer,
            "confidence": round(probs[best], 4),
            "scores": [round(s, 4) for s in scores],
            "question_truncated": truncated,
        })
    return rows


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def run_abstain(model, tok, cases: list[Case], device: str = "cpu",
                threshold: float = 0.34) -> list[dict]:
    """Same scoring as MCQ; the model abstains under the confidence gate."""
    rows = run_mcq(model, tok, cases, device)
    by_id = {c.case_id: c for c in cases}
    for row in rows:
        case = by_id[row["case_id"]]
        row["should_answer"] = case.should_answer
        row["abstained"] = row["confidence"] < threshold
        row["correct_abstain"] = (not case.should_answer) and row["abstained"]
        row["wrong_abstain"] = case.should_answer and row["abstained"]
    return rows


def run_cite(store, cases: list[Case]) -> list[dict]:
    from ryko_lm.knowledge.citation import verify_quote

    rows = []
    for case in cases:
        verdict = verify_quote(store, case.quote)
        rows.append({
            "case_id": case.case_id, "domain": case.domain, "source": case.source,
            "verdict": verdict.status, "expected": case.expect,
            "match": verdict.status == case.expect,
            "score": round(verdict.score, 4),
        })
    return rows


# -------------------------------------------------------------------- report
def summarize(mcq_rows: list[dict], abstain_rows: list[dict],
              cite_rows: list[dict]) -> dict:
    """Per-capability breakdown — never one aggregate number."""
    report: dict = {"schema": SCHEMA}

    def by_domain(rows: list[dict], key: str) -> dict:
        out: dict[str, dict] = {}
        for row in rows:
            bucket = out.setdefault(row["domain"], {"n": 0, "hits": 0})
            bucket["n"] += 1
            bucket["hits"] += 1 if row[key] else 0
        return {d: {"n": v["n"], "rate": round(v["hits"] / v["n"], 4) if v["n"] else None}
                for d, v in sorted(out.items())}

    if mcq_rows:
        report["mcq"] = {
            "n": len(mcq_rows),
            "accuracy": round(sum(r["correct"] for r in mcq_rows) / len(mcq_rows), 4),
            "per_domain": by_domain(mcq_rows, "correct"),
        }
    if abstain_rows:
        n_answer = sum(1 for r in abstain_rows if r["should_answer"])
        n_shy = len(abstain_rows) - n_answer
        report["abstain"] = {
            "n": len(abstain_rows),
            "correct_abstain_rate": round(
                sum(r["correct_abstain"] for r in abstain_rows) / n_shy, 4) if n_shy else None,
            "wrong_abstain_rate": round(
                sum(r["wrong_abstain"] for r in abstain_rows) / n_answer, 4) if n_answer else None,
            "per_domain": {},
        }
        for row in abstain_rows:
            bucket = report["abstain"]["per_domain"].setdefault(
                row["domain"], {"n": 0, "correct_abstain": 0, "wrong_abstain": 0})
            bucket["n"] += 1
            bucket["correct_abstain"] += 1 if row["correct_abstain"] else 0
            bucket["wrong_abstain"] += 1 if row["wrong_abstain"] else 0
    if cite_rows:
        verdicts = {}
        for r in cite_rows:
            verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        report["cite"] = {
            "n": len(cite_rows),
            "match_rate": round(sum(r["match"] for r in cite_rows) / len(cite_rows), 4),
            "verdict_counts": dict(sorted(verdicts.items())),
            "per_domain": {},
        }
        for row in cite_rows:
            bucket = report["cite"]["per_domain"].setdefault(
                row["domain"], {"n": 0, "hits": 0})
            bucket["n"] += 1
            bucket["hits"] += 1 if row["match"] else 0
        report["cite"]["per_domain"] = {
            d: {"n": v["n"], "rate": round(v["hits"] / v["n"], 4)}
            for d, v in sorted(report["cite"]["per_domain"].items())}
    return report


def write_report(report: dict, model_meta: dict, out_base: str | Path) -> tuple[Path, Path]:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_meta, "results": report}
    json_path = out_base.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# بوابات التقييم العلمي — نتائج", "",
             f"- النموذج: `{model_meta.get('checkpoint', '?')}` "
             f"({model_meta.get('params_M', '?')}M معلم)",
             f"- المعيار: `{report.get('schema', SCHEMA)}`", ""]
    if "mcq" in report:
        lines += [f"## اختيار متعدد (دقة: {report['mcq']['accuracy']:.1%} على {report['mcq']['n']})", ""]
        lines += ["| المجال | n | الدقة |", "|---|---|---|"]
        lines += [f"| {d} | {v['n']} | {v['rate']:.1%} |"
                  for d, v in report["mcq"]["per_domain"].items() if v["rate"] is not None]
        lines.append("")
    if "abstain" in report:
        lines += ["## الامتناع", "",
                  f"- امتناع صائب (يستحسن): {report['abstain']['correct_abstain_rate']}",
                  f"- امتناع خاطئ (فوّت إجابة معروفة): {report['abstain']['wrong_abstain_rate']}", ""]
        dom = report["abstain"].get("per_domain") or {}
        if dom:
            lines += ["| مجال الامتناع | n | امتناع صائب | امتناع خاطئ |", "|---|---|---|---|"]
            lines += [f"| {d} | {v['n']} | {v['correct_abstain']} | {v['wrong_abstain']} |"
                      for d, v in dom.items()]
            lines.append("")
    if "cite" in report:
        lines += ["## تدقيق الاقتباس", "",
                  f"- تطابق الحكم المتوقع: {report['cite']['match_rate']:.1%}",
                  f"- توزيع الأحكام: {report['cite']['verdict_counts']}", ""]
        cdom = report["cite"].get("per_domain") or {}
        if cdom:
            lines += ["| مجال الاقتباس | n | التطابق |", "|---|---|---|"]
            lines += [f"| {d} | {v['n']} | {v['rate']:.1%} |"
                      for d, v in cdom.items() if v["rate"] is not None]
            lines.append("")
    md_path = out_base.with_suffix(".md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ryko_bench", description="RYKO benchmark gates — mcq / abstain / cite.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run benchmark cases against a checkpoint")
    p_run.add_argument("--benchmark", type=Path, action="append", required=True)
    p_run.add_argument("--checkpoint", type=Path, required=True)
    p_run.add_argument("--tokenizer", type=Path, required=True)
    p_run.add_argument("--config", type=Path, default=None)
    p_run.add_argument("--store", type=Path, default=None,
                       help="knowledge store for cite cases")
    p_run.add_argument("--abstain-threshold", type=float, default=0.34)
    p_run.add_argument("--out", type=Path, default=Path("docs/results/benchmarks"))
    p_run.add_argument("--device", default="cpu")

    args = parser.parse_args(argv)

    from ..model.ryko_model import RykoLM
    from ..tokenizer.loader import RykoTokenizer

    cases: list[Case] = []
    for path in args.benchmark:
        cases.extend(load_benchmark(path))

    model = RykoLM.load(args.checkpoint, device=args.device, config=args.config)
    model.eval()
    tok = RykoTokenizer.from_file(args.tokenizer)

    mcq = [c for c in cases if c.kind == "mcq"]
    abstain = [c for c in cases if c.kind == "abstain"]
    cite = [c for c in cases if c.kind == "cite"]

    mcq_rows = run_mcq(model, tok, mcq, args.device) if mcq else []
    abstain_rows = (run_abstain(model, tok, abstain, args.device, args.abstain_threshold)
                    if abstain else [])
    cite_rows = []
    if cite:
        if args.store is None:
            raise BenchmarkError("cite cases need --store")
        from ..knowledge.store import KnowledgeStore

        store = KnowledgeStore(str(args.store))
        cite_rows = run_cite(store, cite)
        store.close()

    report = summarize(mcq_rows, abstain_rows, cite_rows)
    model_meta = {
        "checkpoint": str(args.checkpoint),
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "abstain_threshold": args.abstain_threshold,
        "cases": len(cases),
    }
    json_path, md_path = write_report(report, model_meta, args.out)
    print(f"[bench] cases={len(cases)} -> {json_path}")
    print(f"[bench] report -> {md_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
