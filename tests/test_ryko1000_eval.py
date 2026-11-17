"""RYKO-1000 evaluation runner tests — probe materialization, base/rag
modes, determinism, and the pinned ryko-small parameter contract (52.2M)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import TOKENIZER_JSON, requires_tokenizer

from ryko_lm.eval.ryko1000_eval import materialize_probes, run_eval, \
    to_bench_cases
from ryko_lm.eval.verify_ryko1000 import load_ryko1000
from ryko_lm.knowledge.citation import PROTOCOL as CITE_PROTOCOL
from ryko_lm.knowledge.citation import verify_quote
from ryko_lm.knowledge.store import KnowledgeStore
from ryko_lm.model.ryko_model import RykoLM, get_preset

# distinct lexical signatures so every passage anchors its own quote;
# TAILS end with 4 words unique to the passage so cross-book splices can
# never pass even the 0.82 token-coverage partial path
WORDS = ("الطهارة", "الصلاة", "الزكاة", "الصوم", "الحج", "البيع", "الربا",
         "الوقف", "النكاح", "الطلاق", "الشهادات", "الجهاد", "الأيمان",
         "الوصية", "الميراث", "القضاء", "الأدلة", "العلم", "الفتوى",
         "الاجتهاد", "الخلاف", "المصالح", "البيان", "الإعجاز", "السنن",
         "اللغة", "الصرف", "النحو", "البلاغة", "الأدب", "التاريخ", "الرجال")
FILL = ("وفي هذا الباب بيان المسألة الفقهية على مذهب الجمهور مع ذكر "
        "الأدلة من الكتاب والسنة، والخلاصة أن الأمر فيها واسع عند أهل "
        "العلم والمحققين المعتبرين في الفتوى والاجتهاد")


def _passage_text(b: int, i: int) -> str:
    head = (f"{WORDS[(b * 7 + i) % len(WORDS)]} "
            f"{WORDS[(b * 13 + i) % len(WORDS)]} "
            f"{WORDS[(b * 29 + i) % len(WORDS)]}")
    tail = (f"{WORDS[(b * 31 + i) % len(WORDS)]} "
            f"{WORDS[(b * 37 + i) % len(WORDS)]} "
            f"{WORDS[(b * 41 + i) % len(WORDS)]} "
            f"{WORDS[(b * 43 + i) % len(WORDS)]}")
    return (f"باب {WORDS[(b * 3 + i) % len(WORDS)]} المتضمن أصلا {head} "
            f"وهذا مقتضى الفصل {WORDS[(b * 5 + i) % len(WORDS)]} " + FILL +
            f" وخاتمة المقطع {tail}")


@pytest.fixture(scope="module")
def store_path(tmp_path_factory) -> Path:
    """Synthetic store ON DISK — run_eval opens its own connection."""
    path = tmp_path_factory.mktemp("data") / "store.sqlite"
    s = KnowledgeStore(str(path))
    s.add_source("shamela", "Shamela", "MIT", "declared-open")
    for b in range(2):
        s.add_book(f"bk{b}", "shamela",
                   f"كتاب {WORDS[b]} المبين رقم {b}", category="فقه")
        for i in range(6):
            s.add_passage(f"bk{b}", _passage_text(b, i), page_num=i, seq=i)
    s.close()
    return path


@pytest.fixture()
def store(store_path):
    s = KnowledgeStore(str(store_path), persist_vectors=False)
    yield s
    s.close()


def _src(store: KnowledgeStore, pid: str) -> dict:
    p = store.passage(pid)
    return {"book_id": p["book_id"], "title_ar": p["title_ar"],
            "page_num": p["page_num"], "passage_id": pid,
            "protocol": CITE_PROTOCOL, "quote_sha256": ""}


def _case(store: KnowledgeStore, case_id: str, family: str, kind: str,
          pid: str, **extra) -> dict:
    passage = store.passage(pid)
    quote = " ".join(passage["text"].split())[:80]
    base = {
        "schema": "ryko1000/1", "case_id": case_id, "family": family,
        "kind": kind, "source": "test-seed",
        "domain": "اختبار", "context": "",
        "sources": [_src(store, pid)],
        "supporting_text": [{"passage_id": pid, "quote": quote}],
        "reasoning_steps": ["خطوة اختبار"],
        "objections": [], "failure_probes": extra.get("probes", []),
        "acceptance": {"criterion": extra.get("criterion", "mcq-answer"),
                       "min_citation_score": 1.0},
        "verification": {"status": "passed"},
        "meta": {**(extra.get("meta") or {}), "passage_id": pid},
    }
    if kind == "mcq":
        base["question"] = extra["question"]
        base["options"] = extra["options"]
        base["answer"] = extra["options"].index(extra["correct"])
        base["expected_answer"] = extra["correct"]
        base["quote"] = ""
        base["expect"] = None
        base["should_answer"] = None
    elif kind == "cite":
        base["quote"] = extra["quote"]
        base["expect"] = extra["expect"]
        base["expected_answer"] = extra.get(
            "expected", quote if extra["expect"] == "verified" else "رفض")
        base["question"] = ""
        base["options"] = []
        base["answer"] = None
        base["should_answer"] = None
    else:  # abstain
        base["question"] = extra["question"]
        base["should_answer"] = extra["should_answer"]
        base["options"] = extra.get("options", [])
        base["answer"] = extra.get("answer")
        base["expected_answer"] = extra["expected"]
        base["quote"] = ""
        base["expect"] = None
    return base


@pytest.fixture()
def seed(store, tmp_path) -> Path:
    pids = [p["passage_id"] for p in store.iter_passages()]
    cases = [
        _case(store, "ry10-fahm-001", "fahm", "mcq", pids[0],
              question="أي المتابعات تكمل هذا النص؟",
              options=[f"خيار {WORDS[i]}" for i in (1, 2, 3, 4)],
              correct=f"خيار {WORDS[1]}",
              probes=[{"kind": "wrong_attribution",
                       "probe": "نسبة المقطع إلى كتاب غير مصدره",
                       "must": "reject"}]),
        _case(store, "ry10-fahm-002", "fahm", "mcq", pids[1],
              question="أي المتابعات تكمل هذا النص الثاني؟",
              options=[f"خيار {WORDS[i]}" for i in (5, 6, 7, 8)],
              correct=f"خيار {WORDS[5]}",
              probes=[{"kind": "wrong_attribution",
                       "probe": "نسبة المقطع إلى كتاب غير مصدره",
                       "must": "reject"}]),
        _case(store, "ry10-naql-001", "naql", "cite", pids[2],
              quote=" ".join(store.passage(pids[2])["text"].split())[:80],
              expect="verified",
              probes=[{"kind": "splice",
                       "probe": "اقتباس مركّب من نصين مختلفين",
                       "must": "not_found"}]),
        _case(store, "ry10-naql-002", "naql", "cite", pids[3],
              quote=" ".join(store.passage(pids[3])["text"].split())[:80],
              expect="verified",
              probes=[{"kind": "splice",
                       "probe": "اقتباس مركّب من نصين مختلفين",
                       "must": "not_found"}]),
        _case(store, "ry10-mustalahat-001", "mustalahat", "mcq", pids[4],
              question="ما مدلول المصطلح في هذا النص؟",
              options=["تعريف أ", "تعريف ب", "تعريف ج", "تعريف د"],
              correct="تعريف أ", meta={"term": WORDS[0]},
              probes=[{"kind": "evidenceless",
                       "probe": "نص لا يحوي المصطلح أصلاً",
                       "must": "reject"}]),
        _case(store, "ry10-mustalahat-002", "mustalahat", "mcq", pids[5],
              question="ما مدلول المصطلح في النص الثاني؟",
              options=["تعريف أ", "تعريف ب", "تعريف ج", "تعريف د"],
              correct="تعريف ب", meta={"term": WORDS[1]},
              probes=[{"kind": "evidenceless",
                       "probe": "نص لا يحوي المصطلح أصلاً",
                       "must": "reject"}]),
        _case(store, "ry10-yaqin-001", "yaqin", "abstain", pids[0],
              question="ما رأي العلماء في الكمبيوتر الحديث؟",
              should_answer=False, expected="امتناع لعدم السند",
              options=["لا يجوز عند الجمهور", "يكره عند الشافعية",
                       "يستحب عند الحنابلة", "واجب عند المالكية"],
              meta={"absence_token": "الكمبيوتر"},
              criterion="abstain-must-refuse",
              probes=[{"kind": "evidenceless",
                       "probe": "سؤال بلا سند في النواة",
                       "must": "abstain"}]),
    ]
    path = tmp_path / "seed.jsonl"
    path.write_text("\n".join(json.dumps(c, ensure_ascii=False)
                              for c in cases) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------- contracts
def test_ryko_small_param_contract():
    """The 52.2M decision number is an architecture fact — pinned here."""
    model = RykoLM(get_preset("small"))
    n = sum(p.numel() for p in model.parameters())
    assert n == 52_179_456
    assert round(n / 1e6, 1) == 52.2


def test_to_bench_cases_conversion(seed):
    cases = load_ryko1000(seed)
    bench = to_bench_cases(cases)
    assert len(bench) == len(cases)
    assert {c.kind for c in bench} == {"mcq", "cite", "abstain"}
    assert all(c.meta.get("family") for c in bench)


def test_wrong_book_never_verifies(store):
    """The wrong_attribution probe premise holds on the synthetic store."""
    pid = next(store.iter_passages())["passage_id"]
    text = store.passage(pid)["text"]
    quote = " ".join(text.split())[:80]
    control = verify_quote(store, quote, write_status=False)
    assert control.status == "verified"
    scoped = verify_quote(store, quote, book_id="bk1", write_status=False)
    assert scoped.status != "verified"


def test_probe_materialization_compliant(store, seed):
    cases = load_ryko1000(seed)
    report = materialize_probes(store, cases)
    assert report["materialized"] == report["probes_total"]
    assert report["compliance"] == 1.0
    for kind in ("wrong_attribution", "splice", "evidenceless"):
        assert report["by_kind"][kind]["rate"] == 1.0
    assert report["failures"] == []


@requires_tokenizer
def test_run_eval_base_and_rag(store_path, seed, tmp_path):
    """Full runner smoke on a tiny checkpoint; report contract + the two
    prompt modes both produce their sections."""
    model = RykoLM(get_preset("tiny"))
    model.eval()
    ckpt = tmp_path / "tiny.pt"
    model.save(ckpt)
    report = run_eval(ckpt, TOKENIZER_JSON, store_path, seed,
                      modes=["base", "rag"])
    assert report["schema"] == "ryko1000-eval/1"
    assert report["model"]["params_M"] == 11.54
    assert report["cite_alignment"]["match_rate"] == 1.0
    assert report["probes"]["compliance"] == 1.0
    assert report["base"]["mcq"]["n"] == 4
    assert report["base"]["abstain"]["n"] == 1
    assert report["rag"]["mcq"]["n"] == report["base"]["mcq"]["n"]
    assert "retrieval_mean" in report["rag"]["latency_ms"]
    assert 0.0 <= report["base"]["mcq"]["accuracy"] <= 1.0


@requires_tokenizer
def test_run_eval_deterministic_no_latency(store_path, seed, tmp_path):
    """Two identical runs produce identical reports EXCEPT the measured
    latency (wall-clock is a measurement, not a contract quantity)."""
    model = RykoLM(get_preset("tiny"))
    model.eval()
    ckpt = tmp_path / "tiny.pt"
    model.save(ckpt)
    rep1 = run_eval(ckpt, TOKENIZER_JSON, store_path, seed, modes=["base"])
    rep2 = run_eval(ckpt, TOKENIZER_JSON, store_path, seed, modes=["base"])
    for rep in (rep1, rep2):
        rep["base"]["latency_ms"].clear()
        for row in rep["base"]["rows"] + rep["base"]["abstain_rows"]:
            row["ms"] = 0.0  # wall-clock is a measurement, not a contract
    assert json.dumps(rep1, ensure_ascii=False, sort_keys=True) == \
        json.dumps(rep2, ensure_ascii=False, sort_keys=True)
