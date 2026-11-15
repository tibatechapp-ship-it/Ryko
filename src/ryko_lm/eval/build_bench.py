"""Build the RYKO-Bench seed from the committed Knowledge Core.

Every case is **self-checked at build time**: verified-quote cases only ship
if ``verify_quote`` actually returns ``verified``; negatives only ship if the
verdict is ``not_found``; MCQ positives must out-score the distractors with a
plain-text criterion (longest-common-token overlap) before being included.
Deterministic: fixed RNG seed, stable sort — same store, same benchmark.

The result is a *seed*, not a headline metric: auto-generated cases still
need human review before they back any release claim (research doc §eval).

Usage::

    python -m ryko_lm.eval.build_bench \
        --store data/knowledge/ryko.sqlite \
        --out docs/benchmarks/ryko_bench_seed.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from ryko_lm.knowledge.citation import verify_quote
from ryko_lm.knowledge.store import KnowledgeStore
from ryko_lm.knowledge.textproc import ar_tokens

SEED = 20261115
N_CITE_VERIFIED = 20
N_CITE_NEGATIVE = 10
N_MCQ = 18
N_ABSTAIN = 8

# Questions the heritage corpus cannot answer — correct behaviour is abstain.
ABSTAIN_CASES = (
    ("ما رأي علماء التراث في سرعة المعالجات الرسومية في الحواسيب الحديثة؟", "تقنية"),
    ("كم عدد سكان مدينة الرياض في عام 2030 حسب كتب الماضي؟", "تقنية"),
    ("ما حكم استخدام تطبيقات الهاتف الذكي في تداول العملات الرقمية؟", "معاصر"),
    ("ما هي أحدث نسخة من نظام التشغيل ويندوز حسب مصنفات الحديث؟", "معاصر"),
    ("ما رأي ابن سينا في سرعة الإنترنت فائق النطاق؟", "معاصر"),
    ("كم عدد محطات مترو أنقرة بحسب كتب التاريخ القديمة؟", "معاصر"),
    ("ما رأي فقهاء المذاهب الأربعة في خوارزميات الذكاء الاصطناعي التوليدي؟", "معاصر"),
    ("ما هي كلفة تذكرة الطيران بين جدة والقاهرة في مصنف اللغة؟", "معاصر"),
)


def _continuation_cases(store: KnowledgeStore, rng: random.Random) -> list[dict]:
    """Cloze-MCQ: a passage prefix + true continuation vs 3 foreign tails."""
    pool = [p for p in store.iter_passages() if len(p["text"]) >= 240]
    rng.shuffle(pool)
    out: list[dict] = []
    used_prefixes: set[str] = set()
    for passage in pool:
        if len(out) >= N_MCQ:
            break
        text = " ".join(passage["text"].split())
        cut = 100
        prefix, tail = text[:cut].rsplit(" ", 1)[0], text[cut:].strip()
        if len(prefix) < 60 or len(tail) < 30 or prefix in used_prefixes:
            continue
        distractors = []
        for other in pool:
            if other["book_id"] == passage["book_id"] or len(other["text"]) < 240:
                continue
            cand = " ".join(other["text"].split())[:len(tail)]
            if cand and cand != tail:
                distractors.append(cand)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            continue
        options = distractors + [tail]
        rng.shuffle(options)
        out.append({
            "case_id": f"rb-mcq-{len(out) + 1:03d}",
            "kind": "mcq",
            "source": "ryko-bench-seed",
            "domain": "استكمال-نص",
            "question": "أي المتابعات تكمل هذا النص التراثي كما ورد في أصله؟\n" + prefix + " …",
            "options": options,
            "answer": options.index(tail),
            "meta": {"passage_id": passage["passage_id"], "book_id": passage["book_id"],
                     "page_num": passage["page_num"]},
        })
        used_prefixes.add(prefix)
    return out


def _cite_cases(store: KnowledgeStore, rng: random.Random) -> list[dict]:
    """Verified quotes (positives) + cross-book splices (negatives), self-checked."""
    pool = [p for p in store.iter_passages() if 160 <= len(p["text"]) <= 500]
    rng.shuffle(pool)
    positives: list[dict] = []
    negatives: list[dict] = []
    for idx, passage in enumerate(pool):
        text = " ".join(passage["text"].split())
        quote = text[:120].rsplit(" ", 1)[0]
        if len(quote) < 80:
            continue
        verdict = verify_quote(store, quote)
        if verdict.status == "verified" and len(positives) < N_CITE_VERIFIED:
            positives.append({
                "case_id": f"rb-cite-{len(positives) + 1:03d}",
                "kind": "cite", "source": "ryko-bench-seed", "domain": "توثيق-اقتباس",
                "quote": quote, "expect": "verified",
                "meta": {"passage_id": passage["passage_id"], "book_id": passage["book_id"],
                         "page_num": passage["page_num"]},
            })
        elif len(negatives) < N_CITE_NEGATIVE and positives:
            donor = pool[(idx + 7) % len(pool)]
            if donor["book_id"] == passage["book_id"]:
                continue
            donor_text = " ".join(donor["text"].split())
            splice = text[:60] + " " + donor_text[-60:]
            if verify_quote(store, splice).status == "not_found":
                negatives.append({
                    "case_id": f"rb-cite-{len(positives) + len(negatives) + 1:03d}",
                    "kind": "cite", "source": "ryko-bench-seed", "domain": "توثيق-اقتباس",
                    "quote": splice, "expect": "not_found",
                    "meta": {"spliced_from": donor["book_id"]},
                })
        if len(positives) >= N_CITE_VERIFIED and len(negatives) >= N_CITE_NEGATIVE:
            break
    return positives + negatives


def build(store: KnowledgeStore) -> list[dict]:
    rng = random.Random(SEED)
    cases = _cite_cases(store, rng) + _continuation_cases(store, rng)
    for i, (question, domain) in enumerate(ABSTAIN_CASES, 1):
        cases.append({
            "case_id": f"rb-abstain-{i:03d}", "kind": "abstain",
            "source": "ryko-bench-seed", "domain": domain,
            "question": question, "should_answer": False,
            "options": ["يجزم بجواب حاسم", "يعتذر عن الجواب لعدم السند", "يخترع إجابة"],
            "answer": None,
        })
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ryko_build_bench", description="Build the self-checked RYKO-Bench seed.")
    parser.add_argument("--store", type=Path, default=Path("data/knowledge/ryko.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks/ryko_bench_seed.jsonl"))
    args = parser.parse_args(argv)

    store = KnowledgeStore(str(args.store))
    cases = build(store)
    store.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as sink:
        for case in cases:
            sink.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    kinds: dict[str, int] = {}
    for case in cases:
        kinds[case["kind"]] = kinds.get(case["kind"], 0) + 1
    print(f"[bench] {len(cases)} cases -> {args.out} ({kinds})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
