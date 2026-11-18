#!/usr/bin/env python3
"""Build the linguistics probe set (بوابة القياس اللغوية) — rb-ling families.

Extends the RYKO-Bench discipline to the لغة (linguistics) layer: every case
is **self-checked at build time** against the Knowledge Core — attribution
options come from stored claims/author rows (never guessed), book cases come
from committed passages, and the correct answer is derived from the store
before shuffling.  Deterministic: fixed RNG seed (20261115), stable order —
same store, same probes (byte-identical double build is asserted by tests).

Families (all ``mcq``, scored by the unmodified ``ryko_bench`` runner):

* ``rb-ling-origin``   — who authored a curated linguistic rule/definition
  (distractors: other lugha originators);
* ``rb-ling-author``   — who wrote a committed lugha book
  (distractors: other lugha authors);
* ``rb-ling-book``     — which committed lugha book owns a passage excerpt
  (distractors: other lugha book titles);
* ``rb-ling-section``  — which باب (nahw/sarf/balagha/maajim/fiqh-lugha/adab/
  munhaj) a committed lugha book belongs to.

This is a *seed* probe set like the main bench: auto-generated, needs human
review before any release claim.

Usage::

    python scripts/build_ling_probes.py \
        --db data/knowledge/ryko.sqlite \
        --manifest data/knowledge/lugha_manifest.json \
        --out docs/benchmarks/ling_probes.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ryko_lm.knowledge.store import KnowledgeStore  # noqa: E402

SEED = 20261115

# Closed section vocabulary — must mirror lugha_manifest.schools exactly.
SECTION_LABELS: tuple[tuple[str, str], ...] = (
    ("nahw", "النحو"),
    ("sarf", "الصرف"),
    ("balagha", "البلاغة"),
    ("maajim", "المعاجم"),
    ("fiqh-lugha", "فقه اللغة واللسانيات"),
    ("adab", "الأدب"),
    ("munhaj", "المقررات والمناهج"),
)
_SECTION_LABEL_MAP = dict(SECTION_LABELS)

N_ORIGIN_MCQ = 16
N_AUTHOR_MCQ = 12
N_BOOK_MCQ = 12
N_SECTION_MCQ = 12

# claims.meta.source marker for the lugha layer (ideas.seed_ideas contract)
LUGHA_CLAIM_MARKER = "lugha_manifest"


def _lugha_claims(store: KnowledgeStore) -> list[dict]:
    rows = store.conn.execute(
        "SELECT claim_id, claim_text, meta FROM claims").fetchall()
    out = []
    for r in rows:
        meta = json.loads(r["meta"]) if r["meta"] else {}
        if meta.get("layer") == "lugha" and meta.get("originator"):
            out.append({"claim_id": r["claim_id"], "text": r["claim_text"],
                        "originator": meta["originator"],
                        "school": meta.get("school")})
    return out


def _lugha_books(store: KnowledgeStore) -> list[dict]:
    rows = store.conn.execute(
        "SELECT b.book_id, b.title_ar, b.category, a.name_ar AS author "
        "FROM books b LEFT JOIN authors a ON a.author_id = b.author_id "
        "WHERE b.source_id='lugha'").fetchall()
    out = []
    for r in rows:
        title = " ".join((r["title_ar"] or "").split())
        if not title:
            continue
        cat = r["category"] or ""
        section = cat.rsplit("/", 1)[1] if "/" in cat else None
        out.append({"book_id": r["book_id"], "title": title,
                    "author": (r["author"] or "").strip(),
                    "section": section if section in _SECTION_LABEL_MAP else None})
    return out


def _origin_cases(store: KnowledgeStore, rng: random.Random) -> list[dict]:
    claims = _lugha_claims(store)
    originators = sorted({c["originator"] for c in claims})
    if len(claims) < 4 or len(originators) < 4:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    pool = claims[:]
    rng.shuffle(pool)
    for c in pool:
        if len(out) >= N_ORIGIN_MCQ:
            break
        if c["text"] in seen:
            continue
        distractors = [o for o in originators if o != c["originator"]]
        rng.shuffle(distractors)
        if len(distractors) < 3:
            continue
        options = distractors[:3] + [c["originator"]]
        rng.shuffle(options)
        out.append({
            "case_id": f"rb-ling-origin-{len(out) + 1:03d}",
            "kind": "mcq", "source": "ling-probes-seed",
            "domain": "نسبة-القاعدة-اللغوية",
            "question": "من صاحب هذه القاعدة/التعريف في علم العربية؟\n«" + c["text"] + "»",
            "options": options,
            "answer": options.index(c["originator"]),
            "meta": {"claim_id": c["claim_id"], "school": c["school"]},
        })
        seen.add(c["text"])
    return out


def _author_cases(books: list[dict], rng: random.Random) -> list[dict]:
    named = [b for b in books if b["author"]]
    authors = sorted({b["author"] for b in named})
    if len(named) < 4 or len(authors) < 4:
        return []
    titles_set = {b["title"] for b in named}
    out: list[dict] = []
    seen: set[str] = set()
    pool = named[:]
    rng.shuffle(pool)
    for b in pool:
        if len(out) >= N_AUTHOR_MCQ:
            break
        if b["title"] in seen:
            continue
        distractors = [a for a in authors if a != b["author"]]
        rng.shuffle(distractors)
        if len(distractors) < 3:
            continue
        options = distractors[:3] + [b["author"]]
        rng.shuffle(options)
        out.append({
            "case_id": f"rb-ling-author-{len(out) + 1:03d}",
            "kind": "mcq", "source": "ling-probes-seed",
            "domain": "نسبة-المؤلف",
            "question": "من مؤلف «" + b["title"] + "» من كتب طبقة اللغة المثبتة؟",
            "options": options,
            "answer": options.index(b["author"]),
            "meta": {"book_id": b["book_id"]},
        })
        seen.add(b["title"])
    return out


def _book_cases(store: KnowledgeStore, books: list[dict],
                rng: random.Random) -> list[dict]:
    lugha_ids = {b["book_id"] for b in books}
    titles_by_id = {b["book_id"]: b["title"] for b in books}
    if len(titles_by_id) < 4:
        return []
    pool = [p for p in store.iter_passages()
            if p["book_id"] in lugha_ids and len(p["text"]) >= 160]
    rng.shuffle(pool)
    out: list[dict] = []
    used: set[str] = set()
    for passage in pool:
        if len(out) >= N_BOOK_MCQ:
            break
        correct = titles_by_id.get(passage["book_id"])
        if not correct or correct in used:
            continue
        distractors = [t for t in sorted(titles_by_id.values()) if t != correct]
        rng.shuffle(distractors)
        if len(distractors) < 3:
            continue
        options = distractors[:3] + [correct]
        rng.shuffle(options)
        text = " ".join(passage["text"].split())
        excerpt = text[:140].rsplit(" ", 1)[0]
        if len(excerpt) < 80:
            continue
        out.append({
            "case_id": f"rb-ling-book-{len(out) + 1:03d}",
            "kind": "mcq", "source": "ling-probes-seed",
            "domain": "نسبة-الكتاب-اللغوي",
            "question": ("أي الكتب اللغوية المثبتة ينتمي إليه هذا المقطع؟\n«"
                         + excerpt + " …»"),
            "options": options,
            "answer": options.index(correct),
            "meta": {"passage_id": passage["passage_id"],
                     "book_id": passage["book_id"]},
        })
        used.add(correct)
    return out


def _section_cases(books: list[dict], rng: random.Random) -> list[dict]:
    labeled = [b for b in books if b["section"]]
    if len(labeled) < 4:
        return []
    labels = [lbl for _, lbl in SECTION_LABELS]
    out: list[dict] = []
    seen: set[str] = set()
    pool = labeled[:]
    rng.shuffle(pool)
    for b in pool:
        if len(out) >= N_SECTION_MCQ:
            break
        if b["title"] in seen:
            continue
        correct = _SECTION_LABEL_MAP[b["section"]]
        distractors = [lbl for lbl in labels if lbl != correct]
        rng.shuffle(distractors)
        options = distractors[:3] + [correct]
        rng.shuffle(options)
        out.append({
            "case_id": f"rb-ling-section-{len(out) + 1:03d}",
            "kind": "mcq", "source": "ling-probes-seed",
            "domain": "باب-اللغة",
            "question": "إلى أي باب من أبواب طبقة اللغة يعود «" + b["title"] + "»؟",
            "options": options,
            "answer": options.index(correct),
            "meta": {"book_id": b["book_id"], "section": b["section"]},
        })
        seen.add(b["title"])
    return out


def build(store: KnowledgeStore) -> list[dict]:
    rng = random.Random(SEED)
    books = _lugha_books(store)
    if not books:
        raise SystemExit("[error] no lugha books in store — run "
                         "scripts/build_lugha_db.py first")
    cases = (_origin_cases(store, rng) + _author_cases(books, rng)
             + _book_cases(store, books, rng) + _section_cases(books, rng))
    if not cases:
        raise SystemExit("[error] no probe cases passed self-checks")
    return cases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=Path("data/knowledge/ryko-full.sqlite"),
                    help="the DB with lugha passages (full local DB) — the "
                         "committed DB carries the lugha claims + catalog "
                         "metadata only under the 100MiB wall")
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/knowledge/lugha_manifest.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("docs/benchmarks/ling_probes.jsonl"))
    args = ap.parse_args(argv)

    store = KnowledgeStore(str(args.db))
    cases = build(store)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")

    from collections import Counter
    by_domain = Counter(c["domain"] for c in cases)
    print(f"[probes] wrote {len(cases)} cases -> {args.out}")
    for dom, n in sorted(by_domain.items()):
        print(f"[probes]   {dom}: {n}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
