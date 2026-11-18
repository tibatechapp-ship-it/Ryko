"""RYKO-1000 linguistic expansion — the append-only appender (v1.22.0).

Reads the committed seed (``docs/benchmarks/ryko1000_seed.jsonl``) as a
FROZEN artifact pinned by sha256 and appends the three linguistic families
(``irab`` الإعراب والنحو، ``sarf`` الصرف، ``balagha`` البلاغة) x 100 cases
from the lugha layer of the unified store (core + full lugha passages).

Contract (docs/results/ryko1000_ling_expansion.md, v1.22.0):

* **Frozen prefix** — the first 1,000 lines are byte-frozen:
  ``FROZEN_PREFIX_SHA256`` is computed over the first 1,000 lines joined
  with ``\\n`` (NO trailing newline).  Any drift fails the build loudly
  BEFORE any write (tested against a corrupted pin).
* **Append-only** — new cases carry sequential numbering 001..100 per
  family; nothing in the frozen prefix is ever rewritten.
* **Fresh RNG offsets** — SEED_BASE+20/21/22 (the 10..19 range is taken by
  the ten core families; offsets are never reused).
* **Pattern quotas per family (3+11+43+43 = 100):**
  - نسبة-القاعدة-اللغوية (3) — rule attribution from the SHARED lugha claim
    pool: every claim is attributed at most ONCE across the three families
    (9 distinct rules total); the madhhab gate matches the answer against
    the claim's committed ``meta.originator``.
  - نسبة-المؤلف-اللغوي (11) — author attribution from the committed catalog
    (books.meta author), PARTITIONED across the three families over the
    book-id-sorted catalog (irab 0..10, sarf 11..21, balagha 22..32) so no
    question repeats across families; the madhhab gate matches the answer
    against the core author catalog.
  - نسبة-الكتاب-اللغوي (43) — anchored passage -> its committed title (the
    madhhab gate matches the book's stored title).
  - استكمال-النص-اللغوي (43) — anchored passage -> its literal continuation
    (the quotes/citations gates check verbatim containment).
* **Per-family book quota cap** — no single book may take more than 40% of
  the family's two passage-consuming patterns (43+43 = 86 slots).
* **Fail loud** — any shortfall (claims, authors, passages) aborts the
  build; nothing is silently approximated.
* **Dual build** — two runs over the same store + seed are byte-identical
  (deterministic pools, fixed iteration order, per-family RNG).

The builder is NOT the judge: every case is self-checked at build time
(``verification.status = "build-selfchecked"``) and the independent
verifier (``verify_ryko1000``) re-checks the finished 1,300-case seed
through the seven gates on contract-scoped stores.

Usage::

    python -m ryko_lm.eval.build_ryko1000_ling \
        --store data/knowledge/ryko-verify-full.sqlite \
        --seed docs/benchmarks/ryko1000_seed.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from ryko_lm.knowledge.citation import PROTOCOL as CITE_PROTOCOL
from ryko_lm.knowledge.store import KnowledgeStore
from ryko_lm.eval.build_ryko1000 import (
    SEED_BASE,
    _STD_OBJECTION,
    _STD_REASONING,
    _norm,
    _record,
    _trim,
)

BUILDER_VERSION = "1"

FROZEN_PREFIX_LINES = 1_000
FROZEN_PREFIX_SHA256 = (
    "f365c371d2ac468f67972c82f9cbc560f007b209216a1953a33f6ba5a82c4d01")

LINGUA_FAMILIES: tuple[tuple[str, str], ...] = (
    ("irab", "الإعراب والنحو"),
    ("sarf", "الصرف"),
    ("balagha", "البلاغة"),
)
LINGUA_FAMILY_IDS = tuple(f for f, _ in LINGUA_FAMILIES)
N_PER_LING_FAMILY = 100

# Fresh RNG offsets — SEED_BASE+20/21/22 (10..19 are taken by the core
# families; offsets are never reused — append-only RNG discipline).
RNG_OFFSETS = {"irab": 20, "sarf": 21, "balagha": 22}

# Pattern quotas per family: 3 rules + 11 authors + 43 books + 43
# continuations = 100 (9/33/129/129 across the three families).
N_RULE, N_AUTHOR, N_BOOK, N_CONT = 3, 11, 43, 43
assert N_RULE + N_AUTHOR + N_BOOK + N_CONT == N_PER_LING_FAMILY

# Author partition over the book-id-sorted catalog: irab 0..10,
# sarf 11..21, balagha 22..32 (33 catalog entries, no cross-family repeat).
AUTHOR_PARTITION = {"irab": (0, 11), "sarf": (11, 22), "balagha": (22, 33)}

# Per-family book quota cap: 40% of the two passage-consuming patterns'
# share (N_BOOK + N_CONT = 86 slots) — ceil, so the cap is an upper bound.
BOOK_QUOTA_RATIO = 0.40
BOOK_QUOTA_CAP = -(-(N_BOOK + N_CONT) * 40 // 100)  # ceil(86 * 0.40) = 35

# Passage pool admission thresholds (same construction floors as the core
# builder's book-attribution / continuation fillers).
MIN_BOOK_TEXT = 160
MIN_CONT_TEXT = 240
MIN_EXCERPT = 80
MIN_CONT_PREFIX = 60
MIN_CONT_TAIL = 30

DOMAIN_RULE = "نسبة-القاعدة-اللغوية"
DOMAIN_AUTHOR = "نسبة-المؤلف-اللغوي"
DOMAIN_BOOK = "نسبة-الكتاب-اللغوي"
DOMAIN_CONT = "استكمال-النص-اللغوي"

_MCQ_PROBE = {"kind": "wrong_attribution",
              "probe": "نسبة المقطع إلى كتاب غير مصدره",
              "must": "reject"}


# ----------------------------------------------------------------- contract
def frozen_prefix_hash(lines: list[str]) -> str:
    """The contract hash: first 1,000 lines joined with \\n, no trailing \\n."""
    return hashlib.sha256(
        "\n".join(lines[:FROZEN_PREFIX_LINES]).encode("utf-8")).hexdigest()


def load_frozen_seed(seed_path: Path) -> list[str]:
    """Read the committed seed and enforce the frozen-prefix contract."""
    lines = seed_path.read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in lines if ln.strip()]
    if len(lines) != FROZEN_PREFIX_LINES:
        raise ValueError(
            f"seed has {len(lines)} lines vs the frozen {FROZEN_PREFIX_LINES} "
            "— the appender only extends the committed 1,000-case seed")
    got = frozen_prefix_hash(lines)
    if got != FROZEN_PREFIX_SHA256:
        raise ValueError(
            "FROZEN-PREFIX CONTRACT BROKEN: seed sha256 "
            f"{got} != pinned {FROZEN_PREFIX_SHA256} — refusing to append")
    return lines


# ------------------------------------------------------------------- pools
def _lugha_book_rows(store: KnowledgeStore) -> list[dict]:
    """Catalog lugha books with authors, fixed by book_id order."""
    rows = [b for b in store.iter_books()
            if b["source_id"] == "lugha" and b.get("author_id")]
    rows.sort(key=lambda b: b["book_id"])
    return rows


def _author_names(store: KnowledgeStore) -> dict[str, str]:
    names: dict[str, str] = {}
    for row in store.conn.execute("SELECT author_id, name_ar FROM authors"):
        names[row["author_id"]] = (row["name_ar"] or "").strip()
    return names


def _lugha_passage_pool(store: KnowledgeStore,
                        used: set[str]) -> list[dict]:
    """Deterministic lugha passage pool (book-id order), excluding ``used``."""
    pool: list[dict] = []
    for b in _lugha_book_rows(store):
        for p in store.iter_passages(book_id=b["book_id"]):
            if p["passage_id"] not in used and (p.get("text") or "").strip():
                pool.append(p)
    return pool


def _used_from_frozen(cases: list[dict]) -> tuple[set[str], set[str]]:
    """Passage/claim ids consumed by the frozen prefix (never recycled)."""
    used_p, used_c = set(), set()
    for c in cases:
        for s in c.get("sources") or []:
            if s.get("passage_id"):
                used_p.add(s["passage_id"])
        meta = c.get("meta") or {}
        if meta.get("passage_id"):
            used_p.add(meta["passage_id"])
        if meta.get("claim_id"):
            used_c.add(meta["claim_id"])
    return used_p, used_c


def _lugha_claims(store: KnowledgeStore, used_claims: set[str]) -> list[dict]:
    """Usable lugha rules (layer='lugha', originator committed, unused)."""
    out: list[dict] = []
    for c in store.iter_claims(layer="lugha"):
        if c["claim_id"] in used_claims:
            continue
        meta = {}
        try:
            meta = json.loads(c.get("meta") or "{}")
        except (json.JSONDecodeError, TypeError):
            meta = {}
        originator = (meta.get("originator") or "").strip()
        if not originator or len(_norm(c["claim_text"])) < 25:
            continue
        c["originator"] = originator
        out.append(c)
    return out


def _mcq_options(rng: random.Random, correct: str,
                 distractor_pool: list[str]) -> list[str]:
    """correct + 3 distinct foreign options, shuffled (deterministic)."""
    pool = [d for d in dict.fromkeys(distractor_pool) if d != correct]
    rng.shuffle(pool)
    if len(pool) < 3:
        raise ValueError("not enough distinct distractors for an MCQ")
    options = pool[:3] + [correct]
    rng.shuffle(options)
    return options


# ------------------------------------------------------------------ fillers
def _fill_rules(family: str, rng: random.Random, claims: list[dict],
                claim_pool_originators: list[str], n: int,
                start_index: int, log: list[str]) -> list[dict]:
    """Rule attribution from the SHARED pool — no claim attributed twice.

    ``log`` collects the claim ids consumed here so the caller can drop them
    from the shared pool for the next family (each rule attributed once).
    """
    out: list[dict] = []
    claims = list(claims)
    rng.shuffle(claims)
    for claim in claims:
        if len(out) >= n:
            break
        correct = claim["originator"]
        options = _mcq_options(rng, correct, claim_pool_originators)
        text = _norm(claim["claim_text"])
        out.append(_record(
            family,
            case_id=f"ry10-{family}-{start_index + len(out):03d}",
            kind="mcq", domain=DOMAIN_RULE,
            question=("لأي منشئ أو أصل تُنسب هذه القاعدة اللغوية كما هي "
                      "مثبتة في نواة المعرفة؟\n«" + _trim(text, 170) + "»"),
            options=options, answer=options.index(correct),
            expected_answer=correct,
            sources=[],
            reasoning=_STD_REASONING["mcq"], objections=_STD_OBJECTION,
            probes=[{"kind": "wrong_attribution",
                     "probe": "نسبة القاعدة لمنشئ غير مثبت في النواة",
                     "must": "reject"}],
            meta={"claim_id": claim["claim_id"], "originator": correct,
                  "ling_pattern": "rule"},
        ))
        log.append(claim["claim_id"])
    if len(out) < n:
        raise ValueError(
            f"{family}: rule pool exhausted ({len(out)}/{n} constructible) — "
            "the shared lugha claim pool cannot attribute a rule twice")
    return out


def _fill_authors(family: str, rng: random.Random,
                  catalog: list[dict], author_names: dict[str, str],
                  passages_by_book: dict[str, list[dict]],
                  all_author_names: list[str], n: int, start_index: int,
                  used: set[str], book_count: dict[str, int]) -> list[dict]:
    """Author attribution from the book-id-sorted catalog partition."""
    lo, hi = AUTHOR_PARTITION[family]
    slots = catalog[lo:hi]
    if len(slots) < n:
        raise ValueError(
            f"{family}: author partition needs {n} catalog entries "
            f"(slots {lo}..{hi}), catalog has {len(slots)} usable")
    out: list[dict] = []
    for book in slots:
        if len(out) >= n:
            break
        candidates = [p for p in passages_by_book.get(book["book_id"], [])
                      if len(_norm(p["text"])) >= MIN_EXCERPT]
        rng.shuffle(candidates)
        if not candidates:
            raise ValueError(
                f"{family}: catalog book {book['book_id']} has no usable "
                "passage for author attribution — fail loud, never guess")
        passage = candidates[0]
        correct = author_names.get(book["author_id"], "")
        if not correct:
            raise ValueError(
                f"{family}: author {book['author_id']!r} missing from the "
                "committed catalog")
        options = _mcq_options(rng, correct, all_author_names)
        excerpt = _trim(_norm(passage["text"]), 140)
        used.add(passage["passage_id"])
        book_count[book["book_id"]] = book_count.get(book["book_id"], 0) + 1
        out.append(_record(
            family,
            case_id=f"ry10-{family}-{start_index + len(out):03d}",
            kind="mcq", domain=DOMAIN_AUTHOR,
            question=("من مؤلف هذا الكتاب اللغوي كما هو مثبت في كتالوج "
                      "نواة المعرفة؟\n«" + excerpt + " …»"),
            options=options, answer=options.index(correct),
            expected_answer=correct,
            sources=[{"book_id": book["book_id"],
                      "title_ar": _norm(book["title_ar"]),
                      "protocol": CITE_PROTOCOL,
                      "quote_sha256": ""}],
            supporting=[{"passage_id": passage["passage_id"],
                         "quote": excerpt}],
            reasoning=_STD_REASONING["mcq"], objections=_STD_OBJECTION,
            probes=[{"kind": "wrong_attribution",
                     "probe": "نسبة الكتاب إلى مؤلف غير مثبت في الكتالوج",
                     "must": "reject"}],
            meta={"book_id": book["book_id"], "author_id": book["author_id"],
                  "passage_id": passage["passage_id"],
                  "ling_pattern": "author"},
        ))
    return out


def _fill_books(family: str, rng: random.Random, pool: list[dict],
                titles_by_id: dict[str, str], fresh_titles: list[str],
                n: int, start_index: int, used: set[str],
                book_count: dict[str, int],
                book_cap: int) -> list[dict]:
    """Book attribution: anchored lugha passage -> its committed title."""
    out: list[dict] = []
    for passage in pool:
        if len(out) >= n:
            break
        if passage["passage_id"] in used:
            continue
        if book_count.get(passage["book_id"], 0) >= book_cap:
            continue
        correct = titles_by_id.get(passage["book_id"], "")
        if not correct or correct not in fresh_titles:
            continue
        distractors = [t for t in fresh_titles if t != correct]
        rng.shuffle(distractors)
        if len(distractors) < 3:
            continue
        options = distractors[:3] + [correct]
        rng.shuffle(options)
        excerpt = _trim(_norm(passage["text"]), 140)
        if len(excerpt) < MIN_EXCERPT:
            continue
        used.add(passage["passage_id"])
        book_count[passage["book_id"]] = \
            book_count.get(passage["book_id"], 0) + 1
        out.append(_record(
            family,
            case_id=f"ry10-{family}-{start_index + len(out):03d}",
            kind="mcq", domain=DOMAIN_BOOK,
            question=("أي الكتب اللغوية المثبتة في نواة المعرفة ينتمي إليه "
                      "هذا المقطع؟\n«" + excerpt + " …»"),
            options=options, answer=options.index(correct),
            expected_answer=correct,
            sources=[_book_src_ok(passage, titles_by_id.get(
                passage["book_id"], ""))],
            supporting=[{"passage_id": passage["passage_id"],
                         "quote": excerpt}],
            reasoning=_STD_REASONING["mcq"], objections=_STD_OBJECTION,
            probes=[_MCQ_PROBE],
            meta={"passage_id": passage["passage_id"],
                  "book_id": passage["book_id"],
                  "ling_pattern": "book"},
        ))
    if len(out) < n:
        raise ValueError(
            f"{family}: book-attribution pool exhausted ({len(out)}/{n})")
    return out


def _book_src_ok(passage: dict, title: str = "") -> dict:
    """A source entry the verifier can resolve on the unified store.

    Kept module-level so ``_fill_books`` stays deterministic and the
    passage-anchor contract (ryko-cite/1 + resolved book) is testable.
    """
    return {"passage_id": passage["passage_id"],
            "book_id": passage["book_id"],
            "title_ar": _norm(title or passage.get("title_ar") or ""),
            "page_num": passage.get("page_num"),
            "protocol": CITE_PROTOCOL,
            "quote_sha256": ""}


def _fill_continuations(family: str, rng: random.Random, pool: list[dict],
                        titles_by_id: dict[str, str],
                        n: int, start_index: int, used: set[str],
                        book_count: dict[str, int],
                        book_cap: int) -> list[dict]:
    """Cloze-MCQ: lugha passage prefix + true continuation vs 3 foreign tails.

    Mirrors the core builder's continuation construction (word-boundary cut,
    rotated distractor cursor so the three foreign tails differ across
    cases) over the lugha pool, under the same family book quota cap.
    """
    ordered = [p for p in pool if p["passage_id"] not in used
               and len(p["text"]) >= MIN_CONT_TEXT
               and book_count.get(p["book_id"], 0) < book_cap]
    out: list[dict] = []
    for idx, passage in enumerate(ordered):
        if len(out) >= n:
            break
        if passage["passage_id"] in used:
            continue
        if book_count.get(passage["book_id"], 0) >= book_cap:
            continue
        text = _norm(passage["text"])
        cut = 100
        prefix, tail = text[:cut].rsplit(" ", 1)[0], text[cut:].strip()
        if len(prefix) < MIN_CONT_PREFIX or len(tail) < MIN_CONT_TAIL:
            continue
        tail = _trim(tail, 220)
        if len(tail) < MIN_CONT_TAIL:
            continue
        distractors: list[str] = []
        for k in range(len(ordered)):
            other = ordered[(idx * 5 + 11 + k) % len(ordered)]
            if other["book_id"] == passage["book_id"]:
                continue
            cand = _trim(_norm(other["text"]), len(tail))
            if cand and cand != tail and cand not in distractors:
                distractors.append(cand)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            continue
        options = distractors + [tail]
        rng.shuffle(options)
        used.add(passage["passage_id"])
        book_count[passage["book_id"]] = \
            book_count.get(passage["book_id"], 0) + 1
        out.append(_record(
            family,
            case_id=f"ry10-{family}-{start_index + len(out):03d}",
            kind="mcq", domain=DOMAIN_CONT,
            question=("أي المتابعات تكمل هذا النص اللغوي كما ورد في أصله؟\n"
                      + prefix + " …"),
            options=options, answer=options.index(tail),
            expected_answer=tail,
            sources=[_book_src_ok(passage, titles_by_id.get(
                passage["book_id"], ""))],
            supporting=[{"passage_id": passage["passage_id"],
                         "quote": tail}],
            reasoning=_STD_REASONING["mcq"], objections=_STD_OBJECTION,
            probes=[_MCQ_PROBE],
            meta={"passage_id": passage["passage_id"],
                  "book_id": passage["book_id"],
                  "page_num": passage.get("page_num"),
                  "ling_pattern": "continuation"},
        ))
    if len(out) < n:
        raise ValueError(
            f"{family}: continuation pool exhausted ({len(out)}/{n})")
    return out


# -------------------------------------------------------------------- build
def build_ling_cases(store: KnowledgeStore,
                     frozen_cases: list[dict]) -> list[dict]:
    """Build the 300 linguistic cases (deterministic, self-checked)."""
    used_p, used_c = _used_from_frozen(frozen_cases)

    claim_pool = _lugha_claims(store, used_c)
    claim_originators = sorted({c["originator"] for c in claim_pool})
    if len(claim_pool) < N_RULE * len(LINGUA_FAMILY_IDS):
        raise ValueError(
            f"shared rule pool needs {N_RULE * len(LINGUA_FAMILY_IDS)} "
            f"usable lugha claims, has {len(claim_pool)}")

    catalog = _lugha_book_rows(store)
    author_names = _author_names(store)
    if len(catalog) < max(hi for _, hi in AUTHOR_PARTITION.values()):
        raise ValueError(
            f"catalog needs {max(hi for _, hi in AUTHOR_PARTITION.values())} "
            f"authored lugha books, has {len(catalog)}")
    all_author_names = sorted({author_names.get(b["author_id"], "")
                               for b in catalog} - {""})

    passages_by_book: dict[str, list[dict]] = {}
    pool_all = _lugha_passage_pool(store, used_p)
    for p in pool_all:
        passages_by_book.setdefault(p["book_id"], []).append(p)
    titles_by_id = {b["book_id"]: _norm(b["title_ar"] or "")
                    for b in catalog}
    fresh_titles = sorted({t for t in titles_by_id.values() if t})
    if len(fresh_titles) < 4:
        raise ValueError("fewer than 4 distinct lugha titles")

    book_cap = BOOK_QUOTA_CAP
    cases: list[dict] = []
    consumed_claims: list[str] = []
    # ONE shared used-passage set across the three families — a passage is
    # consumed at most once GLOBALLY (the core builder's discipline; separate
    # per-family sets shipped identical continuation cases across families —
    # Jaccard 1.000 pairs the dedup gate rightly rejected).  The book quota
    # cap stays PER-FAMILY (its contract is intra-family diversity).
    used: set[str] = set()
    for family, _ar in LINGUA_FAMILIES:
        rng = random.Random(SEED_BASE + RNG_OFFSETS[family])
        book_count: dict[str, int] = {}
        start = 1
        # 1) rules (3) — from the shared pool; consumed claims are removed
        #    so no rule is attributed twice across the three families
        family_claims = [c for c in claim_pool
                         if c["claim_id"] not in consumed_claims]
        rules = _fill_rules(family, rng, family_claims, claim_originators,
                            N_RULE, start, consumed_claims)
        cases += rules
        start += len(rules)
        # 2) authors (11) — catalog partition slots
        authors = _fill_authors(family, rng, catalog, author_names,
                                passages_by_book, all_author_names,
                                N_AUTHOR, start, used, book_count)
        cases += authors
        start += len(authors)
        # 3) books (43) + 4) continuations (43) — shared family pool and
        #    the per-family book quota cap (40% of the 86 passage slots)
        family_pool = [p for p in pool_all if p["passage_id"] not in used]
        books = _fill_books(family, rng, family_pool, titles_by_id,
                            fresh_titles, N_BOOK, start, used, book_count,
                            book_cap)
        cases += books
        start += len(books)
        conts = _fill_continuations(family, rng, family_pool, titles_by_id,
                                    N_CONT, start, used, book_count, book_cap)
        cases += conts
        if len(cases) < (LINGUA_FAMILY_IDS.index(family) + 1) \
                * N_PER_LING_FAMILY:
            raise ValueError(f"family {family!r}: built fewer than 100 cases")
    if len(cases) != N_PER_LING_FAMILY * len(LINGUA_FAMILY_IDS):
        raise ValueError(
            f"expected {N_PER_LING_FAMILY * len(LINGUA_FAMILY_IDS)} ling "
            f"cases, built {len(cases)}")
    if len(consumed_claims) != len(set(consumed_claims)):
        raise ValueError("a lugha rule was attributed twice — contract broken")
    ids = [c["case_id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("ling case_id uniqueness broken")
    for family in LINGUA_FAMILY_IDS:
        fam_ids = [i for i in ids if i.startswith(f"ry10-{family}-")]
        expected = [f"ry10-{family}-{n:03d}"
                    for n in range(1, N_PER_LING_FAMILY + 1)]
        if fam_ids != expected:
            raise ValueError(
                f"family {family!r}: numbering must be sequential 001..100")
    for c in cases:
        if c["kind"] != "mcq":
            raise ValueError(f"{c['case_id']}: ling families are all-mcq "
                             "(abstain lives only in yaqin)")
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ryko_build_ryko1000_ling",
        description="Append the linguistic families to RYKO-1000 "
                    "(frozen-prefix contract, fail-loud).")
    parser.add_argument("--store", type=Path,
                        default=Path("data/knowledge/ryko-verify-full.sqlite"))
    parser.add_argument("--seed", type=Path,
                        default=Path("docs/benchmarks/ryko1000_seed.jsonl"))
    parser.add_argument("--out", type=Path, default=None,
                        help="output path (default: in-place append to "
                             "--seed via an atomic replace)")
    args = parser.parse_args(argv)
    out_path = args.out or args.seed

    frozen_lines = load_frozen_seed(args.seed)
    frozen_cases = [json.loads(ln) for ln in frozen_lines]

    store = KnowledgeStore(str(args.store), persist_vectors=False)
    try:
        ling_cases = build_ling_cases(store, frozen_cases)
    finally:
        store.close()

    out_lines = frozen_lines + [
        json.dumps(c, ensure_ascii=False, sort_keys=True) for c in ling_cases]
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)

    fams: dict[str, int] = {}
    patterns: dict[str, int] = {}
    for c in ling_cases:
        fams[c["family"]] = fams.get(c["family"], 0) + 1
        patterns[(c.get("meta") or {}).get("ling_pattern", "?")] = \
            patterns.get((c.get("meta") or {}).get("ling_pattern", "?"), 0) + 1
    print(f"[ryko1000-ling] {len(ling_cases)} cases appended -> {out_path} "
          f"(total {len(out_lines)})")
    print(f"[ryko1000-ling] families: {fams}")
    print(f"[ryko1000-ling] patterns: "
          f"{dict(sorted(patterns.items()))}")
    print(f"[ryko1000-ling] frozen prefix intact: "
          f"{frozen_prefix_hash(frozen_lines) == FROZEN_PREFIX_SHA256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
