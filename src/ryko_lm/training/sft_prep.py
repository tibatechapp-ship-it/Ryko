"""RYKO-1000 → SFT / DPO preparation (Phase 5 + Phase 9 data packages).

Turns the verified RYKO-1000 seed into frozen, deterministic training
inputs — the "الانتقال إلى التدريب" step of docs/RYKO1000.md — with the
same receipt discipline as the dataset itself:

1. ``split_ryko1000``      — per-family 85/7/8 (train/dev/test) stratified
   split with a dedicated RNG seed; the **test split is held out forever**
   from every training artifact and the manifest records each file's
   sha256 so any later training run can prove what it trained on.
2. ``build_sft``           — prompt/completion pairs from train+dev only:
   MCQ answers, verbatim citation verdicts (ryko-cite/1), refusal and
   must-answer abstention — balancing the over-refusal the v1.17.0 run
   measured (wrong-abstain 68.75%).
3. ``build_dpo``           — preference pairs derived STRUCTURALLY from the
   seed's ``failure_probes``: chosen = the grounded, cited behaviour;
   rejected = the exact failure mode the probe encodes (endorsing a
   cross-book splice, accepting a wrong attribution, fabricating an
   answer without evidence, or wrongly refusing an evidenced answer).
   Provenance: every pair carries its case_id + probe kind + must-behaviour.

The test split is a hard exclusion: both builders assert no test case_id
leaks into a training artifact (leakage is fatal, not a warning).

Usage::

    python -m ryko_lm.training.sft_prep split \\
        --seed docs/benchmarks/ryko1000_seed.jsonl --out data/training/ryko1000_split
    python -m ryko_lm.training.sft_prep sft \\
        --split data/training/ryko1000_split --store data/knowledge/ryko.sqlite \\
        --out data/training
    python -m ryko_lm.training.sft_prep dpo \\
        --split data/training/ryko1000_split --store data/knowledge/ryko.sqlite \\
        --out data/training
    python -m ryko_lm.training.sft_prep dpo-r2 \\
        --seed docs/benchmarks/ryko1000_seed.jsonl \\
        --split data/training/ryko1000_split --store data/knowledge/ryko.sqlite \\
        --round1 data/training/ryko1000_dpo_train.jsonl --out data/training
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from ryko_lm.knowledge.store import KnowledgeStore
from ryko_lm.eval.ryko1000_eval import _probe_quote

SCHEMA_SPLIT = "ryko1000-split/1"
SCHEMA_SFT = "ryko1000-sft/1"
SCHEMA_DPO = "ryko1000-dpo/1"
SCHEMA_DPO_R2 = "ryko1000-dpo-r2/1"
SPLIT_SEED = 20261118
N_TRAIN, N_DEV, N_TEST = 85, 7, 8
SFT_REFUSE = "لا سند موثقاً في نواة المعرفة للإجابة على هذا السؤال — أُمتنع."
FABRICATED = "الجواب: نعم، بحسب المصنفات المثبتة في نواة المعرفة."
DPO_REFUSE = "لا سند — أُمتنع."

# ---------------------------------------------------------------- round-2
# Pins (fail loud before ANY write): the round-2 bundle extends the
# FROZEN round-1 file and the FROZEN v1.22.0 seed.  Any drift in either
# aborts the build — the append-only discipline is enforced here, not
# trusted.
LING_FAMILIES = ("irab", "sarf", "balagha")
SEED_SHA_R2 = "a5613cb020067a21d1be3bb3ddc3858e5477f9f548915e808f592cbba05f1d25"
ROUND1_SHA_REAL = "457fb4ffe3de3bf04a97fb36e1b4a4a71bdd1f63a31f469754f3c964f126cba5"
ROUND1_SHA_R2 = ROUND1_SHA_REAL


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(seed_path: Path) -> list[dict]:
    from ryko_lm.eval.verify_ryko1000 import load_ryko1000
    return load_ryko1000(seed_path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")


# ------------------------------------------------------------------ split
def split_ryko1000(seed_path: Path, out_dir: Path,
                   proportions: tuple[int, int, int] = (N_TRAIN, N_DEV, N_TEST),
                   split_seed: int = SPLIT_SEED) -> dict:
    """Deterministic per-family split + manifest (sha256-frozen).

    Production default is 85/7/8; smaller pools (tests, family slices)
    pass explicit ``proportions`` — the contract stays identical: the
    test bucket is held out from every downstream training artifact.
    """
    n_train, n_dev, n_test = proportions
    cases = _load(seed_path)
    by_fam: dict[str, list[dict]] = {}
    for rec in cases:
        by_fam.setdefault(rec["family"], []).append(rec)

    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    per_family: dict[str, dict] = {}
    for fi, family in enumerate(sorted(by_fam)):
        rng = random.Random(split_seed + fi)
        pool = list(by_fam[family])
        rng.shuffle(pool)
        take = {"train": pool[:n_train],
                "dev": pool[n_train:n_train + n_dev],
                "test": pool[n_train + n_dev:n_train + n_dev + n_test]}
        if len(take["test"]) < 1:
            raise ValueError(f"family {family!r}: pool too small for a "
                             f"{n_train}/{n_dev}/{n_test} split")
        for name, rows in take.items():
            buckets[name].extend(rows)
            per_family.setdefault(family, {})[name] = len(rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    for name in ("train", "dev", "test"):
        path = out_dir / f"ryko1000_{name}.jsonl"
        _write_jsonl(path, buckets[name])
        files[name] = {"path": str(path), "n": len(buckets[name]),
                       "sha256": _sha256(path)}
    manifest = {
        "schema": SCHEMA_SPLIT,
        "split_seed": split_seed,
        "proportions": [n_train, n_dev, n_test],
        "per_family": per_family,
        "files": files,
        "test_held_out": True,
        "seed_source": {"path": str(seed_path),
                        "sha256": _sha256(seed_path)},
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return manifest


def _load_split(split_dir: Path) -> dict[str, list[dict]]:
    split_dir = Path(split_dir)
    out: dict[str, list[dict]] = {}
    for name in ("train", "dev", "test"):
        path = split_dir / f"ryko1000_{name}.jsonl"
        out[name] = [json.loads(line) for line in
                     path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
    return out


# -------------------------------------------------------------------- SFT
def _sft_pair(rec: dict, store: KnowledgeStore) -> dict | None:
    """One prompt/completion pair; None = structurally unpairable."""
    kind = rec["kind"]
    meta = rec.get("meta") or {}
    src = (rec.get("sources") or [{}])[0]
    title = src.get("title_ar") or ""
    page = src.get("page_num")
    ref = f"«{title}» ({src.get('book_id', '')}"
    if page is not None:
        ref += f"، ص {page}"
    ref += ")"
    family = rec["family"]

    if kind == "mcq":
        prompt = f"السؤال: {rec.get('question', '')}\nالجواب:"
        return {"prompt": prompt,
                "completion": f" {rec.get('expected_answer', '')}"}
    if kind == "cite":
        if rec.get("expect") == "verified":
            prompt = ("وثّق الاقتباس الآتي وفق نواة المعرفة "
                      "(ryko-cite/1):\n"
                      f"«{rec.get('quote', '')}»\nالحكم:")
            return {"prompt": prompt, "completion": f"مُوثَّق حرفياً في {ref}."}
        return {"prompt": ("وثّق الاقتباس الآتي وفق نواة المعرفة "
                           "(ryko-cite/1):\n"
                           f"«{rec.get('quote', '')}»\nالحكم:"),
                "completion": "الاقتباس مركّب غير موجود في نواة "
                              "المعرفة — يُرفض ويُمتنع."}
    if kind == "abstain":
        prompt = f"السؤال: {rec.get('question', '')}\nالجواب:"
        if rec.get("should_answer") is False:
            return {"prompt": prompt, "completion": SFT_REFUSE}
        # in-domain must-answer: the curated originator IS the answer.
        # A citation line is added ONLY when a real source exists — never
        # fabricating a «السند» for cases whose claims carry no anchor.
        answer = rec.get("expected_answer") or str(meta.get("originator") or "")
        if not answer:
            return None
        if title:
            return {"prompt": prompt,
                    "completion": f" {answer} — السند: {ref}."}
        return {"prompt": prompt, "completion": f" {answer}"}
    return None


def build_sft(split_dir: Path, store_path: Path, out_dir: Path) -> dict:
    """Prompt/completion pairs from train+dev ONLY (test stays held out)."""
    store = KnowledgeStore(str(store_path), persist_vectors=False)
    splits = _load_split(Path(split_dir))
    train_ids = {r["case_id"] for r in splits["train"]}
    dev_ids = {r["case_id"] for r in splits["dev"]}
    test_ids = {r["case_id"] for r in splits["test"]}
    out_dir = Path(out_dir)

    counts: dict[str, dict] = {}
    files: dict[str, dict] = {}
    for name, rows in (("train", splits["train"]), ("dev", splits["dev"])):
        pairs: list[dict] = []
        for rec in rows:
            pair = _sft_pair(rec, store)
            if pair is None:
                continue
            pairs.append({
                "id": f"sft-{rec['case_id']}",
                "schema": SCHEMA_SFT,
                "family": rec["family"], "split": name,
                "provenance": {"case_id": rec["case_id"],
                               "criterion": rec.get("acceptance", {})
                               .get("criterion")},
                **pair,
            })
        path = out_dir / f"ryko1000_sft_{name}.jsonl"
        _write_jsonl(path, pairs)
        files[name] = {"path": str(path), "n": len(pairs),
                       "sha256": _sha256(path)}
        counts[name] = {
            "cases": len(rows),
            "pairs": len(pairs),
            "by_family": _count_by(pairs, "family"),
        }
    store.close()

    leaked = sorted((train_ids | dev_ids) & test_ids)
    if leaked:
        raise ValueError(f"test-split leakage into training artifacts: "
                         f"{leaked[:5]}")
    manifest = {
        "schema": SCHEMA_SFT,
        "splits_used": ["train", "dev"],
        "test_split": "held out — never consumed by SFT/DPO builders",
        "files": files,
        "counts": counts,
        "leakage_check": {"leaked": len(leaked), "fatal": True},
    }
    (out_dir / "sft_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return manifest


def _count_by(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return dict(sorted(out.items()))


# -------------------------------------------------------------------- DPO
def _dpo_pairs(rec: dict, store: KnowledgeStore,
               books: list[dict]) -> list[dict]:
    """Preference pairs from ONE case's failure probes (train split only)."""
    family = rec["family"]
    src = (rec.get("sources") or [{}])[0]
    meta = rec.get("meta") or {}
    pairs: list[dict] = []

    def pid_of() -> str | None:
        if src.get("passage_id"):
            return src["passage_id"]
        st = rec.get("supporting_text") or [{}]
        if st and st[0].get("passage_id"):
            return st[0]["passage_id"]
        mpids = meta.get("passage_ids")
        return mpids[0] if mpids else None

    for probe in rec.get("failure_probes") or []:
        kind, must = probe.get("kind"), probe.get("must")
        base = {"schema": SCHEMA_DPO, "family": family,
                "probe_kind": kind, "must": must,
                "provenance": {"case_id": rec["case_id"]}}
        if kind == "wrong_attribution":
            pid = pid_of()
            if not pid:
                continue
            passage = store.passage(pid)
            if not passage:
                continue
            quote = _probe_quote(passage["text"])
            if quote is None:
                continue
            own = set(meta.get("book_ids") or [src.get("book_id")])
            wrong = next((b for b in books if b["book_id"] not in own),
                         None)
            if wrong is None:
                continue
            correct_title = (src.get("title_ar") or
                             passage.get("title_ar") or "")
            if not correct_title:
                continue
            pairs.append({**base,
                "id": f"dpo-{rec['case_id']}-wrong_attribution",
                "prompt": (f"من أي كتاب هذا المقطع؟\n«{quote}»\nالكتاب:"),
                "chosen": f" «{correct_title}»",
                "rejected": f" «{wrong['title_ar']}»"})
        elif kind == "splice":
            quote = rec.get("quote") or ""
            if not quote:
                continue
            title = src.get("title_ar") or ""
            pairs.append({**base,
                "id": f"dpo-{rec['case_id']}-splice",
                "prompt": ("وثّق الاقتباس الآتي وفق نواة المعرفة "
                           "(ryko-cite/1):\n"
                           f"«{quote}»\nالحكم:"),
                "chosen": "الاقتباس مركّب غير موجود في نواة المعرفة "
                          "— يُرفض ويُمتنع.",
                "rejected": f"مُوثَّق حرفياً في «{title}»."})
        elif kind == "evidenceless":
            if must == "abstain":
                pairs.append({**base,
                    "id": f"dpo-{rec['case_id']}-evidenceless",
                    "prompt": f"السؤال: {rec.get('question', '')}\nالجواب:",
                    "chosen": SFT_REFUSE,
                    "rejected": FABRICATED})
            else:  # must == reject (term-absent probe, mustalahat)
                term = meta.get("term")
                pid = pid_of()
                if not term or not pid:
                    continue
                passage = store.passage(pid)
                if not passage:
                    continue
                terms_meta = meta.get("foreign_term")
                foreign = terms_meta or term
                pairs.append({**base,
                    "id": f"dpo-{rec['case_id']}-evidenceless",
                    "prompt": (f"ما مدلول مصطلح «{foreign}» في هذا النص؟\n"
                               f"«{passage['text'][:160]}»\nالمدلول:"),
                    "chosen": "هذا النص لا يحوي المصطلح أصلاً — لا مدلول "
                              "يُستخرج منه ويُرفض السؤال.",
                    "rejected": "ورد المصطلح في النص بمعنى محدد يستقيم "
                                "الاستدلال به."})
    return pairs


def build_dpo(split_dir: Path, store_path: Path, out_dir: Path) -> dict:
    """Preference pairs from failure_probes — TRAIN split only.

    The must-answer (should_answer=True) cases contribute the
    **over-refusal counter-pair**: chosen = answer with citation,
    rejected = the wrong abstention — the direct treatment for the
    68.75% wrong-abstain weakness measured on RYKO-Bench.
    """
    store = KnowledgeStore(str(store_path), persist_vectors=False)
    books = [{"book_id": b["book_id"],
              "title_ar": (b.get("title_ar") or "").strip()}
             for b in store.iter_books()]
    splits = _load_split(Path(split_dir))
    test_ids = {r["case_id"] for r in splits["test"]}

    pairs: list[dict] = []
    for rec in splits["train"]:
        pairs.extend(_dpo_pairs(rec, store, books))
        # over-refusal counter-pairs from must-answer cases
        if rec["kind"] == "abstain" and rec.get("should_answer") is True:
            src = (rec.get("sources") or [{}])[0]
            answer = rec.get("expected_answer") or \
                str((rec.get("meta") or {}).get("originator") or "")
            if answer:
                chosen = f" {answer}"
                if src.get("title_ar"):
                    chosen += f" — السند: «{src['title_ar']}»."
                pairs.append({
                    "schema": SCHEMA_DPO,
                    "family": rec["family"],
                    "probe_kind": "over_refusal",
                    "must": "answer",
                    "id": f"dpo-{rec['case_id']}-over_refusal",
                    "prompt": f"السؤال: {rec.get('question', '')}\nالجواب:",
                    "chosen": chosen,
                    "rejected": "لا سند — أُمتنع.",
                    "provenance": {"case_id": rec["case_id"]},
                })
    store.close()

    leaked = sorted({p["provenance"]["case_id"] for p in pairs}
                    & test_ids)
    if leaked:
        raise ValueError(f"test-split leakage into DPO pairs: {leaked[:5]}")
    out_dir = Path(out_dir)
    path = out_dir / "ryko1000_dpo_train.jsonl"
    _write_jsonl(path, pairs)
    manifest = {
        "schema": SCHEMA_DPO,
        "source": "failure_probes of the train split (+ over-refusal "
                  "counter-pairs from must-answer cases)",
        "files": {"train": {"path": str(path), "n": len(pairs),
                            "sha256": _sha256(path)}},
        "counts": {"by_family": _count_by(pairs, "family"),
                   "by_probe_kind": _count_by(pairs, "probe_kind")},
        "leakage_check": {"leaked": len(leaked), "fatal": True},
        "test_split": "held out — never consumed by SFT/DPO builders",
    }
    (out_dir / "dpo_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return manifest


# --------------------------------------------------------------- DPO r2
def _ling_distractor(rec: dict, authors: list[dict], books: list[dict],
                     cont_pool: dict[str, list[dict]],
                     rule_pool: list[str]) -> str:
    """Wrong originator for a ling case's wrong_attribution probe.

    Closed-catalog only (the committed author/book catalog and the closed
    rule-originator pool of the seed itself) — no invented attribution.
    Deterministic first-hit in fixed catalog order, mirroring the round-1
    ``next(b for b in books if b["book_id"] not in own)`` protocol.
    """
    meta = rec.get("meta") or {}
    pattern = meta.get("ling_pattern")
    ea = rec["expected_answer"]
    if pattern == "author":
        own = meta.get("author_id")
        if not own:
            raise ValueError(f"{rec['case_id']}: author case without "
                             f"meta.author_id")
        wrong = next((a for a in authors if a["author_id"] != own), None)
        if wrong is None:
            raise ValueError(f"{rec['case_id']}: no distractor author")
        return str(wrong["name_ar"])
    if pattern == "book":
        own = meta.get("book_id")
        if not own:
            raise ValueError(f"{rec['case_id']}: book case without "
                             f"meta.book_id")
        wrong = next((b for b in books if b["book_id"] != own
                      and (b.get("title_ar") or "") != ea), None)
        if wrong is None:
            raise ValueError(f"{rec['case_id']}: no distractor book")
        return str(wrong["title_ar"])
    if pattern == "continuation":
        own = meta.get("book_id")
        if not own:
            raise ValueError(f"{rec['case_id']}: continuation case without "
                             f"meta.book_id")
        others = cont_pool.get(rec["family"]) or []
        wrong = next((c for c in others
                      if c["case_id"] != rec["case_id"]
                      and (c.get("meta") or {}).get("book_id") != own
                      and c["expected_answer"] != ea), None)
        if wrong is None:
            raise ValueError(f"{rec['case_id']}: no distractor continuation")
        return str(wrong["expected_answer"])
    if pattern == "rule":
        wrong = next((o for o in rule_pool if o != ea), None)
        if wrong is None:
            raise ValueError(f"{rec['case_id']}: no distractor originator")
        return wrong
    raise ValueError(f"{rec['case_id']}: unknown ling_pattern {pattern!r}")


def build_dpo_r2(seed_path: Path, split_dir: Path, store_path: Path,
                 round1_path: Path, out_dir: Path) -> dict:
    """Round-2 preference bundle (v1.23.0): the FROZEN round-1 file plus
    the linguistic families' embedded probes — the pre-approved treatment
    published with v1.22.0 (``docs/results/ryko1000_ling_expansion.md``).

    New pairs (all deterministic, closed-catalog distractors):
    1. ``wrong_attribution`` — one per linguistic case (300): MCQ-canonical
       prompt; chosen = the documented answer; rejected = a wrong
       originator from the committed author/book catalog (author/book/
       continuation patterns) or the closed rule-originator pool (rule).
    2. ``over_refusal`` — the dev-split must-answer abstain counter-pairs
       (dev is inside the round-1 SFT contract; round-1 DPO used train
       only, round 2 extends the over-refusal treatment to dev).

    MEASURED ITERATION (r2a): the first composition also emitted one
    anti-refusal pair per linguistic case (300 ``over_refusal`` pairs).
    It cut over-refusal 0.6875 -> 0.5938 but BROKE the correct-abstain
    gate (1.0 -> 0.9412) — the anti-refusal pressure inverted the
    evidenceless/anti-refusal balance round 1 proved (143:27).  The
    composition was corrected: the linguistic probes stay, the flood is
    dropped; the round-1 bundle's own 27 + the dev 2 carry the over-
    refusal treatment.  Measured r2a report:
    ``docs/results/ryko1000_eval_dpo_r2a.md``.

    The round-1 rows are copied BYTE-IDENTICALLY (the file's sha256 is
    pinned and verified before any write); the original test bucket stays
    a hard exclusion (leakage fatal).
    """
    seed_path = Path(seed_path)
    seed_sha = _sha256(seed_path)
    if seed_sha != SEED_SHA_R2:
        raise ValueError(
            "seed drift: sha256 "
            f"{seed_sha[:16]}… != pinned {SEED_SHA_R2[:16]}… — the v1.22.0 "
            "seed is an append-only frozen artifact")
    r1_path = Path(round1_path)
    r1_sha = _sha256(r1_path)
    if r1_sha != ROUND1_SHA_R2:
        raise ValueError(
            "round-1 pairs drift: sha256 "
            f"{r1_sha[:16]}… != pinned {ROUND1_SHA_R2[:16]}… — the round-1 "
            "bundle is frozen and must not be regenerated")
    r1_bytes = r1_path.read_bytes()
    r1_rows = sum(1 for l in r1_bytes.decode("utf-8").splitlines()
                  if l.strip())
    # the 792-count invariant is a REAL-bundle contract; synthetic tests
    # monkeypatch the pin, so the count check follows the pin
    if ROUND1_SHA_R2 == ROUND1_SHA_REAL and r1_rows != 792:
        raise ValueError(f"round-1 bundle must hold exactly 792 pairs, "
                         f"found {r1_rows}")

    store = KnowledgeStore(str(store_path), persist_vectors=False)
    authors = list(store.iter_authors())
    books = list(store.iter_books())
    store.close()
    if not authors or not books:
        raise ValueError("empty author/book catalog in the committed store")

    cases = _load(seed_path)
    ling = [c for c in cases
            if c["family"] in LING_FAMILIES
            and (c.get("meta") or {}).get("ling_pattern")]
    # uniformity invariant: every linguistic family contributes the same
    # count (100 on the real seed — the manifest pins the exact totals)
    per_fam: dict[str, int] = {}
    for c in ling:
        per_fam[c["family"]] = per_fam.get(c["family"], 0) + 1
    if not ling or len(set(per_fam.values())) != 1:
        raise ValueError(f"uneven linguistic families: {per_fam!r}")

    # closed rule-originator pool across the whole seed (deterministic)
    rule_pool = sorted({c["expected_answer"] for c in ling
                        if (c.get("meta") or {}).get("ling_pattern")
                        == "rule"})
    cont_pool: dict[str, list[dict]] = {}
    for c in ling:
        if (c.get("meta") or {}).get("ling_pattern") == "continuation":
            cont_pool.setdefault(c["family"], []).append(c)
    for fam in cont_pool:
        cont_pool[fam].sort(key=lambda c: c["case_id"])

    new_pairs: list[dict] = []
    for rec in ling:
        probes = rec.get("failure_probes") or []
        if len(probes) != 1 or probes[0].get("kind") != "wrong_attribution":
            raise ValueError(f"{rec['case_id']}: expected exactly one "
                             f"wrong_attribution probe, got {probes!r}")
        if not rec.get("expected_answer"):
            raise ValueError(f"{rec['case_id']}: empty expected_answer")
        family = rec["family"]
        prompt = f"السؤال: {rec.get('question', '')}\nالجواب:"
        chosen = f" {rec['expected_answer']}"
        # 1) wrong-attribution pair (the embedded probe)
        # (single line: a newline inside a replacement field is a
        #  SyntaxError on Python 3.11 — CI runs 3.11, see run #67)
        distractor = _ling_distractor(
            rec, authors, books, cont_pool, rule_pool)
        rejected = f" {distractor}"
        if chosen == rejected:
            raise ValueError(f"{rec['case_id']}: chosen == rejected")
        new_pairs.append({
            "schema": SCHEMA_DPO, "family": family,
            "probe_kind": "wrong_attribution", "must": "reject",
            "provenance": {"case_id": rec["case_id"],
                           "ling_pattern": rec["meta"]["ling_pattern"]},
            "id": f"dpo-{rec['case_id']}-wrong_attribution",
            "prompt": prompt, "chosen": chosen, "rejected": rejected,
        })

    # dev-split must-answer abstain counter-pairs (over-refusal treatment,
    # extended to dev — the round-1 SFT contract already consumes dev)
    dev_path = Path(split_dir) / "ryko1000_dev.jsonl"
    test_path = Path(split_dir) / "ryko1000_test.jsonl"
    test_ids = {json.loads(l)["case_id"] for l in
                test_path.read_text(encoding="utf-8").splitlines()
                if l.strip()}
    for line in dev_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["kind"] == "abstain" and rec.get("should_answer") is True:
            src = (rec.get("sources") or [{}])[0]
            answer = rec.get("expected_answer") or \
                str((rec.get("meta") or {}).get("originator") or "")
            if not answer:
                continue
            chosen = f" {answer}"
            if src.get("title_ar"):
                chosen += f" — السند: «{src['title_ar']}»."
            new_pairs.append({
                "schema": SCHEMA_DPO, "family": rec["family"],
                "probe_kind": "over_refusal", "must": "answer",
                "id": f"dpo-{rec['case_id']}-over_refusal",
                "prompt": f"السؤال: {rec.get('question', '')}\nالجواب:",
                "chosen": chosen, "rejected": DPO_REFUSE,
                "provenance": {"case_id": rec["case_id"], "split": "dev"},
            })

    leaked = sorted({p["provenance"]["case_id"] for p in new_pairs}
                    & test_ids)
    if leaked:
        raise ValueError(f"test-split leakage into DPO round-2 pairs: "
                         f"{leaked[:5]}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ryko1000_dpo_train_r2.jsonl"
    path.write_bytes(r1_bytes + b"".join(
        (json.dumps(p, ensure_ascii=False) + "\n").encode("utf-8")
        for p in new_pairs))
    manifest = {
        "schema": SCHEMA_DPO_R2,
        "source": "frozen round-1 bundle + linguistic families' embedded "
                  "wrong_attribution probes (300 cases) + over-refusal "
                  "counter-pairs (dev must-answer) — the v1.22.0 "
                  "pre-approved treatment; the r2a anti-refusal flood was "
                  "measured to break the correct-abstain gate and dropped "
                  "(docs/results/ryko1000_eval_dpo_r2a.md)",
        "files": {"train": {"path": str(path),
                            "n": r1_rows + len(new_pairs),
                            "sha256": _sha256(path)}},
        "sources": {
            "round1": {"path": str(r1_path), "n": r1_rows, "sha256": r1_sha},
            "seed": {"path": str(seed_path),
                     "n": len(cases), "sha256": seed_sha},
            "store_catalog": {"path": str(store_path),
                              "authors": len(authors),
                              "books": len(books)},
        },
        "counts": {
            "round1_pairs": r1_rows,
            "new_pairs": len(new_pairs),
            "by_family": _count_by(new_pairs, "family"),
            "by_probe_kind": _count_by(new_pairs, "probe_kind"),
        },
        "leakage_check": {"leaked": len(leaked), "fatal": True,
                          "scope": "original 10-family test bucket"},
        "test_split": "held out — never consumed by SFT/DPO builders",
    }
    (out_dir / "dpo_manifest_r2.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return manifest


# --------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ryko_sft_prep",
        description="RYKO-1000 SFT/DPO preparation (phases 5 + 9).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_split = sub.add_parser("split", help="85/7/8 per-family split")
    p_split.add_argument("--seed", type=Path,
                         default=Path("docs/benchmarks/ryko1000_seed.jsonl"))
    p_split.add_argument("--out", type=Path,
                         default=Path("data/training/ryko1000_split"))
    p_split.add_argument("--train", type=int, default=N_TRAIN)
    p_split.add_argument("--dev", type=int, default=N_DEV)
    p_split.add_argument("--test", type=int, default=N_TEST)

    p_sft = sub.add_parser("sft", help="prompt/completion pairs (train+dev)")
    p_sft.add_argument("--split", type=Path,
                       default=Path("data/training/ryko1000_split"))
    p_sft.add_argument("--store", type=Path,
                       default=Path("data/knowledge/ryko.sqlite"))
    p_sft.add_argument("--out", type=Path, default=Path("data/training"))

    p_dpo = sub.add_parser("dpo", help="preference pairs from failure probes")
    p_dpo.add_argument("--split", type=Path,
                       default=Path("data/training/ryko1000_split"))
    p_dpo.add_argument("--store", type=Path,
                       default=Path("data/knowledge/ryko.sqlite"))
    p_dpo.add_argument("--out", type=Path, default=Path("data/training"))

    p_r2 = sub.add_parser(
        "dpo-r2",
        help="round-2 bundle: frozen round-1 + linguistic probes + "
             "over-refusal counter-pairs (v1.23.0)")
    p_r2.add_argument("--seed", type=Path,
                      default=Path("docs/benchmarks/ryko1000_seed.jsonl"))
    p_r2.add_argument("--split", type=Path,
                      default=Path("data/training/ryko1000_split"))
    p_r2.add_argument("--store", type=Path,
                      default=Path("data/knowledge/ryko.sqlite"))
    p_r2.add_argument("--round1", type=Path,
                      default=Path("data/training/ryko1000_dpo_train.jsonl"))
    p_r2.add_argument("--out", type=Path, default=Path("data/training"))

    args = parser.parse_args(argv)
    if args.cmd == "split":
        manifest = split_ryko1000(args.seed, args.out,
                                  proportions=(args.train, args.dev,
                                               args.test))
        print(f"[split] train={manifest['files']['train']['n']} "
              f"dev={manifest['files']['dev']['n']} "
              f"test={manifest['files']['test']['n']} "
              f"-> {args.out}")
    elif args.cmd == "sft":
        manifest = build_sft(args.split, args.store, args.out)
        print(f"[sft] train pairs={manifest['files']['train']['n']} "
              f"dev pairs={manifest['files']['dev']['n']} "
              f"-> {args.out}")
    else:
        if args.cmd == "dpo-r2":
            manifest = build_dpo_r2(args.seed, args.split, args.store,
                                    args.round1, args.out)
            print(f"[dpo-r2] pairs={manifest['files']['train']['n']} "
                  f"(round1 792 + new {manifest['counts']['new_pairs']}) "
                  f"-> {args.out}")
        else:
            manifest = build_dpo(args.split, args.store, args.out)
            print(f"[dpo] pairs={manifest['files']['train']['n']} "
                  f"-> {args.out}")
        print(json.dumps(manifest["counts"]["by_probe_kind"],
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
