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
SPLIT_SEED = 20261118
N_TRAIN, N_DEV, N_TEST = 85, 7, 8
SFT_REFUSE = "لا سند موثقاً في نواة المعرفة للإجابة على هذا السؤال — أُمتنع."
FABRICATED = "الجواب: نعم، بحسب المصنفات المثبتة في نواة المعرفة."


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
        manifest = build_dpo(args.split, args.store, args.out)
        print(f"[dpo] pairs={manifest['files']['train']['n']} "
              f"-> {args.out}")
        print(json.dumps(manifest["counts"]["by_probe_kind"],
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
