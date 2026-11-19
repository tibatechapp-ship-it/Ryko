#!/usr/bin/env python3
"""Build the bundle DPO preference pairs (أفضليات الحزمة — v1.26.0).

Two deterministic pair files for the frugal balance rounds over the argrel
proof shard:

* ``bundle_dpo_pairs_ling.jsonl`` — linguistics-only balance round:
  the committed v1.23.0 DPO-r2 linguistics preferences (irab/sarf/balagha,
  minus the 7 rows quoting a probe case text) + the 2 over-refusal
  counter-pairs + NEW attribution preferences derived from
  ``ling_sft_{train,val}.jsonl`` (chosen = store gold, rejected = a plausible
  wrong attribution picked deterministically from the store catalog);
* ``bundle_dpo_pairs_mixed.jsonl`` — the same, plus NEW argument-relation
  preferences derived from ``argrel_sft_{train,val}.jsonl`` (chosen = gold
  relation, rejected = a wrong verb from the closed 8-verb vocabulary) so a
  DPO balance round cannot wash out علاقة-حجاج.

Probe-leakage gate is fatal (same policy as the bundle SFT pack).

Usage::

    PYTHONPATH=src python scripts/build_bundle_dpo.py \\
        --store data/knowledge/ryko.sqlite \\
        --out data/training
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ryko_lm.knowledge.store import KnowledgeStore  # noqa: E402

SECTION_LABELS: tuple[tuple[str, str], ...] = (
    ("nahw", "النحو"),
    ("sarf", "الصرف"),
    ("balagha", "البلاغة"),
    ("maajim", "المعاجم"),
    ("fiqh-lugha", "فقه اللغة واللسانيات"),
    ("adab", "الأدب"),
    ("munhaj", "المقررات والمناهج"),
)
_LABELS = [lbl for _, lbl in SECTION_LABELS]

# Closed relation-verb vocabulary — mirrors the علاقة-حجاج bench options.
RELATION_VERBS = ("يدعم", "يعترض", "يرد على", "ينقل عن", "يرجّح",
                  "يلخّص", "يعيد الصياغة", "يناقض")

LING_R2_FAMILIES = ("irab", "sarf", "balagha", "yaqin")
ALL_R2_FAMILIES = LING_R2_FAMILIES + ("akhta", "fahm", "ikhtilaf",
                                      "itirad", "multisource", "naql",
                                      "tarjih")
SCHEMA = "ryko-bundle-dpo/1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm(text: str) -> str:
    return re.sub(r"[\s\u064B-\u0652\u0670«»(){}.,،؛:!?\-—]+", "", text)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _probe_exclusions(probes_path: Path) -> dict:
    claim_ids, book_ids, passage_ids, quotes = set(), set(), set(), set()
    for case in _load_jsonl(probes_path):
        meta = case.get("meta", {})
        if meta.get("claim_id"):
            claim_ids.add(meta["claim_id"])
        if meta.get("book_id"):
            book_ids.add(meta["book_id"])
        if meta.get("passage_id"):
            passage_ids.add(meta["passage_id"])
        for quoted in re.findall(r"«(.+?)»", case.get("question", ""), re.S):
            quotes.add(_norm(quoted))
    return {"claim_ids": claim_ids, "book_ids": book_ids,
            "passage_ids": passage_ids, "quotes": quotes,
            "title_norms": {q for q in quotes if len(q) < 25}}


def _quote_hits(prompt: str, excl: dict) -> str | None:
    np_ = _norm(prompt)
    quoted = [_norm(q) for q in re.findall(r"«(.+?)»", prompt, re.S)]
    for q in excl["quotes"]:
        if not q:
            continue
        if len(q) >= 25:
            if q in np_:
                return "long-quote"
        elif any(q == nq for nq in quoted):
            return "title"
    return None


def _prov_hits(row: dict, excl: dict) -> str | None:
    prov = json.dumps(row.get("provenance", {}), ensure_ascii=False)
    for key, ids in (("claim_id", excl["claim_ids"]),
                     ("book_id", excl["book_ids"]),
                     ("passage_id", excl["passage_ids"])):
        for i in ids:
            if i and i in prov:
                return key
    return None


def _swap_verb(completion: str, wrong: str) -> str:
    """Replace the leading relation verb (may be multi-word) of an argrel
    completion — longest-prefix match against the closed vocabulary."""
    t = completion.strip()
    for verb in sorted(RELATION_VERBS, key=len, reverse=True):
        if t.startswith(verb):
            rest = t[len(verb):]
            return " " + wrong + rest
    raise ValueError(f"argrel completion without a closed-vocabulary verb: "
                     f"{completion[:60]}")


def _wrong_verb(gold_completion: str, index: int) -> str:
    """Pick a wrong verb from the closed vocabulary, never the gold one
    (gold verb matched by longest prefix — verbs may be multi-word)."""
    t = gold_completion.strip()
    gold_verb = next(v for v in sorted(RELATION_VERBS, key=len, reverse=True)
                     if t.startswith(v))
    pool = [v for v in RELATION_VERBS if v != gold_verb]
    return pool[index % len(pool)]


def build(store: KnowledgeStore, probes_path: Path, ling_train: Path,
          ling_val: Path, argrel_train: Path, argrel_val: Path,
          dpo_r2: Path, r2_mode: str = "ling", ling_repeat: int = 1,
          argrel_repeat: int = 1) -> dict:
    excl = _probe_exclusions(probes_path)

    # Catalog pools for plausible wrong answers (store-derived, never guessed).
    authors = sorted({r["author"].strip()
                      for r in store.conn.execute(
                          "SELECT a.name_ar AS author FROM authors a "
                          "JOIN books b ON b.author_id = a.author_id "
                          "WHERE b.source_id='lugha' AND a.name_ar IS NOT NULL")
                      if r["author"].strip()})
    originators = sorted({json.loads(meta or "{}")["originator"]
                          for cid, meta in store.conn.execute(
                              "SELECT claim_id, meta FROM claims")
                          if (json.loads(meta or "{}").get("layer") == "lugha"
                              and json.loads(meta or "{}").get("originator"))})

    ling_rows = _load_jsonl(ling_train) + _load_jsonl(ling_val)
    argrel_rows = _load_jsonl(argrel_train) + _load_jsonl(argrel_val)

    ling_prefs: list[dict] = []
    for i, row in enumerate(ling_rows):
        gold = row["completion"]
        fam = row["family"]
        if fam == "نسبة-المؤلف":
            pool = [a for a in authors if a != gold.lstrip(" ")]
            wrong = pool[i % len(pool)] if pool else None
        elif fam == "باب-اللغة":
            pool = [l for l in _LABELS if l != gold.lstrip(" ")]
            wrong = pool[i % len(pool)] if pool else None
        else:  # نسبة-القاعدة-اللغوية
            pool = [o for o in originators if o != gold.lstrip(" ")]
            wrong = pool[i % len(pool)] if pool else None
        if not wrong:
            continue
        ling_prefs.append({
            "schema": SCHEMA, "id": f"pref-{row['id']}",
            "family": row["family"],
            "provenance": {**row.get("provenance", {}),
                           "source_pair": row["id"]},
            "prompt": row["prompt"], "chosen": gold, "rejected": " " + wrong,
            "split": row.get("split", "train"),
        })

    argrel_prefs: list[dict] = []
    for i, row in enumerate(argrel_rows):
        gold = row["completion"]
        wrong = _wrong_verb(gold, i)
        argrel_prefs.append({
            "schema": SCHEMA, "id": f"pref-{row['id']}",
            "family": row["family"],
            "provenance": {**row.get("provenance", {}),
                           "source_pair": row["id"]},
            "prompt": row["prompt"],
            "chosen": gold, "rejected": _swap_verb(gold, wrong),
            "split": row.get("split", "train"),
        })

    # Committed r2 pairs (ling-only or full), minus probe overlap.
    r2_families = LING_R2_FAMILIES if r2_mode == "ling" else ALL_R2_FAMILIES
    kept_r2: list[dict] = []
    r2_excluded: list[str] = []
    for row in _load_jsonl(dpo_r2):
        if row.get("family") not in r2_families:
            continue
        if _quote_hits(row["prompt"], excl):
            r2_excluded.append(row["id"])
            continue
        kept_r2.append({
            "schema": SCHEMA, "id": row["id"], "family": row["family"],
            "probe_kind": row.get("probe_kind"),
            "provenance": row.get("provenance", {}),
            "prompt": row["prompt"], "chosen": row["chosen"],
            "rejected": row["rejected"], "split": "train",
        })

    ling_file = ling_prefs * ling_repeat + kept_r2
    mixed_file = (ling_prefs * ling_repeat
                  + argrel_prefs * argrel_repeat + kept_r2)

    # Fatal leakage gate over every emitted prompt.
    leaks: list[str] = []
    for row in ling_file + argrel_prefs:
        tag = _quote_hits(row["prompt"], excl) or _prov_hits(row, excl)
        if tag:
            leaks.append(f"{tag}: {row['id']}")
    if leaks:
        raise SystemExit("[error] probe leakage in DPO pairs:\n  "
                         + "\n  ".join(leaks[:10]))

    for rows in (ling_file, argrel_prefs, mixed_file):
        rows.sort(key=lambda r: r["id"])

    return {"ling_prefs": ling_prefs, "argrel_prefs": argrel_prefs,
            "kept_r2": kept_r2, "r2_excluded": r2_excluded,
            "ling_file": ling_file, "mixed_file": mixed_file,
            "authors_pool": len(authors), "originators_pool": len(originators),
            "r2_mode": r2_mode, "ling_repeat": ling_repeat,
            "argrel_repeat": argrel_repeat}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path,
                    default=Path("data/knowledge/ryko.sqlite"))
    ap.add_argument("--probes", type=Path,
                    default=Path("docs/benchmarks/ling_probes.jsonl"))
    ap.add_argument("--ling-train", type=Path,
                    default=Path("data/training/ling_sft_train.jsonl"))
    ap.add_argument("--ling-val", type=Path,
                    default=Path("data/training/ling_sft_val.jsonl"))
    ap.add_argument("--argrel-train", type=Path,
                    default=Path("data/training/argrel_sft_train.jsonl"))
    ap.add_argument("--argrel-val", type=Path,
                    default=Path("data/training/argrel_sft_val.jsonl"))
    ap.add_argument("--dpo-r2", type=Path,
                    default=Path("data/training/ryko1000_dpo_train_r2.jsonl"))
    ap.add_argument("--r2-mode", type=str, default="ling",
                    choices=["ling", "full"])
    ap.add_argument("--ling-repeat", type=int, default=1)
    ap.add_argument("--argrel-repeat", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("data/training"))
    args = ap.parse_args(argv)

    store = KnowledgeStore(str(args.store))
    built = build(store, args.probes, args.ling_train, args.ling_val,
                  args.argrel_train, args.argrel_val, args.dpo_r2,
                  args.r2_mode, args.ling_repeat, args.argrel_repeat)
    store.close()

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False,
                                    sort_keys=True) + "\n")

    ling_path = args.out / "bundle_dpo_pairs_ling.jsonl"
    mixed_path = args.out / "bundle_dpo_pairs_mixed.jsonl"
    write(ling_path, built["ling_file"])
    write(mixed_path, built["mixed_file"])

    fam_ling = Counter(r["family"] for r in built["ling_prefs"])
    fam_argrel = Counter(r["family"] for r in built["argrel_prefs"])
    manifest = {
        "schema": "ryko-bundle-dpo-manifest/1",
        "files": {
            "ling": {"path": str(ling_path), "n": len(built["ling_file"]),
                     "sha256": _sha256(ling_path)},
            "mixed": {"path": str(mixed_path), "n": len(built["mixed_file"]),
                      "sha256": _sha256(mixed_path)},
        },
        "new_ling_prefs": dict(sorted(fam_ling.items())),
        "new_argrel_prefs": dict(sorted(fam_argrel.items())),
        "kept_r2": {"n": len(built["kept_r2"]),
                    "mode": built["r2_mode"],
                    "families": "all" if built["r2_mode"] == "full"
                    else sorted(LING_R2_FAMILIES),
                    "excluded_probe_overlap": len(built["r2_excluded"]),
                    "excluded_ids": built["r2_excluded"]},
        "repeats": {"ling": built["ling_repeat"],
                    "argrel": built["argrel_repeat"]},
        "pools": {"authors": built["authors_pool"],
                  "originators": built["originators_pool"]},
        "sources": {str(p): _sha256(Path(p)) for p in (
            args.probes, args.ling_train, args.ling_val,
            args.argrel_train, args.argrel_val, args.dpo_r2)},
        "leakage_check": {"probe_leaks": 0, "policy": "fatal gate; 0 tolerated"},
        "determinism": "stable sort by id, store-derived wrong answers, "
                       "no RNG",
        "note": "v1.26.0 build corrects the argrel pref rejected-verb bug "
                "(some rows were no-op chosen==rejected in the first "
                "build used by experiments 5/6 — the defensive signal "
                "holds with the fully-effective 105 prefs)",
    }
    (args.out / "bundle_dpo_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    print(f"[bundle-dpo] ling={len(built['ling_file'])} "
          f"(new prefs {len(built['ling_prefs'])}x{built['ling_repeat']} "
          f"+ r2[{built['r2_mode']}] kept {len(built['kept_r2'])}, "
          f"probe-overlap excluded {len(built['r2_excluded'])}) | "
          f"mixed={len(built['mixed_file'])} (+argrel prefs "
          f"{len(built['argrel_prefs'])}x{built['argrel_repeat']}) "
          f"| leakage 0")
    print(f"[bundle-dpo] ling prefs by family: "
          f"{dict(sorted(fam_ling.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
