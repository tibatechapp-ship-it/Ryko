# Knowledge Core — seed store

`ryko.sqlite` is the **committed منقحة-core** of the RYKO Knowledge Core
(`docs/KNOWLEDGE_CORE.md`, `docs/ISLAMIC_THOUGHT.md`): the v1.3.0 Shamela
seed (10 books, 11,674 passages, 3.85M chars) **plus** 13 curated classics
of the four منقحة schools — 23 books, 77,141 passages, ~30M characters,
31 school-attributed ideas.

`ryko-full.sqlite` (gitignored, ~250 MB) additionally carries **all** 31
in-scope fikr books (41 books / 227,062 passages / 91.9M chars) — rebuild
locally with `make fikr-collect && make fikr-build-full`.

Since v1.9.0 the committed core also carries the **الاقتصاد الإسلامي
والحقوق** layer (kind `ryko-econom-manifest` → source `econom`):
11 collected books (14,335 passages, 5.7M chars), of which 7 curated
core books (~5,075 passages) stay in the committed store — see
`docs/ECONOMICS_RIGHTS.md`; rebuild the full variant with
`make econom-collect && make econom-build-full`.

Since v1.16.0 the committed core also carries the **التراث الشامل
المُنتقى** layer (kind `ryko-turath-manifest` → source `turath`): the SAME
wide stratified selection the corpus pipeline merged into v5.2
(`collect_shamela --per-cat 12 --max-books 600 --max-file-mb 3
--keep-pages`), imported with page-level provenance under a
budgeted-breadth policy — **28 committed books (8,233 passages / 2.5M
chars, 28 of 40 Shamela categories)** inside a 2.5M-char TEXT budget that
keeps the committed DB ≤ ~96 MB (GitHub's 100 MB blob limit).  The FULL
tier carries all **358 collected books (373,543 passages / 127.6M chars)**
in the gitignored `ryko-wide-full.sqlite` with its committed receipt —
rebuild with `make turath-collect && make turath-build-full`.  The layer
attributes **no claims**: attributed ideas stay the exclusive discipline
of the two منقحة layers.  The store now holds 4 sources / 58 books /
90,449 passages / 87 claims (31 fikr + 56 econom) / 35 relations.

| file | purpose |
|---|---|
| `ryko.sqlite` | committed منقحة-core store (schema v1) |
| `ryko.receipt.json` | signed provenance receipt — verify offline: `python -m ryko_lm.provenance verify data/knowledge/ryko.receipt.json` |
| `fikr_manifest.json` | منقحة thought map (v2): policy, 4 schools, 40 books, 31 ideas, reading order |
| `econom_manifest.json` | الاقتصاد والحقوق map (v2): policy, 4 sections, 25 books, 56 ideas, reading order, scale |
| `turath_manifest.json` | التراث الشامل map (v1.16.0): selection policy, budget, 28 committed books, per-book provenance |
| `turath_manifest_full.json` | full-tier map: all 358 collected books (receipt committed, DB rebuildable) |
| `ryko-full.sqlite` | full local store (not committed) |
| `ryko-wide-full.sqlite` | full turath store (not committed, ~355 MB) |

The raw collector JSONL is **not** committed (re-fetchable, see the command
in `docs/KNOWLEDGE_CORE.md` §6).  Ingest is idempotent — re-running reports
zero new passages.

**Commit rule (v1.9.0 lesson):** `scripts/vacuum_knowledge.py` must run
as the *last* step before any `git add` of the DB — running the test
suite or the server re-persists the derived `passage_vec` table and
grows the file past GitHub's 100 MB limit (89.2 MB → ~194 MB).  After
vacuum, regenerate `ryko.receipt.json` (no search in between) so the
receipt sha256 matches the committed bytes.

**Byte-neutrality (v1.16.0 hardening):** read-only consumers no longer
dirty the committed DB at all — CLI read commands and the bench runner
open with `persist_vectors=False`, `verify_quote` gained
`write_status=False` for evaluation, and `KnowledgeStore` only rewrites
schema/meta when the schema actually changed.  A full bench run over the
committed DB now leaves its sha256 byte-identical (verified in-repo).

Rights: Shamela v4 is MIT-declared by its dataset publisher; editorial
rights of individual editions remain with their holders.  The declaration
lives inside the store itself (`sources` table) — rights travel with the
data, not with the README.
