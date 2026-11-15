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
`make econom-collect && make econom-build-full`.  The store now holds
3 sources / 30 books / 82,216 passages / 87 claims (31 fikr + 56 econom).

| file | purpose |
|---|---|
| `ryko.sqlite` | committed منقحة-core store (schema v1) |
| `ryko.receipt.json` | signed provenance receipt — verify offline: `python -m ryko_lm.provenance verify data/knowledge/ryko.receipt.json` |
| `fikr_manifest.json` | منقحة thought map (v2): policy, 4 schools, 40 books, 31 ideas, reading order |
| `econom_manifest.json` | الاقتصاد والحقوق map (v2): policy, 4 sections, 25 books, 56 ideas, reading order, scale |
| `ryko-full.sqlite` | full local store (not committed) |

The raw collector JSONL is **not** committed (re-fetchable, see the command
in `docs/KNOWLEDGE_CORE.md` §6).  Ingest is idempotent — re-running reports
zero new passages.

**Commit rule (v1.9.0 lesson):** `scripts/vacuum_knowledge.py` must run
as the *last* step before any `git add` of the DB — running the test
suite or the server re-persists the derived `passage_vec` table and
grows the file past GitHub's 100 MB limit (89.2 MB → ~194 MB).  After
vacuum, regenerate `ryko.receipt.json` (no search in between) so the
receipt sha256 matches the committed bytes.

Rights: Shamela v4 is MIT-declared by its dataset publisher; editorial
rights of individual editions remain with their holders.  The declaration
lives inside the store itself (`sources` table) — rights travel with the
data, not with the README.
