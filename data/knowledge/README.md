# Knowledge Core — seed store

`ryko.sqlite` is the **committed منقحة-core** of the RYKO Knowledge Core
(`docs/KNOWLEDGE_CORE.md`, `docs/ISLAMIC_THOUGHT.md`): the v1.3.0 Shamela
seed (10 books, 11,674 passages, 3.85M chars) **plus** 13 curated classics
of the four منقحة schools — 23 books, 77,141 passages, ~30M characters,
31 school-attributed ideas.

`ryko-full.sqlite` (gitignored, ~250 MB) additionally carries **all** 31
in-scope fikr books (41 books / 227,062 passages / 91.9M chars) — rebuild
locally with `make fikr-collect && make fikr-build-full`.

| file | purpose |
|---|---|
| `ryko.sqlite` | committed منقحة-core store (schema v1) |
| `ryko.receipt.json` | signed provenance receipt — verify offline: `python -m ryko_lm.provenance verify data/knowledge/ryko.receipt.json` |
| `fikr_manifest.json` | منقحة thought map (v2): policy, 4 schools, 40 books, 31 ideas, reading order |
| `ryko-full.sqlite` | full local store (not committed) |

The raw collector JSONL is **not** committed (re-fetchable, see the command
in `docs/KNOWLEDGE_CORE.md` §6).  Ingest is idempotent — re-running reports
zero new passages.

Rights: Shamela v4 is MIT-declared by its dataset publisher; editorial
rights of individual editions remain with their holders.  The declaration
lives inside the store itself (`sources` table) — rights travel with the
data, not with the README.
