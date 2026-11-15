# Knowledge Core — seed store

`ryko.sqlite` is the **committed seed** of the RYKO Knowledge Core
(`docs/KNOWLEDGE_CORE.md`): 10 real heritage books from Shamela v4 with
page-level provenance — 11,674 passages, 3.85M characters.

| file | purpose |
|---|---|
| `ryko.sqlite` | authoritative local store (schema v1) |
| `ryko.receipt.json` | signed provenance receipt — verify offline: `python -m ryko_lm.provenance verify data/knowledge/ryko.receipt.json` |

The raw collector JSONL is **not** committed (re-fetchable, see the command
in `docs/KNOWLEDGE_CORE.md` §6).  Ingest is idempotent — re-running reports
zero new passages.

Rights: Shamela v4 is MIT-declared by its dataset publisher; editorial
rights of individual editions remain with their holders.  The declaration
lives inside the store itself (`sources` table) — rights travel with the
data, not with the README.
