"""Drop derived passage_vec + VACUUM the knowledge DB (repo-copy hygiene).

The committed copy of ``data/knowledge/ryko.sqlite`` ships with an EMPTY
``passage_vec`` table (derived data — the hybrid index rebuilds and
re-persists vectors automatically on first search, ~10s for 66K passages).
Running the test-suite or the server locally re-persists them and grows the
file from ~83MB to ~170MB, which would break the next push (GitHub's 100MB
limit).  Run this script before any ``git add`` that includes the DB::

    python scripts/vacuum_knowledge.py [--db data/knowledge/ryko.sqlite]
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/knowledge/ryko.sqlite")
    args = parser.parse_args()

    path = Path(args.db)
    if not path.exists():
        print(f"[vacuum] no DB at {path}")
        return 1
    db = sqlite3.connect(path)
    n = db.execute("SELECT COUNT(*) FROM passage_vec").fetchone()[0]
    db.execute("DELETE FROM passage_vec")
    db.commit()
    db.execute("VACUUM")
    db.close()
    print(f"[vacuum] dropped {n} derived vectors, vacuumed {path} "
          f"({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
