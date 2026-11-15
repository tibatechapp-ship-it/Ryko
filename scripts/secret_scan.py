#!/usr/bin/env python3
"""Secret gate CLI — thin wrapper so the gate runs before `pip install -e .`.

CI already runs gitleaks after push; this gate exists to stop a secret
**before** it leaves the machine::

    python scripts/secret_scan.py --tree .        # whole repo
    python scripts/secret_scan.py --diff HEAD~1..HEAD
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ryko_lm.security.secret_scan import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
