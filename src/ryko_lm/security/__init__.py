"""RYKO security gate — local secret scanning (ZYKO pattern).

Lazy re-exports (PEP 562): eagerly importing ``secret_scan`` here would make
``python -m ryko_lm.security.secret_scan`` re-execute an already-imported
module (RuntimeWarning + double module state).  Import either the submodule
directly or these names — both work identically.
"""

from typing import Any

__all__ = [
    "EXEMPT_MARKER",
    "SecretHit",
    "added_lines",
    "scan_diff",
    "scan_text",
    "scan_to_json",
    "scan_tree",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from ryko_lm.security import secret_scan as _ss

        return getattr(_ss, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
