"""CI grammar guard: f-strings must parse on Python 3.11.

Root cause chain (CI run #67): the sandbox develops on 3.12+, where
PEP 701 allows a newline inside an f-string replacement field; CI runs
3.11, where that is a hard SyntaxError — and it killed test collection
before a single test ran.

Two-mode checker (stdlib only, works on the interpreter it runs under):

- On 3.12+ (local dev): tokenize emits FSTRING_START/END, so each
  f-string literal is spanned precisely; a literal whose source has a
  newline while inside a replacement field (brace depth >= 1, nested
  string literals skipped, {{ }} escapes honoured) is flagged.
  Implicit concatenation of single-line f-strings across lines is
  valid on 3.11 and correctly NOT flagged.
- On 3.11 (CI): the tokenizer cannot see f-strings as structure — but
  that is fine: on 3.11 the offender is a plain SyntaxError, so
  ast.parse() over every file catches exactly this class.

Exit 0 = clean, exit 1 = at least one offender (printed file:line).
"""
import ast
import pathlib
import sys
import tokenize

BASES_DEFAULT = ("src", "tests")


def _newline_inside_field(seg: str) -> bool:
    """True if a newline occurs while inside a {replacement field}."""
    i, n = 0, len(seg)
    depth = 0
    while i < n:
        c = seg[i]
        if depth >= 1 and c in "\"'":
            # skip a (possibly prefixed) string literal nested in a field
            j = i
            while j > 0 and seg[j - 1].isalpha():  # prefix letters (f, r, b, u)
                j -= 1
            quote = seg[i]
            if seg[i:i + 3] == quote * 3:
                q = quote * 3
            else:
                q = quote
            k = i + len(q)
            while k < n:
                if seg[k] == "\\":
                    k += 2
                    continue
                if seg.startswith(q, k):
                    break
                k += 1
            i = k + len(q)
            continue
        if seg.startswith("{{", i) or seg.startswith("}}", i):
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth = max(0, depth - 1)
        elif c == "\n" and depth >= 1:
            return True
        i += 1
    return False


def _spans_by_tokenize(src: str):
    lines = src.splitlines()
    toks = list(tokenize.generate_tokens(
        iter(src.splitlines(keepends=True)).__next__))
    spans = []
    depth = 0
    start = None
    for t in toks:
        if t.type == tokenize.FSTRING_START:
            if depth == 0:
                start = t
            depth += 1
        elif t.type == tokenize.FSTRING_END:
            depth -= 1
            if depth == 0 and start is not None:
                s_line, e_line = start.start[0], t.end[0]
                if e_line > s_line:
                    seg = "\n".join(
                        [lines[s_line - 1][start.start[1] - 1:]]
                        + [lines[l] for l in range(s_line, e_line - 1)]
                        + [lines[e_line - 1][:t.end[1]]]
                    )
                    if _newline_inside_field(seg):
                        spans.append((s_line, e_line))
                start = None
    return spans


def scan_tree(bases) -> list[tuple[str, int, int]]:
    """Return [(path, start_line, end_line)] of 3.11-invalid f-strings."""
    has_fstring_tokens = hasattr(tokenize, "FSTRING_START")
    bad: list[tuple[str, int, int]] = []
    for base in bases:
        root = pathlib.Path(base)
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.py")):
            s = str(p)
            if "__pycache__" in s or "egg-info" in s:
                continue
            src = p.read_text(encoding="utf-8")
            if has_fstring_tokens:
                for a, b in _spans_by_tokenize(src):
                    bad.append((s, a, b))
            else:
                # on 3.11 the offender fails ast.parse outright
                try:
                    ast.parse(src, filename=s)
                except SyntaxError as e:
                    bad.append((s, e.lineno or 0,
                                e.end_lineno or e.lineno or 0))
    return bad


def main(argv) -> int:
    bases = argv[1:] or list(BASES_DEFAULT)
    bad = scan_tree(bases)
    print("py3.11-invalid f-strings:", len(bad))
    for path, a, b in bad:
        print(f"{path}:{a}-{b}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
