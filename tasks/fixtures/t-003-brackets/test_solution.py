from solution import is_balanced


def test_simple_pairs():
    assert is_balanced("()[]{}") is True


def test_nested():
    assert is_balanced("{[()]}") is True


def test_wrong_order():
    assert is_balanced("([)]") is False


def test_unmatched_open():
    assert is_balanced("(()") is False


def test_empty_string():
    assert is_balanced("") is True


def test_ignores_other_chars():
    assert is_balanced("a(b)c[d]e{f}") is True
