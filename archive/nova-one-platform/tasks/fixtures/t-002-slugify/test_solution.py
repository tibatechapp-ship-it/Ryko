from solution import slugify


def test_basic_english():
    assert slugify("Hello World!") == "hello-world"


def test_collapses_whitespace_and_punct():
    assert slugify("  A   B!! C  ") == "a-b-c"


def test_preserves_unicode_words():
    assert slugify("مرحبا World 123") == "مرحبا-world-123"


def test_empty_and_none():
    assert slugify("") == ""
    assert slugify(None) == ""
