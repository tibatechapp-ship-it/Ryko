import pytest

from solution import fib


def test_base_cases():
    assert fib(0) == 0
    assert fib(1) == 1


def test_small_values():
    assert [fib(n) for n in range(10)] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


def test_negative_raises():
    with pytest.raises(ValueError):
        fib(-1)


def test_rejects_non_int():
    with pytest.raises(TypeError):
        fib("3")  # type: ignore[arg-type]
