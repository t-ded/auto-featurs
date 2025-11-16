from utils.utils import order_preserving_unique


def test_order_preserving_unique() -> None:
    assert order_preserving_unique(()) == []
    assert order_preserving_unique([1, 2, 3]) == [1, 2, 3]
    assert order_preserving_unique([1, 2, 3, 1, 2, 3]) == [1, 2, 3]
    assert order_preserving_unique(range(1, 4)) == [1, 2, 3]
