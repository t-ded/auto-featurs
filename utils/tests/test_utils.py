from core.base.column_specification import ColumnSpecification
from utils.utils import get_names_from_column_specs
from utils.utils import order_preserving_unique


def test_order_preserving_unique() -> None:
    assert order_preserving_unique(()) == []
    assert order_preserving_unique([1, 2, 3]) == [1, 2, 3]
    assert order_preserving_unique([1, 2, 3, 1, 2, 3]) == [1, 2, 3]
    assert order_preserving_unique(range(1, 4)) == [1, 2, 3]


def test_get_names_from_column_specs() -> None:
    assert get_names_from_column_specs([]) == []
    assert get_names_from_column_specs(['a']) == ['a']
    assert get_names_from_column_specs(['a', ColumnSpecification.numeric(name='b')]) == ['a', 'b']
