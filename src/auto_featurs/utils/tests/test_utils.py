from datetime import timedelta

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.utils.utils import format_timedelta
from auto_featurs.utils.utils import get_names_from_column_specs
from auto_featurs.utils.utils import get_valid_param_options
from auto_featurs.utils.utils import order_preserving_unique


def test_order_preserving_unique() -> None:
    assert order_preserving_unique(()) == []
    assert order_preserving_unique([1, 2, 3]) == [1, 2, 3]
    assert order_preserving_unique([1, 2, 3, 1, 2, 3]) == [1, 2, 3]
    assert order_preserving_unique(range(1, 4)) == [1, 2, 3]


def test_get_names_from_column_specs() -> None:
    assert get_names_from_column_specs([]) == []
    assert get_names_from_column_specs(['a']) == ['a']
    assert get_names_from_column_specs(['a', ColumnSpecification.numeric(name='b')]) == ['a', 'b']


def test_get_valid_param_options() -> None:
    assert get_valid_param_options(['A', None, 'B']) == (['A', 'B'], False)
    assert get_valid_param_options([['A', 'B'], None, ['B', 'C']]) == ([['A', 'B'], ['B', 'C']], False)
    assert get_valid_param_options([[], ['A', 'B']]) == ([['A', 'B']], False)
    assert get_valid_param_options([['A', 'B']]) == ([['A', 'B']], True)


def test_format_timedelta() -> None:
    assert format_timedelta(timedelta(0)) == "0s"
    assert format_timedelta(timedelta(seconds=45)) == "45s"
    assert format_timedelta(timedelta(minutes=3, seconds=15)) == "3m15s"
    assert format_timedelta(timedelta(hours=5, minutes=2, seconds=1)) == "5h2m1s"
    assert format_timedelta(timedelta(days=7, hours=5)) == "7d5h"
    assert format_timedelta(timedelta(days=37, hours=5)) == "1mo7d5h"
