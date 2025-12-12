from datetime import timedelta

import polars as pl

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.utils.utils import default_true_filtering_condition
from auto_featurs.utils.utils import filtering_condition_to_string
from auto_featurs.utils.utils import format_timedelta
from auto_featurs.utils.utils import get_names_from_column_specs
from auto_featurs.utils.utils import get_valid_param_options
from auto_featurs.utils.utils import order_preserving_unique


def test_default_true_filtering_condition() -> None:
    assert default_true_filtering_condition(None).meta.eq(pl.lit(True))
    assert default_true_filtering_condition(pl.lit(False)).meta.eq(pl.lit(False))


def test_filtering_condition_to_string() -> None:
    assert filtering_condition_to_string(None) == ''
    assert filtering_condition_to_string(pl.lit(True)) == ''
    assert filtering_condition_to_string(
        (
                (pl.col('a') > 10) &
                (pl.col('b') == 'foo') &
                (
                        (pl.col('c') == 100) |
                        (pl.col('d') == 200)
                )
        ).alias('complicated_filter'),
    ) == '_where_complicated_filter'


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
