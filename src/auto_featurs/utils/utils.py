from collections.abc import Iterable
from collections.abc import Sequence
from datetime import timedelta
from typing import Optional

import polars as pl

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.utils.constants import SECONDS_IN_DAY
from auto_featurs.utils.constants import SECONDS_IN_HOUR
from auto_featurs.utils.constants import SECONDS_IN_MINUTE
from auto_featurs.utils.constants import SECONDS_IN_MONTH
from auto_featurs.utils.constants import SECONDS_IN_YEAR

LIT_TRUE = pl.lit(True)


def default_true_filtering_condition(filtering_condition: Optional[pl.Expr]) -> pl.Expr:
    return filtering_condition if filtering_condition is not None else LIT_TRUE


def filtering_condition_to_string(filtering_condition: Optional[pl.Expr]) -> str:
    if filtering_condition is None or filtering_condition.meta.eq(LIT_TRUE):
        return ''
    return f'_where_{filtering_condition.meta.output_name()}'


def order_preserving_unique[T](iterable: Iterable[T]) -> list[T]:
    seen: set[T] = set()
    result: list[T] = []
    for value in iterable:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def get_names_from_column_specs(columns: Iterable[str | ColumnSpecification]) -> list[str]:
    return [column.name if isinstance(column, ColumnSpecification) else column for column in columns]


def get_valid_param_options[T](param_options: Sequence[Optional[T]]) -> tuple[list[T], bool]:
    valid_options = [option for option in param_options if option]
    all_valid = len(valid_options) == len(param_options)
    return valid_options, all_valid


def format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())

    years, total_seconds = divmod(total_seconds, SECONDS_IN_YEAR)
    months, total_seconds = divmod(total_seconds, SECONDS_IN_MONTH)
    days, total_seconds = divmod(total_seconds, SECONDS_IN_DAY)
    hours, total_seconds = divmod(total_seconds, SECONDS_IN_HOUR)
    minutes, total_seconds = divmod(total_seconds, SECONDS_IN_MINUTE)
    seconds = total_seconds

    result = ''
    if years:
        result += f'{years}y'
    if months:
        result += f'{months}mo'
    if days:
        result += f'{days}d'
    if hours:
        result += f'{hours}h'
    if minutes:
        result += f'{minutes}m'
    if seconds:
        result += f'{seconds}s'

    return result or '0s'
