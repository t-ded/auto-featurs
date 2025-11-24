from collections.abc import Iterable
from datetime import timedelta

from auto_featurs.core.base.column_specification import ColumnSpecification
from auto_featurs.utils.constants import SECONDS_IN_DAY
from auto_featurs.utils.constants import SECONDS_IN_HOUR
from auto_featurs.utils.constants import SECONDS_IN_MINUTE
from auto_featurs.utils.constants import SECONDS_IN_MONTH
from auto_featurs.utils.constants import SECONDS_IN_YEAR


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
