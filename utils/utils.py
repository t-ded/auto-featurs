from collections.abc import Iterable

from core.base.column_specification import ColumnSpecification


def order_preserving_unique[T](iterable: Iterable[T]) -> list[T]:
    seen: set[T] = set()
    result: list[T] = []
    for value in iterable:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def get_names_from_column_specs(columns: Iterable[ColumnSpecification]) -> list[str]:
    return [column.name for column in columns]
