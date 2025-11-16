from collections.abc import Iterable


def order_preserving_unique[T](iterable: Iterable[T]) -> list[T]:
    seen: set[T] = set()
    result: list[T] = []
    for value in iterable:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
