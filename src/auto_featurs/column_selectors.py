from auto_featurs.base.column_specification import NameContains
from auto_featurs.base.column_specification import NameEndsWith
from auto_featurs.base.column_specification import NameRegex
from auto_featurs.base.column_specification import NameStartsWith


def name_contains(value: str) -> NameContains:
    return NameContains(value)


def name_starts_with(value: str) -> NameStartsWith:
    return NameStartsWith(value)


def name_ends_with(value: str) -> NameEndsWith:
    return NameEndsWith(value)


def name_matches(regex: str, flags: int = 0) -> NameRegex:
    return NameRegex(regex, flags)
