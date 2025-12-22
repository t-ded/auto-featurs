from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from enum import auto
import re
from typing import overload


class ColumnType(Enum):
    NUMERIC = 'numeric'
    BOOLEAN = 'boolean'
    NOMINAL = 'nominal'
    ORDINAL = 'ordinal'
    TEXT = 'text'
    DATETIME = 'datetime'

    @classmethod
    def ANY(cls) -> set[ColumnType]:  # noqa: N802
        return set(cls)

    def __and__(self, other: object) -> ColumnSelector:
        if isinstance(other, ColumnRole):
            return self.as_selector() & other.as_selector()
        elif isinstance(other, ColumnSelector):
            return self.as_selector() & other
        else:
            raise TypeError(f'Cannot add {type(other)} to ColumnType')

    @overload
    def __or__(self, other: ColumnType) -> ColumnTypeSelector:
        ...

    @overload
    def __or__(self, other: ColumnRole) -> ColumnSelector:
        ...

    @overload
    def __or__(self, other: ColumnSelector) -> ColumnSelector:
        ...

    def __or__(self, other: object) -> ColumnSelector:
        if isinstance(other, ColumnType):
            return ColumnTypeSelector(frozenset([self, other]))
        elif isinstance(other, ColumnRole):
            return self.as_selector() | other.as_selector()
        elif isinstance(other, ColumnSelector):
            return self.as_selector() | other
        else:
            raise TypeError(f'Cannot add {type(other)} to ColumnType')

    def __invert__(self) -> ColumnSelector:
        return ~self.as_selector()

    def as_selector(self) -> ColumnTypeSelector:
        return ColumnTypeSelector(frozenset([self]))


class ColumnRole(Enum):
    LABEL = auto()
    IDENTIFIER = auto()
    TIME_INFO = auto()
    FEATURE = auto()

    @classmethod
    def ANY(cls) -> set[ColumnRole]:  # noqa: N802
        return set(cls)

    def __and__(self, other: object) -> ColumnSelector:
        if isinstance(other, ColumnType):
            return self.as_selector() & other.as_selector()
        elif isinstance(other, ColumnSelector):
            return self.as_selector() & other
        else:
            raise TypeError(f'Cannot add {type(other)} to ColumnRole')

    @overload
    def __or__(self, other: ColumnRole) -> ColumnRoleSelector:
        ...

    @overload
    def __or__(self, other: ColumnType) -> ColumnSelector:
        ...

    @overload
    def __or__(self, other: ColumnSelector) -> ColumnSelector:
        ...

    def __or__(self, other: object) -> ColumnSelector:
        if isinstance(other, ColumnRole):
            return ColumnRoleSelector(frozenset([self, other]))
        elif isinstance(other, ColumnType):
            return self.as_selector() | other.as_selector()
        elif isinstance(other, ColumnSelector):
            return self.as_selector() | other
        else:
            raise TypeError(f'Cannot add {type(other)} to ColumnRole')

    def __invert__(self) -> ColumnSelector:
        return ~self.as_selector()

    def as_selector(self) -> ColumnRoleSelector:
        return ColumnRoleSelector(frozenset([self]))


@dataclass(kw_only=True, frozen=True, slots=True)
class ColumnSpecification:
    name: str
    column_type: ColumnType
    column_role: ColumnRole = ColumnRole.FEATURE

    @classmethod
    def numeric(cls, name: str, role: ColumnRole = ColumnRole.FEATURE) -> ColumnSpecification:
        return ColumnSpecification(name=name, column_type=ColumnType.NUMERIC, column_role=role)

    @classmethod
    def boolean(cls, name: str, role: ColumnRole = ColumnRole.FEATURE) -> ColumnSpecification:
        return ColumnSpecification(name=name, column_type=ColumnType.BOOLEAN, column_role=role)

    @classmethod
    def nominal(cls, name: str, role: ColumnRole = ColumnRole.FEATURE) -> ColumnSpecification:
        return ColumnSpecification(name=name, column_type=ColumnType.NOMINAL, column_role=role)

    @classmethod
    def ordinal(cls, name: str, role: ColumnRole = ColumnRole.FEATURE) -> ColumnSpecification:
        return ColumnSpecification(name=name, column_type=ColumnType.ORDINAL, column_role=role)

    @classmethod
    def text(cls, name: str, role: ColumnRole = ColumnRole.FEATURE) -> ColumnSpecification:
        return ColumnSpecification(name=name, column_type=ColumnType.TEXT, column_role=role)

    @classmethod
    def datetime(cls, name: str, role: ColumnRole = ColumnRole.FEATURE) -> ColumnSpecification:
        return ColumnSpecification(name=name, column_type=ColumnType.DATETIME, column_role=role)


class ColumnSelector(ABC):
    def __and__(self, other: object) -> ColumnSelector:
        if isinstance(other, ColumnType | ColumnRole):
            return self & other.as_selector()
        elif isinstance(other, ColumnSelector):
            return _And(self, other)
        else:
            raise ValueError(f'Cannot add {type(other)} to ColumnSelector')

    def __or__(self, other: object) -> ColumnSelector:
        if isinstance(other, ColumnType | ColumnRole):
            return self | other.as_selector()
        elif isinstance(other, ColumnSelector):
            return _Or(self, other)
        else:
            raise ValueError(f'Cannot add {type(other)} to ColumnSelector')


    def __invert__(self) -> ColumnSelector:
        return _Not(self)

    @abstractmethod
    def matches(self, column: ColumnSpecification) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class _And(ColumnSelector):
    left: ColumnSelector
    right: ColumnSelector

    def matches(self, column: ColumnSpecification) -> bool:
        return self.left.matches(column) and self.right.matches(column)


@dataclass(frozen=True)
class _Or(ColumnSelector):
    left: ColumnSelector
    right: ColumnSelector

    def matches(self, column: ColumnSpecification) -> bool:
        return self.left.matches(column) or self.right.matches(column)


@dataclass(frozen=True)
class _Not(ColumnSelector):
    selector: ColumnSelector

    def matches(self, column: ColumnSpecification) -> bool:
        return not self.selector.matches(column)


@dataclass(frozen=True)
class NameContains(ColumnSelector):
    value: str

    def matches(self, column: ColumnSpecification) -> bool:
        return self.value in column.name


@dataclass(frozen=True)
class NameStartsWith(ColumnSelector):
    value: str

    def matches(self, column: ColumnSpecification) -> bool:
        return column.name.startswith(self.value)


@dataclass(frozen=True)
class NameEndsWith(ColumnSelector):
    value: str

    def matches(self, column: ColumnSpecification) -> bool:
        return column.name.endswith(self.value)


@dataclass(frozen=True)
class NameRegex(ColumnSelector):
    pattern: str
    flags: int = 0

    def matches(self, column: ColumnSpecification) -> bool:
        return bool(re.compile(self.pattern, self.flags).search(column.name))


@dataclass(frozen=True)
class ColumnTypeSelector(ColumnSelector):
    types: frozenset[ColumnType]

    def matches(self, column: ColumnSpecification) -> bool:
        return column.column_type in self.types

    @classmethod
    def any(cls) -> ColumnTypeSelector:
        return cls(frozenset(ColumnType.ANY()))


@dataclass(frozen=True)
class ColumnRoleSelector(ColumnSelector):
    roles: frozenset[ColumnRole]

    def matches(self, column: ColumnSpecification) -> bool:
        return column.column_role in self.roles

    @classmethod
    def any(cls) -> ColumnRoleSelector:
        return cls(frozenset(ColumnRole.ANY()))
