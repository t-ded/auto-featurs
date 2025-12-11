from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Optional

type ColumnSelection = str | Sequence[str] | ColumnType | Sequence[ColumnType]
type ColumnSet = list[ColumnSpecification]


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
            return ColumnSelector(type_selector=ColumnTypeSelector({self}), role_selector=ColumnRoleSelector({other}))
        elif isinstance(other, ColumnRoleSelector):
            return ColumnSelector(type_selector=ColumnTypeSelector({self}), role_selector=other)
        else:
            raise TypeError(f'Cannot add {type(other)} to ColumnType')

    def __invert__(self) -> ColumnTypeSelector:
        return ColumnTypeSelector(types=ColumnType.ANY() - {self})


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
            return ColumnSelector(type_selector=ColumnTypeSelector({other}), role_selector=ColumnRoleSelector({self}))
        elif isinstance(other, ColumnTypeSelector):
            return ColumnSelector(type_selector=other, role_selector=ColumnRoleSelector({self}))
        else:
            raise TypeError(f'Cannot add {type(other)} to ColumnRole')

    def __invert__(self) -> ColumnRoleSelector:
        return ColumnRoleSelector(roles=ColumnRole.ANY() - {self})


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


class ColumnTypeSelector:
    def __init__(self, types: Optional[set[ColumnType]] = None) -> None:
        self._types = types if types is not None else ColumnType.ANY()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColumnTypeSelector):
            raise TypeError(f'Cannot compare {type(other)} to ColumnTypeSelector')
        return self._types == other.types

    def __and__(self, other: object) -> ColumnSelector:
        if isinstance(other, ColumnRole):
            return ColumnSelector(type_selector=self, role_selector=ColumnRoleSelector({other}))
        elif isinstance(other, ColumnRoleSelector):
            return ColumnSelector(type_selector=self, role_selector=other)
        else:
            raise TypeError(f'Cannot add {type(other)} to ColumnTypeSelector')

    def __invert__(self) -> ColumnTypeSelector:
        return ColumnTypeSelector(types=ColumnType.ANY() - self._types)

    @property
    def types(self) -> set[ColumnType]:
        return self._types


class ColumnRoleSelector:
    def __init__(self, roles: Optional[set[ColumnRole]] = None) -> None:
        self._roles = roles if roles is not None else ColumnRole.ANY()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColumnRoleSelector):
            raise TypeError(f'Cannot compare {type(other)} to ColumnRoleSelector')
        return self._roles == other.roles

    def __and__(self, other: object) -> ColumnSelector:
        if isinstance(other, ColumnType):
            return ColumnSelector(type_selector=ColumnTypeSelector({other}), role_selector=self)
        elif isinstance(other, ColumnTypeSelector):
            return ColumnSelector(type_selector=other, role_selector=self)
        else:
            raise TypeError(f'Cannot add {type(other)} to ColumnRoleSelector')

    def __invert__(self) -> ColumnRoleSelector:
        return ColumnRoleSelector(roles=ColumnRole.ANY() - self._roles)

    @property
    def roles(self) -> set[ColumnRole]:
        return self._roles


class ColumnSelector:
    def __init__(
        self,
        names: Optional[set[str]] = None,
        type_selector: ColumnTypeSelector = ColumnTypeSelector(),
        role_selector: ColumnRoleSelector = ColumnRoleSelector(),
    ) -> None:
        self._names = names
        self._type_selector = type_selector
        self._role_selector = role_selector

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColumnSelector):
            raise TypeError(f'Cannot compare {type(other)} to ColumnSelector')
        return self._names == other.names

    @property
    def names(self) -> Optional[set[str]]:
        return self._names

    @property
    def types(self) -> set[ColumnType]:
        return self._type_selector.types

    @property
    def roles(self) -> set[ColumnRole]:
        return self._role_selector.roles
