from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from enum import auto

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


class ColumnRole(Enum):
    LABEL = auto()
    FEATURE = auto()


@dataclass(kw_only=True, frozen=True, slots=True)
class ColumnSpecification:
    name: str
    column_type: ColumnType
    column_role: ColumnRole = ColumnRole.FEATURE

    @classmethod
    def numeric(cls, name: str) -> ColumnSpecification:
        return ColumnSpecification(name=name, column_type=ColumnType.NUMERIC)

    @classmethod
    def boolean(cls, name: str) -> ColumnSpecification:
        return ColumnSpecification(name=name, column_type=ColumnType.BOOLEAN)

    @classmethod
    def nominal(cls, name: str) -> ColumnSpecification:
        return ColumnSpecification(name=name, column_type=ColumnType.NOMINAL)

    @classmethod
    def ordinal(cls, name: str) -> ColumnSpecification:
        return ColumnSpecification(name=name, column_type=ColumnType.ORDINAL)

    @classmethod
    def text(cls, name: str) -> ColumnSpecification:
        return ColumnSpecification(name=name, column_type=ColumnType.TEXT)

    @classmethod
    def datetime(cls, name: str) -> ColumnSpecification:
        return ColumnSpecification(name=name, column_type=ColumnType.DATETIME)
