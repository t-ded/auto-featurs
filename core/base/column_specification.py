from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from enum import auto


class ColumnType(Enum):
    NUMERIC = auto()
    BOOLEAN = auto()
    NOMINAL = auto()
    ORDINAL = auto()
    TEXT = auto()
    DATETIME = auto()

    @classmethod
    def ANY(cls) -> set[ColumnType]:  # noqa: N802
        return set(cls)


@dataclass(kw_only=True, frozen=True, slots=True)
class ColumnSpecification:
    name: str
    column_type: ColumnType

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
