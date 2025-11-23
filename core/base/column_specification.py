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
    def ANY(cls) -> set[ColumnType]:
        return set(cls)


@dataclass(kw_only=True, frozen=True, slots=True)
class ColumnSpecification:
    name: str
    column_type: ColumnType
