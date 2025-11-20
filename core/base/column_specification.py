from enum import Enum
from enum import auto


class ColumnType(Enum):
    NUMERIC = auto()
    BOOLEAN = auto()
    NOMINAL = auto()
    ORDINAL = auto()
    TEXT = auto()
    DATETIME = auto()
    ANY = object
