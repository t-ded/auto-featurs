from enum import Enum


class ColumnType(Enum):
    NUMERIC = 'NUMERIC'
    BOOLEAN = 'BOOLEAN'
    NOMINAL = 'NOMINAL'
    ORDINAL = 'ORDINAL'
    TEXT = 'TEXT'
    TIME = 'TIME'
    LABEL = 'LABEL'
    ANY = 'ANY'
