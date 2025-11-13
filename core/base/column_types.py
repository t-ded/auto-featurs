from enum import StrEnum


class ColumnType(StrEnum):
    NUMERIC = 'NUMERIC'
    NOMINAL = 'NOMINAL'
    ORDINAL = 'ORDINAL'
    TEXT = 'TEXT'
    TIME = 'TIME'
    LABEL = 'LABEL'
