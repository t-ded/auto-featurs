from auto_featurs.core.base.column_specification import ColumnType


def test_any_column_type() -> None:
    assert ColumnType.ANY() == {
        ColumnType.NUMERIC,
        ColumnType.BOOLEAN,
        ColumnType.NOMINAL,
        ColumnType.ORDINAL,
        ColumnType.TEXT,
        ColumnType.DATETIME,
    }
