from auto_featurs.base.column_specification import ColumnRole
from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType


def test_any_column_type() -> None:
    assert ColumnType.ANY() == {
        ColumnType.NUMERIC,
        ColumnType.BOOLEAN,
        ColumnType.NOMINAL,
        ColumnType.ORDINAL,
        ColumnType.TEXT,
        ColumnType.DATETIME,
    }


def test_spec_creation() -> None:
    assert ColumnSpecification.numeric(name='a', role=ColumnRole.LABEL) == ColumnSpecification(name='a', column_type=ColumnType.NUMERIC, column_role=ColumnRole.LABEL)
