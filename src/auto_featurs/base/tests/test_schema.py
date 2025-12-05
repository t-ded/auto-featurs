from auto_featurs.base.schema import ColumnRole
from auto_featurs.base.schema import ColumnSpecification
from auto_featurs.base.schema import ColumnType
from auto_featurs.base.schema import Schema


def test_any_column_type() -> None:
    assert ColumnType.ANY() == {
        ColumnType.NUMERIC,
        ColumnType.BOOLEAN,
        ColumnType.NOMINAL,
        ColumnType.ORDINAL,
        ColumnType.TEXT,
        ColumnType.DATETIME,
    }


class TestSchema:
    def setup_method(self) -> None:
        self._schema = Schema(
            [
                ColumnSpecification(name='a', column_type=ColumnType.NUMERIC),
                ColumnSpecification(name='b', column_type=ColumnType.ORDINAL, column_role=ColumnRole.LABEL),
            ],
        )

    def test_add(self) -> None:
        schema = Schema([ColumnSpecification(name='c', column_type=ColumnType.NUMERIC)])
        added = self._schema + schema
        assert added.columns == [
            ColumnSpecification(name='a', column_type=ColumnType.NUMERIC),
            ColumnSpecification(name='b', column_type=ColumnType.ORDINAL, column_role=ColumnRole.LABEL),
            ColumnSpecification(name='c', column_type=ColumnType.NUMERIC),
        ]

    def test_column_names(self) -> None:
        assert self._schema.column_names == ['a', 'b']

    def test_num_columns(self) -> None:
        assert self._schema.num_columns == 2

    def test_label_column(self) -> None:
        assert self._schema.label_column == ColumnSpecification(name='b', column_type=ColumnType.ORDINAL, column_role=ColumnRole.LABEL)

    def test_get_column_by_name(self) -> None:
        assert self._schema.get_column_by_name('a') == ColumnSpecification(name='a', column_type=ColumnType.NUMERIC)

    def test_get_columns_of_type(self) -> None:
        assert self._schema.get_columns_of_type(ColumnType.NUMERIC) == [ColumnSpecification(name='a', column_type=ColumnType.NUMERIC)]

    def test_get_columns_from_selection(self) -> None:
        assert self._schema.get_columns_from_selection(['a', 'b']) == [
            ColumnSpecification(name='a', column_type=ColumnType.NUMERIC),
            ColumnSpecification(name='b', column_type=ColumnType.ORDINAL, column_role=ColumnRole.LABEL),
        ]
