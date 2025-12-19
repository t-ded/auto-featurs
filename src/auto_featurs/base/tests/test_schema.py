import re

import pytest

from auto_featurs.base.column_specification import ColumnRole
from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.schema import Schema
import auto_featurs.column_selectors as cs


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
                ColumnSpecification(name='c', column_type=ColumnType.NOMINAL),
            ],
        )

    def test_add(self) -> None:
        schema = Schema([ColumnSpecification(name='d', column_type=ColumnType.NUMERIC)])
        added = self._schema + schema
        assert added.columns == [
            ColumnSpecification(name='a', column_type=ColumnType.NUMERIC),
            ColumnSpecification(name='b', column_type=ColumnType.ORDINAL, column_role=ColumnRole.LABEL),
            ColumnSpecification(name='c', column_type=ColumnType.NOMINAL),
            ColumnSpecification(name='d', column_type=ColumnType.NUMERIC),
        ]

    def test_from_dict(self) -> None:
        spec = {
            ColumnType.NUMERIC: ['age', 'income'],
            ColumnType.ORDINAL: ['education_level'],
        }

        schema = Schema.from_dict(spec, label_col='income')

        assert schema.columns == [
            ColumnSpecification(name='age', column_type=ColumnType.NUMERIC, column_role=ColumnRole.FEATURE),
            ColumnSpecification(name='income', column_type=ColumnType.NUMERIC, column_role=ColumnRole.LABEL),
            ColumnSpecification(name='education_level', column_type=ColumnType.ORDINAL, column_role=ColumnRole.FEATURE),
        ]

    def test_from_dict_missing_label(self) -> None:
        spec = {ColumnType.NUMERIC: ['age', 'income']}

        with pytest.raises(ValueError, match="label_col='not-present' not found in provided columns"):
            Schema.from_dict(spec, label_col='not-present')

    def test_column_names(self) -> None:
        assert self._schema.column_names == ['a', 'b', 'c']

    def test_num_columns(self) -> None:
        assert self._schema.num_columns == 3

    def test_label_column(self) -> None:
        assert self._schema.label_column == ColumnSpecification(name='b', column_type=ColumnType.ORDINAL, column_role=ColumnRole.LABEL)

    def test_drop(self) -> None:
        a = ColumnSpecification(name='a', column_type=ColumnType.NUMERIC)
        b = ColumnSpecification(name='b', column_type=ColumnType.ORDINAL, column_role=ColumnRole.LABEL)
        c = ColumnSpecification(name='c', column_type=ColumnType.NOMINAL)

        without_c = self._schema.drop([b, c])
        assert self._schema.columns == [a, b, c]
        assert without_c.columns == [a]

    def test_get_column_by_name(self) -> None:
        assert self._schema.get_column_by_name('a') == ColumnSpecification(name='a', column_type=ColumnType.NUMERIC)

    def test_get_columns_of_type(self) -> None:
        numeric_subset = [ColumnSpecification(name='a', column_type=ColumnType.NUMERIC)]
        assert self._schema.get_columns_of_type(ColumnType.NUMERIC) == numeric_subset
        assert self._schema.get_columns_of_type(ColumnType.NUMERIC, subset=numeric_subset) == numeric_subset
        with pytest.raises(ValueError, match=re.escape("The following columns in subset not found in schema: ['c']")):
            self._schema.get_columns_of_type(ColumnType.NUMERIC, subset=[ColumnSpecification(name='c', column_type=ColumnType.NUMERIC)])

    def test_get_columns_of_role(self) -> None:
        label_subset = [ColumnSpecification(name='b', column_type=ColumnType.ORDINAL, column_role=ColumnRole.LABEL)]
        assert self._schema.get_columns_of_role(ColumnRole.LABEL) == label_subset
        assert self._schema.get_columns_of_role(ColumnRole.LABEL, subset=label_subset) == label_subset
        with pytest.raises(ValueError, match=re.escape("The following columns in subset not found in schema: ['c']")):
            self._schema.get_columns_of_role(ColumnRole.LABEL, subset=[ColumnSpecification(name='c', column_type=ColumnType.NUMERIC)])

    def test_get_columns_from_selection(self) -> None:
        a = ColumnSpecification(name='a', column_type=ColumnType.NUMERIC)
        b = ColumnSpecification(name='b', column_type=ColumnType.ORDINAL, column_role=ColumnRole.LABEL)
        c = ColumnSpecification(name='c', column_type=ColumnType.NOMINAL)
        assert self._schema.get_columns_from_selection('a') == [a]
        assert self._schema.get_columns_from_selection(['a', 'b']) == [a, b]
        assert self._schema.get_columns_from_selection({'a', 'b'}) == [a, b]
        assert self._schema.get_columns_from_selection(ColumnType.NUMERIC) == [a]
        assert self._schema.get_columns_from_selection([ColumnType.NUMERIC, ColumnType.ORDINAL]) == [a, b]
        assert self._schema.get_columns_from_selection(ColumnRole.FEATURE) == [a, c]
        assert self._schema.get_columns_from_selection([ColumnRole.FEATURE, ColumnRole.LABEL]) == [a, b, c]
        assert self._schema.get_columns_from_selection(ColumnType.NUMERIC | ColumnType.ORDINAL) == [a, b]
        assert self._schema.get_columns_from_selection(~ColumnType.NOMINAL) == [a, b]
        assert self._schema.get_columns_from_selection(ColumnRole.FEATURE | ColumnRole.LABEL) == [a, b, c]
        assert self._schema.get_columns_from_selection(~ColumnRole.LABEL) == [a, c]
        assert self._schema.get_columns_from_selection((ColumnType.NUMERIC | ColumnType.ORDINAL) & ~ColumnRole.LABEL) == [a]
        assert self._schema.get_columns_from_selection((ColumnType.NUMERIC | ColumnType.ORDINAL) & ~(ColumnRole.LABEL | ColumnRole.FEATURE)) == []
        assert self._schema.get_columns_from_selection(cs.name_starts_with('a') | cs.name_starts_with('b') | cs.name_starts_with('d')) == [a, b]
        assert self._schema.get_columns_from_selection(cs.name_starts_with('') & ColumnType.NUMERIC) == [a]
        assert self._schema.get_columns_from_selection(cs.name_starts_with('') | ColumnType.NUMERIC) == [a, b, c]
