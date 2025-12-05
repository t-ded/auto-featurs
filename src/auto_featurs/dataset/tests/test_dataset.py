import polars as pl
import pytest

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.schema import Schema
from auto_featurs.dataset.dataset import Dataset
from auto_featurs.utils.utils import get_names_from_column_specs


class TestDataset:
    def setup_method(self) -> None:
        self._schema = Schema([
            ColumnSpecification(name='a', column_type=ColumnType.NUMERIC),
            ColumnSpecification(name='b', column_type=ColumnType.ORDINAL),
            ColumnSpecification(name='c', column_type=ColumnType.NUMERIC),
        ])
        self._df = pl.DataFrame({'a': [1], 'b': ['x'], 'c': [2]})
        self._ds = Dataset(self._df, schema=self._schema)

    def test_data_is_lazy(self) -> None:
        assert isinstance(self._ds.data, pl.LazyFrame)

    def test_num_columns(self) -> None:
        assert self._ds.num_columns == 3

    def test_drop_columns_outside_schema(self) -> None:
        schema = Schema([ColumnSpecification(name='a', column_type=ColumnType.NUMERIC)])
        ds2 = Dataset(self._df, schema=schema, drop_columns_outside_schema=True)
        out = ds2.collect()
        assert set(out.columns) == {'a'}

    def test_get_columns_of_type(self) -> None:
        cols = self._ds.get_columns_of_type(ColumnType.NUMERIC)
        assert get_names_from_column_specs(cols) == ['a', 'c']

    def test_get_column_by_name(self) -> None:
        col = self._ds.get_column_by_name('b')
        assert col.name == 'b'

    def test_get_column_by_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self._ds.get_column_by_name('missing')

    def test_get_columns_from_selection_single_name(self) -> None:
        cols = self._ds.get_columns_from_selection('a')
        assert get_names_from_column_specs(cols) == ['a']

    def test_get_columns_from_selection_type(self) -> None:
        cols = self._ds.get_columns_from_selection(ColumnType.ORDINAL)
        assert get_names_from_column_specs(cols) == ['b']

    def test_get_columns_from_selection_sequence(self) -> None:
        cols = self._ds.get_columns_from_selection([ColumnType.NUMERIC, ColumnType.ORDINAL])
        assert get_names_from_column_specs(cols) == ['a', 'c', 'b']

    def test_get_combinations_from_selections(self) -> None:
        combos = self._ds.get_combinations_from_selections('a', ColumnType.NUMERIC)
        assert len(combos) == 2
        assert get_names_from_column_specs(combos[0]) == ['a']
        assert get_names_from_column_specs(combos[1]) == ['a', 'c']

    def test_with_columns(self) -> None:
        new = self._ds.with_columns([pl.col('a') + 1])
        assert isinstance(new, Dataset)
        assert new.schema == self._schema

    def test_with_schema(self) -> None:
        extra = Schema([ColumnSpecification(name='d', column_type=ColumnType.ORDINAL)])
        new = self._ds.with_schema(extra)
        assert new.num_columns == 4
        assert new.schema.column_names == ['a', 'b', 'c', 'd']

    def test_with_cached_computation_remains_lazy(self) -> None:
        new = self._ds.with_cached_computation()
        assert isinstance(self._ds.data, pl.LazyFrame)
        assert isinstance(new.data, pl.LazyFrame)

    def test_collect(self) -> None:
        out = self._ds.collect()
        assert isinstance(out, pl.DataFrame)
        assert out.shape == (1, 3)
