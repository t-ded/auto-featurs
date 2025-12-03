import polars as pl
import pytest

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.dataset.dataset import Dataset


class TestDataset:
    def setup_method(self) -> None:
        self._schema = [
            ColumnSpecification(name='a', column_type=ColumnType.NUMERIC),
            ColumnSpecification(name='b', column_type=ColumnType.ORDINAL),
            ColumnSpecification(name='c', column_type=ColumnType.NUMERIC),
        ]
        self._df = pl.DataFrame({'a': [1], 'b': ['x'], 'c': [2]})
        self._ds = Dataset(self._df, schema=self._schema)

    def test_data_is_lazy(self) -> None:
        assert isinstance(self._ds.data, pl.LazyFrame)

    def test_drop_columns_outside_schema(self) -> None:
        schema = [ColumnSpecification(name='a', column_type=ColumnType.NUMERIC)]
        ds2 = Dataset(self._df, schema=schema, drop_columns_outside_schema=True)
        out = ds2.collect()
        assert set(out.columns) == {'a'}

    def test_get_columns_of_type(self) -> None:
        cols = self._ds.get_columns_of_type(ColumnType.NUMERIC)
        assert [c.name for c in cols] == ['a', 'c']

    def test_get_column_by_name(self) -> None:
        col = self._ds.get_column_by_name('b')
        assert col.name == 'b'

    def test_get_column_by_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self._ds.get_column_by_name('missing')

    def test_get_columns_from_selection_single_name(self) -> None:
        out = self._ds.get_columns_from_selection('a')
        assert isinstance(out, list)
        assert out[0].name == 'a'

    def test_get_columns_from_selection_type(self) -> None:
        out = self._ds.get_columns_from_selection(ColumnType.ORDINAL)
        assert [c.name for c in out] == ['b']

    def test_get_columns_from_selection_sequence(self) -> None:
        out = self._ds.get_columns_from_selection([ColumnType.NUMERIC, ColumnType.ORDINAL])
        assert [c.name for c in out] == ['a', 'c', 'b']

    def test_get_combinations_from_selections(self) -> None:
        combos = self._ds.get_combinations_from_selections('a', ColumnType.NUMERIC)
        assert len(combos) == 2
        assert [c.name for c in combos[0]] == ['a']
        assert [c.name for c in combos[1]] == ['a', 'c']

    def test_with_columns(self) -> None:
        new = self._ds.with_columns([pl.col('a') + 1])
        assert isinstance(new, Dataset)
        assert new.schema == self._schema

    def test_with_schema(self) -> None:
        extra = [ColumnSpecification(name='d', column_type=ColumnType.ORDINAL)]
        new = self._ds.with_schema(extra)
        assert len(new.schema) == 4
        assert new.schema[-1].name == 'd'

    def test_with_cached_computation_remains_lazy(self) -> None:
        new = self._ds.with_cached_computation()
        assert isinstance(self._ds.data, pl.LazyFrame)

    def test_collect(self) -> None:
        out = self._ds.collect()
        assert isinstance(out, pl.DataFrame)
        assert out.shape == (1, 3)
