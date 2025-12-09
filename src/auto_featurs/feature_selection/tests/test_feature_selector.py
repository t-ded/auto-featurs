from typing import Optional

import polars as pl
import pytest

from auto_featurs.base.column_specification import ColumnRole
from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.schema import Schema
from auto_featurs.dataset.dataset import Dataset
from auto_featurs.feature_selection.feature_selector import Selector


class TestSelector:
    def setup_method(self) -> None:
        schema = Schema([
            ColumnSpecification(name='x1', column_type=ColumnType.NUMERIC),
            ColumnSpecification(name='x2', column_type=ColumnType.NUMERIC),
            ColumnSpecification(name='x3', column_type=ColumnType.NUMERIC),
            ColumnSpecification(name='x4', column_type=ColumnType.NUMERIC),
            ColumnSpecification(name='z1', column_type=ColumnType.NOMINAL),
            ColumnSpecification(name='z2', column_type=ColumnType.TEXT),
            ColumnSpecification(name='y', column_type=ColumnType.BOOLEAN, column_role=ColumnRole.LABEL),
        ])
        df = pl.DataFrame({
            'x1': [0, 0, 0, 0],
            'x2': [10, 9, 8, 7],
            'x3': [0, 1, 0, 1],
            'x4': [2, 4, 6, 8],
            'z1': ['a', 'b', 'c', 'd'],
            'z2': ['hello', 'world', 'foo', 'bar'],
            'y': [0, 1, 0, 1],
        })
        self._ds = Dataset(df, schema=schema)
        self._selector = Selector()

    @pytest.mark.parametrize(
        ('k', 'frac', 'expected_msg'),
        [
            (None, None, 'Exactly one of k or frac must be specified'),
            (0, None, 'k must be at least 1 but 0 was given.'),
            (None, 2.0, 'frac must be between 0 and 1 but 2.0 was given.'),
        ],
    )
    def test_select_by_correlation_invalid_num_to_select(self, k: Optional[int], frac: Optional[float], expected_msg: str) -> None:
        with pytest.raises(ValueError, match=expected_msg):
            self._selector.select_by_correlation(dataset=self._ds, feature_subset='x1', top_k=k, frac=frac)

    @pytest.mark.parametrize('feature', ['z1', 'z2'])
    def test_select_by_correlation_invalid_feature_type(self, feature: str) -> None:
        with pytest.raises(ValueError, match='Correlation can only be computed for numeric, boolean, ordinal columns'):
            self._selector.select_by_correlation(dataset=self._ds, feature_subset=feature, top_k=1)

    def test_select_by_correlation_invalid_label_type(self) -> None:
        ds = Dataset(
            data=pl.DataFrame({'a': [1, 2], 'label': ['hello', 'world']}),
            schema=Schema(
                [
                    ColumnSpecification(name='a', column_type=ColumnType.NUMERIC),
                    ColumnSpecification(name='label', column_type=ColumnType.TEXT, column_role=ColumnRole.LABEL),
                ],
            ),
        )
        with pytest.raises(ValueError, match='Correlation can only be computed with label column of type numeric, boolean'):
            self._selector.select_by_correlation(dataset=ds, feature_subset='a', top_k=1)

    def test_select_by_correlation_top_k(self) -> None:
        out = self._selector.select_by_correlation(self._ds, ColumnType.NUMERIC, top_k=1)
        assert out == ['x3']

    def test_select_by_correlation_frac(self) -> None:
        out = self._selector.select_by_correlation(self._ds, ColumnType.NUMERIC, frac=0.5)
        assert out == ['x3', 'x2']
