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
            ColumnSpecification(name='y', column_type=ColumnType.ORDINAL, column_role=ColumnRole.LABEL),
        ])
        df = pl.DataFrame({
            'x1': [0, 0, 0, 0],
            'x2': [10, 9, 8, 7],
            'x3': [0, 1, 0, 1],
            'x4': [2, 4, 6, 8],
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
    def test_get_num_to_select_invalid(self, k: Optional[int], frac: Optional[float], expected_msg: str) -> None:
        with pytest.raises(ValueError, match=expected_msg):
            self._selector.select_by_correlation(dataset=self._ds, feature_subset='x1', top_k=k, frac=frac)

    def test_select_by_correlation_top_k(self) -> None:
        out = self._selector.select_by_correlation(self._ds, ColumnType.NUMERIC, top_k=1)
        assert out == ['x3']

    def test_select_by_correlation_frac(self) -> None:
        out = self._selector.select_by_correlation(self._ds, ColumnType.NUMERIC, frac=0.5)
        assert out == ['x3', 'x2']
