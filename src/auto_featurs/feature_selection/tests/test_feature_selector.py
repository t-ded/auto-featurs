from typing import Optional

import polars as pl
import pytest

from auto_featurs.base.column_specification import ColumnRole
from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.schema import Schema
from auto_featurs.dataset.dataset import Dataset
from auto_featurs.feature_selection.feature_selector import FeatureSelector
from auto_featurs.feature_selection.feature_selector import SelectionMethod
from auto_featurs.feature_selection.feature_selector import SelectionReport
from auto_featurs.utils.constants import INFINITY


class TestSelector:
    def setup_method(self) -> None:
        schema = Schema([
            ColumnSpecification(name='x1', column_type=ColumnType.NUMERIC),
            ColumnSpecification(name='x2', column_type=ColumnType.NUMERIC),
            ColumnSpecification(name='x3', column_type=ColumnType.BOOLEAN),
            ColumnSpecification(name='x4', column_type=ColumnType.NUMERIC),
            ColumnSpecification(name='z1', column_type=ColumnType.NOMINAL),
            ColumnSpecification(name='z2', column_type=ColumnType.TEXT),
            ColumnSpecification(name='y', column_type=ColumnType.BOOLEAN, column_role=ColumnRole.LABEL),
        ])
        df = pl.DataFrame({
            'x1': [0, 0, 0, 0],
            'x2': [10, 9, 8, 7],
            'x3': [False, True, False, True],
            'x4': [2, 4, 6, 8],
            'z1': ['a', 'b', 'c', 'd'],
            'z2': ['hello', 'world', 'foo', 'bar'],
            'y': [0, 1, 0, 1],
        })
        self._ds = Dataset(df, schema=schema)
        self._selector = FeatureSelector()

        self._mock_report = SelectionReport(
            feature_names=pl.Series(['a', 'b', 'c', 'd']),
            stat_values=pl.Series([0.0, 0.5, 0.5, 1.0]),
            method=SelectionMethod.CORRELATION,
        )

    @pytest.mark.parametrize(
        ('k', 'frac', 'expected_msg'),
        [
            (None, None, 'Exactly one of k or frac must be specified'),
            (0, None, 'k must be at least 1 but 0 was given.'),
            (None, 2.0, 'frac must be between 0 and 1 but 2.0 was given.'),
        ],
    )
    def test_select_features_invalid_num_to_select(self, k: Optional[int], frac: Optional[float], expected_msg: str) -> None:
        with pytest.raises(ValueError, match=expected_msg):
            self._selector.select_features(report=self._mock_report, top_k=k, frac=frac)

    def test_select_features_top_k(self) -> None:
        out = self._selector.select_features(report=self._mock_report, top_k=1)
        assert out == ['d']

    def test_select_features_frac(self) -> None:
        out = self._selector.select_features(report=self._mock_report, frac=0.5)
        assert out == ['d', 'b']

    @pytest.mark.parametrize('feature', ['z1', 'z2'])
    def test_correlation_report_invalid_feature_type(self, feature: str) -> None:
        with pytest.raises(ValueError, match=f'Correlation can only be computed for numeric, boolean, ordinal columns, but {feature} is of type ColumnType..'):
            self._selector.get_report(dataset=self._ds, feature_subset=feature, method=SelectionMethod.CORRELATION)

    def test_correlation_report_invalid_label_type(self) -> None:
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
            self._selector.get_report(dataset=ds, feature_subset='a', method=SelectionMethod.CORRELATION)

    def test_correlation_report(self) -> None:
        out = self._selector.get_report(self._ds, (ColumnType.NUMERIC | ColumnType.BOOLEAN) & ~ColumnRole.LABEL, method=SelectionMethod.CORRELATION)
        dict_res = dict(out.to_frame().rows())
        assert dict_res['x1'] == 0.0
        assert dict_res['x2'] == 0.4472135954999579
        assert dict_res['x3'] == 1.0
        assert dict_res['x4'] == 0.4472135954999579
        assert len(dict_res) == 4

    @pytest.mark.parametrize('feature', ['z1', 'z2'])
    def test_select_by_ttest_invalid_feature_type(self, feature: str) -> None:
        with pytest.raises(ValueError, match=f'T-Test can only be computed for numeric, boolean, ordinal columns, but {feature} is of type ColumnType..'):
            self._selector.get_report(dataset=self._ds, feature_subset=feature, method=SelectionMethod.T_TEST)

    def test_select_by_ttest_invalid_label_type(self) -> None:
        ds = Dataset(
            data=pl.DataFrame({'a': [1, 2], 'label': [1, 2]}),
            schema=Schema(
                [
                    ColumnSpecification(name='a', column_type=ColumnType.NUMERIC),
                    ColumnSpecification(name='label', column_type=ColumnType.ORDINAL, column_role=ColumnRole.LABEL),
                ],
            ),
        )
        with pytest.raises(ValueError, match='T-Test can only be computed with label column of type boolean'):
            self._selector.get_report(dataset=ds, feature_subset='a', method=SelectionMethod.T_TEST)

    def test_ttest_report(self) -> None:
        out = self._selector.get_report(self._ds, (ColumnType.NUMERIC | ColumnType.BOOLEAN) & ~ColumnRole.LABEL, method=SelectionMethod.T_TEST)
        dict_res = dict(out.to_frame().rows())
        assert dict_res['x1'] == 0.0
        assert dict_res['x2'] == 0.7071067811865475
        assert dict_res['x3'] == INFINITY
        assert dict_res['x4'] == 0.7071067811865475
        assert len(dict_res) == 4
