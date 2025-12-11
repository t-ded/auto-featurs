import numpy as np
import pytest

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.transformers.aggregating_transformers import ArithmeticAggregationTransformer
from auto_featurs.transformers.aggregating_transformers import CountTransformer
from auto_featurs.transformers.aggregating_transformers import FirstValueTransformer
from auto_featurs.transformers.aggregating_transformers import LaggedTransformer
from auto_featurs.transformers.aggregating_transformers import MeanTransformer
from auto_featurs.transformers.aggregating_transformers import ModeTransformer
from auto_featurs.transformers.aggregating_transformers import NumUniqueTransformer
from auto_featurs.transformers.aggregating_transformers import StdTransformer
from auto_featurs.transformers.aggregating_transformers import SumTransformer
from auto_featurs.transformers.aggregating_transformers import ZscoreTransformer
from auto_featurs.transformers.over_wrapper import OverWrapper
from auto_featurs.transformers.rolling_wrapper import RollingWrapper
from auto_featurs.utils.utils_for_tests import BASIC_FRAME
from auto_featurs.utils.utils_for_tests import assert_new_columns_in_frame


class TestOverWrapper:
    def setup_method(self) -> None:
        self._num_group = ['GROUPING_FEATURE_NUM']
        self._num_cat_group = ['GROUPING_FEATURE_NUM', 'GROUPING_FEATURE_CAT_2']

    def test_grouped_count_transform(self) -> None:
        count_transformer = CountTransformer()
        count_over_grouping_num_transformer = OverWrapper(inner_transformer=count_transformer, over_columns=self._num_group)
        count_over_grouping_num_cat_transformer = OverWrapper(inner_transformer=count_transformer, over_columns=self._num_cat_group)

        df = BASIC_FRAME.with_columns(
            count_over_grouping_num_transformer.transform(),
            count_over_grouping_num_cat_transformer.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={
                'count_over_GROUPING_FEATURE_NUM': [1, 3, 2, 3, 2, 3],
                'count_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [1, 2, 2, 1, 2, 2],
            },
        )

    def test_grouped_cumulative_count_transform(self) -> None:
        cum_count_transformer = CountTransformer(cumulative=True)
        cum_count_over_grouping_num_transformer = OverWrapper(inner_transformer=cum_count_transformer, over_columns=self._num_group)
        cum_count_over_grouping_num_cat_transformer = OverWrapper(inner_transformer=cum_count_transformer, over_columns=self._num_cat_group)

        df = BASIC_FRAME.with_columns(
            cum_count_over_grouping_num_transformer.transform(),
            cum_count_over_grouping_num_cat_transformer.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={
                'cum_count_over_GROUPING_FEATURE_NUM': [1, 1, 1, 2, 2, 3],
                'cum_count_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [1, 1, 1, 1, 2, 2],
            },
        )

    def test_grouped_lagged_transform(self) -> None:
        lagged_1_transformer = LaggedTransformer(column=ColumnSpecification.numeric(name='NUMERIC_FEATURE'), lag=1)
        lagged_1_over_grouping_num_transformer = OverWrapper(inner_transformer=lagged_1_transformer, over_columns=self._num_group)
        lagged_1_over_grouping_num_cat_transformer = OverWrapper(inner_transformer=lagged_1_transformer, over_columns=self._num_cat_group)

        df = BASIC_FRAME.with_columns(
            lagged_1_over_grouping_num_transformer.transform(),
            lagged_1_over_grouping_num_cat_transformer.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={
                'NUMERIC_FEATURE_lagged_1_over_GROUPING_FEATURE_NUM': [None, None, None, 1, 2, 3],
                'NUMERIC_FEATURE_lagged_1_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [None, None, None, None, 2, 1],
            },
        )

    def test_grouped_first_value_transform(self) -> None:
        first_value_transformer = FirstValueTransformer(column=ColumnSpecification.numeric(name='NUMERIC_FEATURE'))
        first_value_over_grouping_num_transformer = OverWrapper(inner_transformer=first_value_transformer, over_columns=self._num_group)
        first_value_over_grouping_num_cat_transformer = OverWrapper(inner_transformer=first_value_transformer, over_columns=self._num_cat_group)

        df = BASIC_FRAME.with_columns(
            first_value_over_grouping_num_transformer.transform(),
            first_value_over_grouping_num_cat_transformer.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={
                'NUMERIC_FEATURE_first_value_over_GROUPING_FEATURE_NUM': [0, 1, 2, 1, 2, 1],
                'NUMERIC_FEATURE_first_value_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [0, 1, 2, 3, 2, 1],
            },
        )

    def test_grouped_mode_transform(self) -> None:
        mode_transformer = ModeTransformer(column=ColumnSpecification.boolean(name='BOOL_FEATURE'))
        mode_over_grouping_num_transformer = OverWrapper(inner_transformer=mode_transformer, over_columns=self._num_group)
        mode_over_grouping_num_cat_transformer = OverWrapper(inner_transformer=mode_transformer, over_columns=self._num_cat_group)

        df = BASIC_FRAME.with_columns(
            mode_over_grouping_num_transformer.transform(),
            mode_over_grouping_num_cat_transformer.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={
                'BOOL_FEATURE_mode_over_GROUPING_FEATURE_NUM': [True, False, True, False, True, False],
                'BOOL_FEATURE_mode_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [True, False, True, False, True, False],
            },
        )

    def test_grouped_num_unique_transform(self) -> None:
        num_unique_transformer = NumUniqueTransformer(column=ColumnSpecification.ordinal(name='GROUPING_FEATURE_NUM'))
        num_unique_over_bool_transformer = OverWrapper(inner_transformer=num_unique_transformer, over_columns=['BOOL_FEATURE'])

        df = BASIC_FRAME.with_columns(
            num_unique_over_bool_transformer.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={'GROUPING_FEATURE_NUM_num_unique_over_BOOL_FEATURE': [2, 1, 2, 1, 2, 1]},
        )

    @pytest.mark.parametrize(
        ('inner_transformer_type', 'expected_new_columns'),
        [
            (SumTransformer, {
                'NUMERIC_FEATURE_sum_over_GROUPING_FEATURE_NUM': [0, 9, 6, 9, 6, 9],
                'NUMERIC_FEATURE_sum_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [0, 6, 6, 3, 6, 6],
            }),
            (MeanTransformer, {
                'NUMERIC_FEATURE_mean_over_GROUPING_FEATURE_NUM': [0.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                'NUMERIC_FEATURE_mean_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [0.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            }),
            (StdTransformer, {
                'NUMERIC_FEATURE_std_over_GROUPING_FEATURE_NUM': [None, 2.0, 1.414214, 2.0, 1.414214, 2.0],
                'NUMERIC_FEATURE_std_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [None, 2.828427, 1.414214, None, 1.414214, 2.828427],
            }),
            (ZscoreTransformer, {
                'NUMERIC_FEATURE_z_score_over_GROUPING_FEATURE_NUM': [None, -1.0, -0.707107, 0.0, 0.707107, 1.0],
                'NUMERIC_FEATURE_z_score_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [None, -0.707107, -0.707107, None, 0.707107, 0.707107],
            }),
        ],
    )
    def test_grouped_arithmetic_aggregation_transform(self, inner_transformer_type: type[ArithmeticAggregationTransformer], expected_new_columns: dict[str, list[int] | list[float]]) -> None:
        arithmetic_transformer = inner_transformer_type(column=ColumnSpecification.numeric(name='NUMERIC_FEATURE'))
        arithmetic_over_grouping_num_transformer = OverWrapper(inner_transformer=arithmetic_transformer, over_columns=self._num_group)
        arithmetic_over_grouping_num_cat_transformer = OverWrapper(inner_transformer=arithmetic_transformer, over_columns=self._num_cat_group)

        df = BASIC_FRAME.with_columns(
            arithmetic_over_grouping_num_transformer.transform(),
            arithmetic_over_grouping_num_cat_transformer.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns=expected_new_columns,
        )

    @pytest.mark.parametrize(
        ('inner_transformer_type', 'expected_new_columns'),
        [
            (SumTransformer, {
                'NUMERIC_FEATURE_cum_sum_over_GROUPING_FEATURE_NUM': [0, 1, 2, 4, 6, 9],
                'NUMERIC_FEATURE_cum_sum_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [0, 1, 2, 3, 6, 6],
            }),
            (MeanTransformer, {
                'NUMERIC_FEATURE_cum_mean_over_GROUPING_FEATURE_NUM': [0.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                'NUMERIC_FEATURE_cum_mean_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [0.0, 1.0, 2.0, 3.0, 3.0, 3.0],
            }),
            (StdTransformer, {
                'NUMERIC_FEATURE_cum_std_over_GROUPING_FEATURE_NUM': [0.0, 0.0, 0.0, 1.0, 1.0, 2.236068],
                'NUMERIC_FEATURE_cum_std_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
            }),
            (ZscoreTransformer, {
                'NUMERIC_FEATURE_cum_z_score_over_GROUPING_FEATURE_NUM': [np.nan, np.nan, np.nan, 1.0, 1.0, 0.894427],
                'NUMERIC_FEATURE_cum_z_score_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [np.nan, np.nan, np.nan, np.nan, 1.0, 1.0],
            }),
        ],
    )
    def test_grouped_cumulative_arithmetic_aggregation_transform(
            self,
            inner_transformer_type: type[ArithmeticAggregationTransformer],
            expected_new_columns: dict[str, list[int] | list[float]],
    ) -> None:
        cumulative_arithmetic_transformer = inner_transformer_type(column=ColumnSpecification.numeric(name='NUMERIC_FEATURE'), cumulative=True)
        cumulative_arithmetic_over_grouping_num_transformer = OverWrapper(inner_transformer=cumulative_arithmetic_transformer, over_columns=self._num_group)
        cumulative_arithmetic_over_grouping_num_cat_transformer = OverWrapper(inner_transformer=cumulative_arithmetic_transformer, over_columns=self._num_cat_group)

        df = BASIC_FRAME.with_columns(
            cumulative_arithmetic_over_grouping_num_transformer.transform(),
            cumulative_arithmetic_over_grouping_num_cat_transformer.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns=expected_new_columns,
        )

    def test_over_combined_with_rolling(self) -> None:
        first_value_transformer = FirstValueTransformer(column=ColumnSpecification.numeric(name='NUMERIC_FEATURE'))
        first_value_rolling_transformer = RollingWrapper(inner_transformer=first_value_transformer, index_column=ColumnSpecification.datetime(name='DATE_FEATURE'), time_window='2d1h')
        first_value_rolling_over_transformer = OverWrapper(inner_transformer=first_value_rolling_transformer, over_columns=self._num_group)

        df = BASIC_FRAME.with_columns(first_value_rolling_over_transformer.transform())

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={'NUMERIC_FEATURE_first_value_in_the_last_2d1h_over_GROUPING_FEATURE_NUM': [0, 1, 2, 1, 2, 3]},
        )
