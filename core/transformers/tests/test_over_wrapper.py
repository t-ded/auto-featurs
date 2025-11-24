import pytest

from core.base.column_specification import ColumnSpecification
from core.transformers.aggregating_transformers import ArithmeticAggregationTransformer
from core.transformers.aggregating_transformers import LaggedTransformer
from core.transformers.aggregating_transformers import MeanTransformer
from core.transformers.aggregating_transformers import StdTransformer
from core.transformers.aggregating_transformers import SumTransformer
from core.transformers.over_wrapper import OverWrapper
from utils.utils_for_tests import BASIC_FRAME
from utils.utils_for_tests import assert_new_columns_in_frame


class TestOverWrapper:
    def setup_method(self) -> None:
        self._num_group = ['GROUPING_FEATURE_NUM']
        self._num_cat_group = ['GROUPING_FEATURE_NUM', 'GROUPING_FEATURE_CAT_2']

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

    @pytest.mark.parametrize(
        'inner_transformer_type, expected_new_columns',
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
