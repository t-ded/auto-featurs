from core.base.column_specification import ColumnSpecification
from core.transformers.aggregating_transformers import LaggedTransformer
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
