import pytest

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.transformers.aggregating_transformers import ArithmeticAggregationTransformer
from auto_featurs.transformers.aggregating_transformers import CountTransformer
from auto_featurs.transformers.aggregating_transformers import FirstValueTransformer
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


class TestRollingWrapper:
    def setup_method(self) -> None:
        self._index_col = ColumnSpecification.datetime(name='DATE_FEATURE')
        self._time_window = '2d1h'

    def test_rolling_count_transform(self) -> None:
        count_transformer = CountTransformer()
        count_rolling_transformer = RollingWrapper(inner_transformer=count_transformer, index_column=self._index_col, time_window=self._time_window)

        df = BASIC_FRAME.with_columns(count_rolling_transformer.transform())

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={'count_in_the_last_2d1h': [1, 2, 3, 3, 3, 3]},
        )

    def test_rolling_first_value_transform(self) -> None:
        first_value_transformer = FirstValueTransformer(column=ColumnSpecification.numeric(name='NUMERIC_FEATURE'))
        first_value_rolling_transformer = RollingWrapper(inner_transformer=first_value_transformer, index_column=self._index_col, time_window=self._time_window)

        df = BASIC_FRAME.with_columns(first_value_rolling_transformer.transform())

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={'NUMERIC_FEATURE_first_value_in_the_last_2d1h': [0, 0, 0, 1, 2, 3]},
        )

    def test_rolling_mode_transform(self) -> None:
        mode_transformer = ModeTransformer(column=ColumnSpecification.ordinal(name='GROUPING_FEATURE_CAT_2'))
        mode_rolling_transformer = RollingWrapper(inner_transformer=mode_transformer, index_column=self._index_col, time_window=self._time_window)

        df = BASIC_FRAME.with_columns(mode_rolling_transformer.transform())

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={'GROUPING_FEATURE_CAT_2_mode_in_the_last_2d1h': ['CONSONANT', 'VOWEL', 'CONSONANT', 'CONSONANT', 'CONSONANT', 'CONSONANT']},
        )

    def test_rolling_num_unique_transform(self) -> None:
        num_unique_transformer = NumUniqueTransformer(column=ColumnSpecification.numeric(name='GROUPING_FEATURE_NUM'))
        num_unique_rolling_transformer = RollingWrapper(inner_transformer=num_unique_transformer, index_column=self._index_col, time_window=self._time_window)

        df = BASIC_FRAME.with_columns(num_unique_rolling_transformer.transform())

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={'GROUPING_FEATURE_NUM_num_unique_in_the_last_2d1h': [1, 2, 3, 2, 2, 2]},
        )

    @pytest.mark.parametrize(
        ('inner_transformer_type', 'expected_new_columns'),
        [
            (SumTransformer, {'NUMERIC_FEATURE_sum_in_the_last_2d1h': [0, 1, 3, 6, 9, 12]}),
            (MeanTransformer, {'NUMERIC_FEATURE_mean_in_the_last_2d1h': [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]}),
            (StdTransformer, {'NUMERIC_FEATURE_std_in_the_last_2d1h': [None, 0.707107, 1.0, 1.0, 1.0, 1.0]}),
            (ZscoreTransformer, {'NUMERIC_FEATURE_z_score_in_the_last_2d1h': [None, 0.707107, 1.0, 1.0, 1.0, 1.0]}),
        ],
    )
    def test_rolling_arithmetic_aggregation_transform(self, inner_transformer_type: type[ArithmeticAggregationTransformer], expected_new_columns: dict[str, list[int] | list[float]]) -> None:
        arithmetic_transformer = inner_transformer_type(column=ColumnSpecification.numeric(name='NUMERIC_FEATURE'))
        arithmetic_rolling_transformer = RollingWrapper(inner_transformer=arithmetic_transformer, index_column=self._index_col, time_window=self._time_window)

        df = BASIC_FRAME.with_columns(arithmetic_rolling_transformer.transform())

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns=expected_new_columns,
        )

    def test_rolling_combined_with_over(self) -> None:
        first_value_transformer = FirstValueTransformer(column=ColumnSpecification.numeric(name='NUMERIC_FEATURE'))
        first_value_over_transformer = OverWrapper(inner_transformer=first_value_transformer, over_columns=['GROUPING_FEATURE_NUM'])
        first_value_over_rolling_transformer = RollingWrapper(inner_transformer=first_value_over_transformer, index_column=self._index_col, time_window=self._time_window)

        df = BASIC_FRAME.with_columns(first_value_over_rolling_transformer.transform())

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={'NUMERIC_FEATURE_first_value_over_GROUPING_FEATURE_NUM_in_the_last_2d1h': [0, 1, 2, 1, 2, 3]},
        )
