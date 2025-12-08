import numpy as np
import polars as pl
import pytest

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
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
from auto_featurs.utils.utils_for_tests import BASIC_FRAME
from auto_featurs.utils.utils_for_tests import assert_new_columns_in_frame


class TestCountTransformer:
    def setup_method(self) -> None:
        self._count_transformer = CountTransformer()
        self._cumulative_count_transformer = CountTransformer(cumulative=True)
        self._filtered_count_transformer = CountTransformer(filtering_condition=pl.col('NUMERIC_FEATURE').ge(2).alias('NUMERIC_FEATURE_GE_2'))
        self._filtered_cumulative_count_transformer = CountTransformer(cumulative=True, filtering_condition=pl.col('NUMERIC_FEATURE').ge(2).alias('NUMERIC_FEATURE_GE_2'))

    def test_name_and_output_type(self) -> None:
        assert self._count_transformer.output_column_specification == ColumnSpecification.numeric(name='count')
        assert self._cumulative_count_transformer.output_column_specification == ColumnSpecification.numeric(name='cum_count')
        assert self._filtered_count_transformer.output_column_specification == ColumnSpecification.numeric(name='count_where_NUMERIC_FEATURE_GE_2')
        assert self._filtered_cumulative_count_transformer.output_column_specification == ColumnSpecification.numeric(name='cum_count_where_NUMERIC_FEATURE_GE_2')

    def test_count_transform(self) -> None:
        df = BASIC_FRAME.with_columns(self._count_transformer.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns={'count': [6, 6, 6, 6, 6, 6]})

    def test_cum_count_transform(self) -> None:
        df = BASIC_FRAME.with_columns(self._cumulative_count_transformer.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns={'cum_count': [1, 2, 3, 4, 5, 6]})

    def test_filtered_count_transform(self) -> None:
        df = BASIC_FRAME.with_columns(self._filtered_count_transformer.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns={'count_where_NUMERIC_FEATURE_GE_2': [4, 4, 4, 4, 4, 4]})

    def test_filtered_cum_count_transform(self) -> None:
        df = BASIC_FRAME.with_columns(self._filtered_cumulative_count_transformer.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns={'cum_count_where_NUMERIC_FEATURE_GE_2': [0, 0, 1, 2, 3, 4]})


class TestLaggedTransformer:
    def setup_method(self) -> None:
        self._lagged_1_categorical_transformer = LaggedTransformer(column=ColumnSpecification.ordinal(name='CATEGORICAL_FEATURE'), lag=1)
        self._lagged_1_transformer = LaggedTransformer(column=ColumnSpecification.numeric(name='NUMERIC_FEATURE'), lag=1)
        self._lagged_2_transformer = LaggedTransformer(column=ColumnSpecification.numeric(name='NUMERIC_FEATURE'), lag=2)

    def test_name_and_output_type(self) -> None:
        assert self._lagged_1_categorical_transformer.output_column_specification == ColumnSpecification.ordinal(name='CATEGORICAL_FEATURE_lagged_1')
        assert self._lagged_1_transformer.output_column_specification == ColumnSpecification.numeric(name='NUMERIC_FEATURE_lagged_1')
        assert self._lagged_2_transformer.output_column_specification == ColumnSpecification.numeric(name='NUMERIC_FEATURE_lagged_2')

    def test_lagged_transform(self) -> None:
        df = BASIC_FRAME.with_columns(
            self._lagged_1_categorical_transformer.transform(),
            self._lagged_1_transformer.transform(),
            self._lagged_2_transformer.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={
                'CATEGORICAL_FEATURE_lagged_1': [None, 'A', 'B', 'C', 'D', 'E'],
                'NUMERIC_FEATURE_lagged_1': [None, 0, 1, 2, 3, 4],
                'NUMERIC_FEATURE_lagged_2': [None, None, 0, 1, 2, 3],
            },
        )

    def test_fill_value(self) -> None:
        lagged_transformer = LaggedTransformer(column=ColumnSpecification(name='NUMERIC_FEATURE', column_type=ColumnType.NUMERIC), lag=2, fill_value=0)

        df = BASIC_FRAME.with_columns(lagged_transformer.transform())

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={'NUMERIC_FEATURE_lagged_2': [0, 0, 0, 1, 2, 3]},
        )


class TestFirstValueTransformer:
    def setup_method(self) -> None:
        self._first_value_transformer_ordinal = FirstValueTransformer(column=ColumnSpecification.ordinal(name='CATEGORICAL_FEATURE'))
        self._first_value_transformer_numeric = FirstValueTransformer(column=ColumnSpecification.numeric(name='NUMERIC_FEATURE'))
        self._filtered_first_value_transformer_numeric = FirstValueTransformer(
            column=ColumnSpecification.numeric(name='NUMERIC_FEATURE'),
            filtering_condition=~pl.col('BOOL_FEATURE').alias('NOT_BOOL_FEATURE'),
        )

    def test_name_and_output_type(self) -> None:
        assert self._first_value_transformer_ordinal.output_column_specification == ColumnSpecification.ordinal(name='CATEGORICAL_FEATURE_first_value')
        assert self._first_value_transformer_numeric.output_column_specification == ColumnSpecification.numeric(name='NUMERIC_FEATURE_first_value')
        assert self._filtered_first_value_transformer_numeric.output_column_specification == ColumnSpecification.numeric(name='NUMERIC_FEATURE_first_value_where_NOT_BOOL_FEATURE')

    def test_first_value_transform(self) -> None:
        df = BASIC_FRAME.with_columns(
            self._first_value_transformer_ordinal.transform(),
            self._first_value_transformer_numeric.transform(),
            self._filtered_first_value_transformer_numeric.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={
                'CATEGORICAL_FEATURE_first_value': ['A', 'A', 'A', 'A', 'A', 'A'],
                'NUMERIC_FEATURE_first_value': [0, 0, 0, 0, 0, 0],
                'NUMERIC_FEATURE_first_value_where_NOT_BOOL_FEATURE': [1, 1, 1, 1, 1, 1],
            },
        )


class TestModeTransformer:
    def setup_method(self) -> None:
        self._mode_transformer_ordinal = ModeTransformer(column=ColumnSpecification.ordinal(name='GROUPING_FEATURE_NUM'))
        self._mode_transformer_bool = ModeTransformer(column=ColumnSpecification.boolean(name='BOOL_FEATURE'))
        self._filtered_mode_transformer_ordinal = ModeTransformer(column=ColumnSpecification.ordinal(name='GROUPING_FEATURE_NUM'), filtering_condition=pl.col('BOOL_FEATURE'))

    def test_name_and_output_type(self) -> None:
        assert self._mode_transformer_ordinal.output_column_specification == ColumnSpecification.ordinal(name='GROUPING_FEATURE_NUM_mode')
        assert self._mode_transformer_bool.output_column_specification == ColumnSpecification.boolean(name='BOOL_FEATURE_mode')
        assert self._filtered_mode_transformer_ordinal.output_column_specification == ColumnSpecification.ordinal(name='GROUPING_FEATURE_NUM_mode_where_BOOL_FEATURE')

    def test_mode_transform(self) -> None:
        df = BASIC_FRAME.with_columns(
            self._mode_transformer_ordinal.transform(),
            self._mode_transformer_bool.transform(),
            self._filtered_mode_transformer_ordinal.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={
                'GROUPING_FEATURE_NUM_mode': ['ODD', 'ODD', 'ODD', 'ODD', 'ODD', 'ODD'],
                'BOOL_FEATURE_mode': [True, True, True, True, True, True],
                'GROUPING_FEATURE_NUM_mode_where_BOOL_FEATURE': ['EVEN', 'EVEN', 'EVEN', 'EVEN', 'EVEN', 'EVEN'],
            },
        )


class TestNumUniqueTransformer:
    def setup_method(self) -> None:
        self._num_unique_transformer_ordinal = NumUniqueTransformer(column=ColumnSpecification.ordinal(name='GROUPING_FEATURE_NUM'))
        self._num_unique_transformer_numeric = NumUniqueTransformer(column=ColumnSpecification.numeric(name='NUMERIC_FEATURE'))
        self._filtered_num_unique_transformer_ordinal = NumUniqueTransformer(column=ColumnSpecification.ordinal(name='GROUPING_FEATURE_NUM'), filtering_condition=pl.col('BOOL_FEATURE'))

    def test_name_and_output_type(self) -> None:
        assert self._num_unique_transformer_ordinal.output_column_specification == ColumnSpecification.numeric(name='GROUPING_FEATURE_NUM_num_unique')
        assert self._num_unique_transformer_numeric.output_column_specification == ColumnSpecification.numeric(name='NUMERIC_FEATURE_num_unique')
        assert self._filtered_num_unique_transformer_ordinal.output_column_specification == ColumnSpecification.numeric(name='GROUPING_FEATURE_NUM_num_unique_where_BOOL_FEATURE')

    def test_num_unique_transform(self) -> None:
        df = BASIC_FRAME.with_columns(
            self._num_unique_transformer_ordinal.transform(),
            self._num_unique_transformer_numeric.transform(),
            self._filtered_num_unique_transformer_ordinal.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={
                'GROUPING_FEATURE_NUM_num_unique': [3, 3, 3, 3, 3, 3],
                'NUMERIC_FEATURE_num_unique': [6, 6, 6, 6, 6, 6],
                'GROUPING_FEATURE_NUM_num_unique_where_BOOL_FEATURE': [2, 2, 2, 2, 2, 2],
            },
        )


class TestArithmeticAggregationTransformers:
    @pytest.mark.parametrize(
        ('transformer_type', 'expected_new_columns'),
        [
            (SumTransformer, {'NUMERIC_FEATURE_sum': [15, 15, 15, 15, 15, 15]}),
            (MeanTransformer, {'NUMERIC_FEATURE_mean': [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]}),
            (StdTransformer, {'NUMERIC_FEATURE_std': [1.870829, 1.870829, 1.870829, 1.870829, 1.870829, 1.870829]}),
            (ZscoreTransformer, {'NUMERIC_FEATURE_z_score': [-1.3363059905528512, -0.8017835943317106, -0.2672611981105702, 0.2672611981105702, 0.8017835943317106, 1.3363059905528512]}),
        ],
    )
    def test_basic_arithmetic_aggregation(self, transformer_type: type[ArithmeticAggregationTransformer], expected_new_columns: dict[str, list[int] | list[float]]) -> None:
        transformer = transformer_type(column='NUMERIC_FEATURE')
        df = BASIC_FRAME.with_columns(transformer.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns=expected_new_columns)

    @pytest.mark.parametrize(
        ('transformer_type', 'expected_new_columns'),
        [
            (SumTransformer, {'NUMERIC_FEATURE_cum_sum': [0, 1, 3, 6, 10, 15]}),
            (MeanTransformer, {'NUMERIC_FEATURE_cum_mean': [0.0, 0.5, 1, 1.5, 2, 2.5]}),
            (StdTransformer, {'NUMERIC_FEATURE_cum_std': [0.0, 0.5, 1.118034, 1.870829, 2.738613, 3.708099]}),
            (ZscoreTransformer, {'NUMERIC_FEATURE_cum_z_score': [np.nan, 1.0, 0.8944271819998318, 0.8017835943317106, 0.7302966866804473, 0.6741999067446689]}),
        ],
    )
    def test_cumulative_arithmetic_aggregation(self, transformer_type: type[ArithmeticAggregationTransformer], expected_new_columns: dict[str, list[int] | list[float]]) -> None:
        transformer = transformer_type(column='NUMERIC_FEATURE', cumulative=True)
        df = BASIC_FRAME.with_columns(transformer.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns=expected_new_columns)

    @pytest.mark.parametrize(
        ('transformer_type', 'expected_new_columns'),
        [
            (SumTransformer, {'NUMERIC_FEATURE_sum_where_BOOL_FEATURE': [6, 6, 6, 6, 6, 6]}),
            (MeanTransformer, {'NUMERIC_FEATURE_mean_where_BOOL_FEATURE': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]}),
            (StdTransformer, {'NUMERIC_FEATURE_std_where_BOOL_FEATURE': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]}),
            (
                ZscoreTransformer,
                {'NUMERIC_FEATURE_z_score_where_BOOL_FEATURE': [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]},
            ),
        ],
    )
    def test_filtered_arithmetic_aggregation(self, transformer_type: type[ArithmeticAggregationTransformer], expected_new_columns: dict[str, list[int] | list[float]]) -> None:
        transformer = transformer_type(column='NUMERIC_FEATURE', filtering_condition=pl.col('BOOL_FEATURE'))
        df = BASIC_FRAME.with_columns(transformer.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns=expected_new_columns)
