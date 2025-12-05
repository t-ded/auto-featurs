from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.schema import Schema
from auto_featurs.dataset.dataset import Dataset
from auto_featurs.pipeline.optimizer import OptimizationLevel
from auto_featurs.pipeline.pipeline import Pipeline
from auto_featurs.transformers.aggregating_transformers import ArithmeticAggregations
from auto_featurs.transformers.comparison_transformers import Comparisons
from auto_featurs.transformers.datetime_transformers import SeasonalOperation
from auto_featurs.transformers.numeric_transformers import ArithmeticOperation
from auto_featurs.transformers.numeric_transformers import PolynomialTransformer
from auto_featurs.utils.utils_for_tests import BASIC_FRAME
from auto_featurs.utils.utils_for_tests import assert_new_columns_in_frame


class TestPipeline:
    def setup_method(self) -> None:
        df = pl.LazyFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5]})
        self._simple_dataset = Dataset(data=df, schema=Schema([ColumnSpecification.numeric(name='NUMERIC_FEATURE')]))

    def test_transformers_from_init(self) -> None:
        pipeline = Pipeline(
            dataset=self._simple_dataset,
            transformers=[[PolynomialTransformer(column='NUMERIC_FEATURE', degree=2)]],
        )

        res = pipeline.collect()

        assert_frame_equal(
            res,
            pl.DataFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5], 'NUMERIC_FEATURE_pow_2': [0, 1, 4, 9, 16, 25]}),
        )

    def test_basic_layering(self) -> None:
        pipeline = Pipeline(dataset=self._simple_dataset)
        pipeline = pipeline.with_polynomial(subset=ColumnType.NUMERIC, degrees=[2])
        pipeline = pipeline.with_new_layer()
        pipeline = pipeline.with_polynomial(subset=ColumnType.NUMERIC, degrees=[2])

        res = pipeline.collect()

        assert_frame_equal(
            res,
            pl.DataFrame({
                'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5],
                'NUMERIC_FEATURE_pow_2': [0, 1, 4, 9, 16, 25],
                'NUMERIC_FEATURE_pow_2_pow_2': [0, 1, 16, 81, 256, 625],
            }),
        )

    def test_collect_plan(self) -> None:
        pipeline = Pipeline(dataset=self._simple_dataset)
        pipeline = pipeline.with_polynomial(subset=ColumnType.NUMERIC, degrees=[2])

        res = pipeline.collect_plan()

        assert_frame_equal(
            res.data,
            pl.LazyFrame({
                'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5],
                'NUMERIC_FEATURE_pow_2': [0, 1, 4, 9, 16, 25],
            }),
        )

    def test_pipeline_is_not_changed_inplace(self) -> None:
        pipeline = Pipeline(dataset=self._simple_dataset)
        pipeline_with_polynomial = pipeline.with_polynomial(subset=ColumnType.NUMERIC, degrees=[2])

        res = pipeline.collect()
        res_with_polynomial = pipeline_with_polynomial.collect()

        assert_frame_equal(res, pl.DataFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5]}))
        assert_frame_equal(res_with_polynomial, pl.DataFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5], 'NUMERIC_FEATURE_pow_2': [0, 1, 4, 9, 16, 25]}))

    @pytest.mark.parametrize(
        'optimization_level',
        [
            OptimizationLevel.NONE,
            OptimizationLevel.SKIP_SELF,
            OptimizationLevel.DEDUPLICATE_COMMUTATIVE,
        ],
    )
    def test_pipeline_optimization(self, optimization_level: OptimizationLevel) -> None:
        pipeline = Pipeline(
            dataset=Dataset(
                data=BASIC_FRAME,
                schema=Schema([
                    ColumnSpecification.numeric(name='NUMERIC_FEATURE'),
                    ColumnSpecification.numeric(name='NUMERIC_FEATURE_2'),
                ]),
            ),
            optimization_level=optimization_level,
        )
        pipeline = pipeline.with_arithmetic(left_subset=ColumnType.NUMERIC, right_subset=ColumnType.NUMERIC, operations=[ArithmeticOperation.ADD, ArithmeticOperation.SUBTRACT])

        expected_new_columns = {
            'NUMERIC_FEATURE_add_NUMERIC_FEATURE': [0, 2, 4, 6, 8, 10],
            'NUMERIC_FEATURE_add_NUMERIC_FEATURE_2': [0, 0, 0, 0, 0, 0],
            'NUMERIC_FEATURE_2_add_NUMERIC_FEATURE': [0, 0, 0, 0, 0, 0],
            'NUMERIC_FEATURE_2_add_NUMERIC_FEATURE_2': [0, -2, -4, -6, -8, -10],
            'NUMERIC_FEATURE_subtract_NUMERIC_FEATURE': [0, 0, 0, 0, 0, 0],
            'NUMERIC_FEATURE_subtract_NUMERIC_FEATURE_2': [0, 2, 4, 6, 8, 10],
            'NUMERIC_FEATURE_2_subtract_NUMERIC_FEATURE': [0, -2, -4, -6, -8, -10],
            'NUMERIC_FEATURE_2_subtract_NUMERIC_FEATURE_2': [0, 0, 0, 0, 0, 0],
        }

        if optimization_level >= OptimizationLevel.SKIP_SELF:
            expected_new_columns.pop('NUMERIC_FEATURE_add_NUMERIC_FEATURE')
            expected_new_columns.pop('NUMERIC_FEATURE_2_add_NUMERIC_FEATURE_2')
            expected_new_columns.pop('NUMERIC_FEATURE_subtract_NUMERIC_FEATURE')
            expected_new_columns.pop('NUMERIC_FEATURE_2_subtract_NUMERIC_FEATURE_2')
        if optimization_level >= OptimizationLevel.DEDUPLICATE_COMMUTATIVE:
            expected_new_columns.pop('NUMERIC_FEATURE_2_add_NUMERIC_FEATURE')

        res = pipeline.collect()

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=res,
            expected_new_columns=expected_new_columns,
        )

    def test_basic_sample_with_all_transformers(self) -> None:
        pipeline = Pipeline(
            dataset=Dataset(
                data=BASIC_FRAME,
                schema=Schema([
                    ColumnSpecification.numeric(name='NUMERIC_FEATURE'),
                    ColumnSpecification.numeric(name='NUMERIC_FEATURE_2'),
                    ColumnSpecification.ordinal(name='CATEGORICAL_FEATURE'),
                    ColumnSpecification.nominal(name='CATEGORICAL_FEATURE_2'),
                    ColumnSpecification.datetime(name='DATE_FEATURE'),
                    ColumnSpecification.boolean(name='BOOL_FEATURE'),
                ]),
            ),
        )
        pipeline = (
            pipeline
            .with_seasonal(subset='DATE_FEATURE', operations=[SeasonalOperation.DAY_OF_WEEK])
            .with_polynomial(subset=ColumnType.NUMERIC, degrees=[2, 3])
            .with_arithmetic(
                left_subset=ColumnType.NUMERIC, right_subset=ColumnType.NUMERIC,
                operations=[ArithmeticOperation.ADD, ArithmeticOperation.SUBTRACT, ArithmeticOperation.MULTIPLY, ArithmeticOperation.DIVIDE],
            )
            .with_comparison(
                left_subset=ColumnType.NUMERIC, right_subset=ColumnType.NUMERIC,
                comparisons=[Comparisons.EQUAL, Comparisons.GREATER_THAN, Comparisons.GREATER_OR_EQUAL],
            )
            .with_comparison(
                left_subset=[ColumnType.ORDINAL, ColumnType.NOMINAL], right_subset=[ColumnType.ORDINAL, ColumnType.NOMINAL],
                comparisons=[Comparisons.EQUAL, Comparisons.GREATER_THAN, Comparisons.GREATER_OR_EQUAL],
            )
            .with_count(over_columns_combinations=[[], ['GROUPING_FEATURE_NUM'], ['GROUPING_FEATURE_NUM', 'GROUPING_FEATURE_CAT_2']])
            .with_count(over_columns_combinations=[['GROUPING_FEATURE_NUM']], cumulative=True)
            .with_count(
                # over_columns_combinations=[[], ['GROUPING_FEATURE_NUM']],  # TODO: Uncomment this with new polars release (allowing window expressions in aggregation)
                time_windows=['2d', timedelta(days=2, hours=1)],
                index_column_name='DATE_FEATURE',
            )
            .with_lagged(subset=ColumnType.NUMERIC, lags=[1], over_columns_combinations=[[], ['GROUPING_FEATURE_NUM'], ['GROUPING_FEATURE_NUM', 'GROUPING_FEATURE_CAT_2']], fill_value=0)
            .with_lagged(subset=[ColumnType.ORDINAL, ColumnType.NOMINAL], lags=[1, 2], fill_value='missing')
            .with_first_value(subset=[ColumnType.NUMERIC, ColumnType.ORDINAL], over_columns_combinations=[[], ['GROUPING_FEATURE_NUM'], ['GROUPING_FEATURE_NUM', 'GROUPING_FEATURE_CAT_2']])
            .with_mode(subset=[ColumnType.BOOLEAN], over_columns_combinations=[[], ['GROUPING_FEATURE_NUM']])
            .with_num_unique(subset=[ColumnType.BOOLEAN], over_columns_combinations=[[], ['GROUPING_FEATURE_NUM'], ['GROUPING_FEATURE_NUM', 'GROUPING_FEATURE_CAT_2']])
            .with_arithmetic_aggregation(
                subset=ColumnType.NUMERIC,
                aggregations=[ArithmeticAggregations.SUM, ArithmeticAggregations.MEAN, ArithmeticAggregations.STD, ArithmeticAggregations.ZSCORE],
                over_columns_combinations=[['GROUPING_FEATURE_NUM'], ['GROUPING_FEATURE_NUM', 'GROUPING_FEATURE_CAT_2']],
            )
        )

        res = pipeline.collect()

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=res,
            expected_new_columns={
                'DATE_FEATURE_day_of_week': [6, 7, 1, 2, 3, 4],
                'NUMERIC_FEATURE_pow_2': [0, 1, 4, 9, 16, 25],
                'NUMERIC_FEATURE_pow_3': [0, 1, 8, 27, 64, 125],
                'NUMERIC_FEATURE_2_pow_2': [0, 1, 4, 9, 16, 25],
                'NUMERIC_FEATURE_2_pow_3': [0, -1, -8, -27, -64, -125],
                'NUMERIC_FEATURE_add_NUMERIC_FEATURE': [0, 2, 4, 6, 8, 10],
                'NUMERIC_FEATURE_add_NUMERIC_FEATURE_2': [0, 0, 0, 0, 0, 0],
                'NUMERIC_FEATURE_2_add_NUMERIC_FEATURE': [0, 0, 0, 0, 0, 0],
                'NUMERIC_FEATURE_2_add_NUMERIC_FEATURE_2': [0, -2, -4, -6, -8, -10],
                'NUMERIC_FEATURE_subtract_NUMERIC_FEATURE': [0, 0, 0, 0, 0, 0],
                'NUMERIC_FEATURE_subtract_NUMERIC_FEATURE_2': [0, 2, 4, 6, 8, 10],
                'NUMERIC_FEATURE_2_subtract_NUMERIC_FEATURE': [0, -2, -4, -6, -8, -10],
                'NUMERIC_FEATURE_2_subtract_NUMERIC_FEATURE_2': [0, 0, 0, 0, 0, 0],
                'NUMERIC_FEATURE_multiply_NUMERIC_FEATURE': [0, 1, 4, 9, 16, 25],
                'NUMERIC_FEATURE_multiply_NUMERIC_FEATURE_2': [0, -1, -4, -9, -16, -25],
                'NUMERIC_FEATURE_2_multiply_NUMERIC_FEATURE': [0, -1, -4, -9, -16, -25],
                'NUMERIC_FEATURE_2_multiply_NUMERIC_FEATURE_2': [0, 1, 4, 9, 16, 25],
                'NUMERIC_FEATURE_divide_NUMERIC_FEATURE': [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0],
                'NUMERIC_FEATURE_divide_NUMERIC_FEATURE_2': [np.nan, -1.0, -1.0, -1.0, -1.0, -1.0],
                'NUMERIC_FEATURE_2_divide_NUMERIC_FEATURE': [np.nan, -1.0, -1.0, -1.0, -1.0, -1.0],
                'NUMERIC_FEATURE_2_divide_NUMERIC_FEATURE_2': [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0],
                'NUMERIC_FEATURE_equal_NUMERIC_FEATURE': [True, True, True, True, True, True],
                'NUMERIC_FEATURE_equal_NUMERIC_FEATURE_2': [True, False, False, False, False, False],
                'NUMERIC_FEATURE_2_equal_NUMERIC_FEATURE': [True, False, False, False, False, False],
                'NUMERIC_FEATURE_2_equal_NUMERIC_FEATURE_2': [True, True, True, True, True, True],
                'NUMERIC_FEATURE_greater_than_NUMERIC_FEATURE': [False, False, False, False, False, False],
                'NUMERIC_FEATURE_greater_than_NUMERIC_FEATURE_2': [False, True, True, True, True, True],
                'NUMERIC_FEATURE_2_greater_than_NUMERIC_FEATURE': [False, False, False, False, False, False],
                'NUMERIC_FEATURE_2_greater_than_NUMERIC_FEATURE_2': [False, False, False, False, False, False],
                'NUMERIC_FEATURE_greater_or_equal_NUMERIC_FEATURE': [True, True, True, True, True, True],
                'NUMERIC_FEATURE_greater_or_equal_NUMERIC_FEATURE_2': [True, True, True, True, True, True],
                'NUMERIC_FEATURE_2_greater_or_equal_NUMERIC_FEATURE': [True, False, False, False, False, False],
                'NUMERIC_FEATURE_2_greater_or_equal_NUMERIC_FEATURE_2': [True, True, True, True, True, True],
                'CATEGORICAL_FEATURE_equal_CATEGORICAL_FEATURE': [True, True, True, True, True, True],
                'CATEGORICAL_FEATURE_equal_CATEGORICAL_FEATURE_2': [False, False, False, False, False, False],
                'CATEGORICAL_FEATURE_2_equal_CATEGORICAL_FEATURE': [False, False, False, False, False, False],
                'CATEGORICAL_FEATURE_2_equal_CATEGORICAL_FEATURE_2': [True, True, True, True, True, True],
                'CATEGORICAL_FEATURE_greater_than_CATEGORICAL_FEATURE': [False, False, False, False, False, False],
                'CATEGORICAL_FEATURE_greater_than_CATEGORICAL_FEATURE_2': [False, False, False, True, True, True],
                'CATEGORICAL_FEATURE_2_greater_than_CATEGORICAL_FEATURE': [True, True, True, False, False, False],
                'CATEGORICAL_FEATURE_2_greater_than_CATEGORICAL_FEATURE_2': [False, False, False, False, False, False],
                'CATEGORICAL_FEATURE_greater_or_equal_CATEGORICAL_FEATURE': [True, True, True, True, True, True],
                'CATEGORICAL_FEATURE_greater_or_equal_CATEGORICAL_FEATURE_2': [False, False, False, True, True, True],
                'CATEGORICAL_FEATURE_2_greater_or_equal_CATEGORICAL_FEATURE': [True, True, True, False, False, False],
                'CATEGORICAL_FEATURE_2_greater_or_equal_CATEGORICAL_FEATURE_2': [True, True, True, True, True, True],
                'count': [6, 6, 6, 6, 6, 6],
                'count_over_GROUPING_FEATURE_NUM': [1, 3, 2, 3, 2, 3],
                'count_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [1, 2, 2, 1, 2, 2],
                'cum_count_over_GROUPING_FEATURE_NUM': [1, 1, 1, 2, 2, 3],
                'count_in_the_last_2d': [1, 2, 2, 2, 2, 2],
                'count_in_the_last_2d1h': [1, 2, 3, 3, 3, 3],
                'NUMERIC_FEATURE_lagged_1': [0, 0, 1, 2, 3, 4],
                'NUMERIC_FEATURE_2_lagged_1': [0, 0, -1, -2, -3, -4],
                'NUMERIC_FEATURE_lagged_1_over_GROUPING_FEATURE_NUM': [0, 0, 0, 1, 2, 3],
                'NUMERIC_FEATURE_lagged_1_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [0, 0, 0, 0, 2, 1],
                'NUMERIC_FEATURE_2_lagged_1_over_GROUPING_FEATURE_NUM': [0, 0, 0, -1, -2, -3],
                'NUMERIC_FEATURE_2_lagged_1_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [0, 0, 0, 0, -2, -1],
                'CATEGORICAL_FEATURE_lagged_1': ['missing', 'A', 'B', 'C', 'D', 'E'],
                'CATEGORICAL_FEATURE_lagged_2': ['missing', 'missing', 'A', 'B', 'C', 'D'],
                'CATEGORICAL_FEATURE_2_lagged_1': ['missing', 'F', 'E', 'D', 'C', 'B'],
                'CATEGORICAL_FEATURE_2_lagged_2': ['missing', 'missing', 'F', 'E', 'D', 'C'],
                'NUMERIC_FEATURE_first_value': [0, 0, 0, 0, 0, 0],
                'NUMERIC_FEATURE_2_first_value': [0, 0, 0, 0, 0, 0],
                'CATEGORICAL_FEATURE_first_value': ['A', 'A', 'A', 'A', 'A', 'A'],
                'NUMERIC_FEATURE_first_value_over_GROUPING_FEATURE_NUM': [0, 1, 2, 1, 2, 1],
                'NUMERIC_FEATURE_first_value_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [0, 1, 2, 3, 2, 1],
                'NUMERIC_FEATURE_2_first_value_over_GROUPING_FEATURE_NUM': [0, -1, -2, -1, -2, -1],
                'NUMERIC_FEATURE_2_first_value_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [0, -1, -2, -3, -2, -1],
                'CATEGORICAL_FEATURE_first_value_over_GROUPING_FEATURE_NUM': ['A', 'B', 'C', 'B', 'C', 'B'],
                'CATEGORICAL_FEATURE_first_value_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': ['A', 'B', 'C', 'D', 'C', 'B'],
                'BOOL_FEATURE_mode': [True, True, True, True, True, True],
                'BOOL_FEATURE_mode_over_GROUPING_FEATURE_NUM': [True, False, True, False, True, False],
                'BOOL_FEATURE_num_unique': [2, 2, 2, 2, 2, 2],
                'BOOL_FEATURE_num_unique_over_GROUPING_FEATURE_NUM': [1, 1, 1, 1, 1, 1],
                'BOOL_FEATURE_num_unique_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [1, 1, 1, 1, 1, 1],
                'NUMERIC_FEATURE_sum_over_GROUPING_FEATURE_NUM': [0, 9, 6, 9, 6, 9],
                'NUMERIC_FEATURE_sum_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [0, 6, 6, 3, 6, 6],
                'NUMERIC_FEATURE_2_sum_over_GROUPING_FEATURE_NUM': [0, -9, -6, -9, -6, -9],
                'NUMERIC_FEATURE_2_sum_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [0, -6, -6, -3, -6, -6],
                'NUMERIC_FEATURE_mean_over_GROUPING_FEATURE_NUM': [0.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                'NUMERIC_FEATURE_mean_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [0.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                'NUMERIC_FEATURE_2_mean_over_GROUPING_FEATURE_NUM': [0.0, -3.0, -3.0, -3.0, -3.0, -3.0],
                'NUMERIC_FEATURE_2_mean_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [0.0, -3.0, -3.0, -3.0, -3.0, -3.0],
                'NUMERIC_FEATURE_std_over_GROUPING_FEATURE_NUM': [None, 2.0, 1.414214, 2.0, 1.414214, 2.0],
                'NUMERIC_FEATURE_std_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [None, 2.828427, 1.414214, None, 1.414214, 2.828427],
                'NUMERIC_FEATURE_2_std_over_GROUPING_FEATURE_NUM': [None, 2.0, 1.414214, 2.0, 1.414214, 2.0],
                'NUMERIC_FEATURE_2_std_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [None, 2.828427, 1.414214, None, 1.414214, 2.828427],
                'NUMERIC_FEATURE_z_score_over_GROUPING_FEATURE_NUM': [None, -1.0, -0.707107, 0.0, 0.707107, 1.0],
                'NUMERIC_FEATURE_z_score_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [None, -0.707107, -0.707107, None, 0.707107, 0.707107],
                'NUMERIC_FEATURE_2_z_score_over_GROUPING_FEATURE_NUM': [None, 1.0, 0.707107, 0.0, -0.707107, -1.0],
                'NUMERIC_FEATURE_2_z_score_over_GROUPING_FEATURE_NUM_and_GROUPING_FEATURE_CAT_2': [None, 0.707107, 0.707107, None, -0.707107, -0.707107],
            },
        )
