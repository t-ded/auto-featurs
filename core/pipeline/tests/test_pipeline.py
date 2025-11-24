import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from core.base.column_specification import ColumnSpecification
from core.base.column_specification import ColumnType
from core.pipeline.optimizer import OptimizationLevel
from core.pipeline.pipeline import Pipeline
from core.transformers.aggregating_transformers import ArithmeticAggregations
from core.transformers.comparison_transformers import Comparisons
from core.transformers.numeric_transformers import ArithmeticOperation
from core.transformers.numeric_transformers import PolynomialTransformer
from utils.utils_for_tests import BASIC_FRAME
from utils.utils_for_tests import assert_new_columns_in_frame


class TestPipeline:
    def test_transformers_from_init(self) -> None:
        pipeline = Pipeline(
            schema=[ColumnSpecification.numeric(name='NUMERIC_FEATURE')],
            transformers=[[PolynomialTransformer(column='NUMERIC_FEATURE', degree=2)]],
        )
        df = pl.LazyFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5]})

        res = pipeline.collect(df)

        assert_frame_equal(
            res,
            pl.DataFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5], 'NUMERIC_FEATURE_pow_2': [0, 1, 4, 9, 16, 25]}),
        )

    def test_basic_layering(self) -> None:
        pipeline = Pipeline(schema=[ColumnSpecification.numeric(name='NUMERIC_FEATURE')])
        pipeline = pipeline.with_polynomial(subset=ColumnType.NUMERIC, degrees=[2])
        pipeline = pipeline.with_new_layer()
        pipeline = pipeline.with_polynomial(subset=ColumnType.NUMERIC, degrees=[2])
        df = pl.LazyFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5]})

        res = pipeline.collect(df)

        assert_frame_equal(
            res,
            pl.DataFrame({
                'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5],
                'NUMERIC_FEATURE_pow_2': [0, 1, 4, 9, 16, 25],
                'NUMERIC_FEATURE_pow_2_pow_2': [0, 1, 16, 81, 256, 625],
            })
        )

    def test_pipeline_is_not_changed_inplace(self) -> None:
        pipeline = Pipeline(schema=[ColumnSpecification.numeric(name='NUMERIC_FEATURE')])
        pipeline_with_polynomial = pipeline.with_polynomial(subset=ColumnType.NUMERIC, degrees=[2])
        df = pl.LazyFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5]})

        res = pipeline.collect(df)
        res_with_polynomial = pipeline_with_polynomial.collect(df)

        assert_frame_equal(res, pl.DataFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5]}))
        assert_frame_equal(res_with_polynomial, pl.DataFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5], 'NUMERIC_FEATURE_pow_2': [0, 1, 4, 9, 16, 25]}))

    @pytest.mark.parametrize(
        'optimization_level',
        [
            OptimizationLevel.NONE,
            OptimizationLevel.SKIP_SELF,
            OptimizationLevel.DEDUPLICATE_COMMUTATIVE,
        ]
    )
    def test_pipeline_optimization(self, optimization_level: OptimizationLevel) -> None:
        pipeline = Pipeline(
            schema=[
                ColumnSpecification.numeric(name='NUMERIC_FEATURE'),
                ColumnSpecification.numeric(name='NUMERIC_FEATURE_2'),
            ],
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

        res = pipeline.collect(BASIC_FRAME)

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=res,
            expected_new_columns=expected_new_columns,
        )

    def test_basic_sample_with_all_transformers(self) -> None:
        pipeline = Pipeline(
            schema=[
                ColumnSpecification.numeric(name='NUMERIC_FEATURE'),
                ColumnSpecification.numeric(name='NUMERIC_FEATURE_2'),
                ColumnSpecification.ordinal(name='CATEGORICAL_FEATURE'),
                ColumnSpecification.nominal(name='CATEGORICAL_FEATURE_2'),
            ],
        )
        pipeline = (
            pipeline
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
            .with_lagged(subset=ColumnType.NUMERIC, lags=[1], over_columns_combinations=[[], ['GROUPING_FEATURE_NUM'], ['GROUPING_FEATURE_NUM','GROUPING_FEATURE_CAT_2']], fill_value=0)
            .with_lagged(subset=[ColumnType.ORDINAL, ColumnType.NOMINAL], lags=[1, 2], fill_value='missing')
            .with_arithmetic_aggregation(
                subset=ColumnType.NUMERIC,
                aggregations=[ArithmeticAggregations.SUM, ArithmeticAggregations.MEAN, ArithmeticAggregations.STD],
                over_columns_combinations=[['GROUPING_FEATURE_NUM'], ['GROUPING_FEATURE_NUM','GROUPING_FEATURE_CAT_2']],
            )
        )

        res = pipeline.collect(BASIC_FRAME)

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=res,
            expected_new_columns={
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
            },
        )
