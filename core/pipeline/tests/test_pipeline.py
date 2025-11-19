import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from core.base.column_types import ColumnType
from core.pipeline.pipeline import Pipeline
from core.transformers.comparison_transformers import Comparisons
from core.transformers.numeric_transformers import ArithmeticOperation
from core.transformers.numeric_transformers import PolynomialTransformer
from utils.utils_for_tests import BASIC_FRAME
from utils.utils_for_tests import assert_new_columns_in_frame


class TestPipeline:
    def test_transformers_from_init(self) -> None:
        pipeline = Pipeline(
            column_types={'NUMERIC_FEATURE': ColumnType.NUMERIC},
            transformers=[[PolynomialTransformer(column='NUMERIC_FEATURE', degree=2)]],
        )
        df = pl.LazyFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5]})

        res = pipeline.collect(df)

        assert_frame_equal(
            res,
            pl.DataFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5], 'NUMERIC_FEATURE_pow_2': [0, 1, 4, 9, 16, 25]}),
        )

    def test_basic_layering(self) -> None:
        pipeline = Pipeline(column_types={'NUMERIC_FEATURE': ColumnType.NUMERIC})
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
        pipeline = Pipeline(column_types={'NUMERIC_FEATURE': ColumnType.NUMERIC})
        pipeline_with_polynomial = pipeline.with_polynomial(subset=ColumnType.NUMERIC, degrees=[2])
        df = pl.LazyFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5]})

        res = pipeline.collect(df)
        res_with_polynomial = pipeline_with_polynomial.collect(df)

        assert_frame_equal(res, pl.DataFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5]}))
        assert_frame_equal(res_with_polynomial, pl.DataFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5], 'NUMERIC_FEATURE_pow_2': [0, 1, 4, 9, 16, 25]}))

    def test_basic_sample_with_all_transformers(self) -> None:
        pipeline = Pipeline(column_types={'NUMERIC_FEATURE': ColumnType.NUMERIC, 'NUMERIC_FEATURE_2': ColumnType.NUMERIC})
        pipeline = pipeline.with_polynomial(subset=ColumnType.NUMERIC, degrees=[2, 3])
        pipeline = pipeline.with_arithmetic(
            left_subset=ColumnType.NUMERIC, right_subset=ColumnType.NUMERIC,
            operations=[ArithmeticOperation.ADD, ArithmeticOperation.SUBTRACT, ArithmeticOperation.MULTIPLY, ArithmeticOperation.DIVIDE],
        )
        # TODO: Add tests for other types of input columns
        pipeline = pipeline.with_comparison(
            left_subset=ColumnType.NUMERIC, right_subset=ColumnType.NUMERIC,
            comparisons=[Comparisons.EQUAL, Comparisons.GREATER_THAN, Comparisons.GREATER_OR_EQUAL],
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
                'NUMERIC_FEATURE_add_NUMERIC_FEATURE_2': [0, 0, 0, 0, 0, 0],
                'NUMERIC_FEATURE_2_add_NUMERIC_FEATURE': [0, 0, 0, 0, 0, 0],
                'NUMERIC_FEATURE_subtract_NUMERIC_FEATURE_2': [0, 2, 4, 6, 8, 10],
                'NUMERIC_FEATURE_2_subtract_NUMERIC_FEATURE': [0, -2, -4, -6, -8, -10],
                'NUMERIC_FEATURE_multiply_NUMERIC_FEATURE_2': [0, -1, -4, -9, -16, -25],
                'NUMERIC_FEATURE_2_multiply_NUMERIC_FEATURE': [0, -1, -4, -9, -16, -25],
                'NUMERIC_FEATURE_divide_NUMERIC_FEATURE_2': [np.nan, -1.0, -1.0, -1.0, -1.0, -1.0],
                'NUMERIC_FEATURE_2_divide_NUMERIC_FEATURE': [np.nan, -1.0, -1.0, -1.0, -1.0, -1.0],
                'NUMERIC_FEATURE_equal_NUMERIC_FEATURE_2': [True, False, False, False, False, False],
                'NUMERIC_FEATURE_2_equal_NUMERIC_FEATURE': [True, False, False, False, False, False],
                'NUMERIC_FEATURE_greater_than_NUMERIC_FEATURE_2': [False, True, True, True, True, True],
                'NUMERIC_FEATURE_2_greater_than_NUMERIC_FEATURE': [False, False, False, False, False, False],
                'NUMERIC_FEATURE_greater_or_equal_NUMERIC_FEATURE_2': [True, True, True, True, True, True],
                'NUMERIC_FEATURE_2_greater_or_equal_NUMERIC_FEATURE': [True, False, False, False, False, False],
            },
        )
