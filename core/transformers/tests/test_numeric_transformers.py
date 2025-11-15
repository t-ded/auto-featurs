import polars as pl
from polars.testing import assert_frame_equal

from core.base.column_types import ColumnType
from core.transformers.numeric_transformers import PolynomialTransformer


class TestPolynomialTransformer:
    def setup_method(self) -> None:
        self._df = pl.LazyFrame(
            {
                'NUMERIC_FEATURE': [1, 2, 3, 4, 5],
                'NUMERIC_FEATURE_2': [-1, -2, -3, -4, -5],
                'CATEGORICAL_FEATURE': ['A', 'B', 'C', 'D', 'E'],
            },
        )

    def test_new_column_type(self) -> None:
        pol_transformer = PolynomialTransformer(column='NUMERIC_FEATURE', degree=2)
        assert pol_transformer.new_column_type() == ('NUMERIC_FEATURE_pow_2', ColumnType.NUMERIC)

    def test_basic_polynomial_transformation(self) -> None:
        feature_2_polynomial_transformer_degree_2 = PolynomialTransformer(column='NUMERIC_FEATURE', degree=2)
        df = self._df.with_columns(feature_2_polynomial_transformer_degree_2.transform())
        assert_frame_equal(
            df,
            pl.LazyFrame(
                {
                    'NUMERIC_FEATURE': [1, 2, 3, 4, 5],
                    'NUMERIC_FEATURE_2': [-1, -2, -3, -4, -5],
                    'CATEGORICAL_FEATURE': ['A', 'B', 'C', 'D', 'E'],
                    'NUMERIC_FEATURE_pow_2': [1, 4, 9, 16, 25],
                },
            ),
        )

    def test_combination_of_polynomial_transformations(self) -> None:
        transformers = [
            PolynomialTransformer(column='NUMERIC_FEATURE', degree=2),
            PolynomialTransformer(column='NUMERIC_FEATURE', degree=3),
            PolynomialTransformer(column='NUMERIC_FEATURE_2', degree=2),
            PolynomialTransformer(column='NUMERIC_FEATURE_2', degree=3),
        ]
        df = self._df.with_columns(*[transformer.transform() for transformer in transformers])
        assert_frame_equal(
            df,
            pl.LazyFrame(
                {
                    'NUMERIC_FEATURE': [1, 2, 3, 4, 5],
                    'NUMERIC_FEATURE_2': [-1, -2, -3, -4, -5],
                    'CATEGORICAL_FEATURE': ['A', 'B', 'C', 'D', 'E'],
                    'NUMERIC_FEATURE_pow_2': [1, 4, 9, 16, 25],
                    'NUMERIC_FEATURE_pow_3': [1, 8, 27, 64, 125],
                    'NUMERIC_FEATURE_2_pow_2': [1, 4, 9, 16, 25],
                    'NUMERIC_FEATURE_2_pow_3': [-1, -8, -27, -64, -125],
                },
            ),
        )
