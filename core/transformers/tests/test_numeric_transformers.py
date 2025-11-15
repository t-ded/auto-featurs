import polars as pl
from polars.testing import assert_frame_equal

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

    def test_basic_polynomial_transformation(self) -> None:
        numeric_features_polynomial_transformer_degree_2 = PolynomialTransformer(columns=['NUMERIC_FEATURE', 'NUMERIC_FEATURE_2'], degree=2)
        df = self._df.with_columns(numeric_features_polynomial_transformer_degree_2.transform())
        assert_frame_equal(
            df,
            pl.LazyFrame(
                {
                    'NUMERIC_FEATURE': [1, 2, 3, 4, 5],
                    'NUMERIC_FEATURE_2': [-1, -2, -3, -4, -5],
                    'CATEGORICAL_FEATURE': ['A', 'B', 'C', 'D', 'E'],
                    'NUMERIC_FEATURE_pow_2': [1, 4, 9, 16, 25],
                    'NUMERIC_FEATURE_2_pow_2': [1, 4, 9, 16, 25],
                },
            ),
        )

    def test_combination_of_polynomial_transformations(self) -> None:
        numeric_features_polynomial_transformer_degree_2 = PolynomialTransformer(columns=['NUMERIC_FEATURE', 'NUMERIC_FEATURE_2'], degree=2)
        numeric_features_polynomial_transformer_degree_3 = PolynomialTransformer(columns=['NUMERIC_FEATURE', 'NUMERIC_FEATURE_2'], degree=3)
        df = self._df.with_columns(numeric_features_polynomial_transformer_degree_2.transform(), numeric_features_polynomial_transformer_degree_3.transform())
        assert_frame_equal(
            df,
            pl.LazyFrame(
                {
                    'NUMERIC_FEATURE': [1, 2, 3, 4, 5],
                    'NUMERIC_FEATURE_2': [-1, -2, -3, -4, -5],
                    'CATEGORICAL_FEATURE': ['A', 'B', 'C', 'D', 'E'],
                    'NUMERIC_FEATURE_pow_2': [1, 4, 9, 16, 25],
                    'NUMERIC_FEATURE_2_pow_2': [1, 4, 9, 16, 25],
                    'NUMERIC_FEATURE_pow_3': [1, 8, 27, 64, 125],
                    'NUMERIC_FEATURE_2_pow_3': [-1, -8, -27, -64, -125],
                },
            ),
        )

    def test_polynomial_transformation_for_subset(self) -> None:
        feature_2_polynomial_transformer_degree_2 = PolynomialTransformer(columns=['NUMERIC_FEATURE_2'], degree=2)
        df = self._df.with_columns(feature_2_polynomial_transformer_degree_2.transform())
        assert_frame_equal(
            df,
            pl.LazyFrame(
                {
                    'NUMERIC_FEATURE': [1, 2, 3, 4, 5],
                    'NUMERIC_FEATURE_2': [-1, -2, -3, -4, -5],
                    'CATEGORICAL_FEATURE': ['A', 'B', 'C', 'D', 'E'],
                    'NUMERIC_FEATURE_2_pow_2': [1, 4, 9, 16, 25],
                },
            ),
        )
