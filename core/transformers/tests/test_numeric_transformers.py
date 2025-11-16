
from core.base.column_types import ColumnType
from core.transformers.numeric_transformers import PolynomialTransformer
from utils.utils_for_tests import BASIC_FRAME, assert_new_columns_in_frame


class TestPolynomialTransformer:
    def setup_method(self) -> None:
        self._feature_2_polynomial_transformer_degree_2 = PolynomialTransformer(column='NUMERIC_FEATURE', degree=2)

    def test_new_column_type(self) -> None:
        assert self._feature_2_polynomial_transformer_degree_2.new_column_type() == ('NUMERIC_FEATURE_pow_2', ColumnType.NUMERIC)

    def test_basic_polynomial_transformation(self) -> None:
        df = BASIC_FRAME.with_columns(self._feature_2_polynomial_transformer_degree_2.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns={'NUMERIC_FEATURE_pow_2': [0, 1, 4, 9, 16, 25]})

    def test_combination_of_polynomial_transformations(self) -> None:
        transformers = [
            self._feature_2_polynomial_transformer_degree_2,
            PolynomialTransformer(column='NUMERIC_FEATURE', degree=3),
            PolynomialTransformer(column='NUMERIC_FEATURE_2', degree=2),
            PolynomialTransformer(column='NUMERIC_FEATURE_2', degree=3),
        ]
        df = BASIC_FRAME.with_columns(*[transformer.transform() for transformer in transformers])
        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={
                'NUMERIC_FEATURE_pow_2': [0, 1, 4, 9, 16, 25],
                'NUMERIC_FEATURE_pow_3': [0, 1, 8, 27, 64, 125],
                'NUMERIC_FEATURE_2_pow_2': [0, 1, 4, 9, 16, 25],
                'NUMERIC_FEATURE_2_pow_3': [0, -1, -8, -27, -64, -125],
            },
        )
