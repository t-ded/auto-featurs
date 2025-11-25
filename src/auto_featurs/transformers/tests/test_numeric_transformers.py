import numpy as np
import pytest

from auto_featurs.transformers.numeric_transformers import AddTransformer
from auto_featurs.transformers.numeric_transformers import ArithmeticTransformer
from auto_featurs.transformers.numeric_transformers import DivideTransformer
from auto_featurs.transformers.numeric_transformers import MultiplyTransformer
from auto_featurs.transformers.numeric_transformers import PolynomialTransformer
from auto_featurs.transformers.numeric_transformers import SubtractTransformer
from auto_featurs.utils.utils_for_tests import BASIC_FRAME
from auto_featurs.utils.utils_for_tests import assert_new_columns_in_frame


class TestPolynomialTransformer:
    def setup_method(self) -> None:
        self._feature_2_polynomial_transformer_degree_2 = PolynomialTransformer(column='NUMERIC_FEATURE', degree=2)

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


class TestArithmeticTransformers:
    @pytest.mark.parametrize(
        ('transformer_type', 'expected_new_columns'),
        [
            (AddTransformer, {'NUMERIC_FEATURE_add_NUMERIC_FEATURE_2': [0, 0, 0, 0, 0, 0]}),
            (SubtractTransformer, {'NUMERIC_FEATURE_subtract_NUMERIC_FEATURE_2': [0, 2, 4, 6, 8, 10]}),
            (MultiplyTransformer, {'NUMERIC_FEATURE_multiply_NUMERIC_FEATURE_2': [0, -1, -4, -9, -16, -25]}),
            (DivideTransformer, {'NUMERIC_FEATURE_divide_NUMERIC_FEATURE_2': [np.nan, -1.0, -1.0, -1.0, -1.0, -1.0]}),
        ],
    )
    def test_basic_arithmetic_transformation(self, transformer_type: type[ArithmeticTransformer], expected_new_columns: dict[str, list[int] | list[float]]) -> None:
        transformer = transformer_type(left_column='NUMERIC_FEATURE', right_column='NUMERIC_FEATURE_2')
        df = BASIC_FRAME.with_columns(transformer.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns=expected_new_columns)

    @pytest.mark.parametrize(
        ('transformer_type', 'expected_new_columns'),
        [
            (AddTransformer, {'NUMERIC_FEATURE_2_add_NUMERIC_FEATURE': [0, 0, 0, 0, 0, 0]}),
            (SubtractTransformer, {'NUMERIC_FEATURE_2_subtract_NUMERIC_FEATURE': [0, -2, -4, -6, -8, -10]}),
            (MultiplyTransformer, {'NUMERIC_FEATURE_2_multiply_NUMERIC_FEATURE': [0, -1, -4, -9, -16, -25]}),
            (DivideTransformer, {'NUMERIC_FEATURE_2_divide_NUMERIC_FEATURE': [np.nan, -1.0, -1.0, -1.0, -1.0, -1.0]}),
        ],
    )
    def test_basic_arithmetic_transformation_opposite_order(self, transformer_type: type[ArithmeticTransformer], expected_new_columns: dict[str, list[int] | list[float]]) -> None:
        transformer = transformer_type(left_column='NUMERIC_FEATURE_2', right_column='NUMERIC_FEATURE')
        df = BASIC_FRAME.with_columns(transformer.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns=expected_new_columns)
