import numpy as np
import pytest

from auto_featurs.transformers.numeric_transformers import AddTransformer
from auto_featurs.transformers.numeric_transformers import ArithmeticTransformer
from auto_featurs.transformers.numeric_transformers import CosTransformer
from auto_featurs.transformers.numeric_transformers import DivideTransformer
from auto_featurs.transformers.numeric_transformers import LogTransformer
from auto_featurs.transformers.numeric_transformers import MinMaxScaler
from auto_featurs.transformers.numeric_transformers import MultiplyTransformer
from auto_featurs.transformers.numeric_transformers import PolynomialTransformer
from auto_featurs.transformers.numeric_transformers import SinTransformer
from auto_featurs.transformers.numeric_transformers import StandardScaler
from auto_featurs.transformers.numeric_transformers import SubtractTransformer
from auto_featurs.utils.constants import INFINITY
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


class TestLogTransformer:
    def setup_method(self) -> None:
        self._natural_log_transformer = LogTransformer(column='NUMERIC_FEATURE')
        self._log_10_transformer = LogTransformer(column='NUMERIC_FEATURE', base=10)

    def test_basic_log_transformation(self) -> None:
        df = BASIC_FRAME.with_columns(
            self._natural_log_transformer.transform(),
            self._log_10_transformer.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={
                'NUMERIC_FEATURE_ln': [-INFINITY, 0.0, 0.69314718, 1.09861229, 1.38629436, 1.60943791],
                'NUMERIC_FEATURE_log10': [-INFINITY, 0.0, 0.30103, 0.47712125, 0.60205999, 0.69897],
            },
        )


class TestGoniometricTransformers:
    def setup_method(self) -> None:
        self._sin_transformer = SinTransformer(column='NUMERIC_FEATURE')
        self._cos_transformer = CosTransformer(column='NUMERIC_FEATURE')

    def test_basic_sin_cos_transformation(self) -> None:
        df = BASIC_FRAME.with_columns(
            self._sin_transformer.transform(),
            self._cos_transformer.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={
                'NUMERIC_FEATURE_sin': [0.0, 0.84147098, 0.90929743, 0.14112001, -0.7568025, -0.95892427],
                'NUMERIC_FEATURE_cos': [1.0, 0.54030231, -0.41614684, -0.9899925, -0.65364362, 0.28366219],
            },
        )


class TestScalers:
    def setup_method(self) -> None:
        self._standard_scaler = StandardScaler(column='NUMERIC_FEATURE')
        self._min_max_scaler = MinMaxScaler(column='NUMERIC_FEATURE')

    def test_basic_scaling_transformation(self) -> None:
        df = BASIC_FRAME.with_columns(
            self._standard_scaler.transform(),
            self._min_max_scaler.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={
                'NUMERIC_FEATURE_standard_scaled': [-1.336306, -0.801784, -0.267261, 0.267261, 0.801784, 1.336306],
                'NUMERIC_FEATURE_minmax_scaled': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
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

    @pytest.mark.parametrize(
        ('transformer_type', 'expected_new_columns'),
        [
            (AddTransformer, {'NUMERIC_FEATURE_add_BOOL_FEATURE': [1, 1, 3, 3, 5, 5]}),
            (SubtractTransformer, {'NUMERIC_FEATURE_subtract_BOOL_FEATURE': [-1, 1, 1, 3, 3, 5]}),
            (MultiplyTransformer, {'NUMERIC_FEATURE_multiply_BOOL_FEATURE': [0, 0, 2, 0, 4, 0]}),
            (DivideTransformer, {'NUMERIC_FEATURE_divide_BOOL_FEATURE': [0.0, INFINITY, 2.0, INFINITY, 4.0, INFINITY]}),
        ],
    )
    def test_basic_arithmetic_transformation_numeric_and_boolean(self, transformer_type: type[ArithmeticTransformer], expected_new_columns: dict[str, list[int] | list[float]]) -> None:
        transformer = transformer_type(left_column='NUMERIC_FEATURE', right_column='BOOL_FEATURE')
        df = BASIC_FRAME.with_columns(transformer.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns=expected_new_columns)
