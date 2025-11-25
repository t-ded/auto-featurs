import pytest

from auto_featurs.transformers.comparison_transformers import ComparisonTransformer
from auto_featurs.transformers.comparison_transformers import EqualTransformer
from auto_featurs.transformers.comparison_transformers import GreaterOrEqualTransformer
from auto_featurs.transformers.comparison_transformers import GreaterThanTransformer
from auto_featurs.utils.utils_for_tests import BASIC_FRAME
from auto_featurs.utils.utils_for_tests import assert_new_columns_in_frame


class TestComparisonTransformers:
    @pytest.mark.parametrize(
        ('transformer_type', 'expected_new_columns'),
        [
            (EqualTransformer, {'NUMERIC_FEATURE_equal_NUMERIC_FEATURE_2': [True, False, False, False, False, False]}),
            (GreaterThanTransformer, {'NUMERIC_FEATURE_greater_than_NUMERIC_FEATURE_2': [False, True, True, True, True, True]}),
            (GreaterOrEqualTransformer, {'NUMERIC_FEATURE_greater_or_equal_NUMERIC_FEATURE_2': [True, True, True, True, True, True]}),
        ],
    )
    def test_basic_arithmetic_transformation(self, transformer_type: type[ComparisonTransformer], expected_new_columns: dict[str, list[int] | list[float]]) -> None:
        transformer = transformer_type(left_column='NUMERIC_FEATURE', right_column='NUMERIC_FEATURE_2')
        df = BASIC_FRAME.with_columns(transformer.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns=expected_new_columns)

    @pytest.mark.parametrize(
        ('transformer_type', 'expected_new_columns'),
        [
            (EqualTransformer, {'NUMERIC_FEATURE_2_equal_NUMERIC_FEATURE': [True, False, False, False, False, False]}),
            (GreaterThanTransformer, {'NUMERIC_FEATURE_2_greater_than_NUMERIC_FEATURE': [False, False, False, False, False, False]}),
            (GreaterOrEqualTransformer, {'NUMERIC_FEATURE_2_greater_or_equal_NUMERIC_FEATURE': [True, False, False, False, False, False]}),
        ],
    )
    def test_basic_arithmetic_transformation_opposite_order(self, transformer_type: type[ComparisonTransformer], expected_new_columns: dict[str, list[int] | list[float]]) -> None:
        transformer = transformer_type(left_column='NUMERIC_FEATURE_2', right_column='NUMERIC_FEATURE')
        df = BASIC_FRAME.with_columns(transformer.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns=expected_new_columns)
