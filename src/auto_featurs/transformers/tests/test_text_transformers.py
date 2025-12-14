import pytest

from auto_featurs.transformers.text_transformers import DamerauLevenshteinSimilarityTransformer
from auto_featurs.transformers.text_transformers import TextSimilarityTransformer
from auto_featurs.utils.utils_for_tests import BASIC_FRAME
from auto_featurs.utils.utils_for_tests import assert_new_columns_in_frame


class TestTextSimilarityTransformers:
    @pytest.mark.parametrize(
        ('transformer_type', 'expected_new_columns'),
        [
            (DamerauLevenshteinSimilarityTransformer, {'TEXT_FEATURE_damerau_levenshtein_text_similarity_TEXT_FEATURE_2': [1.0, 0.142857, 0.714286, 0.5, 0.428571, 0.875]}),
        ],
    )
    def test_basic_text_similarity_transformation(self, transformer_type: type[TextSimilarityTransformer], expected_new_columns: dict[str, list[float] | list[int]]) -> None:
        transformer = transformer_type(left_column='TEXT_FEATURE', right_column='TEXT_FEATURE_2')
        df = BASIC_FRAME.with_columns(transformer.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns=expected_new_columns)

    @pytest.mark.parametrize(
        ('transformer_type', 'expected_new_columns'),
        [
            (DamerauLevenshteinSimilarityTransformer, {'TEXT_FEATURE_2_damerau_levenshtein_text_similarity_TEXT_FEATURE': [1.0, 0.142857, 0.714286, 0.5, 0.428571, 0.875]}),
        ],
    )
    def test_basic_text_similarity_transformation_opposite_order(self, transformer_type: type[TextSimilarityTransformer], expected_new_columns: dict[str, list[float] | list[int]]) -> None:
        transformer = transformer_type(left_column='TEXT_FEATURE_2', right_column='TEXT_FEATURE')
        df = BASIC_FRAME.with_columns(transformer.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns=expected_new_columns)
