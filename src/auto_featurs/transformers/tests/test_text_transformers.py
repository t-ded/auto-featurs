import polars as pl
import pytest

from auto_featurs.transformers.text_transformers import CharacterEntropyTransformer
from auto_featurs.transformers.text_transformers import CommonPatterns
from auto_featurs.transformers.text_transformers import DamerauLevenshteinSimilarityTransformer
from auto_featurs.transformers.text_transformers import JaccardSimilarityTransformer
from auto_featurs.transformers.text_transformers import JaroSimilarityTransformer
from auto_featurs.transformers.text_transformers import JaroWinklerSimilarityTransformer
from auto_featurs.transformers.text_transformers import EmailDomainExtractionTransformer
from auto_featurs.transformers.text_transformers import PatternInput
from auto_featurs.transformers.text_transformers import TextCountMatchesTransformer
from auto_featurs.transformers.text_transformers import TextLengthTransformer
from auto_featurs.transformers.text_transformers import TextSimilarityTransformer
from auto_featurs.utils.utils_for_tests import BASIC_FRAME
from auto_featurs.utils.utils_for_tests import assert_new_columns_in_frame


class TestTextSimilarityTransformers:
    @pytest.mark.parametrize(
        ('transformer_type', 'expected_new_columns'),
        [
            (DamerauLevenshteinSimilarityTransformer, {'TEXT_FEATURE_damerau_levenshtein_text_similarity_TEXT_FEATURE_2': [1.0, 0.142857, 0.714286, 0.5, 0.428571, 0.875]}),
            (JaccardSimilarityTransformer, {'TEXT_FEATURE_jaccard_text_similarity_TEXT_FEATURE_2': [1.0, 0.5, 0.333333, 0.461538, 0.333333, 0.625]}),
            (JaroSimilarityTransformer, {'TEXT_FEATURE_jaro_text_similarity_TEXT_FEATURE_2': [1.0, 0.428571, 0.809524, 0.690476, 0.809524, 0.958333]}),
            (JaroWinklerSimilarityTransformer, {'TEXT_FEATURE_jaro_winkler_text_similarity_TEXT_FEATURE_2': [1.0, 0.428571, 0.809524, 0.690476, 0.866667, 0.970833]}),
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
            (JaccardSimilarityTransformer, {'TEXT_FEATURE_2_jaccard_text_similarity_TEXT_FEATURE': [1.0, 0.5, 0.333333, 0.461538, 0.333333, 0.625]}),
            (JaroSimilarityTransformer, {'TEXT_FEATURE_2_jaro_text_similarity_TEXT_FEATURE': [1.0, 0.428571, 0.809524, 0.690476, 0.809524, 0.958333]}),
            (JaroWinklerSimilarityTransformer, {'TEXT_FEATURE_2_jaro_winkler_text_similarity_TEXT_FEATURE': [1.0, 0.428571, 0.809524, 0.690476, 0.866667, 0.970833]}),
        ],
    )
    def test_basic_text_similarity_transformation_opposite_order(self, transformer_type: type[TextSimilarityTransformer], expected_new_columns: dict[str, list[float] | list[int]]) -> None:
        transformer = transformer_type(left_column='TEXT_FEATURE_2', right_column='TEXT_FEATURE')
        df = BASIC_FRAME.with_columns(transformer.transform())
        assert_new_columns_in_frame(original_frame=BASIC_FRAME, new_frame=df, expected_new_columns=expected_new_columns)


class TestTextExtractionTransformers:
    def setup_method(self) -> None:
        self._frame = pl.DataFrame(
            {
                'TEXT_FEATURE': ['john.doe@example.com', 'USER123!!!', 'aaaBBB111', 'straße café 42'],
                'EMAIL': ['louis@gmail.com', 'user@seznam.cz', 'john.doe@email.com', 'london@gov.co.uk'],
            },
        )

    def test_text_length_transformer(self) -> None:
        transformer = TextLengthTransformer(column='TEXT_FEATURE')
        df = self._frame.with_columns(transformer.transform())
        assert_new_columns_in_frame(
            original_frame=self._frame,
            new_frame=df,
            expected_new_columns={'TEXT_FEATURE_length_chars': [20, 10, 9, 14]},
        )

    def test_email_domain_extraction_transformers(self) -> None:
        transformer = EmailDomainExtractionTransformer(column='EMAIL')
        df = self._frame.with_columns(transformer.transform())
        assert_new_columns_in_frame(
            original_frame=self._frame,
            new_frame=df,
            expected_new_columns={'EMAIL_email_domain': ['gmail.com', 'seznam.cz', 'email.com', 'gov.co.uk']},
        )

    def test_character_entropy_transformer(self) -> None:
        transformer = CharacterEntropyTransformer(column='TEXT_FEATURE')
        df = self._frame.with_columns(transformer.transform())
        assert_new_columns_in_frame(
            original_frame=self._frame,
            new_frame=df,
            expected_new_columns={'TEXT_FEATURE_character_entropy': [3.64644, 2.84644, 1.58496, 3.52164]},
        )

    @pytest.mark.parametrize(
        ('pattern', 'expected_new_columns'),
        [
            (r'\d', {'TEXT_FEATURE_count_digits': [0, 3, 3, 2]}),
            (CommonPatterns.CONSECUTIVE_DIGITS, {'TEXT_FEATURE_count_consecutive_digits': [0, 1, 1, 0]}),
            (r'[A-Z]', {'TEXT_FEATURE_count_uppercase': [0, 4, 3, 0]}),
            (r'[^A-Za-z0-9]', {'TEXT_FEATURE_count_non_alphanumeric': [3, 3, 0, 4]}),
            (r'\s', {'TEXT_FEATURE_count_whitespace': [0, 0, 0, 2]}),
            (r'[^\x00-\x7F]', {'TEXT_FEATURE_count_non_ascii': [0, 0, 0, 2]}),
        ],
    )
    def test_count_matches_transformer(self, pattern: PatternInput, expected_new_columns: dict[str, list[int]]) -> None:
        transformer = TextCountMatchesTransformer(column='TEXT_FEATURE', pattern=pattern)
        df = self._frame.with_columns(transformer.transform())
        assert_new_columns_in_frame(original_frame=self._frame, new_frame=df, expected_new_columns=expected_new_columns)
