from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import polars as pl
import polars_ds as pds  # type: ignore[import-untyped]

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.column_specification import ColumnTypeSelector
from auto_featurs.transformers.base import Transformer


class TextSimilarityTransformer(Transformer, ABC):
    def __init__(self, left_column: str | ColumnSpecification, right_column: str | ColumnSpecification, **kwargs: Any) -> None:
        self._left_column = left_column if isinstance(left_column, str) else left_column.name
        self._right_column = right_column if isinstance(right_column, str) else right_column.name

    def input_type(self) -> tuple[ColumnTypeSelector, ColumnTypeSelector]:
        return ColumnType.TEXT.as_selector(), ColumnType.TEXT.as_selector()

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_{self._dist_str}_text_similarity_{self._right_column}')

    @property
    @abstractmethod
    def _dist_str(self) -> str:
        raise NotImplementedError()


class DamerauLevenshteinSimilarityTransformer(TextSimilarityTransformer):
    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _transform(self) -> pl.Expr:
        return pds.str_d_leven(self._left_column, self._right_column, return_sim=True)

    @property
    def _dist_str(self) -> str:
        return 'damerau_levenshtein'


class JaccardSimilarityTransformer(TextSimilarityTransformer):
    def __init__(self, left_column: str | ColumnSpecification, right_column: str | ColumnSpecification, substr_size: int = 2, **kwargs: Any) -> None:
        super().__init__(left_column, right_column)
        self._substr_size = substr_size

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _transform(self) -> pl.Expr:
        return pds.str_jaccard(self._left_column, self._right_column, substr_size=self._substr_size)

    @property
    def _dist_str(self) -> str:
        return 'jaccard'


class JaroSimilarityTransformer(TextSimilarityTransformer):
    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _transform(self) -> pl.Expr:
        return pds.str_jaro(self._left_column, self._right_column)

    @property
    def _dist_str(self) -> str:
        return 'jaro'


class JaroWinklerSimilarityTransformer(TextSimilarityTransformer):
    def __init__(self, left_column: str | ColumnSpecification, right_column: str | ColumnSpecification, weight: float = 0.1, **kwargs: Any) -> None:
        super().__init__(left_column, right_column)
        self._weight = weight

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _transform(self) -> pl.Expr:
        return pds.str_jw(self._left_column, self._right_column, weight=self._weight)

    @property
    def _dist_str(self) -> str:
        return 'jaro_winkler'


class TextSimilarity(Enum):
    DAMERAU_LEVENSHTEIN = DamerauLevenshteinSimilarityTransformer
    JACCARD = JaccardSimilarityTransformer
    JARO = JaroSimilarityTransformer
    JARO_WINKLER = JaroWinklerSimilarityTransformer


class TextExtractionTransformer(Transformer, ABC):
    def __init__(self, column: str | ColumnSpecification):
        self._column = column if isinstance(column, str) else column.name

    def input_type(self) -> ColumnTypeSelector:
        return ColumnType.TEXT.as_selector()

    @classmethod
    def is_commutative(cls) -> bool:
        return True


class TextLengthTransformer(TextExtractionTransformer):
    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _transform(self) -> pl.Expr:
        return pl.col(self._column).str.len_chars()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column}_length_chars')


class EmailDomainExtractionTransformer(TextExtractionTransformer):
    def _return_type(self) -> ColumnType:
        return ColumnType.NOMINAL

    def _transform(self) -> pl.Expr:
        return pl.col(self._column).str.extract(r'@(.+)$', 1)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column}_email_domain')


class CharacterEntropyTransformer(TextExtractionTransformer):
    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _transform(self) -> pl.Expr:
        return (
            pl.col(self._column)
            .str.split('')
            .list.eval(
                pl.element()
                .value_counts()
                .struct.field('count')
                .entropy(base=2),
            )
            .list.first()
        )

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column}_character_entropy')


class TextExtraction(Enum):
    LENGTH = TextLengthTransformer
    EMAIL_DOMAIN = EmailDomainExtractionTransformer
    CHARACTER_ENTROPY = CharacterEntropyTransformer


type PatternInput = str | CommonPatterns | tuple[str, str]


@dataclass(frozen=True)
class _ResolvedPattern:
    regex: str
    name: str


class TextCountMatchesTransformer(TextExtractionTransformer):
    def __init__(self, column: str | ColumnSpecification, pattern: PatternInput) -> None:
        super().__init__(column)
        resolved = self._resolve_pattern(pattern)
        self._regex = resolved.regex
        self._human_readable = resolved.name

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _transform(self) -> pl.Expr:
        return pl.col(self._column).str.count_matches(self._regex)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column}_count_{self._human_readable}')

    @staticmethod
    def _resolve_pattern(pattern: PatternInput) -> _ResolvedPattern:
        if isinstance(pattern, CommonPatterns):
            return _ResolvedPattern(
                regex=pattern.value,
                name=pattern.name.lower(),
            )

        if isinstance(pattern, tuple):
            regex, name = pattern
            return _ResolvedPattern(regex=regex, name=name)

        if isinstance(pattern, str):
            for common_pattern in CommonPatterns:
                if common_pattern.value == pattern:
                    return _ResolvedPattern(regex=common_pattern.value, name=common_pattern.name.lower())

            return _ResolvedPattern(regex=pattern, name=pattern)

        raise TypeError(f'Unsupported pattern type: {type(pattern)}')


class CommonPatterns(Enum):
    DIGITS = r'\d'
    LETTER = r'[A-Za-z]'
    UPPERCASE = r'[A-Z]'
    LOWERCASE = r'[a-z]'
    NON_ALPHANUMERIC = r'[^A-Za-z0-9]'
    WHITESPACE = r'\s'

    CONSECUTIVE_DIGITS = r'\d{3,}'
    CONSECUTIVE_LETTERS = r'[A-Za-z]{5,}'

    SPECIAL_SYMBOLS = r'[!@#$%^&*_=+|~<>]'
    PUNCTUATION = r'[.,;:!?]'

    DOT = r'\.'
    SLASH = r'/'
    AT_SIGN = r'@'
    HYPHEN = r'-'
    UNDERSCORE = r'_'

    NON_ASCII = r'[^\x00-\x7F]'
    ZERO_WIDTH = r'[\u200B-\u200D\uFEFF]'
