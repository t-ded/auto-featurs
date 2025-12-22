from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

import polars as pl
import polars_ds as pds  # type: ignore[import-untyped]

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.column_specification import ColumnTypeSelector
from auto_featurs.transformers.base import Transformer


class TextSimilarityTransformer(Transformer, ABC):
    def __init__(self, left_column: str | ColumnSpecification, right_column: str | ColumnSpecification) -> None:
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


class TextSimilarity(Enum):
    DAMERAU_LEVENSHTEIN = DamerauLevenshteinSimilarityTransformer


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


class TextExtraction(Enum):
    LENGTH = TextLengthTransformer
    EMAIL_DOMAIN = EmailDomainExtractionTransformer


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
