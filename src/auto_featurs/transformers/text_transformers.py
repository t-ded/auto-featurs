from abc import ABC
from abc import abstractmethod
from enum import Enum

import polars as pl
import polars_ds as pds

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
