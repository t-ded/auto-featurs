from abc import ABC
from enum import Enum

import polars as pl

from auto_featurs.core.base.column_specification import ColumnSpecification
from auto_featurs.core.base.column_specification import ColumnType
from auto_featurs.core.transformers.base import Transformer


class ComparisonTransformer(Transformer, ABC):
    def __init__(self, left_column: str | ColumnSpecification, right_column: str | ColumnSpecification) -> None:
        self._left_column = left_column if isinstance(left_column, str) else left_column.name
        self._right_column = right_column if isinstance(right_column, str) else right_column.name

    def input_type(self) -> tuple[set[ColumnType], set[ColumnType]]:
        return ColumnType.ANY(), ColumnType.ANY()

    def _return_type(self) -> ColumnType:
        return ColumnType.BOOLEAN


class EqualTransformer(ComparisonTransformer):
    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _transform(self) -> pl.Expr:
        return pl.col(self._left_column) == pl.col(self._right_column)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_equal_{self._right_column}')


class GreaterThanTransformer(ComparisonTransformer):
    @classmethod
    def is_commutative(cls) -> bool:
        return False

    def _transform(self) -> pl.Expr:
        return pl.col(self._left_column) > pl.col(self._right_column)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_greater_than_{self._right_column}')


class GreaterOrEqualTransformer(ComparisonTransformer):
    @classmethod
    def is_commutative(cls) -> bool:
        return False

    def _transform(self) -> pl.Expr:
        return pl.col(self._left_column) >= pl.col(self._right_column)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_greater_or_equal_{self._right_column}')


class Comparisons(Enum):
    EQUAL = EqualTransformer
    GREATER_THAN = GreaterThanTransformer
    GREATER_OR_EQUAL = GreaterOrEqualTransformer
