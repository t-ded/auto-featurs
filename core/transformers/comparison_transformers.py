from abc import ABC

import polars as pl

from core.base.column_types import ColumnType
from core.transformers.base import Transformer


class ComparisonTransformer(Transformer, ABC):
    def __init__(self, left_column: str, right_column: str) -> None:
        self._left_column = left_column
        self._right_column = right_column

    def input_type(self) -> tuple[ColumnType, ...]:
        return ColumnType.ANY, ColumnType.ANY

    def _return_type(self) -> ColumnType:
        return ColumnType.BOOLEAN


class EqualTransformer(ComparisonTransformer):
    def _transform(self) -> pl.Expr:
        return pl.col(self._left_column) == pl.col(self._right_column)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_equal_{self._right_column}')


class GreaterThanTransformer(ComparisonTransformer):
    def _transform(self) -> pl.Expr:
        return pl.col(self._left_column) > pl.col(self._right_column)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_greater_than_{self._right_column}')


class GreaterOrEqualTransformer(ComparisonTransformer):
    def _transform(self) -> pl.Expr:
        return pl.col(self._left_column) >= pl.col(self._right_column)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_greater_or_equal_{self._right_column}')
