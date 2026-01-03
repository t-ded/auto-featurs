from abc import ABC
from enum import Enum

import polars as pl

from auto_featurs.base.column_specification import ColumnNameOrSpec
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.column_specification import ColumnTypeSelector
from auto_featurs.transformers.base import Transformer
from auto_featurs.utils.utils import parse_column_name


class ComparisonTransformer(Transformer, ABC):
    def __init__(self, left_column: ColumnNameOrSpec, right_column: ColumnNameOrSpec) -> None:
        self._left_column = parse_column_name(left_column)
        self._right_column = parse_column_name(right_column)

    def input_type(self) -> tuple[ColumnTypeSelector, ColumnTypeSelector]:
        return ColumnTypeSelector.any(), ColumnTypeSelector.any()

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
