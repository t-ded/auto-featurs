from abc import ABC
from enum import Enum

import polars as pl

from core.base.column_specification import ColumnType
from core.transformers.base import Transformer


class PolynomialTransformer(Transformer):
    def __init__(self, column: str, degree: int) -> None:
        self._column = column
        self._degree = degree

    def input_type(self) -> set[ColumnType]:
        return {ColumnType.NUMERIC}

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _transform(self) -> pl.Expr:
        return pl.col(self._column).pow(self._degree)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.name.suffix(f'_pow_{self._degree}')


class ArithmeticTransformer(Transformer, ABC):
    def __init__(self, left_column: str, right_column: str) -> None:
        self._left_column = left_column
        self._right_column = right_column

    def input_type(self) -> tuple[set[ColumnType], set[ColumnType]]:
        return {ColumnType.NUMERIC}, {ColumnType.NUMERIC}

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC


class AddTransformer(ArithmeticTransformer):
    def _transform(self) -> pl.Expr:
        return pl.col(self._left_column) + pl.col(self._right_column)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_add_{self._right_column}')


class SubtractTransformer(ArithmeticTransformer):
    def _transform(self) -> pl.Expr:
        return pl.col(self._left_column) - pl.col(self._right_column)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_subtract_{self._right_column}')


class MultiplyTransformer(ArithmeticTransformer):
    def _transform(self) -> pl.Expr:
        return pl.col(self._left_column) * pl.col(self._right_column)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_multiply_{self._right_column}')


class DivideTransformer(ArithmeticTransformer):
    def _transform(self) -> pl.Expr:
        return pl.col(self._left_column) / pl.col(self._right_column)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_divide_{self._right_column}')


class ArithmeticOperation(Enum):
    ADD = AddTransformer
    SUBTRACT = SubtractTransformer
    MULTIPLY = MultiplyTransformer
    DIVIDE = DivideTransformer