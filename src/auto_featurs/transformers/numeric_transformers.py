from abc import ABC
from enum import Enum
import math

import polars as pl

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.column_specification import ColumnTypeSelector
from auto_featurs.transformers.base import Transformer


class NumericTransformer(Transformer, ABC):
    def __init__(self, column: str | ColumnSpecification) -> None:
        self._column = column if isinstance(column, str) else column.name

    def input_type(self) -> ColumnTypeSelector:
        return ColumnType.NUMERIC.as_selector()

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC


class PolynomialTransformer(NumericTransformer):
    def __init__(self, column: str | ColumnSpecification, *, degree: int) -> None:
        super().__init__(column)
        self._degree = degree

    def _transform(self) -> pl.Expr:
        return pl.col(self._column).pow(self._degree)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.name.suffix(f'_pow_{self._degree}')


class LogTransformer(NumericTransformer):
    def __init__(self, column: str | ColumnSpecification, *, base: float = math.e) -> None:
        super().__init__(column)
        self._base = base

    def _transform(self) -> pl.Expr:
        return pl.col(self._column).log(self._base)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        suffix = '_ln' if self._base == math.e else f'_log{self._base}'
        return transform.name.suffix(suffix)


class SinTransformer(NumericTransformer):
    def _transform(self) -> pl.Expr:
        return pl.col(self._column).sin()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.name.suffix('_sin')


class CosTransformer(NumericTransformer):
    def _transform(self) -> pl.Expr:
        return pl.col(self._column).cos()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.name.suffix('_cos')


class Goniometric(Enum):
    SIN = SinTransformer
    COS = CosTransformer


class StandardScaler(NumericTransformer):
    def _transform(self) -> pl.Expr:
        col = pl.col(self._column)
        return (col - col.mean()) / col.std()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column}_standard_scaled')


class MinMaxScaler(NumericTransformer):
    def _transform(self) -> pl.Expr:
        col = pl.col(self._column)
        return (col - col.min()) / (col.max() - col.min())

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column}_minmax_scaled')


class Scaling(Enum):
    STANDARD = StandardScaler
    MIN_MAX = MinMaxScaler


class ArithmeticTransformer(Transformer, ABC):
    def __init__(self, left_column: str | ColumnSpecification, right_column: str | ColumnSpecification) -> None:
        self._left_column = left_column if isinstance(left_column, str) else left_column.name
        self._right_column = right_column if isinstance(right_column, str) else right_column.name

    def input_type(self) -> tuple[ColumnTypeSelector, ColumnTypeSelector]:
        return ColumnType.NUMERIC | ColumnType.BOOLEAN, ColumnType.NUMERIC | ColumnType.BOOLEAN

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC


class AddTransformer(ArithmeticTransformer):
    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _transform(self) -> pl.Expr:
        return pl.col(self._left_column) + pl.col(self._right_column)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_add_{self._right_column}')


class SubtractTransformer(ArithmeticTransformer):
    @classmethod
    def is_commutative(cls) -> bool:
        return False

    def _transform(self) -> pl.Expr:
        return pl.col(self._left_column) - pl.col(self._right_column)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_subtract_{self._right_column}')


class MultiplyTransformer(ArithmeticTransformer):
    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _transform(self) -> pl.Expr:
        return pl.col(self._left_column) * pl.col(self._right_column)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_multiply_{self._right_column}')


class DivideTransformer(ArithmeticTransformer):
    @classmethod
    def is_commutative(cls) -> bool:
        return False

    def _transform(self) -> pl.Expr:
        return pl.col(self._left_column) / pl.col(self._right_column)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_divide_{self._right_column}')


class ArithmeticOperation(Enum):
    ADD = AddTransformer
    SUBTRACT = SubtractTransformer
    MULTIPLY = MultiplyTransformer
    DIVIDE = DivideTransformer
