from abc import ABC
from enum import Enum
from typing import Optional

import polars as pl
from polars._typing import IntoExpr

from core.base.column_specification import ColumnSpecification
from core.base.column_specification import ColumnType
from core.transformers.base import Transformer


class AggregatingTransformer(Transformer, ABC):
    pass


class LaggedTransformer(AggregatingTransformer):
    def __init__(self, column: ColumnSpecification, lag: int, fill_value: Optional[IntoExpr] = None) -> None:
        self._column = column
        self._lag = lag
        self._fill_value = fill_value

    def input_type(self) -> set[ColumnType]:
        return ColumnType.ANY()

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return self._column.column_type

    def _transform(self) -> pl.Expr:
        return pl.col(self._column.name).shift(self._lag, fill_value=self._fill_value)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column.name}_lagged_{self._lag}')


class FirstValueTransformer(AggregatingTransformer):
    def __init__(self, column: ColumnSpecification) -> None:
        self._column = column

    def input_type(self) -> set[ColumnType]:
        return ColumnType.ANY()

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return self._column.column_type

    def _transform(self) -> pl.Expr:
        return pl.col(self._column.name).first()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column.name}_first_value')


class ArithmeticAggregationTransformer(AggregatingTransformer, ABC):
    def __init__(self, column: str | ColumnSpecification, cumulative: bool = False) -> None:
        self._column = column if isinstance(column, str) else column.name
        self._cumulative = cumulative

    def input_type(self) -> set[ColumnType]:
        return {ColumnType.NUMERIC}

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC


class CountTransformer(ArithmeticAggregationTransformer):
    def _transform(self) -> pl.Expr:
        if self._cumulative:
            return pl.col(self._column).cum_count()
        return pl.col(self._column).len()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        operation = 'cum_count' if self._cumulative else 'count'
        return transform.alias(f'{self._column}_{operation}')


class SumTransformer(ArithmeticAggregationTransformer):
    def _transform(self) -> pl.Expr:
        if self._cumulative:
            return pl.col(self._column).cum_sum()
        return pl.col(self._column).sum()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        operation = 'cum_sum' if self._cumulative else 'sum'
        return transform.alias(f'{self._column}_{operation}')


class MeanTransformer(ArithmeticAggregationTransformer):
    def _transform(self) -> pl.Expr:
        col = pl.col(self._column)
        if self._cumulative:
            cum_sum = col.cum_sum()
            cum_count = col.cum_count()
            return cum_sum.truediv(cum_count)
        return col.mean()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        operation = 'cum_mean' if self._cumulative else 'mean'
        return transform.alias(f'{self._column}_{operation}')


class StdTransformer(ArithmeticAggregationTransformer):
    def _transform(self) -> pl.Expr:
        col = pl.col(self._column)
        if self._cumulative:
            cum_sum = col.cum_sum()
            cum_count = col.cum_count()
            cum_mean = cum_sum.truediv(cum_count)

            mean_diff = col - cum_mean
            cum_sum_squared_mean_diff = mean_diff.pow(2).cum_sum()

            return cum_sum_squared_mean_diff.sqrt()
        return col.std()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        operation = 'cum_std' if self._cumulative else 'std'
        return transform.alias(f'{self._column}_{operation}')


class ArithmeticAggregations(Enum):
    COUNT = CountTransformer
    SUM = SumTransformer
    MEAN = MeanTransformer
    STD = StdTransformer
