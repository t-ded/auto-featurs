from abc import ABC
from typing import Any

import polars as pl

from core.base.column_specification import ColumnSpecification
from core.base.column_specification import ColumnType
from core.transformers.base import Transformer


class AggregatingTransformer(Transformer, ABC):
    pass


class LaggedTransformer(AggregatingTransformer):
    def __init__(self, column: ColumnSpecification, lag: int, fill_value: Any = None) -> None:
        self._column = column
        self._lag = lag
        self._fill_value = fill_value

    def input_type(self) -> set[ColumnType] | tuple[set[ColumnType], ...]:
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
