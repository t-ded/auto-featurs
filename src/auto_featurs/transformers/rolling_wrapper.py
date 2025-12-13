from datetime import timedelta
from typing import Any

import polars as pl

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.column_specification import ColumnTypeSelector
from auto_featurs.transformers.aggregating_transformers import AggregatingTransformer
from auto_featurs.utils.utils import format_timedelta


class RollingWrapper[AT: AggregatingTransformer](AggregatingTransformer):
    def __init__(self, inner_transformer: AT, index_column: ColumnSpecification, time_window: str | timedelta, *args: Any) -> None:
        if index_column.column_type != ColumnType.DATETIME:
            raise ValueError(f'Currently only {ColumnType.DATETIME} columns are supported for rolling aggregation but {index_column.column_type} was passed for {index_column.name}.')

        self._inner_transformer = inner_transformer
        self._index_column = index_column
        self._time_window = time_window

    def input_type(self) -> ColumnTypeSelector | tuple[ColumnTypeSelector, ...]:
        return self._inner_transformer.input_type()

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return self._inner_transformer.output_column_specification.column_type

    def _transform(self) -> pl.Expr:
        agg_expr = self._inner_transformer.transform()
        return agg_expr.last().rolling(index_column=self._index_column.name, period=self._time_window)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        time_window = format_timedelta(self._time_window) if isinstance(self._time_window, timedelta) else self._time_window
        return transform.name.suffix(f'_in_the_last_{time_window}')
