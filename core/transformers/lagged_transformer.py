from collections.abc import Iterable
from typing import Any
from typing import Optional

import polars as pl

from core.base.column_specification import ColumnSpecification
from core.base.column_specification import ColumnType
from core.transformers.base import Transformer
from utils.utils import get_names_from_column_specs


class LaggedTransformer(Transformer):
    def __init__(self, column: ColumnSpecification, lag: int, over_columns: Optional[Iterable[str | ColumnSpecification]] = None, fill_value: Any = None) -> None:
        self._column = column
        self._over_columns: list[str] = get_names_from_column_specs(over_columns) if over_columns else []
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
        lag_expr = pl.col(self._column.name).shift(self._lag, fill_value=self._fill_value)
        if self._over_columns:
            return lag_expr.over(self._over_columns)
        return lag_expr

    def _name(self, transform: pl.Expr) -> pl.Expr:
        lag_name = f'{self._column.name}_lagged_{self._lag}'
        over_name = ('_over_' + '_and_'.join(self._over_columns)) if self._over_columns else ''
        return transform.alias(lag_name + over_name)
