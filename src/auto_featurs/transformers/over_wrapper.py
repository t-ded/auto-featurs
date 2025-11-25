from collections.abc import Iterable
from typing import Any

import polars as pl

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.transformers.aggregating_transformers import AggregatingTransformer
from auto_featurs.utils.utils import get_names_from_column_specs


class OverWrapper(AggregatingTransformer):
    def __init__(self, inner_transformer: AggregatingTransformer, over_columns: Iterable[str | ColumnSpecification], *args: Any) -> None:
        self._inner_transformer = inner_transformer
        self._over_columns: list[str] = get_names_from_column_specs(over_columns)

    def input_type(self) -> set[ColumnType] | tuple[set[ColumnType], ...]:
        return self._inner_transformer.input_type()

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return self._inner_transformer.output_column_specification.column_type

    def _transform(self) -> pl.Expr:
        agg_expr = self._inner_transformer.transform()
        return agg_expr.over(self._over_columns)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        over_name = '_over_' + '_and_'.join(self._over_columns)
        return transform.name.suffix(over_name)
