from __future__ import annotations

import logging
from collections.abc import Sequence

import polars as pl
from more_itertools import flatten

from auto_featurs.base.column_specification import ColumnRole
from auto_featurs.base.column_specification import ColumnSelection
from auto_featurs.base.column_specification import ColumnSet
from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.column_specification import Schema
from auto_featurs.utils.utils import order_preserving_unique

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, data: pl.LazyFrame | pl.DataFrame, schema: Schema, drop_columns_outside_schema: bool = False) -> None:
        self._data = data.lazy()
        self._schema: Schema = schema
        if drop_columns_outside_schema:
            self._select_columns_in_schema()

    def _select_columns_in_schema(self) -> None:
        data_cols = set(self._data.collect_schema().names())
        schema_cols = {col.name for col in self._schema}
        columns_outside_schema = data_cols - schema_cols

        if columns_outside_schema:
            logger.warning(f'Dropping columns not present in schema: {', '.join(sorted(columns_outside_schema))}')
            self._data = self._data.drop(columns_outside_schema, strict=False)

    @property
    def data(self) -> pl.LazyFrame:
        return self._data

    @property
    def schema(self) -> Schema:
        return self._schema

    def get_combinations_from_selections(self, *subsets: ColumnSelection) -> list[ColumnSet]:
        return [self.get_columns_from_selection(subset) for subset in subsets]

    def get_columns_from_selection(self, subset: ColumnSelection) -> ColumnSet:
        match subset:
            case ColumnType():
                return self.get_columns_of_type(subset)
            case str():
                return [self.get_column_by_name(subset)]
            case Sequence():
                return order_preserving_unique(flatten([self.get_columns_from_selection(col) for col in subset]))
            case _:
                raise ValueError(f'Unexpected subset type: {type(subset)}')

    def get_columns_of_type(self, column_type: ColumnType) -> ColumnSet:
        return [col_spec for col_spec in self._schema if col_spec.column_type == column_type]

    def get_column_by_name(self, column_name: str) -> ColumnSpecification:
        for col_spec in self._schema:
            if col_spec.name == column_name:
                return col_spec
        raise KeyError(f'Column "{column_name}" not found in schema.')

    def get_label_column(self) -> ColumnSpecification:
        for col_spec in self._schema:
            if col_spec.column_role == ColumnRole.LABEL:
                return col_spec
        raise ValueError('No label column found in schema.')

    def with_columns(self, new_columns: list[pl.Expr]) -> Dataset:
        return Dataset(self._data.with_columns(*new_columns), self._schema)

    def with_schema(self, new_schema: Schema) -> Dataset:
        return Dataset(self._data, self._schema + new_schema)

    def with_cached_computation(self) -> Dataset:
        return Dataset(self._data.cache(), self._schema)

    def collect(self) -> pl.DataFrame:
        return self._data.collect()
