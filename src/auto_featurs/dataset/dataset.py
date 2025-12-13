from __future__ import annotations

import logging

import polars as pl

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.schema import ColumnSelection
from auto_featurs.base.schema import ColumnSet
from auto_featurs.base.schema import Schema

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, data: pl.LazyFrame | pl.DataFrame, schema: Schema, drop_columns_outside_schema: bool = False) -> None:
        self._data = data.lazy()
        self._schema: Schema = schema
        if drop_columns_outside_schema:
            self._select_columns_in_schema()

    def _select_columns_in_schema(self) -> None:
        data_cols = set(self._data.collect_schema().names())
        schema_cols = set(self._schema.column_names)
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

    @property
    def num_columns(self) -> int:
        return self._schema.num_columns

    def get_combinations_from_selections(self, *subsets: ColumnSelection) -> list[ColumnSet]:
        return [self.get_columns_from_selection(subset) for subset in subsets]

    def get_columns_from_selection(self, subset: ColumnSelection) -> ColumnSet:
        return self._schema.get_columns_from_selection(subset)

    def get_columns_of_type(self, column_type: ColumnType) -> ColumnSet:
        return self._schema.get_columns_of_type(column_type)

    def get_column_by_name(self, column_name: str) -> ColumnSpecification:
        return self._schema.get_column_by_name(column_name)

    def get_label_column(self) -> ColumnSpecification:
        return self._schema.label_column

    def with_columns(self, new_columns: list[pl.Expr]) -> Dataset:
        return Dataset(self._data.with_columns(*new_columns), self._schema)

    def with_schema(self, new_schema: Schema) -> Dataset:
        return Dataset(self._data, self._schema + new_schema)

    def with_cached_computation(self) -> Dataset:
        return Dataset(self._data.cache(), self._schema)

    def collect(self) -> pl.DataFrame:
        return self._data.collect()
