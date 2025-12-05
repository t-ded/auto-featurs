from __future__ import annotations

from collections.abc import Sequence

from more_itertools import flatten

from auto_featurs.base.column_specification import ColumnRole
from auto_featurs.base.column_specification import ColumnSelection
from auto_featurs.base.column_specification import ColumnSet
from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.utils.utils import get_names_from_column_specs
from auto_featurs.utils.utils import order_preserving_unique


class Schema:
    def __init__(self, columns: list[ColumnSpecification]) -> None:
        self._columns = columns

    def __add__(self, other: object) -> Schema:
        if not isinstance(other, Schema):
            raise TypeError(f'Cannot add {type(other)} to Schema')
        return Schema(self._columns + other.columns)

    @property
    def columns(self) -> list[ColumnSpecification]:
        return self._columns

    @property
    def column_names(self) -> list[str]:
        return get_names_from_column_specs(self._columns)

    @property
    def num_columns(self) -> int:
        return len(self._columns)

    @property
    def label_column(self) -> ColumnSpecification:
        for col_spec in self._columns:
            if col_spec.column_role == ColumnRole.LABEL:
                return col_spec
        raise ValueError('No label column found in schema.')

    def get_column_by_name(self, column_name: str) -> ColumnSpecification:
        for col_spec in self._columns:
            if col_spec.name == column_name:
                return col_spec
        raise KeyError(f'Column "{column_name}" not found in schema.')

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
        return [col_spec for col_spec in self._columns if col_spec.column_type == column_type]
