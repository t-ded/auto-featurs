from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Schema):
            raise TypeError(f'Cannot compare {type(other)} to Schema')
        return self._columns == other.columns

    @classmethod
    def from_dict(cls, spec: dict[ColumnType, list[str]], *, label_col: Optional[str] = None) -> Schema:
        columns: list[ColumnSpecification] = []

        for col_type, names in spec.items():
            for name in names:
                role = ColumnRole.LABEL if name == label_col else ColumnRole.FEATURE
                columns.append(ColumnSpecification(name=name, column_type=col_type, column_role=role))

        if label_col is not None and all(col.name != label_col for col in columns):
            raise ValueError(f'label_col={label_col!r} not found in provided columns')

        return cls(columns)

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

    def get_columns_of_type(self, column_type: ColumnType, subset: Optional[ColumnSet] = None) -> ColumnSet:
        if subset is None:
            subset = self._columns
        else:
            self._check_subset_in_schema(subset)
        return [col_spec for col_spec in subset if col_spec.column_type == column_type]

    def get_columns_of_role(self, column_role: ColumnRole, subset: Optional[ColumnSet] = None) -> ColumnSet:
        if subset is None:
            subset = self._columns
        else:
            self._check_subset_in_schema(subset)
        return [col_spec for col_spec in subset if col_spec.column_role == column_role]

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

    def _check_subset_in_schema(self, subset: ColumnSet) -> None:
        not_present = [col for col in subset if col not in self._columns]
        if not_present:
            not_present_names = sorted(get_names_from_column_specs(subset))
            raise ValueError(f'The following columns in subset not found in schema: {not_present_names}')
