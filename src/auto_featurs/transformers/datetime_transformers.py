from abc import ABC
from enum import Enum
from typing import Literal
from typing import assert_never

import polars as pl

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.transformers.base import Transformer


class SeasonalTransformer(Transformer, ABC):
    def __init__(self, column: str | ColumnSpecification) -> None:
        self._column = column if isinstance(column, str) else column.name

    def input_type(self) -> set[ColumnType]:
        return {ColumnType.DATETIME}

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return ColumnType.ORDINAL


class HourOfDayTransformer(SeasonalTransformer):
    def _transform(self) -> pl.Expr:
        return pl.col(self._column).dt.hour()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column}_hour_of_day')


class DayOfWeekTransformer(SeasonalTransformer):
    def _transform(self) -> pl.Expr:
        return pl.col(self._column).dt.weekday()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column}_day_of_week')


class MonthOfYearTransformer(SeasonalTransformer):
    def _transform(self) -> pl.Expr:
        return pl.col(self._column).dt.month()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column}_month_of_year')


class SeasonalOperation(Enum):
    HOUR_OF_DAY = HourOfDayTransformer
    DAY_OF_WEEK = DayOfWeekTransformer
    MONTH_OF_YEAR = MonthOfYearTransformer


class TimeDiffTransformer(Transformer):
    def __init__(self, left_column: str | ColumnSpecification, right_column: str | ColumnSpecification, unit: Literal['s', 'h', 'd'] = 'd') -> None:
        self._left_column = left_column if isinstance(left_column, str) else left_column.name
        self._right_column = right_column if isinstance(right_column, str) else right_column.name
        self._unit = unit

    def input_type(self) -> tuple[set[ColumnType], set[ColumnType]]:
        return {ColumnType.DATETIME}, {ColumnType.DATETIME}

    @classmethod
    def is_commutative(cls) -> bool:
        return False

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _transform(self) -> pl.Expr:
        diff = pl.col(self._left_column).sub(pl.col(self._right_column))
        match self._unit:
            case 's':
                return diff.dt.total_seconds()
            case 'h':
                return diff.dt.total_hours()
            case 'd':
                return diff.dt.total_days()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        suffix: str = ''
        match self._unit:
            case 's':
                suffix = '_total_seconds'
            case 'h':
                suffix = '_total_hours'
            case 'd':
                suffix = '_total_days'
            case _:
                assert_never(self._unit)

        return transform.alias(f'{self._left_column}_subtract_{self._right_column}{suffix}')
