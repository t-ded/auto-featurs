from abc import ABC
from enum import Enum

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
