import math
from abc import ABC
from enum import Enum
from typing import Literal
from typing import Optional
from typing import assert_never

import polars as pl

from auto_featurs.base.column_specification import ColumnNameOrSpec
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.column_specification import ColumnTypeSelector
from auto_featurs.transformers.base import Transformer
from auto_featurs.utils.utils import parse_column_name


class SeasonalTransformer(Transformer, ABC):
    def __init__(self, column: ColumnNameOrSpec, angular: bool = False, gon_transformation: Optional[Literal['sin', 'cos']] = None) -> None:
        if not angular and gon_transformation is not None:
            raise ValueError('gon_transformation can be used only with angular=True')

        self._column = parse_column_name(column)
        self._angular = angular
        self._gon_transformation = gon_transformation

    def input_type(self) -> ColumnTypeSelector:
        return ColumnType.DATETIME.as_selector()

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return ColumnType.ORDINAL if self._gon_transformation is None else ColumnType.NUMERIC

    def _gon_transform(self, expr: pl.Expr) -> pl.Expr:
        match self._gon_transformation:
            case None:
                return expr
            case 'sin':
                return expr.sin()
            case 'cos':
                return expr.cos()

    def _suffix(self) -> str:
        angular_suffix = '_angular' if self._angular else ''
        gon_transf_suffix = f'_{self._gon_transformation}' if self._gon_transformation is not None else ''
        return angular_suffix + gon_transf_suffix


class HourOfDayTransformer(SeasonalTransformer):
    def _transform(self) -> pl.Expr:
        res = pl.col(self._column).dt.hour()
        if self._angular:
            res = res.mul(2 * math.pi).truediv(24)
        return self._gon_transform(res)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column}_hour_of_day' + self._suffix())


class DayOfWeekTransformer(SeasonalTransformer):
    def _transform(self) -> pl.Expr:
        res = pl.col(self._column).dt.weekday()
        if self._angular:
            res = res.sub(1).mul(2 * math.pi).truediv(7)
        return self._gon_transform(res)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column}_day_of_week' + self._suffix())


class MonthOfYearTransformer(SeasonalTransformer):
    def _transform(self) -> pl.Expr:
        res = pl.col(self._column).dt.month()
        if self._angular:
            res = res.sub(1).mul(2 * math.pi).truediv(12)
        return self._gon_transform(res)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column}_month_of_year' + self._suffix())


class SeasonalOperation(Enum):
    HOUR_OF_DAY = HourOfDayTransformer
    DAY_OF_WEEK = DayOfWeekTransformer
    MONTH_OF_YEAR = MonthOfYearTransformer


class TimeDiffTransformer(Transformer):
    def __init__(self, left_column: ColumnNameOrSpec, right_column: ColumnNameOrSpec, unit: Literal['s', 'h', 'd'] = 'd') -> None:
        self._left_column = parse_column_name(left_column)
        self._right_column = parse_column_name(right_column)
        self._unit = unit

    def input_type(self) -> tuple[ColumnTypeSelector, ColumnTypeSelector]:
        return ColumnType.DATETIME.as_selector(), ColumnType.DATETIME.as_selector()

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
        unit_str: str = ''
        match self._unit:
            case 's':
                unit_str = 'seconds'
            case 'h':
                unit_str = 'hours'
            case 'd':
                unit_str = 'days'
            case _:
                assert_never(self._unit)

        return transform.alias(f'{self._left_column}_total_{unit_str}_diff_{self._right_column}')
