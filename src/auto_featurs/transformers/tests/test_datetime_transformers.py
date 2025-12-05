from datetime import UTC
from datetime import datetime
from typing import Optional

import polars as pl

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.transformers.datetime_transformers import DayOfWeekTransformer
from auto_featurs.transformers.datetime_transformers import HourOfDayTransformer
from auto_featurs.transformers.datetime_transformers import MonthOfYearTransformer


class TestSeasonalTransformers:
    def setup_method(self) -> None:
        self._hour_of_day_transformer = HourOfDayTransformer(column=ColumnSpecification.datetime(name='DATE_FEATURE'))
        self._day_of_week_transformer = DayOfWeekTransformer(column=ColumnSpecification.datetime(name='DATE_FEATURE'))
        self._month_of_year_transformer = MonthOfYearTransformer(column=ColumnSpecification.datetime(name='DATE_FEATURE'))

    def test_name_and_output_type(self) -> None:
        assert self._hour_of_day_transformer.output_column_specification == ColumnSpecification.ordinal(name='DATE_FEATURE_hour_of_day')
        assert self._day_of_week_transformer.output_column_specification == ColumnSpecification.ordinal(name='DATE_FEATURE_day_of_week')
        assert self._month_of_year_transformer.output_column_specification == ColumnSpecification.ordinal(name='DATE_FEATURE_month_of_year')

    @staticmethod
    def _get_df(months: Optional[list[int]] = None, days: Optional[list[int]] = None, hours: Optional[list[int]] = None) -> pl.DataFrame:
        months = months or [1, 1, 1]
        days = days or [1, 1, 1]
        hours = hours or [1, 1, 1]

        datetimes: list[datetime] = []
        for month, day, hour in zip(months, days, hours, strict=True):
            datetimes.append(datetime(year=2018, month=month, day=day, hour=hour, tzinfo=UTC))

        return pl.DataFrame({'DATE_FEATURE': datetimes})

    def test_hour_of_day_transform(self) -> None:
        hours = [20, 21, 22]
        df = self._get_df(hours=hours)
        df = df.with_columns(self._hour_of_day_transformer.transform())
        assert df['DATE_FEATURE_hour_of_day'].to_list() == hours

    def test_day_of_week_transform(self) -> None:
        days = [1, 2, 3]
        df = self._get_df(days=days)
        df = df.with_columns(self._day_of_week_transformer.transform())
        assert df['DATE_FEATURE_day_of_week'].to_list() == days

    def test_month_of_year_transform(self) -> None:
        months = [10, 11, 12]
        df = self._get_df(months=months)
        df = df.with_columns(self._month_of_year_transformer.transform())
        assert df['DATE_FEATURE_month_of_year'].to_list() == months
