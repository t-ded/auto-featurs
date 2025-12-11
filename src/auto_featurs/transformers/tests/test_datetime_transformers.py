from datetime import UTC
from datetime import datetime
from typing import Optional

import polars as pl

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.transformers.datetime_transformers import DayOfWeekTransformer
from auto_featurs.transformers.datetime_transformers import HourOfDayTransformer
from auto_featurs.transformers.datetime_transformers import MonthOfYearTransformer
from auto_featurs.transformers.datetime_transformers import TimeDiffTransformer


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


class TestTimeDiffTransformer:
    def setup_method(self) -> None:
        self._time_diff_transformer_seconds = TimeDiffTransformer(left_column='DATE_FEATURE', right_column='DATE_FEATURE_2', unit='s')
        self._time_diff_transformer_hours = TimeDiffTransformer(left_column='DATE_FEATURE', right_column='DATE_FEATURE_2', unit='h')
        self._time_diff_transformer_days = TimeDiffTransformer(left_column='DATE_FEATURE', right_column='DATE_FEATURE_2', unit='d')

    def test_name_and_output_type(self) -> None:
        assert self._time_diff_transformer_seconds.output_column_specification == ColumnSpecification.numeric(name='DATE_FEATURE_subtract_DATE_FEATURE_2_total_seconds')
        assert self._time_diff_transformer_hours.output_column_specification == ColumnSpecification.numeric(name='DATE_FEATURE_subtract_DATE_FEATURE_2_total_hours')
        assert self._time_diff_transformer_days.output_column_specification == ColumnSpecification.numeric(name='DATE_FEATURE_subtract_DATE_FEATURE_2_total_days')

    def test_time_diff_transform(self) -> None:
        df = pl.DataFrame(
            {
                'DATE_FEATURE_2': [
                    datetime(2018, 1, 1, 1, 0, 1, tzinfo=UTC),
                    datetime(2018, 1, 2, 2, 0, 2, tzinfo=UTC),
                    datetime(2018, 1, 3, 3, 0, 3, tzinfo=UTC),
                ],
                'DATE_FEATURE': [
                    datetime(2018, 1, 2, 2, 0, 2, tzinfo=UTC),
                    datetime(2018, 1, 3, 3, 0, 3, tzinfo=UTC),
                    datetime(2018, 1, 4, 4, 0, 4, tzinfo=UTC),
                ],
            },
        )

        df = df.with_columns(
            self._time_diff_transformer_seconds.transform(),
            self._time_diff_transformer_hours.transform(),
            self._time_diff_transformer_days.transform(),
        )
        assert df['DATE_FEATURE_subtract_DATE_FEATURE_2_total_seconds'].to_list() == [90_001, 90_001, 90_001]
        assert df['DATE_FEATURE_subtract_DATE_FEATURE_2_total_hours'].to_list() == [25, 25, 25]
        assert df['DATE_FEATURE_subtract_DATE_FEATURE_2_total_days'].to_list() == [1, 1, 1]
