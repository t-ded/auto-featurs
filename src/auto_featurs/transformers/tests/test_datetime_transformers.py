
from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.transformers.datetime_transformers import DayOfWeekTransformer
from auto_featurs.utils.utils_for_tests import BASIC_FRAME
from auto_featurs.utils.utils_for_tests import assert_new_columns_in_frame


class TestDayOfWeekTransformer:
    def setup_method(self) -> None:
        self._day_of_week_transformer = DayOfWeekTransformer(column=ColumnSpecification.datetime(name='DATE_FEATURE'))

    def test_name_and_output_type(self) -> None:
        assert self._day_of_week_transformer.output_column_specification == ColumnSpecification.ordinal(name='DATE_FEATURE_day_of_week')

    def test_day_of_week_transform(self) -> None:
        df = BASIC_FRAME.with_columns(self._day_of_week_transformer.transform())
        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={'DATE_FEATURE_day_of_week': [6, 7, 1, 2, 3, 4]},
        )
