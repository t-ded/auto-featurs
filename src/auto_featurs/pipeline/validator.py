import logging
from collections.abc import Sequence
from datetime import timedelta
from typing import Optional

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.transformers.base import Transformer
from auto_featurs.transformers.over_wrapper import OverWrapper
from auto_featurs.transformers.rolling_wrapper import RollingWrapper

logger = logging.getLogger(__name__)


class Validator:
    @staticmethod
    def validate_time_window_index_column(time_windows: Sequence[Optional[str | timedelta]], index_column: Optional[ColumnSpecification]) -> None:
        if time_windows and time_windows[0] is not None and index_column is None:
            raise ValueError('Time window specified without index column.')
        if index_column is not None and index_column.column_type != ColumnType.DATETIME:
            raise ValueError(f'Currently only {ColumnType.DATETIME} columns are supported for rolling aggregation but {index_column.column_type} was passed for {index_column.name}.')

    @staticmethod
    def validate_transformer_against_input_columns(transformer: Transformer, input_columns: tuple[ColumnSpecification, ...]) -> None:
        if isinstance(transformer, RollingWrapper | OverWrapper):
            return

        expected_column_types_per_column = transformer.input_type()
        iterable_expected_column_types_per_column = (expected_column_types_per_column, ) if isinstance(expected_column_types_per_column, set) else expected_column_types_per_column
        iterable_expected_column_types_per_column = tuple(col_types for col_types in iterable_expected_column_types_per_column if len(col_types) > 0)

        if len(input_columns) != len(iterable_expected_column_types_per_column):
            raise ValueError(
                f'Transformer {transformer} expected {len(iterable_expected_column_types_per_column)} input columns, '
                f'but received {len(input_columns)}.',
            )

        for column, expected_types in zip(input_columns, iterable_expected_column_types_per_column, strict=True):
            if column.column_type not in expected_types:
                raise ValueError(
                    f"Column '{column.name}' has type '{column.column_type}', "
                    f"but transformer {transformer} expected one of '{expected_types}'.",
                )
