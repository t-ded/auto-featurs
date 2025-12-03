from collections.abc import Sequence
from datetime import timedelta
import logging
from typing import Optional

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.transformers.base import Transformer
from auto_featurs.transformers.over_wrapper import OverWrapper
from auto_featurs.transformers.rolling_wrapper import RollingWrapper

logger = logging.getLogger(__name__)


class Validator:
    def __init__(self, raise_on_validation_error: bool = True) -> None:
        self._raise_on_validation_error = raise_on_validation_error

    @staticmethod
    def validate_time_window_index_column(time_windows: Sequence[Optional[str | timedelta]], index_column: Optional[ColumnSpecification]) -> None:
        if time_windows and time_windows[0] is not None and index_column is None:
            raise ValueError('Time window specified without index column.')
        if index_column is not None and index_column.column_type != ColumnType.DATETIME:
            raise ValueError(f'Currently only {ColumnType.DATETIME} columns are supported for rolling aggregation but {index_column.column_type} was passed for {index_column.name}.')

    def validate_transformer_against_input_columns(self, transformer: Transformer, input_columns: tuple[ColumnSpecification, ...]) -> bool:
        if isinstance(transformer, RollingWrapper | OverWrapper):
            return True

        expected_column_types_per_column = transformer.input_type()
        iterable_expected_column_types_per_column = (expected_column_types_per_column, ) if isinstance(expected_column_types_per_column, set) else expected_column_types_per_column
        iterable_expected_column_types_per_column = tuple(col_types for col_types in iterable_expected_column_types_per_column if len(col_types) > 0)

        if len(input_columns) != len(iterable_expected_column_types_per_column):
            self._raise_or_warn(
                f'Transformer {transformer} expected {len(iterable_expected_column_types_per_column)} input columns, '
                f'but received {len(input_columns)}.',
            )
            return False

        for column, expected_types in zip(input_columns, iterable_expected_column_types_per_column, strict=True):
            if column.column_type not in expected_types:
                self._raise_or_warn(
                    f"Column '{column.name}' has type '{column.column_type}', "
                    f"but transformer {transformer} expected one of '{expected_types}'.",
                )
                return False

        return True

    def _raise_or_warn(self, message: str) -> None:
        if self._raise_on_validation_error:
            raise ValueError(message)
        else:
            logger.warning(message)
