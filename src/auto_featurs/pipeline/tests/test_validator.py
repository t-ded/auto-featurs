
import polars as pl
import pytest

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.column_specification import ColumnTypeSelector
from auto_featurs.pipeline.validator import Validator
from auto_featurs.transformers.base import Transformer


class MockInputTypeTransformer(Transformer):
    def __init__(self, expected_types: frozenset[ColumnType] | tuple[frozenset[ColumnType], ...]) -> None:
        self._expected_types = ColumnTypeSelector(expected_types) if isinstance(expected_types, frozenset) else tuple(ColumnTypeSelector(types) for types in expected_types)

    def input_type(self) -> ColumnTypeSelector | tuple[ColumnTypeSelector, ...]:
        return self._expected_types

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return ColumnType.TEXT

    def _transform(self) -> pl.Expr:
        return pl.lit('Mock')

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias('Mock')


class TestValidator:
    def setup_method(self) -> None:
        self._validator = Validator()

    def test_validate_time_window_index_column_raises_without_index_column(self) -> None:
        time_windows = ('1h',)
        index_column = None

        with pytest.raises(ValueError, match='Time window specified without index column'):
            self._validator.validate_time_window_index_column(time_windows, index_column)

    def test_validate_time_window_index_column_passes_when_index_and_first_window_present(self) -> None:
        time_windows = ('1h',)
        index_column = ColumnSpecification(name='ts', column_type=ColumnType.DATETIME)
        self._validator.validate_time_window_index_column(time_windows, index_column)

    def test_validate_transformer_wrong_number_of_columns(self) -> None:
        transformer = MockInputTypeTransformer((frozenset([ColumnType.NUMERIC]), frozenset([ColumnType.ORDINAL])))
        input_columns = (ColumnSpecification(name='a', column_type=ColumnType.NUMERIC),)

        with pytest.raises(ValueError, match='expected 2 input columns'):
            self._validator.validate_transformer_against_input_columns(transformer, input_columns)

    def test_validate_transformer_wrong_column_type(self) -> None:
        transformer = MockInputTypeTransformer(frozenset([ColumnType.NUMERIC]))
        input_columns = (ColumnSpecification(name='a', column_type=ColumnType.ORDINAL),)

        with pytest.raises(ValueError, match='expected one of'):
            self._validator.validate_transformer_against_input_columns(transformer, input_columns)

    def test_validate_transformer_valid_input(self) -> None:
        transformer = MockInputTypeTransformer(frozenset([ColumnType.NUMERIC]))
        input_columns = (ColumnSpecification(name='a', column_type=ColumnType.NUMERIC),)
        self._validator.validate_transformer_against_input_columns(transformer, input_columns)
