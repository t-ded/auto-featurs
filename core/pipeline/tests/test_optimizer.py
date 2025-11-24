from collections.abc import Iterable

import polars as pl

from core.base.column_specification import ColumnSpecification
from core.base.column_specification import ColumnType
from core.pipeline.optimizer import OptimizationLevel
from core.pipeline.optimizer import Optimizer
from core.transformers.base import Transformer
from utils.utils import get_names_from_column_specs


class MockCommutativeTransformer(Transformer):
    def __init__(self, left_column: str, right_column: str) -> None:
        self._left_column = left_column
        self._right_column = right_column

    def input_type(self) -> set[ColumnType] | tuple[set[ColumnType], ...]:
        return {ColumnType.NUMERIC}, {ColumnType.NUMERIC}

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _transform(self) -> pl.Expr:
        return pl.lit('commutative')

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_commutative_mock_{self._right_column}')


class MockNonCommutativeTransformer(Transformer):
    def __init__(self, left_column: str, right_column: str) -> None:
        self._left_column = left_column
        self._right_column = right_column

    def input_type(self) -> set[ColumnType] | tuple[set[ColumnType], ...]:
        return {ColumnType.NUMERIC}, {ColumnType.NUMERIC}

    @classmethod
    def is_commutative(cls) -> bool:
        return False

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _transform(self) -> pl.Expr:
        return pl.lit('non-commutative')

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._left_column}_non_commutative_mock_{self._right_column}')


class TestOptimizer:
    def setup_method(self) -> None:
        self._zero_level_optimizer = Optimizer(OptimizationLevel.NONE)
        self._skip_self_level_optimizer = Optimizer(OptimizationLevel.SKIP_SELF)
        self._deduplicate_commutative_level_optimizer = Optimizer(OptimizationLevel.DEDUPLICATE_COMMUTATIVE)
        a_col = ColumnSpecification.numeric(name='a')
        b_col = ColumnSpecification.numeric(name='b')
        self._input_columns = [(a_col, a_col), (a_col, b_col), (b_col, a_col), (b_col, b_col)]

    @staticmethod
    def _flatten_to_names(col_combinations: Iterable[tuple[ColumnSpecification, ...]]) -> list[list[str]]:
        return [get_names_from_column_specs(combination) for combination in col_combinations]

    def test_deduplicates_transformers_within_layer(self) -> None:
        present_columns: list[ColumnSpecification] = []
        mock_1 = MockCommutativeTransformer('a', 'b')
        mock_2 = MockCommutativeTransformer('a', 'b')
        current_layer_additions = [mock_1, mock_2]

        assert self._zero_level_optimizer.deduplicate_transformers_against_layers(present_columns, current_layer_additions) == [mock_1]

    def test_deduplicates_transformers_across_layers(self) -> None:
        present_columns: list[ColumnSpecification] = [ColumnSpecification.numeric(name='a_commutative_mock_b')]
        mock_1 = MockCommutativeTransformer('a', 'b')
        mock_2 = MockNonCommutativeTransformer('a', 'b')
        current_layer_additions = [mock_1, mock_2]

        assert self._zero_level_optimizer.deduplicate_transformers_against_layers(present_columns, current_layer_additions) == [mock_2]

    def test_zero_level_optimize_input_columns(self) -> None:
        commutative_optimized = self._flatten_to_names(self._zero_level_optimizer.optimize_input_columns(MockCommutativeTransformer, self._input_columns))
        non_commutative_optimized = self._flatten_to_names(self._zero_level_optimizer.optimize_input_columns(MockNonCommutativeTransformer, self._input_columns))

        assert commutative_optimized == [['a', 'a'], ['a', 'b'], ['b', 'a'], ['b', 'b']]
        assert non_commutative_optimized == [['a', 'a'], ['a', 'b'], ['b', 'a'], ['b', 'b']]

    def test_skip_self_level_optimize_input_columns(self) -> None:
        commutative_optimized = self._flatten_to_names(self._skip_self_level_optimizer.optimize_input_columns(MockNonCommutativeTransformer, self._input_columns))
        non_commutative_optimized = self._flatten_to_names(self._skip_self_level_optimizer.optimize_input_columns(MockNonCommutativeTransformer, self._input_columns))

        assert commutative_optimized == [['a', 'b'], ['b', 'a']]
        assert non_commutative_optimized == [['a', 'b'], ['b', 'a']]

    def test_deduplicate_commutative_level_optimize_input_columns(self) -> None:
        commutative_optimized = self._flatten_to_names(self._deduplicate_commutative_level_optimizer.optimize_input_columns(MockCommutativeTransformer, self._input_columns))
        non_commutative_optimized = self._flatten_to_names(self._deduplicate_commutative_level_optimizer.optimize_input_columns(MockNonCommutativeTransformer, self._input_columns))

        assert commutative_optimized == [['a', 'b']]
        assert non_commutative_optimized == [['a', 'b'], ['b', 'a']]
