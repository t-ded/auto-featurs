from collections.abc import Iterable
from collections.abc import Iterator
from enum import IntEnum

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.transformers.base import Transformer


class OptimizationLevel(IntEnum):
    NONE = 0
    SKIP_SELF = 1
    DEDUPLICATE_COMMUTATIVE = 2


class Optimizer:
    def __init__(self, optimization_level: OptimizationLevel) -> None:
        self._optimization_level = optimization_level

    @property
    def optimization_level(self) -> OptimizationLevel:
        return self._optimization_level

    @staticmethod
    def deduplicate_transformers_against_layers(present_columns: Iterable[ColumnSpecification], current_layer_additions: Iterable[Transformer]) -> list[Transformer]:
        deduplicated_current_layer_additions: list[Transformer] = []
        already_present_columns = set(present_columns)

        for transformer in current_layer_additions:
            col_spec = transformer.output_column_specification
            if col_spec not in already_present_columns:
                deduplicated_current_layer_additions.append(transformer)
                already_present_columns.add(col_spec)

        return deduplicated_current_layer_additions

    @staticmethod
    def _deduplicate_input_columns_for_transformer(
            transformer: type[Transformer],
            input_columns_positional_combinations: Iterable[tuple[ColumnSpecification, ...]],
    ) -> Iterator[tuple[ColumnSpecification, ...]]:
        if not transformer.is_commutative():
            yield from input_columns_positional_combinations
        else:
            seen_combinations: set[tuple[ColumnSpecification, ...]] = set()
            for column_combination in input_columns_positional_combinations:
                sorted_column_combination = tuple(sorted(column_combination, key=lambda col: col.name))
                if sorted_column_combination not in seen_combinations:
                    seen_combinations.add(sorted_column_combination)
                    yield column_combination

    @staticmethod
    def _skip_self(input_columns_positional_combinations: Iterable[tuple[ColumnSpecification, ...]]) -> Iterator[tuple[ColumnSpecification, ...]]:
        for column_combination in input_columns_positional_combinations:
            if len(set(column_combination)) == len(column_combination):
                yield column_combination

    def optimize_input_columns(self, transformer: type[Transformer], input_columns_positional_combinations: Iterable[tuple[ColumnSpecification, ...]]) -> Iterator[tuple[ColumnSpecification, ...]]:
        optimized = input_columns_positional_combinations
        if self._optimization_level >= OptimizationLevel.SKIP_SELF:
            optimized = self._skip_self(input_columns_positional_combinations)
        if self._optimization_level >= OptimizationLevel.DEDUPLICATE_COMMUTATIVE:
            optimized = self._deduplicate_input_columns_for_transformer(transformer, optimized)
        yield from optimized
