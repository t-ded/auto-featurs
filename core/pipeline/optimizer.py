from collections.abc import Iterable
from collections.abc import KeysView
from typing import Iterator

from core.transformers.base import Transformer


class Optimizer:
    @staticmethod
    def deduplicate_transformers_against_layers(present_columns: KeysView[str], current_layer_additions: list[Transformer]) -> list[Transformer]:
        deduplicated_current_layer_additions: list[Transformer] = []
        already_present_columns = set(present_columns)

        for transformer in current_layer_additions:
            col_name = transformer.new_column_type()[0]
            if col_name not in present_columns:
                deduplicated_current_layer_additions.append(transformer)
                already_present_columns.add(col_name)

        return deduplicated_current_layer_additions

    @staticmethod
    def _deduplicate_input_columns_for_transformer(transformer: Transformer, input_columns_positional_combinations: Iterable[tuple[str, ...]]) -> Iterator[tuple[str, ...]]:
        if not transformer.is_commutative:
            yield from input_columns_positional_combinations

        seen_combinations: set[tuple[str, ...]] = set()
        for column_combination in input_columns_positional_combinations:
            sorted_column_combination = tuple(sorted(column_combination))
            if sorted_column_combination not in seen_combinations:
                seen_combinations.add(sorted_column_combination)
                yield column_combination

    @staticmethod
    def _skip_self(input_columns_positional_combinations: Iterable[tuple[str, ...]]) -> Iterator[tuple[str, ...]]:
        for column_combination in input_columns_positional_combinations:
            if len(set(column_combination)) == len(column_combination):
                yield column_combination

    def optimize_input_columns(self, transformer: Transformer, input_columns_positional_combinations: Iterable[tuple[str, ...]]) -> Iterator[tuple[str, ...]]:
        optimized = self._skip_self(input_columns_positional_combinations)
        optimized = self._deduplicate_input_columns_for_transformer(transformer, optimized)
        yield from optimized
