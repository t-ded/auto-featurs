from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable, Sequence
from itertools import product
from typing import Any
from typing import Optional

import polars as pl

from core.base.column_specification import ColumnType
from core.transformers.base import Transformer
from core.transformers.comparison_transformers import Comparisons
from core.transformers.numeric_transformers import ArithmeticOperation
from core.transformers.numeric_transformers import PolynomialTransformer
from utils.utils import order_preserving_unique

class Pipeline:
    def __init__(
        self,
        column_types: Optional[dict[str, ColumnType]] = None,
        transformers: Optional[list[list[Transformer]]] = None,
    ) -> None:
        self._transformers: list[list[Transformer]] = transformers or [[]]
        self._column_types: dict[str, ColumnType] = column_types or {}

    def with_polynomial(self, subset: str | Sequence[str] | ColumnType, degrees: Iterable[int]) -> Pipeline:
        selection = self._get_columns_from_subset(subset)

        transformers = self._build_transformers(
            transformer_factory=PolynomialTransformer,
            input_columns=[selection],
            kw_params={'degree': degrees},
        )

        return self._with_added_to_current_layer(transformers)

    def with_arithmetic(self, left_subset: str | Sequence[str] | ColumnType, right_subset: str | Sequence[str] | ColumnType, operations: Iterable[ArithmeticOperation]) -> Pipeline:
        transformers: list[Transformer] = []
        operations = order_preserving_unique(operations)
        left_selection = self._get_columns_from_subset(left_subset)
        right_selection = self._get_columns_from_subset(right_subset)

        for op in operations:
            transformers.extend(
                self._build_transformers(
                    transformer_factory=op.value,
                    input_columns=[left_selection, right_selection],
                ),
            )

        return self._with_added_to_current_layer(transformers)

    def with_comparison(self, left_subset: str | Sequence[str] | ColumnType, right_subset: str | Sequence[str] | ColumnType, comparisons: Iterable[Comparisons]) -> Pipeline:
        transformers: list[Transformer] = []
        comparisons = order_preserving_unique(comparisons)
        left_selection = self._get_columns_from_subset(left_subset)
        right_selection = self._get_columns_from_subset(right_subset)

        for comp in comparisons:
            transformers.extend(
                self._build_transformers(
                    transformer_factory=comp.value,
                    input_columns=[left_selection, right_selection],
                ),
            )

        return self._with_added_to_current_layer(transformers)

    def with_new_layer(self) -> Pipeline:
        current_layer_column_types = self._get_column_types_from_transformers(self._current_layer())
        return Pipeline(column_types=self._column_types | current_layer_column_types, transformers=self._transformers + [[]])

    def collect(self, df: pl.LazyFrame) -> pl.DataFrame:
        for layer in self._transformers:
            exprs = [transformer.transform() for transformer in layer]
            df = df.with_columns(*exprs)
        return df.collect()

    def _with_added_to_current_layer(self, transformers: Transformer | Iterable[Transformer]) -> Pipeline:
        current_layer_additions = [transformers] if isinstance(transformers, Transformer) else list(transformers)
        return Pipeline(column_types=self._column_types, transformers=self._transformers[:-1] + [self._current_layer() + current_layer_additions])

    def _current_layer(self) -> list[Transformer]:
        return self._transformers[-1]

    def _get_columns_from_subset(self, subset: str | Sequence[str] | ColumnType) -> list[str]:
        match subset:
            case ColumnType():
                return self._get_columns_of_type(subset)
            case str():
                return [subset]
            case Sequence():
                return list(subset)
            case _:
                raise ValueError(f'Unexpected subset type: {type(subset)}')

    def _get_columns_of_type(self, column_type: ColumnType) -> list[str]:
        return [col for col, col_type in self._column_types.items() if col_type == column_type]

    @staticmethod
    def _get_column_types_from_transformers(transformers: Iterable[Transformer]) -> dict[str, ColumnType]:
        missing_column_types: dict[str, ColumnType] = {}
        for transformer in transformers:
            col_name, col_type = transformer.new_column_type()
            missing_column_types[col_name] = col_type
        return missing_column_types

    @staticmethod
    def _build_transformers(
        *,
        transformer_factory: Callable[..., Transformer],
        input_columns: Optional[Iterable[Iterable[str]]] = None,
        kw_params: Optional[dict[str, Iterable[Any]]] = None,
        **kwargs,
    ) -> list[Transformer]:

        transformers: list[Transformer] = []

        input_columns = input_columns or [[]]
        kw_params = kw_params or {}

        input_columns_positional_combinations = list(product(*input_columns))
        kw_keys = list(kw_params.keys())
        kw_params_positional_combinations = list(product(*kw_params.values()))

        for column_combination in input_columns_positional_combinations:
            if len(set(column_combination)) != len(column_combination):
                continue
            for kw_params_combination in kw_params_positional_combinations:
                transformer_kwargs = dict(zip(kw_keys, kw_params_combination)) | kwargs
                transformers.append(transformer_factory(*column_combination, **transformer_kwargs))

        return transformers
