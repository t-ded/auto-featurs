from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import product
from typing import Any
from typing import Optional

from more_itertools import flatten
import polars as pl

from core.base.column_specification import ColumnSpecification
from core.base.column_specification import ColumnType
from core.pipeline.optimizer import OptimizationLevel
from core.pipeline.optimizer import Optimizer
from core.transformers.aggregating_transformers import LaggedTransformer
from core.transformers.base import Transformer
from core.transformers.comparison_transformers import Comparisons
from core.transformers.numeric_transformers import ArithmeticOperation
from core.transformers.numeric_transformers import PolynomialTransformer
from core.transformers.over_wrapper import OverWrapper
from utils.utils import order_preserving_unique

ColumnSelection = str | Sequence[str] | ColumnType | Sequence[ColumnType]
ColumnSet = list[ColumnSpecification]
TransformerLayers = list[list[Transformer]]
Schema = list[ColumnSpecification]


class Pipeline:
    def __init__(
        self,
        schema: Optional[Schema] = None,
        transformers: Optional[TransformerLayers] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.NONE,
    ) -> None:
        self._schema: Schema = schema or []
        self._transformers: TransformerLayers = transformers or [[]]
        self._optimizer = Optimizer(optimization_level)

    def with_polynomial(self, subset: ColumnSelection, degrees: Iterable[int]) -> Pipeline:
        input_columns = self._get_combinations_from_selections(subset)

        transformers = self._build_transformers(
            transformer_factory=PolynomialTransformer,
            input_columns=input_columns,
            kw_params={'degree': degrees},
        )

        return self._with_added_to_current_layer(transformers)

    def with_arithmetic(self, left_subset: ColumnSelection, right_subset: ColumnSelection, operations: Iterable[ArithmeticOperation]) -> Pipeline:
        input_columns = self._get_combinations_from_selections(left_subset, right_subset)
        transformer_types = [op.value for op in order_preserving_unique(operations)]

        transformers = self._build_transformers(
            transformer_factory=transformer_types,
            input_columns=input_columns,
        )

        return self._with_added_to_current_layer(transformers)

    def with_comparison(self, left_subset: ColumnSelection, right_subset: ColumnSelection, comparisons: Iterable[Comparisons]) -> Pipeline:
        input_columns = self._get_combinations_from_selections(left_subset, right_subset)
        transformer_types = [comp.value for comp in order_preserving_unique(comparisons)]

        transformers = self._build_transformers(
            transformer_factory=transformer_types,
            input_columns=input_columns,
        )

        return self._with_added_to_current_layer(transformers)

    def with_lagged(self, subset: ColumnSelection, lags: Iterable[int], over_columns_combinations: Iterable[Iterable[str | ColumnSpecification]] = (), fill_value: Any = None) -> Pipeline:
        input_columns = self._get_combinations_from_selections(subset)
        over_combinations = list(over_columns_combinations)

        all_transformers: list[Transformer] = []

        lagged_transformers = self._build_transformers(
            transformer_factory=LaggedTransformer,
            input_columns=input_columns,
            kw_params={'lag': lags},
            fill_value=fill_value,
        )

        non_empty_over_columns_combinations = [combination for combination in over_combinations if combination]
        if not over_combinations or len(non_empty_over_columns_combinations) != len(over_combinations):
            all_transformers.extend(lagged_transformers)

        if non_empty_over_columns_combinations:
            lagged_over_transformers = self._build_transformers(
                transformer_factory=OverWrapper,
                input_columns=None,
                kw_params={'inner_transformer': lagged_transformers, 'over_columns': non_empty_over_columns_combinations},
            )
            all_transformers.extend(lagged_over_transformers)

        return self._with_added_to_current_layer(all_transformers)

    def with_new_layer(self) -> Pipeline:
        new_layer_schema = self._get_schema_from_transformers(self._current_layer())
        return Pipeline(
            schema=self._schema + new_layer_schema,
            transformers=self._transformers + [[]],
            optimization_level=self._optimizer.optimization_level,
        )

    def collect(self, df: pl.LazyFrame) -> pl.DataFrame:
        for layer in self._transformers:
            exprs = [transformer.transform() for transformer in layer]
            df = df.with_columns(*exprs)
        return df.collect()

    def _with_added_to_current_layer(self, transformers: Transformer | Iterable[Transformer]) -> Pipeline:
        current_layer_additions = [transformers] if isinstance(transformers, Transformer) else list(transformers)
        current_layer_additions = self._optimizer.deduplicate_transformers_against_layers(self._schema, current_layer_additions)
        return Pipeline(
            schema=self._schema,
            transformers=self._transformers[:-1] + [self._current_layer() + current_layer_additions],
            optimization_level=self._optimizer.optimization_level,
        )

    def _current_layer(self) -> list[Transformer]:
        return self._transformers[-1]

    def _get_combinations_from_selections(self, *subsets: ColumnSelection) -> list[ColumnSet]:
        return [self._get_columns_from_selection(subset) for subset in subsets]

    def _get_columns_from_selection(self, subset: ColumnSelection) -> ColumnSet:
        match subset:
            case ColumnType():
                return self._get_columns_of_type(subset)
            case str():
                return [self._get_column_by_name(subset)]
            case Sequence():
                return order_preserving_unique(flatten([self._get_columns_from_selection(col) for col in subset]))
            case _:
                raise ValueError(f'Unexpected subset type: {type(subset)}')

    def _get_columns_of_type(self, column_type: ColumnType) -> ColumnSet:
        return [col_spec for col_spec in self._schema if col_spec.column_type == column_type]

    def _get_column_by_name(self, column_name: str) -> ColumnSpecification:
        for col_spec in self._schema:
            if col_spec.name == column_name:
                return col_spec
        raise KeyError(f'Column "{column_name}" not found in schema.')

    @staticmethod
    def _get_schema_from_transformers(transformers: Iterable[Transformer]) -> Schema:
        return [transformer.output_column_specification for transformer in transformers]

    def _build_transformers[T: Transformer](
        self,
        *,
        transformer_factory: type[T] | list[type[T]],
        input_columns: Optional[list[ColumnSet]] = None,
        kw_params: Optional[dict[str, Iterable[Any]]] = None,
        **kwargs,
    ) -> list[T]:

        transformers: list[T] = []

        factories = transformer_factory if isinstance(transformer_factory, list) else [transformer_factory]
        input_columns = input_columns or []
        kw_params = kw_params or {}

        input_columns_positional_combinations: list[tuple[ColumnSpecification, ...]] = list(product(*input_columns))
        kw_keys = list(kw_params.keys())
        kw_params_positional_combinations = list(product(*kw_params.values()))

        for transformer_factory in factories:
            optimized_combinations = self._optimizer.optimize_input_columns(transformer_factory, input_columns_positional_combinations)
            for column_combination in optimized_combinations:
                for kw_params_combination in kw_params_positional_combinations:
                    transformer_kwargs = dict(zip(kw_keys, kw_params_combination)) | kwargs
                    transformers.append(transformer_factory(*column_combination, **transformer_kwargs))

        return transformers
