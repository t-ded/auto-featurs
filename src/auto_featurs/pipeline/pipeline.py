from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from itertools import product
from typing import Any
from typing import Optional

import polars as pl

from auto_featurs.base.column_specification import ColumnSelection
from auto_featurs.base.column_specification import ColumnSet
from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import Schema
from auto_featurs.dataset.dataset import Dataset
from auto_featurs.pipeline.optimizer import OptimizationLevel
from auto_featurs.pipeline.optimizer import Optimizer
from auto_featurs.pipeline.validator import Validator
from auto_featurs.transformers.aggregating_transformers import AggregatingTransformer
from auto_featurs.transformers.aggregating_transformers import ArithmeticAggregations
from auto_featurs.transformers.aggregating_transformers import CountTransformer
from auto_featurs.transformers.aggregating_transformers import FirstValueTransformer
from auto_featurs.transformers.aggregating_transformers import LaggedTransformer
from auto_featurs.transformers.aggregating_transformers import NumUniqueTransformer
from auto_featurs.transformers.base import Transformer
from auto_featurs.transformers.comparison_transformers import Comparisons
from auto_featurs.transformers.numeric_transformers import ArithmeticOperation
from auto_featurs.transformers.numeric_transformers import PolynomialTransformer
from auto_featurs.transformers.over_wrapper import OverWrapper
from auto_featurs.transformers.rolling_wrapper import RollingWrapper
from auto_featurs.utils.utils import get_valid_param_options
from auto_featurs.utils.utils import order_preserving_unique

type TransformerLayers = list[list[Transformer]]


class Pipeline:
    def __init__(
        self,
        dataset: Dataset,
        transformers: Optional[TransformerLayers] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.NONE,
    ) -> None:
        self._dataset = dataset
        self._transformers: TransformerLayers = transformers or [[]]
        self._optimizer = Optimizer(optimization_level)
        self._validator = Validator()

    def with_polynomial(self, subset: ColumnSelection, degrees: Sequence[int]) -> Pipeline:
        input_columns = self._dataset.get_combinations_from_selections(subset)

        transformers = self._build_transformers(
            transformer_factory=PolynomialTransformer,
            input_columns=input_columns,
            kw_params={'degree': degrees},
        )

        return self._with_added_to_current_layer(transformers)

    def with_arithmetic(self, left_subset: ColumnSelection, right_subset: ColumnSelection, operations: Sequence[ArithmeticOperation]) -> Pipeline:
        input_columns = self._dataset.get_combinations_from_selections(left_subset, right_subset)
        transformer_types = [op.value for op in order_preserving_unique(operations)]

        transformers = self._build_transformers(
            transformer_factory=transformer_types,
            input_columns=input_columns,
        )

        return self._with_added_to_current_layer(transformers)

    def with_comparison(self, left_subset: ColumnSelection, right_subset: ColumnSelection, comparisons: Sequence[Comparisons]) -> Pipeline:
        input_columns = self._dataset.get_combinations_from_selections(left_subset, right_subset)
        transformer_types = [comp.value for comp in order_preserving_unique(comparisons)]

        transformers = self._build_transformers(
            transformer_factory=transformer_types,
            input_columns=input_columns,
        )

        return self._with_added_to_current_layer(transformers)

    def with_count(
            self,
            over_columns_combinations: Sequence[Sequence[str | ColumnSpecification]] = (),
            time_windows: Sequence[Optional[str | timedelta]] = (),
            index_column_name: Optional[str] = None,
            cumulative: bool = False,
    ) -> Pipeline:
        aggregating_transformers = self._build_aggregated_transformers(
            subset=None,
            transformer_factory=CountTransformer,
            over_columns_combinations=over_columns_combinations,
            time_windows=time_windows,
            index_column_name=index_column_name,
            cumulative=cumulative,
        )
        return self._with_added_to_current_layer(aggregating_transformers)

    def with_lagged(self, subset: ColumnSelection, lags: Sequence[int], over_columns_combinations: Sequence[Sequence[str | ColumnSpecification]] = (), fill_value: Any = None) -> Pipeline:
        lagged_transformers = self._build_aggregated_transformers(
            subset=subset,
            transformer_factory=LaggedTransformer,
            over_columns_combinations=over_columns_combinations,
            kw_params={'lag': lags},
            fill_value=fill_value,
        )
        return self._with_added_to_current_layer(lagged_transformers)

    def with_first_value(
            self,
            subset: ColumnSelection,
            over_columns_combinations: Sequence[Sequence[str | ColumnSpecification]] = (),
            time_windows: Sequence[Optional[str | timedelta]] = (),
            index_column_name: Optional[str] = None,
    ) -> Pipeline:
        first_value_transformers = self._build_aggregated_transformers(
            subset=subset,
            transformer_factory=FirstValueTransformer,
            over_columns_combinations=over_columns_combinations,
            time_windows=time_windows,
            index_column_name=index_column_name,
        )
        return self._with_added_to_current_layer(first_value_transformers)

    def with_num_unique(
            self,
            subset: ColumnSelection,
            over_columns_combinations: Sequence[Sequence[str | ColumnSpecification]] = (),
            time_windows: Sequence[Optional[str | timedelta]] = (),
            index_column_name: Optional[str] = None,
    ) -> Pipeline:
        num_unique_transformers = self._build_aggregated_transformers(
            subset=subset,
            transformer_factory=NumUniqueTransformer,
            over_columns_combinations=over_columns_combinations,
            time_windows=time_windows,
            index_column_name=index_column_name,
        )
        return self._with_added_to_current_layer(num_unique_transformers)

    def with_arithmetic_aggregation(
            self,
            subset: ColumnSelection,
            aggregations: Sequence[ArithmeticAggregations],
            over_columns_combinations: Sequence[Sequence[str | ColumnSpecification]] = (),
            time_windows: Sequence[Optional[str | timedelta]] = (),
            index_column_name: Optional[str] = None,
            cumulative: bool = False,
    ) -> Pipeline:
        transformer_types = [op.value for op in order_preserving_unique(aggregations)]
        arithmetic_aggregation_transformers = self._build_aggregated_transformers(
            subset=subset,
            transformer_factory=transformer_types,
            over_columns_combinations=over_columns_combinations,
            time_windows=time_windows,
            index_column_name=index_column_name,
            cumulative=cumulative,
        )
        return self._with_added_to_current_layer(arithmetic_aggregation_transformers)

    def with_new_layer(self) -> Pipeline:
        new_layer_schema = self._get_schema_from_transformers(self._current_layer())
        return Pipeline(
            dataset=self._dataset.with_schema(new_schema=new_layer_schema),
            transformers=self._transformers + [[]],
            optimization_level=self._optimizer.optimization_level,
        )

    def collect_plan(self, cache_computation: bool = False) -> Dataset:
        dataset = self._dataset
        for layer in self._transformers:
            exprs = [transformer.transform() for transformer in layer]
            dataset = dataset.with_columns(new_columns=exprs)

        if cache_computation:
            return dataset.with_cached_computation()
        return dataset

    def collect(self) -> pl.DataFrame:
        updated_dataset = self.collect_plan()
        return updated_dataset.collect()

    def _with_added_to_current_layer(self, transformers: Transformer | Sequence[Transformer]) -> Pipeline:
        current_layer_additions = [transformers] if isinstance(transformers, Transformer) else list(transformers)
        current_layer_additions = self._optimizer.deduplicate_transformers_against_layers(self._dataset.schema, current_layer_additions)
        return Pipeline(
            dataset=self._dataset,
            transformers=self._transformers[:-1] + [self._current_layer() + current_layer_additions],
            optimization_level=self._optimizer.optimization_level,
        )

    def _current_layer(self) -> list[Transformer]:
        return self._transformers[-1]

    @staticmethod
    def _get_schema_from_transformers(transformers: Sequence[Transformer]) -> Schema:
        return [transformer.output_column_specification for transformer in transformers]

    def _build_aggregated_transformers[AT: AggregatingTransformer](
            self,
            *,
            subset: Optional[ColumnSelection],
            transformer_factory: type[AT] | list[type[AT]],
            over_columns_combinations: Sequence[Sequence[str | ColumnSpecification]] = (),
            time_windows: Sequence[Optional[str | timedelta]] = (),
            index_column_name: Optional[str] = None,
            **kwargs: Any,
    ) -> list[AT | OverWrapper[AT] | RollingWrapper[AT | OverWrapper[AT]]]:
        index_column = self._dataset.get_column_by_name(index_column_name) if index_column_name else None
        self._validator.validate_time_window_index_column(time_windows, index_column)
        input_columns = self._dataset.get_combinations_from_selections(subset) if subset is not None else None

        aggregating_transformers = self._build_transformers(
            transformer_factory=transformer_factory,
            input_columns=input_columns,
            **kwargs,
        )

        aggregations_over = self._get_over_transformers(aggregating_transformers=aggregating_transformers, over_columns_combinations=over_columns_combinations)
        rolling_aggregations_over = self._get_rolling_transformers(aggregating_transformers=aggregations_over, index_column=index_column, time_windows=time_windows)
        return rolling_aggregations_over

    def _get_over_transformers[AT: AggregatingTransformer](
            self,
            aggregating_transformers: Sequence[AT],
            over_columns_combinations: Sequence[Sequence[str | ColumnSpecification]],
    ) -> list[AT | OverWrapper[AT]]:
        if not over_columns_combinations:
            return list(aggregating_transformers)

        all_transformers: list[AT | OverWrapper[AT]] = []

        valid_over_columns_combinations, all_are_valid = get_valid_param_options(over_columns_combinations)
        if not all_are_valid:
            all_transformers.extend(aggregating_transformers)

        if valid_over_columns_combinations:
            aggregated_over_transformers = self._build_transformers(
                transformer_factory=OverWrapper,
                input_columns=None,
                kw_params={'inner_transformer': aggregating_transformers, 'over_columns': valid_over_columns_combinations},
            )
            all_transformers.extend(aggregated_over_transformers)

        return all_transformers

    def _get_rolling_transformers[AT: AggregatingTransformer](
            self,
            aggregating_transformers: Sequence[AT],
            index_column: Optional[ColumnSpecification],
            time_windows: Sequence[Optional[str | timedelta]],
    ) -> list[AT | RollingWrapper[AT]]:
        if index_column is None or not time_windows:
            return list(aggregating_transformers)

        all_transformers: list[AT | RollingWrapper[AT]] = []

        valid_time_windows, all_are_valid = get_valid_param_options(time_windows)
        if not all_are_valid:
            all_transformers.extend(aggregating_transformers)

        if valid_time_windows:
            aggregated_rolling_transformers = self._build_transformers(
                transformer_factory=RollingWrapper,
                input_columns=None,
                kw_params={'inner_transformer': aggregating_transformers, 'time_window': valid_time_windows},
                index_column=index_column,
            )
            all_transformers.extend(aggregated_rolling_transformers)

        return all_transformers

    def _build_transformers[T: Transformer](
        self,
        *,
        transformer_factory: type[T] | list[type[T]],
        input_columns: Optional[Sequence[ColumnSet]] = None,
        kw_params: Optional[dict[str, Sequence[Any]]] = None,
        **kwargs: Any,
    ) -> list[T]:

        transformers: list[T] = []

        factories = transformer_factory if isinstance(transformer_factory, list) else [transformer_factory]
        input_columns = input_columns or []
        kw_params = kw_params or {}

        input_columns_positional_combinations: list[tuple[ColumnSpecification, ...]] = list(product(*input_columns))
        kw_keys = list(kw_params.keys())
        kw_params_positional_combinations = list(product(*kw_params.values()))

        for factory in factories:
            optimized_combinations = self._optimizer.optimize_input_columns(factory, input_columns_positional_combinations)
            for column_combination in optimized_combinations:
                for kw_params_combination in kw_params_positional_combinations:
                    transformer_kwargs = dict(zip(kw_keys, kw_params_combination, strict=True)) | kwargs
                    transformer = factory(*column_combination, **transformer_kwargs)
                    self._validator.validate_transformer_against_input_columns(transformer, column_combination)
                    transformers.append(transformer)

        return transformers
