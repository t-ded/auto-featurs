from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from itertools import product
from typing import Any
from typing import Optional

import polars as pl
from more_itertools import flatten

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.pipeline.optimizer import OptimizationLevel
from auto_featurs.pipeline.optimizer import Optimizer
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

    def with_polynomial(self, subset: ColumnSelection, degrees: Sequence[int]) -> Pipeline:
        input_columns = self._get_combinations_from_selections(subset)

        transformers = self._build_transformers(
            transformer_factory=PolynomialTransformer,
            input_columns=input_columns,
            kw_params={'degree': degrees},
        )

        return self._with_added_to_current_layer(transformers)

    def with_arithmetic(self, left_subset: ColumnSelection, right_subset: ColumnSelection, operations: Sequence[ArithmeticOperation]) -> Pipeline:
        input_columns = self._get_combinations_from_selections(left_subset, right_subset)
        transformer_types = [op.value for op in order_preserving_unique(operations)]

        transformers = self._build_transformers(
            transformer_factory=transformer_types,
            input_columns=input_columns,
        )

        return self._with_added_to_current_layer(transformers)

    def with_comparison(self, left_subset: ColumnSelection, right_subset: ColumnSelection, comparisons: Sequence[Comparisons]) -> Pipeline:
        input_columns = self._get_combinations_from_selections(left_subset, right_subset)
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
        index_column = self._get_column_by_name(index_column_name) if index_column_name else None
        self._validate_time_window_index_column(time_windows, index_column)

        count_transformers = self._build_transformers(transformer_factory=CountTransformer, cumulative=cumulative)

        count_over = self._get_over_transformers(aggregating_transformers=count_transformers, over_columns_combinations=over_columns_combinations)
        rolling_count_over = self._get_rolling_transformers(aggregating_transformers=count_over, index_column=index_column, time_windows=time_windows)
        return self._with_added_to_current_layer(rolling_count_over)

    def with_lagged(self, subset: ColumnSelection, lags: Sequence[int], over_columns_combinations: Sequence[Sequence[str | ColumnSpecification]] = (), fill_value: Any = None) -> Pipeline:
        input_columns = self._get_combinations_from_selections(subset)

        lagged_transformers = self._build_transformers(
            transformer_factory=LaggedTransformer,
            input_columns=input_columns,
            kw_params={'lag': lags},
            fill_value=fill_value,
        )

        lagged_over = self._get_over_transformers(aggregating_transformers=lagged_transformers, over_columns_combinations=over_columns_combinations)
        return self._with_added_to_current_layer(lagged_over)

    def with_first_value(
            self,
            subset: ColumnSelection,
            over_columns_combinations: Sequence[Sequence[str | ColumnSpecification]] = (),
            time_windows: Sequence[Optional[str | timedelta]] = (),
            index_column_name: Optional[str] = None,
    ) -> Pipeline:
        index_column = self._get_column_by_name(index_column_name) if index_column_name else None
        self._validate_time_window_index_column(time_windows, index_column)
        input_columns = self._get_combinations_from_selections(subset)

        first_value_transformers = self._build_transformers(
            transformer_factory=FirstValueTransformer,
            input_columns=input_columns,
        )

        first_value_over = self._get_over_transformers(aggregating_transformers=first_value_transformers, over_columns_combinations=over_columns_combinations)
        rolling_first_value_over = self._get_rolling_transformers(aggregating_transformers=first_value_over, index_column=index_column, time_windows=time_windows)
        return self._with_added_to_current_layer(rolling_first_value_over)

    def with_num_unique(
            self,
            subset: ColumnSelection,
            over_columns_combinations: Sequence[Sequence[str | ColumnSpecification]] = (),
            time_windows: Sequence[Optional[str | timedelta]] = (),
            index_column_name: Optional[str] = None,
    ) -> Pipeline:
        index_column = self._get_column_by_name(index_column_name) if index_column_name else None
        self._validate_time_window_index_column(time_windows, index_column)
        input_columns = self._get_combinations_from_selections(subset)

        num_unique_transformers = self._build_transformers(
            transformer_factory=NumUniqueTransformer,
            input_columns=input_columns,
        )

        num_unique_over = self._get_over_transformers(aggregating_transformers=num_unique_transformers, over_columns_combinations=over_columns_combinations)
        rolling_num_unique_over = self._get_rolling_transformers(aggregating_transformers=num_unique_over, index_column=index_column, time_windows=time_windows)
        return self._with_added_to_current_layer(rolling_num_unique_over)

    def with_arithmetic_aggregation(
            self,
            subset: ColumnSelection,
            aggregations: Sequence[ArithmeticAggregations],
            over_columns_combinations: Sequence[Sequence[str | ColumnSpecification]] = (),
            time_windows: Sequence[Optional[str | timedelta]] = (),
            index_column_name: Optional[str] = None,
            cumulative: bool = False,
    ) -> Pipeline:
        index_column = self._get_column_by_name(index_column_name) if index_column_name else None
        self._validate_time_window_index_column(time_windows, index_column)
        input_columns = self._get_combinations_from_selections(subset)
        transformer_types = [op.value for op in order_preserving_unique(aggregations)]

        aggregating_transformers = self._build_transformers(
            transformer_factory=transformer_types,
            input_columns=input_columns,
            cumulative=cumulative,
        )

        arithmetic_aggregations_over = self._get_over_transformers(aggregating_transformers=aggregating_transformers, over_columns_combinations=over_columns_combinations)
        rolling_aggregations_over = self._get_rolling_transformers(aggregating_transformers=arithmetic_aggregations_over, index_column=index_column, time_windows=time_windows)
        return self._with_added_to_current_layer(rolling_aggregations_over)

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

    def _with_added_to_current_layer(self, transformers: Transformer | Sequence[Transformer]) -> Pipeline:
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
    def _get_schema_from_transformers(transformers: Sequence[Transformer]) -> Schema:
        return [transformer.output_column_specification for transformer in transformers]

    @staticmethod
    def _validate_time_window_index_column(time_windows: Sequence[Optional[str | timedelta]], index_column: Optional[ColumnSpecification]) -> None:
        if time_windows and time_windows[0] is not None and index_column is None:
            raise ValueError('Time window specified without index column.')
        if index_column is not None and index_column.column_type != ColumnType.DATETIME:
            raise ValueError(f'Currently only {ColumnType.DATETIME} columns are supported for rolling aggregation but {index_column.column_type} was passed for {index_column.name}.')

    def _get_over_transformers(
            self,
            aggregating_transformers: Sequence[AggregatingTransformer],
            over_columns_combinations: Sequence[Sequence[str | ColumnSpecification]],
    ) -> list[AggregatingTransformer | OverWrapper]:
        if not over_columns_combinations:
            return list(aggregating_transformers)

        all_transformers: list[AggregatingTransformer | OverWrapper] = []

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

    def _get_rolling_transformers(
            self,
            aggregating_transformers: Sequence[AggregatingTransformer],
            index_column: Optional[ColumnSpecification],
            time_windows: Sequence[Optional[str | timedelta]],
    ) -> list[AggregatingTransformer | RollingWrapper]:
        if index_column is None or not time_windows:
            return list(aggregating_transformers)

        all_transformers: list[AggregatingTransformer | RollingWrapper] = []

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

        for transformer_factory in factories:
            optimized_combinations = self._optimizer.optimize_input_columns(transformer_factory, input_columns_positional_combinations)
            for column_combination in optimized_combinations:
                for kw_params_combination in kw_params_positional_combinations:
                    transformer_kwargs = dict(zip(kw_keys, kw_params_combination, strict=True)) | kwargs
                    transformers.append(transformer_factory(*column_combination, **transformer_kwargs))

        return transformers
