from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import Any
from typing import Optional

import polars as pl
from polars._typing import IntoExpr

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.column_specification import ColumnTypeSelector
from auto_featurs.transformers.base import Transformer
from auto_featurs.utils.utils import default_true_filtering_condition
from auto_featurs.utils.utils import filtering_condition_to_string


class CumulativeOptions(Enum):
    NONE = 'none'
    EXCLUSIVE = 'exclusive'
    INCLUSIVE = 'inclusive'

    def __str__(self) -> str:
        return f'{self.value}_cum_' if self != CumulativeOptions.NONE else ''


class AggregatingTransformer(Transformer, ABC):
    pass


class CountTransformer(AggregatingTransformer):
    def __init__(self, cumulative: CumulativeOptions = CumulativeOptions.NONE, filtering_condition: Optional[pl.Expr] = None) -> None:
        self._cumulative = cumulative
        self._filtering_condition = filtering_condition

    def input_type(self) -> ColumnTypeSelector:
        return ColumnTypeSelector()

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _transform(self) -> pl.Expr:
        if self._filtering_condition is not None:
            match self._cumulative:
                case CumulativeOptions.NONE:
                    return self._filtering_condition.sum()
                case CumulativeOptions.EXCLUSIVE:
                    return self._filtering_condition.cum_sum().shift(1, fill_value=0)
                case CumulativeOptions.INCLUSIVE:
                    return self._filtering_condition.cum_sum()
        else:
            match self._cumulative:
                case CumulativeOptions.NONE:
                    return pl.len()
                case CumulativeOptions.EXCLUSIVE:
                    return pl.int_range(0, pl.len())
                case CumulativeOptions.INCLUSIVE:
                    return pl.int_range(1, pl.len() + 1)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        condition_name = filtering_condition_to_string(self._filtering_condition)
        return transform.alias(str(self._cumulative) + 'count' + condition_name)


class LaggedTransformer(AggregatingTransformer):
    def __init__(self, column: ColumnSpecification, lag: int, fill_value: Optional[IntoExpr] = None) -> None:
        self._column = column
        self._lag = lag
        self._fill_value = fill_value

    def input_type(self) -> ColumnTypeSelector:
        return ColumnTypeSelector.any()

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return self._column.column_type

    def _transform(self) -> pl.Expr:
        return pl.col(self._column.name).shift(self._lag, fill_value=self._fill_value)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column.name}_lagged_{self._lag}')


class FirstValueTransformer(AggregatingTransformer):
    def __init__(self, column: ColumnSpecification, filtering_condition: Optional[pl.Expr] = None) -> None:
        self._column = column
        self._filtering_condition = default_true_filtering_condition(filtering_condition)

    def input_type(self) -> ColumnTypeSelector:
        return ColumnTypeSelector.any()

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return self._column.column_type

    def _transform(self) -> pl.Expr:
        return pl.col(self._column.name).filter(self._filtering_condition).first()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column.name}_first_value' + filtering_condition_to_string(self._filtering_condition))


class ModeTransformer(AggregatingTransformer):
    def __init__(self, column: ColumnSpecification, cumulative: CumulativeOptions = CumulativeOptions.NONE, filtering_condition: Optional[pl.Expr] = None) -> None:
        self._column = column
        self._cumulative = cumulative
        self._filtering_condition = default_true_filtering_condition(filtering_condition)

    def input_type(self) -> ColumnTypeSelector:
        return ColumnTypeSelector.any()

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return self._column.column_type

    def _transform(self) -> pl.Expr:
        col = pl.col(self._column.name)
        if self._cumulative == CumulativeOptions.NONE:
            return col.filter(self._filtering_condition).mode().sort(descending=True).first()
        else:
            cum_value_counts = pl.when(self._filtering_condition).then(pl.int_range(1, pl.len() + 1)).forward_fill().fill_null(0).over(col)
            cum_mode_count = cum_value_counts.cum_max()

            cum_mode = pl.when(cum_value_counts == cum_mode_count).then(col).forward_fill()
            if self._cumulative == CumulativeOptions.EXCLUSIVE:
                cum_mode = cum_mode.shift(1, fill_value=None)

            return cum_mode

    def _name(self, transform: pl.Expr) -> pl.Expr:
        condition_name = filtering_condition_to_string(self._filtering_condition)
        return transform.alias(f'{self._column.name}_{str(self._cumulative)}mode' + condition_name)


class NumUniqueTransformer(AggregatingTransformer):
    def __init__(self, column: ColumnSpecification, cumulative: CumulativeOptions = CumulativeOptions.NONE, filtering_condition: Optional[pl.Expr] = None) -> None:
        self._column = column
        self._cumulative = cumulative
        self._filtering_condition = default_true_filtering_condition(filtering_condition)

    def input_type(self) -> ColumnTypeSelector:
        return ColumnTypeSelector.any()

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _transform(self) -> pl.Expr:
        col = pl.col(self._column.name)
        if self._cumulative == CumulativeOptions.NONE:
            return col.filter(self._filtering_condition).n_unique()
        else:
            cum_n_unique = (col.is_first_distinct() & self._filtering_condition).cum_sum()
            if self._cumulative == CumulativeOptions.EXCLUSIVE:
                cum_n_unique = cum_n_unique.is_first_distinct().cum_sum().shift(1, fill_value=0)

        return cum_n_unique

    def _name(self, transform: pl.Expr) -> pl.Expr:
        condition_name = filtering_condition_to_string(self._filtering_condition)
        return transform.alias(f'{self._column.name}_{str(self._cumulative)}num_unique' + condition_name)


class ArithmeticAggregationTransformer(AggregatingTransformer, ABC):
    def __init__(self, column: str | ColumnSpecification, cumulative: CumulativeOptions = CumulativeOptions.NONE, filtering_condition: Optional[pl.Expr] = None, **kwargs: Any) -> None:
        self._column = column if isinstance(column, str) else column.name
        self._cumulative = cumulative
        self._filtering_condition = default_true_filtering_condition(filtering_condition)

    def input_type(self) -> ColumnTypeSelector:
        return ColumnType.NUMERIC | ColumnType.BOOLEAN

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column}_{self._cumulative}{self._aggregation}' + filtering_condition_to_string(self._filtering_condition))

    @property
    @abstractmethod
    def _aggregation(self) -> str:
        raise NotImplementedError


class SumTransformer(ArithmeticAggregationTransformer):
    def _transform(self) -> pl.Expr:
        col = pl.col(self._column).filter(self._filtering_condition)
        match self._cumulative:
            case CumulativeOptions.NONE:
                return col.sum()
            case CumulativeOptions.EXCLUSIVE:
                return col.cum_sum().shift(1, fill_value=0.0)
            case CumulativeOptions.INCLUSIVE:
                return col.cum_sum()

    @property
    def _aggregation(self) -> str:
        return 'sum'


class QuantileTransformer(ArithmeticAggregationTransformer):
    def __init__(self, column: str | ColumnSpecification, quantile: float, cumulative: CumulativeOptions = CumulativeOptions.NONE, filtering_condition: Optional[pl.Expr] = None) -> None:
        super().__init__(column, cumulative, filtering_condition)
        self._quantile = quantile

    def _transform(self) -> pl.Expr:
        col = pl.col(self._column).filter(self._filtering_condition).cast(pl.Float64)
        match self._cumulative:
            case CumulativeOptions.NONE:
                return col.quantile(self._quantile, interpolation='linear')
            case CumulativeOptions.EXCLUSIVE:
                return col.cumulative_eval(pl.element().quantile(self._quantile, interpolation='linear')).shift(1)
            case CumulativeOptions.INCLUSIVE:
                return col.cumulative_eval(pl.element().quantile(self._quantile, interpolation='linear'))

    @property
    def _aggregation(self) -> str:
        if self._quantile == 0.5:
            return 'median'
        return f'quantile_{int(self._quantile * 100)}'


class MedianTransformer(QuantileTransformer):
    def __init__(self, column: str | ColumnSpecification, cumulative: CumulativeOptions = CumulativeOptions.NONE, filtering_condition: Optional[pl.Expr] = None) -> None:
        super().__init__(column, 0.5, cumulative, filtering_condition)


class MeanTransformer(ArithmeticAggregationTransformer):
    def __init__(self, column: str | ColumnSpecification, cumulative: CumulativeOptions = CumulativeOptions.NONE, filtering_condition: Optional[pl.Expr] = None) -> None:
        super().__init__(column, cumulative, filtering_condition)
        self._sum_transformer = SumTransformer(column, cumulative, filtering_condition)
        self._count_transformer = CountTransformer(cumulative, filtering_condition)

    def _transform(self) -> pl.Expr:
        return self._sum_transformer.transform() / self._count_transformer.transform()

    @property
    def _aggregation(self) -> str:
        return 'mean'


class StdTransformer(ArithmeticAggregationTransformer):
    def __init__(self, column: str | ColumnSpecification, cumulative: CumulativeOptions = CumulativeOptions.NONE, filtering_condition: Optional[pl.Expr] = None) -> None:
        super().__init__(column, cumulative, filtering_condition)
        self._mean_transformer = MeanTransformer(column, cumulative, filtering_condition)

    def _transform(self) -> pl.Expr:
        col = pl.col(self._column).filter(self._filtering_condition)
        match self._cumulative:
            case CumulativeOptions.NONE:
                return col.std()
            case CumulativeOptions.EXCLUSIVE:
                mean_diff = col - self._mean_transformer.transform()
                cum_sum_squared_mean_diff = mean_diff.pow(2).fill_nan(0.0).cum_sum().shift(1, fill_value=0.0)
                return cum_sum_squared_mean_diff.sqrt()
            case CumulativeOptions.INCLUSIVE:
                mean_diff = col - self._mean_transformer.transform()
                cum_sum_squared_mean_diff = mean_diff.pow(2).fill_nan(0.0).cum_sum()
                return cum_sum_squared_mean_diff.sqrt()

    @property
    def _aggregation(self) -> str:
        return 'std'


class ZscoreTransformer(ArithmeticAggregationTransformer):
    def __init__(self, column: str | ColumnSpecification, cumulative: CumulativeOptions = CumulativeOptions.NONE, filtering_condition: Optional[pl.Expr] = None) -> None:
        super().__init__(column, cumulative, filtering_condition)
        self._mean_transformer = MeanTransformer(column, cumulative, filtering_condition)
        self._std_transformer = StdTransformer(column, cumulative, filtering_condition)

    def _transform(self) -> pl.Expr:
        return (pl.col(self._column) - self._mean_transformer.transform()) / self._std_transformer.transform()

    @property
    def _aggregation(self) -> str:
        return 'z_score'


class ArithmeticAggregations(Enum):
    SUM = SumTransformer
    QUANTILE = QuantileTransformer
    MEDIAN = MedianTransformer
    MEAN = MeanTransformer
    STD = StdTransformer
    ZSCORE = ZscoreTransformer
