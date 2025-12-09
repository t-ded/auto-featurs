from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import Optional

import polars as pl
from polars._typing import IntoExpr

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.transformers.base import Transformer
from auto_featurs.utils.utils import default_true_filtering_condition
from auto_featurs.utils.utils import filtering_condition_to_string


class AggregatingTransformer(Transformer, ABC):
    pass


class CountTransformer(AggregatingTransformer):
    def __init__(self, cumulative: bool = False, filtering_condition: Optional[pl.Expr] = None) -> None:
        self._cumulative = cumulative
        self._filtering_condition = filtering_condition

    def input_type(self) -> set[ColumnType]:
        return set()

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _transform(self) -> pl.Expr:
        if self._filtering_condition is not None:
            if self._cumulative:
                return self._filtering_condition.cum_sum()
            else:
                return self._filtering_condition.sum()
        else:
            if self._cumulative:
                return pl.int_range(1, pl.len() + 1)
            else:
                return pl.len()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        count_name = 'cum_count' if self._cumulative else 'count'
        condition_name = filtering_condition_to_string(self._filtering_condition) if self._filtering_condition is not None else ''
        return transform.alias(count_name + condition_name)


class LaggedTransformer(AggregatingTransformer):
    def __init__(self, column: ColumnSpecification, lag: int, fill_value: Optional[IntoExpr] = None) -> None:
        self._column = column
        self._lag = lag
        self._fill_value = fill_value

    def input_type(self) -> set[ColumnType]:
        return ColumnType.ANY()

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

    def input_type(self) -> set[ColumnType]:
        return ColumnType.ANY()

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
    def __init__(self, column: ColumnSpecification, filtering_condition: Optional[pl.Expr] = None) -> None:
        self._column = column
        self._filtering_condition = default_true_filtering_condition(filtering_condition)

    def input_type(self) -> set[ColumnType]:
        return ColumnType.ANY()

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return self._column.column_type

    def _transform(self) -> pl.Expr:
        return pl.col(self._column.name).filter(self._filtering_condition).mode().sort(descending=True).first()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column.name}_mode' + filtering_condition_to_string(self._filtering_condition))


class NumUniqueTransformer(AggregatingTransformer):
    def __init__(self, column: ColumnSpecification, filtering_condition: Optional[pl.Expr] = None) -> None:
        self._column = column
        self._filtering_condition = default_true_filtering_condition(filtering_condition)

    def input_type(self) -> set[ColumnType]:
        return ColumnType.ANY()

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _transform(self) -> pl.Expr:
        return pl.col(self._column.name).filter(self._filtering_condition).n_unique()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f'{self._column.name}_num_unique' + filtering_condition_to_string(self._filtering_condition))


class ArithmeticAggregationTransformer(AggregatingTransformer, ABC):
    def __init__(self, column: str | ColumnSpecification, cumulative: bool = False, filtering_condition: Optional[pl.Expr] = None) -> None:
        self._column = column if isinstance(column, str) else column.name
        self._cumulative = cumulative
        self._filtering_condition = default_true_filtering_condition(filtering_condition)

    def input_type(self) -> set[ColumnType]:
        return {ColumnType.NUMERIC}

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _name(self, transform: pl.Expr) -> pl.Expr:
        operation = f'cum_{self._aggregation}' if self._cumulative else self._aggregation
        return transform.alias(f'{self._column}_{operation}' + filtering_condition_to_string(self._filtering_condition))

    @property
    @abstractmethod
    def _aggregation(self) -> str:
        raise NotImplementedError


class SumTransformer(ArithmeticAggregationTransformer):
    def _transform(self) -> pl.Expr:
        col = pl.col(self._column).filter(self._filtering_condition)
        if self._cumulative:
            return col.cum_sum()
        return col.sum()

    @property
    def _aggregation(self) -> str:
        return 'sum'


class MeanTransformer(ArithmeticAggregationTransformer):
    def _transform(self) -> pl.Expr:
        col = pl.col(self._column).filter(self._filtering_condition)
        if self._cumulative:
            cum_sum = col.cum_sum()
            cum_count = col.cum_count()
            return cum_sum.truediv(cum_count)
        return col.mean()

    @property
    def _aggregation(self) -> str:
        return 'mean'


class StdTransformer(ArithmeticAggregationTransformer):
    def _transform(self) -> pl.Expr:
        col = pl.col(self._column).filter(self._filtering_condition)
        if self._cumulative:
            cum_sum = col.cum_sum()
            cum_count = col.cum_count()
            cum_mean = cum_sum.truediv(cum_count)

            mean_diff = col - cum_mean
            cum_sum_squared_mean_diff = mean_diff.pow(2).cum_sum()

            return cum_sum_squared_mean_diff.sqrt()
        return col.std()

    @property
    def _aggregation(self) -> str:
        return 'std'


class ZscoreTransformer(ArithmeticAggregationTransformer):
    def __init__(self, column: str | ColumnSpecification, cumulative: bool = False, filtering_condition: Optional[pl.Expr] = None) -> None:
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
    MEAN = MeanTransformer
    STD = StdTransformer
    ZSCORE = ZscoreTransformer
