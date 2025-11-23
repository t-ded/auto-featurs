from abc import ABC, abstractmethod
from functools import cached_property

import polars as pl

from core.base.column_specification import ColumnSpecification
from core.base.column_specification import ColumnType


class Transformer(ABC):
    @abstractmethod
    def input_type(self) -> set[ColumnType] | tuple[set[ColumnType], ...]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def is_commutative(cls) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _return_type(self) -> ColumnType:
        raise NotImplementedError

    @abstractmethod
    def _transform(self) -> pl.Expr:
        raise NotImplementedError

    @abstractmethod
    def _name(self, transform: pl.Expr) -> pl.Expr:
        raise NotImplementedError

    def transform(self) -> pl.Expr:
        return self._name(self._transform())

    @cached_property
    def output_column_specification(self) -> ColumnSpecification:
        return ColumnSpecification(
            name=self.transform().meta.output_name(),
            column_type=self._return_type(),
        )
