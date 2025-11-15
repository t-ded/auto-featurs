from abc import ABC, abstractmethod
import polars as pl

from core.base.column_types import ColumnType


class Transformer(ABC):
    @abstractmethod
    def input_type(self) -> ColumnType:
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

    def new_column_type(self) -> tuple[str, ColumnType]:
        return self.transform().meta.output_name(), self._return_type()
