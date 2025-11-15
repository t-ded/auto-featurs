from abc import ABC, abstractmethod
import polars as pl

from core.base.column_types import ColumnType


class Transformer(ABC):
    @abstractmethod
    def return_type(self) -> ColumnType:
        raise NotImplementedError

    @abstractmethod
    def transform(self) -> pl.Expr:
        raise NotImplementedError
