from collections.abc import Iterable

import polars as pl
from polars.selectors import Selector

from core.base.column_types import ColumnType
from core.transformers.base import Transformer


class PolynomialTransformer(Transformer):
    def __init__(self, columns: Selector, degrees: Iterable[int]) -> None:
        self._columns = columns
        self._degrees = degrees

    def return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def transform(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = df.with_columns([self._columns.pow(degree).name.suffix(f'_pow_{degree}') for degree in self._degrees])
        return df
