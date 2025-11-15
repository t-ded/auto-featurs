from collections.abc import Iterable

import polars as pl
from polars.selectors import Selector

from core.base.column_types import ColumnType
from core.transformers.base import Transformer


class PolynomialTransformer(Transformer):
    def __init__(self, columns: Selector, degree: int) -> None:
        self._columns = columns
        self._degree = degree

    def return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def transform(self) -> pl.Expr:
        return self._columns.pow(self._degree).name.suffix(f'_pow_{self._degree}')
