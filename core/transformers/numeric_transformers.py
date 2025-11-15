import polars as pl

from core.base.column_types import ColumnType
from core.transformers.base import Transformer


class PolynomialTransformer(Transformer):
    def __init__(self, column: str, degree: int) -> None:
        self._column = column
        self._degree = degree

    def input_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _transform(self) -> pl.Expr:
        return pl.col(self._column).pow(self._degree)

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.name.suffix(f'_pow_{self._degree}')
