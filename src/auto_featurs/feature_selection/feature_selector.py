import polars as pl
from polars._typing import IntoExpr


class Selector:
    def point_correlation(self, feature_columns: IntoExpr, label_column: IntoExpr) -> pl.Expr:
        pass
