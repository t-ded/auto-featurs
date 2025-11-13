from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

import polars as pl
import polars.selectors as cs

from core.base.column_types import ColumnType
from core.transformers.base import Transformer
from core.transformers.numeric_transformers import PolynomialTransformer


class Pipeline:
    def __init__(
        self,
        column_types: Optional[dict[str, ColumnType]] = None,
        transformers: Optional[list[Transformer]] = None,
    ) -> None:
        self._column_types = column_types or {}
        self._transformers = transformers or []

    def with_polynomial(self, subset: str | Sequence[str] | cs.Selector, degrees: Iterable[int]) -> Pipeline:
        selection = subset if isinstance(subset, cs.Selector) else cs.by_name(subset)
        self._transformers.append(PolynomialTransformer(columns=selection, degrees=degrees))
        return self

    def collect(self, df: pl.LazyFrame) -> pl.DataFrame:
        for transformer in self._transformers:
            df = df.pipe(transformer.transform)
        return df.collect()
