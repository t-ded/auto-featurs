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
        transformers: Optional[list[list[Transformer]]] = None,
    ) -> None:
        self._column_types: dict[str, ColumnType] = column_types or {}
        self._transformers: list[list[Transformer]] = transformers or [[]]

    def with_polynomial(self, subset: str | Sequence[str] | cs.Selector, degrees: Iterable[int]) -> Pipeline:
        selection = subset if isinstance(subset, cs.Selector) else cs.by_name(subset)
        transformers = [PolynomialTransformer(columns=selection, degree=degree) for degree in degrees]
        return self._with_added_to_current_layer(transformers)

    def with_new_layer(self) -> Pipeline:
        return Pipeline(column_types=self._column_types, transformers=self._transformers + [[]])

    def collect(self, df: pl.LazyFrame) -> pl.DataFrame:
        for layer in self._transformers:
            exprs = [transformer.transform() for transformer in layer]
            df = df.with_columns(*exprs)
        return df.collect()

    def _with_added_to_current_layer(self, transformers: Transformer | Iterable[Transformer]) -> Pipeline:
        current_layer_additions = [transformers] if isinstance(transformers, Transformer) else list(transformers)
        return Pipeline(column_types=self._column_types, transformers=self._transformers[:-1] + [self._current_layer() + current_layer_additions])

    def _current_layer(self) -> list[Transformer]:
        return self._transformers[-1]
