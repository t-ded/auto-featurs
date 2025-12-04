from typing import Optional

import polars as pl
from polars import selectors as cs

from auto_featurs.base.column_specification import ColumnSelection
from auto_featurs.dataset.dataset import Dataset
from auto_featurs.utils.utils import get_names_from_column_specs


class Selector:
    def select_by_correlation(self, dataset: Dataset, feature_subset: ColumnSelection, top_k: Optional[int] = None, frac: Optional[float] = None) -> list[str]:
        label_col = dataset.get_label_column().name
        feature_cols = get_names_from_column_specs(dataset.get_columns_from_selection(feature_subset))
        num_to_select = self._get_num_to_select(top_k=top_k, frac=frac, num_cols=len(feature_cols))

        corr = dataset.data.select(self._point_correlation(feature_cols=feature_cols, label_col=label_col))

        to_select: list[str] = (
            corr
            .unpivot(variable_name='feature_col', value_name='abs_corr_with_target')
            .sort(['abs_corr_with_target', 'feature_col'], descending=[True, False])
            .head(num_to_select)
            .select('feature_col')
            .collect()
            .to_series()
            .to_list()
        )

        return to_select

    @staticmethod
    def _point_correlation(feature_cols: list[str], label_col: str) -> pl.Expr:
        return pl.corr(cs.by_name(feature_cols), label_col).fill_nan(0.0).abs()

    @staticmethod
    def _get_num_to_select(top_k: Optional[int], frac: Optional[float], num_cols: int) -> int:
        if (top_k is None) == (frac is None):
            raise ValueError('Exactly one of k or frac must be specified')

        if top_k is not None:
            if top_k < 1:
                raise ValueError(f'k must be at least 1 but {top_k} was given.')
            return top_k

        if frac is not None:
            if not (0 <= frac <= 1):
                raise ValueError(f'frac must be between 0 and 1 but {frac} was given.')
            return int(frac * num_cols)

        raise TypeError()
