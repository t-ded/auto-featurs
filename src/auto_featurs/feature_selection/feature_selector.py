from enum import Enum
from typing import Optional

import polars as pl
from polars import selectors as cs

from auto_featurs.base.column_specification import ColumnSelection
from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.dataset.dataset import Dataset
from auto_featurs.utils.utils import get_names_from_column_specs


class SelectionMethod(Enum):
    CORRELATION = 'Correlation'


SUPPORTED_COLUMN_TYPES = {
    SelectionMethod.CORRELATION: [ColumnType.NUMERIC, ColumnType.BOOLEAN, ColumnType.ORDINAL],
}

SUPPORTED_LABEL_COLUMN_TYPES = {
    SelectionMethod.CORRELATION: [ColumnType.NUMERIC, ColumnType.BOOLEAN],
}


class Selector:
    def select_by_correlation(self, dataset: Dataset, feature_subset: ColumnSelection, top_k: Optional[int] = None, frac: Optional[float] = None) -> list[str]:
        label_col = dataset.get_label_column()
        label_col_name = label_col.name

        feature_cols = dataset.get_columns_from_selection(feature_subset)
        self._check_valid_types(feature_cols, label_col, SelectionMethod.CORRELATION)
        feature_col_names = get_names_from_column_specs(feature_cols)
        num_to_select = self._get_num_to_select(top_k=top_k, frac=frac, num_cols=len(feature_col_names))

        corr = dataset.data.select(self._point_correlation(feature_col_names=feature_col_names, label_col_name=label_col_name))

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
    def _point_correlation(feature_col_names: list[str], label_col_name: str) -> pl.Expr:
        return pl.corr(cs.by_name(feature_col_names), label_col_name).fill_nan(0.0).abs()

    @staticmethod
    def _check_valid_types(feature_cols: list[ColumnSpecification], label_col: ColumnSpecification, operation: SelectionMethod) -> None:
        supported_label_types = SUPPORTED_LABEL_COLUMN_TYPES[operation]
        if label_col.column_type not in supported_label_types:
            raise ValueError(
                f"{operation.value} can only be computed with label column of type {', '.join(col_type.value for col_type in supported_label_types)}, "
                f'but {label_col.name} is of type {label_col.column_type}.',
            )

        supported_types = SUPPORTED_COLUMN_TYPES[operation]
        for col in feature_cols:
            if col.column_type not in supported_types:
                raise ValueError(
                    f"{operation.value} can only be computed for {', '.join(col_type.value for col_type in supported_types)} columns, "
                    f'but {col.name} is of type {col.column_type}.')

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
