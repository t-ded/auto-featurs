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
    T_TEST = 'T-Test'


SUPPORTED_COLUMN_TYPES = {
    SelectionMethod.CORRELATION: [ColumnType.NUMERIC, ColumnType.BOOLEAN, ColumnType.ORDINAL],
    SelectionMethod.T_TEST: [ColumnType.NUMERIC, ColumnType.BOOLEAN, ColumnType.ORDINAL],
}

SUPPORTED_LABEL_COLUMN_TYPES = {
    SelectionMethod.CORRELATION: [ColumnType.NUMERIC, ColumnType.BOOLEAN],
    SelectionMethod.T_TEST: [ColumnType.BOOLEAN],
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

    def select_by_ttest(self, dataset: Dataset, feature_subset: ColumnSelection, top_k: Optional[int] = None, frac: Optional[float] = None) -> list[str]:
        label_col = dataset.get_label_column()
        label_col_name = label_col.name

        feature_cols = dataset.get_columns_from_selection(feature_subset)
        self._check_valid_types(feature_cols, label_col, SelectionMethod.T_TEST)
        feature_col_names = get_names_from_column_specs(feature_cols)
        num_to_select = self._get_num_to_select(top_k=top_k, frac=frac, num_cols=len(feature_col_names))

        t_stats = self._ttest_stat_expr(dataset.data, feature_col_names=feature_col_names, label_col_name=label_col_name)

        to_select: list[str] = (
            t_stats
            .sort(['T_STAT', 'FEATURE_NAME'], descending=[True, False])
            .head(num_to_select)
            .select('FEATURE_NAME')
            .collect()
            .to_series()
            .to_list()
        )

        return to_select

    @staticmethod
    def _ttest_stat_expr(df: pl.LazyFrame | pl.DataFrame, feature_col_names: list[str], label_col_name: str) -> pl.LazyFrame:
        stats = (
            df
            .lazy()
            .group_by(label_col_name)
            .agg(
                cs.by_name(feature_col_names).mean().name.suffix('_mean'),
                cs.by_name(feature_col_names).var().name.suffix('_var'),
                pl.len().alias('count'),
            )
            .with_columns(pl.col(label_col_name).cast(pl.Boolean))
        )

        counts = stats.select(label_col_name, 'count').collect()
        true_count: int = counts.filter(pl.col(label_col_name) == True)['count'].item()
        false_count: int = counts.filter(pl.col(label_col_name) == False)['count'].item()

        t_stats = (
            stats
            .drop('count')
            .unpivot(index=label_col_name, variable_name='FEATURE_STAT', value_name='VALUE')
            .select(
                pl.col('FEATURE_STAT').str.split('_').list.get(0).alias('FEATURE_NAME'),
                label_col_name,
                pl.col('FEATURE_STAT').str.split('_').list.get(1).alias('STAT'),
                'VALUE',
            )
            .with_columns(
                (pl.col(label_col_name).cast(pl.Utf8).str.to_uppercase() + pl.lit('_') + pl.col('STAT').str.to_uppercase()).alias('pivot_col')
            )
            .pivot(on='pivot_col', on_columns=['TRUE_MEAN', 'FALSE_MEAN', 'TRUE_VAR', 'FALSE_VAR'], index='FEATURE_NAME', values='VALUE')
            .with_columns(
                pl.col('TRUE_MEAN').sub(pl.col('FALSE_MEAN')).abs().alias('MEAN_DIFF'),
                pl.col('TRUE_VAR').truediv(true_count).alias('NORMALIZED_TRUE_VAR'),
                pl.col('FALSE_VAR').truediv(false_count).alias('NORMALIZED_FALSE_VAR'),
            )
            .with_columns(pl.col('NORMALIZED_TRUE_VAR').add(pl.col('NORMALIZED_FALSE_VAR')).sqrt().alias('DENOMINATOR'))
            .select(
                'FEATURE_NAME',
                pl.col('MEAN_DIFF').truediv(pl.col('DENOMINATOR')).fill_nan(0.0).alias('T_STAT'),
            )
        )

        return t_stats

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
