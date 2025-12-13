from dataclasses import dataclass
from enum import Enum
from typing import Optional
from typing import assert_never

import polars as pl
from polars import selectors as cs

from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.schema import ColumnSelection
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


@dataclass(kw_only=True, frozen=True, slots=True)
class SelectionReport:
    feature_names: pl.Series
    stat_values: pl.Series
    method: SelectionMethod
    p_values: Optional[pl.Series] = None

    def to_frame(self) -> pl.DataFrame:
        method_value_col_name = self.method.value + ' Value'
        res = {'Feature Name': self.feature_names, method_value_col_name: self.stat_values}
        if self.p_values is not None:
            res['P-Value'] = self.p_values
        return pl.DataFrame(res)


class FeatureSelector:
    def select_features(
            self,
            report: SelectionReport,
            top_k: Optional[int] = None,
            frac: Optional[float] = None,
    ) -> list[str]:
        num_to_select = self._get_num_to_select(top_k=top_k, frac=frac, num_cols=len(report.feature_names))
        order = pl.DataFrame({'stat': report.stat_values, 'name': report.feature_names}).with_row_index(name='idx').sort(['stat', 'name'], descending=[True, False])['idx']
        return report.feature_names[order].head(num_to_select).to_list()

    def get_report(self, dataset: Dataset, feature_subset: ColumnSelection, method: SelectionMethod) -> SelectionReport:
        label_col = dataset.get_label_column()
        feature_cols = dataset.get_columns_from_selection(feature_subset)
        self._check_valid_types(feature_cols, label_col, method)

        label_col_name = label_col.name
        feature_col_names = get_names_from_column_specs(feature_cols)

        match method:
            case SelectionMethod.CORRELATION:
                stats = self._point_correlation(dataset.data, feature_col_names=feature_col_names, label_col_name=label_col_name)
            case SelectionMethod.T_TEST:
                stats = self._ttest_stat_expr(dataset.data, feature_col_names=feature_col_names, label_col_name=label_col_name)
            case _:
                assert_never(method)

        return SelectionReport(feature_names=stats['FEATURE_NAME'], stat_values=stats['STAT_VALUE'], method=method)

    @staticmethod
    def _point_correlation(df: pl.LazyFrame | pl.DataFrame, feature_col_names: list[str], label_col_name: str) -> pl.DataFrame:
        return df.lazy().select(pl.corr(cs.by_name(feature_col_names), label_col_name).fill_nan(0.0).abs()).unpivot(variable_name='FEATURE_NAME', value_name='STAT_VALUE').collect()

    @staticmethod
    def _ttest_stat_expr(df: pl.LazyFrame | pl.DataFrame, feature_col_names: list[str], label_col_name: str) -> pl.DataFrame:
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
        true_count: int = counts.filter(pl.col(label_col_name).eq(True))['count'].item()
        false_count: int = counts.filter(pl.col(label_col_name).eq(False))['count'].item()

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
                (pl.col(label_col_name).cast(pl.Utf8).str.to_uppercase() + pl.lit('_') + pl.col('STAT').str.to_uppercase()).alias('pivot_col'),
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
                pl.col('MEAN_DIFF').truediv(pl.col('DENOMINATOR')).fill_nan(0.0).alias('STAT_VALUE'),
            )
        )

        return t_stats.collect()

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
