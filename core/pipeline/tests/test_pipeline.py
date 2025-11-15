import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal

from core.base.column_types import ColumnType
from core.pipeline.pipeline import Pipeline
from core.transformers.numeric_transformers import PolynomialTransformer


class TestPipeline:
    def test_transformers_from_init(self) -> None:
        pipeline = Pipeline(
            column_types={'NUMERIC_FEATURE': ColumnType.NUMERIC},
            transformers=[[PolynomialTransformer(columns=cs.by_name('NUMERIC_FEATURE'), degree=2)]],
        )
        df = pl.LazyFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5]})

        res = pipeline.collect(df)

        assert_frame_equal(
            res,
            pl.DataFrame({'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5], 'NUMERIC_FEATURE_pow_2': [0, 1, 4, 9, 16, 25]}),
        )

    def test_basic_sample_with_all_transformers(self) -> None:
        pipeline = Pipeline(column_types={'NUMERIC_FEATURE': ColumnType.NUMERIC, 'NUMERIC_FEATURE_2': ColumnType.NUMERIC})
        pipeline = pipeline.with_polynomial(subset=cs.numeric(), degrees=[2, 3])
        df = pl.LazyFrame({
            'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5],
            'NUMERIC_FEATURE_2': [0, -1, -2, -3, -4, -5],
        })

        res = pipeline.collect(df)

        assert_frame_equal(
            res,
            pl.DataFrame({
                'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5],
                'NUMERIC_FEATURE_2': [0, -1, -2, -3, -4, -5],
                'NUMERIC_FEATURE_pow_2': [0, 1, 4, 9, 16, 25],
                'NUMERIC_FEATURE_2_pow_2': [0, 1, 4, 9, 16, 25],
                'NUMERIC_FEATURE_pow_3': [0, 1, 8, 27, 64, 125],
                'NUMERIC_FEATURE_2_pow_3': [0, -1, -8, -27, -64, -125],
            })
        )
