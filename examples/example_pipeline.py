# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "auto-featurs",
#     "polars==1.35.2",
# ]
#
# [tool.uv.sources]
# auto-featurs = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(rf"""
    # Initial Look at the Data
    """)
    return


@app.cell
def _(BASIC_FRAME):
    BASIC_FRAME
    return


@app.cell
def _(mo):
    mo.md(rf"""
    # Pipeline Setup
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Unoptimized
    """)
    return


@app.cell
def _(
    ArithmeticAggregations,
    BASIC_FRAME,
    ColumnSpecification,
    ColumnType,
    Comparisons,
    Pipeline,
):
    pipeline = Pipeline(
        schema=[
            ColumnSpecification.numeric(name='NUMERIC_FEATURE'),
            ColumnSpecification.ordinal(name='CATEGORICAL_FEATURE'),
            ColumnSpecification.datetime(name='DATE_FEATURE'),
        ],
    )

    pipeline = (
        pipeline
        .with_polynomial(subset='NUMERIC_FEATURE', degrees=[2])
        .with_arithmetic_aggregation(
            subset='NUMERIC_FEATURE',
            aggregations=[ArithmeticAggregations.COUNT],
            time_windows=['2d'],
            index_column_name='DATE_FEATURE',
            # over_columns_combinations=[[], ['GROUPING_FEATURE_NUM']],  # TODO: Uncomment this with new polars release (allowing window expressions in aggregation)
        )
        .with_new_layer()
        .with_comparison(
            left_subset=ColumnType.NUMERIC,
            right_subset=ColumnType.NUMERIC,
            comparisons=[Comparisons.EQUAL],
        )
    )

    res = pipeline.collect(df=BASIC_FRAME)
    res
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Optimized
    """)
    return


@app.cell
def _(
    ArithmeticAggregations,
    BASIC_FRAME,
    ColumnSpecification,
    ColumnType,
    Comparisons,
    OptimizationLevel,
    Pipeline,
):
    pipeline_optimized = Pipeline(
        schema=[
            ColumnSpecification.numeric(name='NUMERIC_FEATURE'),
            ColumnSpecification.ordinal(name='CATEGORICAL_FEATURE'),
            ColumnSpecification.datetime(name='DATE_FEATURE'),
        ],
        optimization_level=OptimizationLevel.DEDUPLICATE_COMMUTATIVE,
    )

    pipeline_optimized = (
        pipeline_optimized
        .with_polynomial(subset='NUMERIC_FEATURE', degrees=[2])
        .with_arithmetic_aggregation(
            subset='NUMERIC_FEATURE',
            aggregations=[ArithmeticAggregations.COUNT],
            time_windows=['2d'],
            index_column_name='DATE_FEATURE',
            # over_columns_combinations=[[], ['GROUPING_FEATURE_NUM']],  # TODO: Uncomment this with new polars release (allowing window expressions in aggregation)
        )
        .with_new_layer()
        .with_comparison(
            left_subset=ColumnType.NUMERIC,
            right_subset=ColumnType.NUMERIC,
            comparisons=[Comparisons.EQUAL],
        )
    )

    res_optimized = pipeline_optimized.collect(df=BASIC_FRAME)
    res_optimized
    return


@app.cell
def _(mo):
    mo.md(rf"""
    # Apendix
    """)
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from datetime import date
    return date, mo, pl


@app.cell
def _():
    from auto_featurs.base.column_specification import ColumnSpecification
    from auto_featurs.base.column_specification import ColumnType

    from auto_featurs.pipeline.optimizer import OptimizationLevel
    from auto_featurs.pipeline.pipeline import Pipeline

    from auto_featurs.transformers import ArithmeticAggregations
    from auto_featurs.transformers import Comparisons
    return (
        ArithmeticAggregations,
        ColumnSpecification,
        ColumnType,
        Comparisons,
        OptimizationLevel,
        Pipeline,
    )


@app.cell
def _(date, pl):
    BASIC_FRAME = pl.LazyFrame(
        {
            'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5],
            'CATEGORICAL_FEATURE': ['A', 'B', 'C', 'D', 'E', 'F'],
            'GROUPING_FEATURE_NUM': ['ZERO', 'ODD', 'EVEN', 'ODD', 'EVEN', 'ODD'],
            'DATE_FEATURE': [date(year=2_000, month=1, day=i) for i in range(1, 7)],
        },
    )
    return (BASIC_FRAME,)


if __name__ == "__main__":
    app.run()
