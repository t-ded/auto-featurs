import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from datetime import UTC
    from datetime import datetime
    return UTC, datetime, mo, pl


@app.cell
def _():
    import auto_featurs
    return


@app.cell
def _(mo):
    mo.md(rf"""
    ## Data Setup
    """)
    return


@app.cell
def _(BASIC_FRAME):
    BASIC_FRAME
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Pipeline Setup
    """)
    return


@app.cell
def _(Pipeline):
    pipeline = Pipeline
    return


@app.cell
def _(mo):
    mo.md(rf"""
    ## Apendix
    """)
    return


@app.cell
def _(UTC, datetime, pl):
    BASIC_FRAME = pl.LazyFrame(
        {
            'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5],
            'NUMERIC_FEATURE_2': [0, -1, -2, -3, -4, -5],
            'CATEGORICAL_FEATURE': ['A', 'B', 'C', 'D', 'E', 'F'],
            'CATEGORICAL_FEATURE_2': ['F', 'E', 'D', 'C', 'B', 'A'],
            'GROUPING_FEATURE_NUM': ['ZERO', 'ODD', 'EVEN', 'ODD', 'EVEN', 'ODD'],
            'GROUPING_FEATURE_CAT_2': ['CONSONANT', 'VOWEL', 'CONSONANT', 'CONSONANT', 'CONSONANT', 'VOWEL'],
            'DATE_FEATURE': [datetime(year=2_000, month=1, day=i, tzinfo=UTC) for i in range(1, 7)],
        },
    )
    return (BASIC_FRAME,)


if __name__ == "__main__":
    app.run()
