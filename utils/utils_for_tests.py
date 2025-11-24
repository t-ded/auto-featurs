from typing import Any

import polars as pl
from polars.testing import assert_frame_equal


BASIC_FRAME = pl.LazyFrame(
    {
        'NUMERIC_FEATURE': [0, 1, 2, 3, 4, 5],
        'NUMERIC_FEATURE_2': [0, -1, -2, -3, -4, -5],
        'CATEGORICAL_FEATURE': ['A', 'B', 'C', 'D', 'E', 'F'],
        'CATEGORICAL_FEATURE_2': ['F', 'E', 'D', 'C', 'B', 'A'],
        'GROUPING_FEATURE_NUM': ['ZERO', 'ODD', 'EVEN', 'ODD', 'EVEN', 'ODD'],
        'GROUPING_FEATURE_CAT_2': ['CONSONANT', 'VOWEL', 'CONSONANT', 'CONSONANT', 'CONSONANT', 'VOWEL'],
    },
)


def assert_new_columns_in_frame(original_frame: pl.LazyFrame | pl.DataFrame, new_frame: pl.LazyFrame | pl.DataFrame, expected_new_columns: dict[str, list[Any]]) -> None:
    original_frame = original_frame.lazy()
    new_frame = new_frame.lazy()

    new_columns_frame = pl.LazyFrame(expected_new_columns)
    expected_new_frame = pl.concat([original_frame, new_columns_frame], how='horizontal')

    assert_frame_equal(new_frame, expected_new_frame, check_dtypes=False)
