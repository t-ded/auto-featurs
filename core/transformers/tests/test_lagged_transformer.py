from core.base.column_specification import ColumnSpecification
from core.base.column_specification import ColumnType
from core.transformers.lagged_transformer import LaggedTransformer
from utils.utils_for_tests import BASIC_FRAME
from utils.utils_for_tests import assert_new_columns_in_frame


class TestLaggedTransformer:
    def setup_method(self) -> None:
        self._lagged_1_categorical_transformer = LaggedTransformer(column=ColumnSpecification.ordinal(name='CATEGORICAL_FEATURE'), lag=1)
        self._lagged_1_transformer = LaggedTransformer(column=ColumnSpecification.numeric(name='NUMERIC_FEATURE'), lag=1)
        self._lagged_2_transformer = LaggedTransformer(column=ColumnSpecification.numeric(name='NUMERIC_FEATURE'), lag=2)

    def test_name_and_output_type(self) -> None:
        assert self._lagged_1_categorical_transformer.output_column_specification == ColumnSpecification.ordinal(name='CATEGORICAL_FEATURE_lagged_1')
        assert self._lagged_1_transformer.output_column_specification == ColumnSpecification.numeric(name='NUMERIC_FEATURE_lagged_1')
        assert self._lagged_2_transformer.output_column_specification == ColumnSpecification.numeric(name='NUMERIC_FEATURE_lagged_2')

    def test_lagged_transform(self) -> None:
        df = BASIC_FRAME.with_columns(
            self._lagged_1_categorical_transformer.transform(),
            self._lagged_1_transformer.transform(),
            self._lagged_2_transformer.transform(),
        )

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={
                'CATEGORICAL_FEATURE_lagged_1': [None, 'A', 'B', 'C', 'D', 'E'],
                'NUMERIC_FEATURE_lagged_1': [None, 0, 1, 2, 3, 4],
                'NUMERIC_FEATURE_lagged_2': [None, None, 0, 1, 2, 3],
            },
        )

    def test_fill_value(self) -> None:
        lagged_transformer = LaggedTransformer(column=ColumnSpecification(name='NUMERIC_FEATURE', column_type=ColumnType.NUMERIC), lag=2, fill_value=0)

        df = BASIC_FRAME.with_columns(lagged_transformer.transform())

        assert_new_columns_in_frame(
            original_frame=BASIC_FRAME,
            new_frame=df,
            expected_new_columns={'NUMERIC_FEATURE_lagged_2': [0, 0, 0, 1, 2, 3]},
        )
