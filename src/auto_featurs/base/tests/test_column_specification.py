from auto_featurs.base.column_specification import ColumnRole
from auto_featurs.base.column_specification import ColumnRoleSelector
from auto_featurs.base.column_specification import ColumnSelector
from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs import column_selectors as cs
from auto_featurs.base.column_specification import ColumnTypeSelector


def test_spec_creation() -> None:
    assert ColumnSpecification.numeric(name='a', role=ColumnRole.LABEL) == ColumnSpecification(name='a', column_type=ColumnType.NUMERIC, column_role=ColumnRole.LABEL)


def assert_positives_match(selectors: list[ColumnSelector], column: ColumnSpecification) -> None:
    for selector in selectors:
        assert selector.matches(column), f'Column {column.name} was expected to match selector {selector}'


def assert_negatives_do_not_match(selectors: list[ColumnSelector], column: ColumnSpecification) -> None:
    for selector in selectors:
        assert not selector.matches(column), f'Column {column.name} was expected to not match selector {selector}'


class TestColumnType:
    def test_any_column_type(self) -> None:
        assert ColumnType.ANY() == {
            ColumnType.NUMERIC,
            ColumnType.BOOLEAN,
            ColumnType.NOMINAL,
            ColumnType.ORDINAL,
            ColumnType.TEXT,
            ColumnType.DATETIME,
        }

    def test_and(self) -> None:
        numeric_label_column = ColumnSpecification.numeric(name='abc', role=ColumnRole.LABEL)

        positive_selectors = [
            ColumnType.NUMERIC & ColumnRole.LABEL,
            ColumnType.NUMERIC & cs.name_contains('abc'),
            ColumnType.NUMERIC & cs.name_starts_with('a'),
            ColumnType.NUMERIC & cs.name_ends_with('c'),
        ]

        negative_selectors = [
            ColumnType.TEXT & ColumnRole.LABEL,
            ColumnType.NUMERIC & ColumnRole.FEATURE,
            ColumnType.NUMERIC & cs.name_contains('d'),
            ColumnType.NUMERIC & cs.name_starts_with('b'),
            ColumnType.NUMERIC & cs.name_ends_with('b'),
        ]

        assert_positives_match(positive_selectors, numeric_label_column)
        assert_negatives_do_not_match(negative_selectors, numeric_label_column)

    def test_or(self) -> None:
        numeric_label_column = ColumnSpecification.numeric(name='abc', role=ColumnRole.LABEL)

        positive_selectors = [
            ColumnType.NUMERIC | ColumnType.TEXT,
            ColumnType.TEXT | ColumnRole.LABEL,
            ColumnType.NUMERIC | ColumnRole.LABEL,
            ColumnType.NUMERIC | ColumnRole.FEATURE,
            ColumnType.NUMERIC | cs.name_contains('d'),
            ColumnType.NUMERIC | cs.name_starts_with('b'),
            ColumnType.NUMERIC | cs.name_ends_with('b'),
        ]

        negative_selectors = [
            ColumnType.TEXT | ColumnRole.FEATURE,
            ColumnType.TEXT | cs.name_contains('d'),
            ColumnType.TEXT | cs.name_starts_with('b'),
            ColumnType.TEXT | cs.name_ends_with('b'),
        ]

        assert_positives_match(positive_selectors, numeric_label_column)
        assert_negatives_do_not_match(negative_selectors, numeric_label_column)
        assert type(ColumnType.NUMERIC | ColumnType.TEXT) is ColumnTypeSelector

    def test_invert(self) -> None:
        numeric_label_column = ColumnSpecification.numeric(name='abc', role=ColumnRole.LABEL)

        positive_selectors = [~ColumnType.TEXT]

        negative_selectors = [~ColumnType.NUMERIC]

        assert_positives_match(positive_selectors, numeric_label_column)
        assert_negatives_do_not_match(negative_selectors, numeric_label_column)

    def test_as_selector(self) -> None:
        assert ColumnType.NUMERIC.as_selector() == ColumnTypeSelector(frozenset([ColumnType.NUMERIC]))


class TestColumnRole:
    def test_any_column_type(self) -> None:
        assert ColumnRole.ANY() == {
            ColumnRole.LABEL,
            ColumnRole.IDENTIFIER,
            ColumnRole.TIME_INFO,
            ColumnRole.FEATURE,
        }

    def test_and(self) -> None:
        numeric_label_column = ColumnSpecification.numeric(name='abc', role=ColumnRole.LABEL)

        positive_selectors = [
            ColumnRole.LABEL & ColumnType.NUMERIC,
            ColumnRole.LABEL & cs.name_contains('abc'),
            ColumnRole.LABEL & cs.name_starts_with('a'),
            ColumnRole.LABEL & cs.name_ends_with('c'),
        ]

        negative_selectors = [
            ColumnRole.LABEL & ColumnType.TEXT,
            ColumnRole.LABEL & cs.name_contains('d'),
            ColumnRole.LABEL & cs.name_starts_with('b'),
            ColumnRole.LABEL & cs.name_ends_with('b'),
        ]

        assert_positives_match(positive_selectors, numeric_label_column)
        assert_negatives_do_not_match(negative_selectors, numeric_label_column)

    def test_or(self) -> None:
        numeric_label_column = ColumnSpecification.numeric(name='abc', role=ColumnRole.LABEL)

        positive_selectors = [
            ColumnRole.LABEL | ColumnRole.FEATURE,
            ColumnRole.LABEL | ColumnType.NUMERIC,
            ColumnRole.LABEL | ColumnType.TEXT,
            ColumnRole.LABEL | cs.name_contains('d'),
            ColumnRole.LABEL | cs.name_starts_with('b'),
            ColumnRole.LABEL | cs.name_ends_with('b'),
        ]

        negative_selectors = [
            ColumnRole.FEATURE | ColumnType.TEXT,
            ColumnType.TEXT | cs.name_contains('d'),
            ColumnType.TEXT | cs.name_starts_with('b'),
            ColumnType.TEXT | cs.name_ends_with('b'),
        ]

        assert_positives_match(positive_selectors, numeric_label_column)
        assert_negatives_do_not_match(negative_selectors, numeric_label_column)
        assert type(ColumnRole.LABEL | ColumnRole.FEATURE) is ColumnRoleSelector

    def test_invert(self) -> None:
        numeric_label_column = ColumnSpecification.numeric(name='abc', role=ColumnRole.LABEL)

        positive_selectors = [~ColumnRole.FEATURE]

        negative_selectors = [~ColumnRole.LABEL]

        assert_positives_match(positive_selectors, numeric_label_column)
        assert_negatives_do_not_match(negative_selectors, numeric_label_column)

    def test_as_selector(self) -> None:
        assert ColumnRole.LABEL.as_selector() == ColumnRoleSelector(frozenset([ColumnRole.LABEL]))
