from auto_featurs.base.column_specification import ColumnRole
from auto_featurs.base.column_specification import ColumnRoleSelector
from auto_featurs.base.column_specification import ColumnSelector
from auto_featurs.base.column_specification import ColumnSpecification
from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.column_specification import ColumnTypeSelector


def test_spec_creation() -> None:
    assert ColumnSpecification.numeric(name='a', role=ColumnRole.LABEL) == ColumnSpecification(name='a', column_type=ColumnType.NUMERIC, column_role=ColumnRole.LABEL)


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
        expected_selector = ColumnSelector(
            type_selector=ColumnTypeSelector({ColumnType.NUMERIC}),
            role_selector=ColumnRoleSelector({ColumnRole.LABEL}),
        )
        assert ColumnType.NUMERIC & ColumnRole.LABEL == expected_selector
        assert ColumnType.NUMERIC & ColumnRoleSelector({ColumnRole.LABEL}) == expected_selector

    def test_or(self) -> None:
        expected_selector = ColumnTypeSelector({ColumnType.NUMERIC, ColumnType.TEXT})
        assert ColumnType.NUMERIC | ColumnType.TEXT == expected_selector
        assert ColumnType.NUMERIC | ColumnTypeSelector({ColumnType.TEXT}) == expected_selector

    def test_invert(self) -> None:
        expected_selector = ColumnTypeSelector(types=ColumnType.ANY() - {ColumnType.NUMERIC})
        assert ~ColumnType.NUMERIC == expected_selector


class TestColumnRole:
    def test_any_column_type(self) -> None:
        assert ColumnRole.ANY() == {
            ColumnRole.LABEL,
            ColumnRole.IDENTIFIER,
            ColumnRole.TIME_INFO,
            ColumnRole.FEATURE,
        }

    def test_and(self) -> None:
        expected_selector = ColumnSelector(
            type_selector=ColumnTypeSelector({ColumnType.NUMERIC}),
            role_selector=ColumnRoleSelector({ColumnRole.LABEL}),
        )
        assert ColumnRole.LABEL & ColumnType.NUMERIC == expected_selector
        assert ColumnRole.LABEL & ColumnTypeSelector({ColumnType.NUMERIC}) == expected_selector

    def test_or(self) -> None:
        expected_selector = ColumnRoleSelector({ColumnRole.LABEL, ColumnRole.IDENTIFIER})
        assert ColumnRole.LABEL | ColumnRole.IDENTIFIER == expected_selector
        assert ColumnRole.LABEL | ColumnRoleSelector({ColumnRole.IDENTIFIER}) == expected_selector

    def test_invert(self) -> None:
        expected_selector = ColumnRoleSelector(roles=ColumnRole.ANY() - {ColumnRole.LABEL})
        assert ~ColumnRole.LABEL == expected_selector


class TestColumnTypeSelector:
    def test_eq(self) -> None:
        assert ColumnTypeSelector({ColumnType.NUMERIC}) == ColumnTypeSelector({ColumnType.NUMERIC})
        assert ColumnTypeSelector({ColumnType.NUMERIC}) != ColumnTypeSelector({ColumnType.ORDINAL})

    def test_and(self) -> None:
        expected_selector = ColumnSelector(
            type_selector=ColumnTypeSelector({ColumnType.NUMERIC}),
            role_selector=ColumnRoleSelector({ColumnRole.LABEL}),
        )
        assert ColumnTypeSelector({ColumnType.NUMERIC}) & ColumnRole.LABEL == expected_selector
        assert ColumnTypeSelector({ColumnType.NUMERIC}) & ColumnRoleSelector({ColumnRole.LABEL}) == expected_selector

    def test_or(self) -> None:
        expected_selector = ColumnTypeSelector({ColumnType.NUMERIC, ColumnType.TEXT})
        assert ColumnTypeSelector({ColumnType.NUMERIC}) | ColumnType.TEXT == expected_selector
        assert ColumnTypeSelector({ColumnType.NUMERIC}) | ColumnTypeSelector({ColumnType.TEXT}) == expected_selector

    def test_invert(self) -> None:
        expected_selector = ColumnTypeSelector(types=ColumnType.ANY() - {ColumnType.NUMERIC})
        assert ~ColumnTypeSelector({ColumnType.NUMERIC}) == expected_selector


class TestColumnRoleSelector:
    def test_eq(self) -> None:
        assert ColumnRoleSelector({ColumnRole.LABEL}) == ColumnRoleSelector({ColumnRole.LABEL})
        assert ColumnRoleSelector({ColumnRole.LABEL}) != ColumnRoleSelector({ColumnRole.IDENTIFIER})

    def test_and(self) -> None:
        expected_selector = ColumnSelector(
            type_selector=ColumnTypeSelector({ColumnType.NUMERIC}),
            role_selector=ColumnRoleSelector({ColumnRole.LABEL}),
        )
        assert ColumnRoleSelector({ColumnRole.LABEL}) & ColumnType.NUMERIC == expected_selector
        assert ColumnRoleSelector({ColumnRole.LABEL}) & ColumnTypeSelector({ColumnType.NUMERIC}) == expected_selector

    def test_or(self) -> None:
        expected_selector = ColumnRoleSelector({ColumnRole.LABEL, ColumnRole.IDENTIFIER})
        assert ColumnRoleSelector({ColumnRole.LABEL}) | ColumnRole.IDENTIFIER == expected_selector
        assert ColumnRoleSelector({ColumnRole.LABEL}) | ColumnRoleSelector({ColumnRole.IDENTIFIER}) == expected_selector

    def test_invert(self) -> None:
        expected_selector = ColumnRoleSelector(roles=ColumnRole.ANY() - {ColumnRole.LABEL})
        assert ~ColumnRoleSelector({ColumnRole.LABEL}) == expected_selector
