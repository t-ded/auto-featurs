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
        assert ColumnType.NUMERIC & ColumnRole.LABEL == ColumnSelector(
            type_selector=ColumnTypeSelector({ColumnType.NUMERIC}),
            role_selector=ColumnRoleSelector({ColumnRole.LABEL}),
        )
        assert ColumnType.NUMERIC & ColumnRoleSelector({ColumnRole.LABEL}) == ColumnSelector(
            type_selector=ColumnTypeSelector({ColumnType.NUMERIC}),
            role_selector=ColumnRoleSelector({ColumnRole.LABEL}),
        )

    def test_invert(self) -> None:
        assert ~ColumnType.NUMERIC == ColumnTypeSelector(types=ColumnType.ANY() - {ColumnType.NUMERIC})


class TestColumnRole:
    def test_any_column_type(self) -> None:
        assert ColumnRole.ANY() == {
            ColumnRole.LABEL,
            ColumnRole.IDENTIFIER,
            ColumnRole.TIME_INFO,
            ColumnRole.FEATURE,
        }

    def test_and(self) -> None:
        assert ColumnRole.LABEL & ColumnType.NUMERIC == ColumnSelector(
            type_selector=ColumnTypeSelector({ColumnType.NUMERIC}),
            role_selector=ColumnRoleSelector({ColumnRole.LABEL}),
        )
        assert ColumnRole.LABEL & ColumnTypeSelector({ColumnType.NUMERIC}) == ColumnSelector(
            type_selector=ColumnTypeSelector({ColumnType.NUMERIC}),
            role_selector=ColumnRoleSelector({ColumnRole.LABEL}),
        )

    def test_invert(self) -> None:
        assert ~ColumnRole.LABEL == ColumnRoleSelector(roles=ColumnRole.ANY() - {ColumnRole.LABEL})


class TestColumnTypeSelector:
    def test_eq(self) -> None:
        assert ColumnTypeSelector({ColumnType.NUMERIC}) == ColumnTypeSelector({ColumnType.NUMERIC})
        assert ColumnTypeSelector({ColumnType.NUMERIC}) != ColumnTypeSelector({ColumnType.ORDINAL})

    def test_and(self) -> None:
        assert ColumnTypeSelector({ColumnType.NUMERIC}) & ColumnRole.LABEL == ColumnSelector(
            type_selector=ColumnTypeSelector({ColumnType.NUMERIC}),
            role_selector=ColumnRoleSelector({ColumnRole.LABEL}),
        )
        assert ColumnTypeSelector({ColumnType.NUMERIC}) & ColumnRoleSelector({ColumnRole.LABEL}) == ColumnSelector(
            type_selector=ColumnTypeSelector({ColumnType.NUMERIC}),
            role_selector=ColumnRoleSelector({ColumnRole.LABEL}),
        )

    def test_invert(self) -> None:
        assert ~ColumnTypeSelector({ColumnType.NUMERIC}) == ColumnTypeSelector(types=ColumnType.ANY() - {ColumnType.NUMERIC})


class TestColumnRoleSelector:
    def test_eq(self) -> None:
        assert ColumnRoleSelector({ColumnRole.LABEL}) == ColumnRoleSelector({ColumnRole.LABEL})
        assert ColumnRoleSelector({ColumnRole.LABEL}) != ColumnRoleSelector({ColumnRole.IDENTIFIER})

    def test_and(self) -> None:
        assert ColumnRoleSelector({ColumnRole.LABEL}) & ColumnType.NUMERIC == ColumnSelector(
            type_selector=ColumnTypeSelector({ColumnType.NUMERIC}),
            role_selector=ColumnRoleSelector({ColumnRole.LABEL}),
        )
        assert ColumnRoleSelector({ColumnRole.LABEL}) & ColumnTypeSelector({ColumnType.NUMERIC}) == ColumnSelector(
            type_selector=ColumnTypeSelector({ColumnType.NUMERIC}),
            role_selector=ColumnRoleSelector({ColumnRole.LABEL}),
        )

    def test_invert(self) -> None:
        assert ~ColumnRoleSelector({ColumnRole.LABEL}) == ColumnRoleSelector(roles=ColumnRole.ANY() - {ColumnRole.LABEL})
