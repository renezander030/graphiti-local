import pytest

from kg_mcp.config import Settings
from kg_mcp.restore import file_group, target_group


def _settings(provider: str) -> Settings:
    return Settings.model_validate(
        {"graph": {"groups": ["main", "other"]}, "database": {"provider": provider}}
    )


def test_target_group_on_ladybug_is_the_single_graph():
    assert target_group({"uuid": "u1", "group_id": "main"}, None, _settings("ladybug")) == ""


def test_target_group_takes_the_override_then_the_record():
    settings = _settings("neo4j")
    assert target_group({"uuid": "u1", "group_id": "main"}, None, settings) == "main"
    assert target_group({"uuid": "u1", "group_id": "main"}, "other", settings) == "other"


def test_target_group_without_one_names_the_provider():
    with pytest.raises(ValueError) as caught:
        target_group({"uuid": "u1"}, None, _settings("neo4j"))
    assert str(caught.value) == (
        "record u1 has no group_id; pass --group to choose the target group on neo4j"
    )


def test_file_group_without_one_names_the_file():
    with pytest.raises(ValueError) as caught:
        file_group({"uuid": "u2", "group_id": ""}, None, _settings("ladybug"))
    assert str(caught.value) == (
        "record u2 has no group_id; pass --group to choose the group file it is restored into"
    )


@pytest.mark.parametrize("resolve", [target_group, file_group])
def test_a_group_outside_the_allow_list_is_refused(resolve):
    with pytest.raises(ValueError):
        resolve({"uuid": "u3", "group_id": "elsewhere"}, None, _settings("neo4j"))
