from pathlib import Path

import yaml


def test_openenv_yaml_has_required_fields() -> None:
    doc = yaml.safe_load(Path("openenv.yaml").read_text(encoding="utf-8"))

    assert doc["name"]
    assert doc["entrypoint"]["module"]
    assert doc["entrypoint"]["class"]
    assert "reset" in doc["spec"]["methods"]
    assert "step" in doc["spec"]["methods"]
    assert "state" in doc["spec"]["methods"]
