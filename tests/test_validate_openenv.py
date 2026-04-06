from news_stock_env.validate_openenv import run_validation


def test_validate_openenv_fixture() -> None:
    result = run_validation("tests/fixtures/fixture_dataset.csv")

    assert result["rows"] == 3
    assert result["reward"] == 1.0
    assert result["first_task_id"] == "task-001"
