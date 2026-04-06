from news_stock_env.data_types import Action, DatasetRow, NewsArticle, StockPrediction
from news_stock_env.env import NewsSignalEnv


def _sample_row() -> DatasetRow:
    article = NewsArticle(
        title="Oil rises amid supply concerns",
        description="Energy stocks rally as oil moves higher.",
        source="reuters.com",
        url="https://example.com/oil",
        published_at="2026-04-01T00:00:00Z",
    )
    prediction = StockPrediction(
        gainers=["XOM", "CVX", "SLB", "HAL", "COP"],
        losers=["AAL", "DAL", "UAL", "LUV", "JBLU"],
        assets={"gold": "UP", "silver": "NEUTRAL", "oil": "UP"},
    )
    return DatasetRow(
        id="task-001",
        date="2026-04-01",
        difficulty="easy",
        long_term_context=[article],
        short_term_context=[article],
        stock_predictions=prediction,
    )


def test_env_smoke_cycle() -> None:
    env = NewsSignalEnv(dataset=[_sample_row()])
    observation = env.reset()
    assert observation.task_id == "task-001"

    action = Action(
        gainers=["XOM", "CVX", "SLB", "HAL", "COP"],
        losers=["AAL", "DAL", "UAL", "LUV", "JBLU"],
        assets={"gold": "UP", "silver": "NEUTRAL", "oil": "UP"},
    )

    result = env.step(action)

    assert result.done is True
    assert 0.0 <= result.reward.value <= 1.0
    assert env.state().done is True
