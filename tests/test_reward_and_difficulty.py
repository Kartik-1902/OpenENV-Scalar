from news_stock_env.data_types import Action, NewsArticle, StockPrediction
from news_stock_env.difficulty import (
    distribution_report,
    label_difficulty_approach_1,
    should_recommend_hybrid,
)
from news_stock_env.reward import compute_reward


def test_reward_normalized_bounds() -> None:
    truth = StockPrediction(
        gainers=["A", "B", "C", "D", "E"],
        losers=["V", "W", "X", "Y", "Z"],
        assets={"gold": "UP", "silver": "DOWN", "oil": "NEUTRAL"},
    )
    action = Action(
        gainers=["A", "B", "C", "D", "E"],
        losers=["V", "W", "X", "Y", "Z"],
        assets={"gold": "UP", "silver": "DOWN", "oil": "NEUTRAL"},
    )

    reward = compute_reward(action, truth)
    assert reward.value == 1.0


def test_approach_1_labeling_and_imbalance_trigger() -> None:
    articles = [
        NewsArticle(title="XOM and CVX gain as oil rises", description="gold steady"),
        NewsArticle(title="Oil outlook improves", description="XOM demand jumps"),
        NewsArticle(title="Market wrap", description="CVX sees inflow"),
        NewsArticle(title="Energy rally", description="oil and gold supported"),
        NewsArticle(title="Macro", description="reuters"),
        NewsArticle(title="Commodities", description="silver and oil"),
        NewsArticle(title="Airlines drop", description="AAL and DAL underperform"),
    ]
    truth = StockPrediction(
        gainers=["XOM", "CVX", "SLB", "HAL", "COP"],
        losers=["AAL", "DAL", "UAL", "LUV", "JBLU"],
        assets={"gold": "UP", "silver": "NEUTRAL", "oil": "UP"},
    )

    label = label_difficulty_approach_1(articles, truth)
    assert label == "easy"

    report = distribution_report(["easy", "easy", "easy", "medium"])
    assert should_recommend_hybrid(report) is True
