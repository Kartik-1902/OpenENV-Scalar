from __future__ import annotations

from collections import Counter
from typing import Dict, List

from .config import DIFFICULTY_IMBALANCE_THRESHOLD, FIXED_ASSETS
from .data_types import DifficultyLevel, NewsArticle, StockPrediction


def _article_text(article: NewsArticle) -> str:
    return f"{article.title} {article.description}".lower()


def _match_count(articles: List[NewsArticle], symbols: List[str]) -> int:
    count = 0
    for article in articles:
        text = _article_text(article)
        if any(symbol.lower() in text for symbol in symbols):
            count += 1
    return count


def label_difficulty_approach_1(articles: List[NewsArticle], truth: StockPrediction) -> DifficultyLevel:
    symbols = truth.gainers + truth.losers + list(FIXED_ASSETS.keys())
    matches = _match_count(articles, symbols)

    if matches >= 6:
        return "easy"
    if matches >= 3:
        return "medium"
    return "hard"


def distribution_report(labels: List[DifficultyLevel]) -> Dict[str, float]:
    if not labels:
        return {"easy": 0.0, "medium": 0.0, "hard": 0.0}

    total = float(len(labels))
    counts = Counter(labels)
    return {
        "easy": counts.get("easy", 0) / total,
        "medium": counts.get("medium", 0) / total,
        "hard": counts.get("hard", 0) / total,
    }


def should_recommend_hybrid(report: Dict[str, float]) -> bool:
    min_share = DIFFICULTY_IMBALANCE_THRESHOLD.min_share
    max_share = DIFFICULTY_IMBALANCE_THRESHOLD.max_share
    return any(v < min_share or v > max_share for v in report.values())
