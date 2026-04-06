from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from newsapi import NewsApiClient

from .config import ALL_KEYWORDS, NEWS_API_KEY, SHORT_TERM_ARTICLES_PER_DATE, TRUSTED_DOMAINS
from .data_types import NewsArticle


def _to_articles(raw_articles: list[dict]) -> List[NewsArticle]:
    normalized: List[NewsArticle] = []
    for article in raw_articles:
        source = (article.get("source") or {}).get("name", "") if isinstance(article.get("source"), dict) else ""
        normalized.append(
            NewsArticle(
                title=article.get("title") or "",
                description=article.get("description") or "",
                source=source,
                url=article.get("url") or "",
                published_at=article.get("publishedAt") or "",
            )
        )
    return normalized


def _query() -> str:
    return " OR ".join(ALL_KEYWORDS)


def _client() -> NewsApiClient:
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY is missing. Set it in environment variables.")
    return NewsApiClient(api_key=NEWS_API_KEY)


def get_long_term_context(date_str: str, days_back: int = 7, page_size: int = 30) -> List[NewsArticle]:
    date = datetime.strptime(date_str, "%Y-%m-%d")
    start = (date - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end = date.strftime("%Y-%m-%d")

    try:
        result = _client().get_everything(
            q=_query(),
            from_param=start,
            to=end,
            language="en",
            sort_by="relevancy",
            domains=",".join(TRUSTED_DOMAINS),
            page_size=page_size,
            page=1,
        )
    except Exception:
        return []

    articles = result.get("articles", []) if isinstance(result, dict) else []
    return _to_articles(articles)


def get_short_term_context(date_str: str) -> List[NewsArticle]:
    try:
        result = _client().get_everything(
            q=_query(),
            from_param=date_str,
            to=date_str,
            language="en",
            sort_by="popularity",
            domains=",".join(TRUSTED_DOMAINS),
            page_size=SHORT_TERM_ARTICLES_PER_DATE,
            page=1,
        )
    except Exception:
        return []

    articles = result.get("articles", []) if isinstance(result, dict) else []
    return _to_articles(articles)
