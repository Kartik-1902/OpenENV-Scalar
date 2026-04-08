from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

TRUSTED_DOMAINS = [
    "bbc.co.uk",
    "cnbc.com",
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "apnews.com",
    "theguardian.com",
    "forbes.com",
]

ASSET_KEYWORDS = [
    "gold",
    "silver",
    "oil",
    "crude",
    "rare earth",
    "copper",
    "natural gas",
    "lithium",
]

MACRO_KEYWORDS = [
    "war",
    "sanctions",
    "supply chain",
    "rate hike",
    "inflation",
    "recession",
    "federal reserve",
    "OPEC",
    "interest rate",
    "GDP",
    "unemployment",
    "trade war",
    "tariff",
    "scarcity",
    "demand shock",
    "geopolitical",
]

ALL_KEYWORDS = ASSET_KEYWORDS + MACRO_KEYWORDS

FIXED_ASSETS = {
    "gold": "GLD",
    "silver": "SLV",
    "oil": "USO",
}

SP500_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "BRK-B", "JPM",
    "XOM", "LLY", "UNH", "V", "MA", "PG", "JNJ", "COST", "HD", "MRK",
    "ABBV", "KO", "BAC", "PEP", "CVX", "ADBE", "WMT", "NFLX", "AMD", "CRM",
    "CSCO", "TMO", "ACN", "MCD", "DIS", "ABT", "LIN", "INTU", "CMCSA", "VZ",
    "TXN", "AMAT", "DHR", "QCOM", "NKE", "PFE", "ORCL", "INTC", "IBM", "CAT",
]

SHORT_TERM_ARTICLES_PER_DATE = 50
LONG_TERM_DAYS_BACK = 7
TOP_N_MOVERS = 5
RANDOM_SEED = 42


@dataclass(frozen=True)
class DifficultyImbalanceThreshold:
    min_share: float = 0.15
    max_share: float = 0.60


DIFFICULTY_IMBALANCE_THRESHOLD = DifficultyImbalanceThreshold()
