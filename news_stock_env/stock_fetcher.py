from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import yfinance as yf

from .config import FIXED_ASSETS, SP500_TICKERS, TOP_N_MOVERS
from .data_types import StockPrediction


def _pct_change(open_price: float, close_price: float) -> float:
    if open_price == 0:
        return 0.0
    return ((close_price - open_price) / open_price) * 100.0


def _day_change(ticker: str, start: str, end: str) -> float | None:
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        return None

    if "Open" not in data.columns or "Close" not in data.columns:
        return None

    open_price = float(data["Open"].iloc[0])
    close_price = float(data["Close"].iloc[-1])
    return _pct_change(open_price, close_price)


def _asset_direction(pct: float) -> str:
    if pct > 0.3:
        return "UP"
    if pct < -0.3:
        return "DOWN"
    return "NEUTRAL"


def get_stock_predictions(date_str: str) -> StockPrediction:
    date = datetime.strptime(date_str, "%Y-%m-%d")
    next_day = (date + timedelta(days=1)).strftime("%Y-%m-%d")

    changes: List[Tuple[str, float]] = []
    for ticker in SP500_TICKERS:
        try:
            pct = _day_change(ticker, date_str, next_day)
        except Exception:
            pct = None
        if pct is not None:
            changes.append((ticker, pct))

    if len(changes) < TOP_N_MOVERS * 2:
        raise RuntimeError("Insufficient ticker data for gainers/losers computation.")

    ordered = sorted(changes, key=lambda x: x[1], reverse=True)
    gainers = [ticker for ticker, _ in ordered[:TOP_N_MOVERS]]
    losers = [ticker for ticker, _ in ordered[-TOP_N_MOVERS:]]

    asset_result: Dict[str, str] = {}
    for asset_name, ticker in FIXED_ASSETS.items():
        try:
            pct = _day_change(ticker, date_str, next_day)
        except Exception:
            pct = None
        asset_result[asset_name] = _asset_direction(pct or 0.0)

    return StockPrediction(gainers=gainers, losers=losers, assets=asset_result)
