from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
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


def _extract_change_from_batch(data: pd.DataFrame, ticker: str) -> float | None:
    if data.empty:
        return None

    try:
        if isinstance(data.columns, pd.MultiIndex):
            # Common shape with group_by="ticker": first level is ticker symbol.
            if ticker in data.columns.get_level_values(0):
                ticker_data = data[ticker]
            # Fallback shape: first level is OHLC field and second level is ticker.
            elif ticker in data.columns.get_level_values(1):
                ticker_data = data.xs(ticker, axis=1, level=1)
            else:
                return None

            if "Open" not in ticker_data.columns or "Close" not in ticker_data.columns:
                return None

            open_price = float(ticker_data["Open"].iloc[0])
            close_price = float(ticker_data["Close"].iloc[-1])
            if pd.isna(open_price) or pd.isna(close_price):
                return None
            return _pct_change(open_price, close_price)

        # Single-ticker dataframe fallback.
        if "Open" in data.columns and "Close" in data.columns:
            open_price = float(data["Open"].iloc[0])
            close_price = float(data["Close"].iloc[-1])
            if pd.isna(open_price) or pd.isna(close_price):
                return None
            return _pct_change(open_price, close_price)
    except Exception:
        return None

    return None


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
    batch = yf.download(
        SP500_TICKERS,
        start=date_str,
        end=next_day,
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=False,
    )
    for ticker in SP500_TICKERS:
        pct = _extract_change_from_batch(batch, ticker)
        if pct is not None:
            changes.append((ticker, pct))

    if len(changes) < TOP_N_MOVERS * 2:
        raise RuntimeError("Insufficient ticker data for gainers/losers computation.")

    ordered = sorted(changes, key=lambda x: x[1], reverse=True)
    gainers = [ticker for ticker, _ in ordered[:TOP_N_MOVERS]]
    losers = [ticker for ticker, _ in ordered[-TOP_N_MOVERS:]]

    asset_result: Dict[str, str] = {}
    asset_tickers = list(FIXED_ASSETS.values())
    asset_batch = yf.download(
        asset_tickers,
        start=date_str,
        end=next_day,
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=False,
    )
    for asset_name, ticker in FIXED_ASSETS.items():
        pct = _extract_change_from_batch(asset_batch, ticker)
        asset_result[asset_name] = _asset_direction(pct or 0.0)

    return StockPrediction(gainers=gainers, losers=losers, assets=asset_result)
