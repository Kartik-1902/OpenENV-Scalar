from __future__ import annotations

from typing import Dict, List

from .data_types import Action, Reward, StockPrediction


def _positional_score(predicted: List[str], truth: List[str]) -> float:
    score = 0.0
    for idx, ticker in enumerate(predicted):
        if ticker in truth:
            truth_idx = truth.index(ticker)
            score += 2.0 if truth_idx == idx else 1.0
    return score


def _asset_score(predicted_assets: Dict[str, str], truth_assets: Dict[str, str]) -> float:
    score = 0.0
    for asset in ["gold", "silver", "oil"]:
        if predicted_assets.get(asset) == truth_assets.get(asset):
            score += 1.0
    return score


def compute_reward(action: Action, truth: StockPrediction, repeat_count: int = 0) -> Reward:
    gainers_score = _positional_score(action.gainers, truth.gainers)
    losers_score = _positional_score(action.losers, truth.losers)
    assets_score = _asset_score(action.assets, truth.assets)

    raw_total = gainers_score + losers_score + assets_score
    max_total = 23.0

    progress = raw_total / max_total
    penalty = 0.0

    # Penalize repeated no-improvement loops.
    if repeat_count > 0:
        penalty += min(0.05 * repeat_count, 0.20)

    normalized = max(0.0, min(1.0, progress - penalty))

    return Reward(
        value=normalized,
        progress=max(0.0, min(1.0, progress)),
        penalty=penalty,
        details={
            "gainers_score": gainers_score,
            "losers_score": losers_score,
            "assets_score": assets_score,
            "raw_total": raw_total,
            "max_total": max_total,
        },
    )
