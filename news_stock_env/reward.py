from __future__ import annotations

from typing import Dict, List

from .data_types import Action, Reward, StockPrediction


# (10 movers x 2 max positional points) + (3 assets x 1)
MAX_RAW_REWARD = 23.0
MAX_WITH_MULTIPLIER = MAX_RAW_REWARD * 2.0


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
    progress = round(min(raw_total / MAX_RAW_REWARD, 1.0), 4)
    penalty = 0.0

    # Penalize repeated no-improvement loops.
    if repeat_count > 0:
        penalty += min(0.05 * repeat_count, 0.20)

    normalized = round(max(0.0, min(1.0, progress - penalty)), 4)

    return Reward(
        value=normalized,
        progress=progress,
        penalty=penalty,
        details={
            "gainers_score": gainers_score,
            "losers_score": losers_score,
            "assets_score": assets_score,
            "raw_total": raw_total,
            "max_total": MAX_RAW_REWARD,
        },
    )


if __name__ == "__main__":
    truth = StockPrediction(
        gainers=["A", "B", "C", "D", "E"],
        losers=["V", "W", "X", "Y", "Z"],
        assets={"gold": "UP", "silver": "DOWN", "oil": "NEUTRAL"},
    )

    perfect = Action(
        gainers=["A", "B", "C", "D", "E"],
        losers=["V", "W", "X", "Y", "Z"],
        assets={"gold": "UP", "silver": "DOWN", "oil": "NEUTRAL"},
    )
    perfect_reward = compute_reward(perfect, truth).value

    empty_like = Action(
        gainers=["Q1", "Q2", "Q3", "Q4", "Q5"],
        losers=["R1", "R2", "R3", "R4", "R5"],
        assets={"gold": "NEUTRAL", "silver": "UP", "oil": "DOWN"},
    )
    empty_reward = compute_reward(empty_like, truth).value

    assert perfect_reward == 1.0
    assert empty_reward == 0.0
    assert 0.0 <= perfect_reward <= 1.0
    assert 0.0 <= empty_reward <= 1.0

    print("reward smoke test passed")
