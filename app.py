from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from news_stock_env.data_types import Action, DatasetRow, NewsArticle, StockPrediction
from news_stock_env.env import NewsSignalEnv


app = FastAPI(title="OpenENV Scalar News", version="0.1.0")


class ResetRequest(BaseModel):
    difficulty: str | None = None
    seed: int | None = None


class StepRequest(BaseModel):
    action: str


def _load_rows(dataset_path: str) -> list[DatasetRow]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    frame = pd.read_csv(path)
    rows: list[DatasetRow] = []
    for _, rec in frame.iterrows():
        rows.append(
            DatasetRow(
                id=str(rec["id"]),
                date=str(rec["date"]),
                difficulty=str(rec["difficulty"]),
                long_term_context=[NewsArticle(**a) for a in json.loads(rec["long_term_context"])],
                short_term_context=[NewsArticle(**a) for a in json.loads(rec["short_term_context"])],
                stock_predictions=StockPrediction(**json.loads(rec["stock_predictions"])),
            )
        )
    if not rows:
        raise ValueError("dataset has no rows")
    return rows


_ROWS = _load_rows("data/dataset_submission.csv")
_ENV = NewsSignalEnv(dataset=_ROWS)
_ROWS_BY_TASK_ID = {row.id: row for row in _ROWS}
_LAST_RESPONSE: dict[str, Any] | None = None


def _to_title_case(level: str) -> str:
    return level[:1].upper() + level[1:].lower() if level else level


def _observation_payload(obs: Any, override_difficulty: str | None = None) -> dict[str, Any]:
    difficulty = override_difficulty if override_difficulty else _to_title_case(str(obs.difficulty))
    return {
        "task_id": obs.task_id,
        "long_term_context": [a.model_dump() for a in obs.long_term_context],
        "short_term_context": [a.model_dump() for a in obs.short_term_context],
        "date": obs.date,
        "difficulty": difficulty,
    }


def _reset_for_difficulty(difficulty: str | None) -> tuple[Any, str]:
    if not difficulty:
        obs = _ENV.reset()
        return obs, _to_title_case(str(obs.difficulty))

    normalized = difficulty.strip().lower()
    for _ in range(len(_ROWS)):
        obs = _ENV.reset()
        if str(obs.difficulty).lower() == normalized:
            return obs, _to_title_case(difficulty)

    # Fallback if requested difficulty is not represented in the current dataset.
    obs = _ENV.reset()
    return obs, _to_title_case(difficulty)


def _fallback_action() -> Action:
    return Action(
        gainers=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
        losers=["INTC", "IBM", "ORCL", "NKE", "PFE"],
        assets={"gold": "NEUTRAL", "silver": "NEUTRAL", "oil": "NEUTRAL"},
    )


def _parse_action(raw_action: str) -> Action:
    try:
        payload = json.loads(raw_action)
        return Action(**payload)
    except Exception:
        return _fallback_action()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest | None = None) -> dict[str, Any]:
    global _LAST_RESPONSE

    req = req or ResetRequest()
    if req.seed is not None:
        random.seed(req.seed)

    obs, requested_difficulty = _reset_for_difficulty(req.difficulty)
    payload = {
        "observation": _observation_payload(obs, override_difficulty=requested_difficulty),
        "done": False,
        "reward": 0.0,
    }
    _LAST_RESPONSE = payload
    return payload


@app.post("/step")
def step(req: StepRequest) -> dict[str, Any]:
    global _LAST_RESPONSE

    action = _parse_action(req.action)
    result = _ENV.step(action)
    row = _ROWS_BY_TASK_ID.get(result.observation.task_id)

    payload = {
        "observation": _observation_payload(result.observation),
        "reward": float(result.reward.value),
        "done": bool(result.done),
        "info": {
            "ground_truth": row.stock_predictions.model_dump() if row else {},
            "difficulty": _to_title_case(str(result.observation.difficulty)),
        },
    }
    _LAST_RESPONSE = payload
    return payload


@app.get("/state")
def state() -> dict[str, Any]:
    if _LAST_RESPONSE is None:
        return {"status": "no_episode_started"}
    return _LAST_RESPONSE
