from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .data_types import Action, DatasetRow, NewsArticle, StockPrediction
from .env import NewsSignalEnv

app = FastAPI(title="OpenENV Scalar News", version="0.1.0")


class EvaluateRequest(BaseModel):
    dataset_path: str = "data/dataset.csv"
    task_index: int = 0
    action: dict[str, Any]


def _load_rows(dataset_path: str) -> list[DatasetRow]:
    import pandas as pd

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
    return rows


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate(req: EvaluateRequest) -> dict[str, Any]:
    try:
        rows = _load_rows(req.dataset_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not rows:
        raise HTTPException(status_code=400, detail="dataset has no rows")

    idx = req.task_index % len(rows)
    env = NewsSignalEnv(dataset=[rows[idx]])
    observation = env.reset()

    try:
        action = Action(**req.action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid action: {exc}") from exc

    result = env.step(action)
    return {
        "observation": observation.model_dump(),
        "reward": result.reward.model_dump(),
        "done": result.done,
        "info": result.info,
    }
