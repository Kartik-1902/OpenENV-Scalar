from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from llm_client import call_llm

from .config import RANDOM_SEED
from .data_types import Action, DatasetRow, NewsArticle, StockPrediction
from .env import NewsSignalEnv


def _load_dataset(path: str) -> list[DatasetRow]:
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


def _predict_action(obs_text: str) -> Action:
    content = call_llm(
        [{"role": "user", "content": obs_text}],
        system_prompt=(
            "You predict top 5 gainers, top 5 losers and gold/silver/oil directions. "
            "Return JSON only with keys gainers, losers, assets."
        ),
    )
    content = content or "{}"
    payload: dict[str, Any] = json.loads(content)
    return Action(**payload)


def _compact_observation_text(obs: dict[str, Any], max_articles: int = 12, max_chars: int = 6000) -> str:
    long_articles = obs.get("long_term_context", [])[:max_articles]
    short_articles = obs.get("short_term_context", [])[:max_articles]

    def _titles(items: list[dict[str, Any]]) -> list[str]:
        return [str(item.get("title", "")).strip() for item in items if item.get("title")]

    compact = {
        "task_id": obs.get("task_id"),
        "date": obs.get("date"),
        "difficulty": obs.get("difficulty"),
        "instruction": obs.get("instruction"),
        "long_term_titles": _titles(long_articles),
        "short_term_titles": _titles(short_articles),
        "schema": {
            "gainers": "list[str] length 5",
            "losers": "list[str] length 5",
            "assets": {"gold": "UP|DOWN|NEUTRAL", "silver": "UP|DOWN|NEUTRAL", "oil": "UP|DOWN|NEUTRAL"},
        },
    }
    text = json.dumps(compact)
    return text[:max_chars]


def _predict_with_retries(obs_text: str, retries: int = 4) -> Action:
    delay = 2.0
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            return _predict_action(obs_text)
        except Exception as exc:
            last_error = exc
            # Handle provider throttling with bounded exponential backoff.
            if "429" in str(exc) and attempt < retries:
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)
                continue
            raise

    raise RuntimeError(f"prediction failed after retries: {last_error}")


def run_baseline(dataset_path: str, episodes: int, out_path: str) -> dict:
    random.seed(RANDOM_SEED)

    dataset = _load_dataset(dataset_path)
    if not dataset:
        raise RuntimeError("Dataset is empty.")

    random.shuffle(dataset)
    selected = dataset[: min(episodes, len(dataset))]

    env = NewsSignalEnv(dataset=selected)

    results: list[dict] = []
    total = 0.0
    difficulty_totals: dict[str, float] = defaultdict(float)
    difficulty_counts: Counter[str] = Counter()

    for _ in range(len(selected)):
        obs = env.reset()
        obs_text = _compact_observation_text(obs.model_dump())
        try:
            action = _predict_with_retries(obs_text)
            step_result = env.step(action)
        except Exception as exc:
            # Keep baseline runs reproducible when provider quota/rate limits are hit.
            results.append(
                {
                    "task_id": obs.task_id,
                    "date": obs.date,
                    "difficulty": obs.difficulty,
                    "reward": 0.0,
                    "progress": 0.0,
                    "penalty": 0.0,
                    "error": str(exc),
                }
            )
            difficulty_totals[obs.difficulty] += 0.0
            difficulty_counts[obs.difficulty] += 1
            continue

        total += step_result.reward.value
        difficulty_totals[obs.difficulty] += step_result.reward.value
        difficulty_counts[obs.difficulty] += 1
        results.append(
            {
                "task_id": obs.task_id,
                "date": obs.date,
                "difficulty": obs.difficulty,
                "reward": step_result.reward.value,
                "progress": step_result.reward.progress,
                "penalty": step_result.reward.penalty,
            }
        )

    aggregate = {
        "episodes": len(selected),
        "average_reward": total / len(selected),
        "seed": RANDOM_SEED,
        "provider": "llm_client",
        "model": os.getenv("MODEL_NAME", ""),
        "difficulty_summary": {
            difficulty: {
                "count": difficulty_counts.get(difficulty, 0),
                "average_reward": (
                    difficulty_totals[difficulty] / difficulty_counts[difficulty]
                    if difficulty_counts[difficulty]
                    else 0.0
                ),
            }
            for difficulty in ["easy", "medium", "hard"]
        },
        "runs": results,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline model inference")
    parser.add_argument("--dataset", default="data/dataset.csv")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--out", default="data/results.json")
    args = parser.parse_args()

    result = run_baseline(args.dataset, args.episodes, args.out)
    print(f"average_reward={result['average_reward']:.4f} episodes={result['episodes']}")


if __name__ == "__main__":
    main()
