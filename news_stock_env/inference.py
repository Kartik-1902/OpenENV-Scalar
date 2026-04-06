from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from .config import GEMINI_API_KEY, OPENAI_API_KEY, RANDOM_SEED
from .data_types import Action, DatasetRow, NewsArticle, StockPrediction
from .env import NewsSignalEnv


def _build_client() -> OpenAI:
    if GEMINI_API_KEY:
        # Gemini supports OpenAI-compatible API endpoints.
        return OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    if OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY)

    raise ValueError("Set GEMINI_API_KEY (preferred) or OPENAI_API_KEY.")


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


def _predict_action(client: OpenAI, obs_text: str) -> Action:
    model_name = "gemini-2.0-flash" if GEMINI_API_KEY else "gpt-4o-mini"
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You predict top 5 gainers, top 5 losers and gold/silver/oil directions. "
                    "Return JSON only with keys gainers, losers, assets."
                ),
            },
            {"role": "user", "content": obs_text},
        ],
    )

    content = response.choices[0].message.content or "{}"
    payload: dict[str, Any] = json.loads(content)
    return Action(**payload)


def run_baseline(dataset_path: str, episodes: int, out_path: str) -> dict:
    random.seed(RANDOM_SEED)

    dataset = _load_dataset(dataset_path)
    if not dataset:
        raise RuntimeError("Dataset is empty.")

    random.shuffle(dataset)
    selected = dataset[: min(episodes, len(dataset))]

    env = NewsSignalEnv(dataset=selected)
    client = _build_client()

    results: list[dict] = []
    total = 0.0
    difficulty_totals: dict[str, float] = defaultdict(float)
    difficulty_counts: Counter[str] = Counter()

    for _ in range(len(selected)):
        obs = env.reset()
        obs_text = json.dumps(obs.model_dump())
        action = _predict_action(client, obs_text)
        step_result = env.step(action)

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
        "model": "gemini-2.0-flash" if GEMINI_API_KEY else "gpt-4o-mini",
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
