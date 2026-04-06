from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .data_types import Action, DatasetRow, NewsArticle, StockPrediction
from .env import NewsSignalEnv


def _load_fixture_rows(dataset_path: str) -> list[DatasetRow]:
    import pandas as pd

    frame = pd.read_csv(dataset_path)
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


def validate_openenv_spec(spec_path: str) -> dict:
    doc = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    required = ["name", "entrypoint", "spec"]
    for key in required:
        if key not in doc:
            raise ValueError(f"missing key: {key}")
    return doc


def run_validation(dataset_path: str, spec_path: str = "openenv.yaml") -> dict:
    validate_openenv_spec(spec_path)
    rows = _load_fixture_rows(dataset_path)
    if not rows:
        raise ValueError("fixture dataset is empty")

    env = NewsSignalEnv(dataset=rows)
    observation = env.reset()
    perfect_truth = rows[0].stock_predictions
    action = Action(
        gainers=perfect_truth.gainers,
        losers=perfect_truth.losers,
        assets=perfect_truth.assets,
    )
    result = env.step(action)

    return {
        "rows": len(rows),
        "first_task_id": observation.task_id,
        "reward": result.reward.value,
        "difficulty": observation.difficulty,
        "spec_path": spec_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OpenEnv spec and fixture dataset")
    parser.add_argument("--dataset", default="tests/fixtures/fixture_dataset.csv")
    parser.add_argument("--spec", default="openenv.yaml")
    args = parser.parse_args()

    result = run_validation(args.dataset, args.spec)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
