from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd

from .difficulty import label_difficulty_approach_1
from .news_fetcher import get_long_term_context, get_short_term_context
from .stock_fetcher import get_stock_predictions


def _date_range(start: str, end: str) -> List[str]:
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")
    days: List[str] = []

    cur = start_date
    while cur <= end_date:
        days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return days


def build_rows(start: str, end: str) -> list[dict]:
    rows: list[dict] = []
    for idx, date_str in enumerate(_date_range(start, end), start=1):
        try:
            truth = get_stock_predictions(date_str)
        except Exception:
            continue

        long_term = get_long_term_context(date_str)
        short_term = get_short_term_context(date_str)
        if not short_term:
            continue

        difficulty = label_difficulty_approach_1(short_term, truth)

        rows.append(
            {
                "id": f"task-{idx:05d}",
                "date": date_str,
                "difficulty": difficulty,
                "long_term_context": json.dumps([a.model_dump() for a in long_term]),
                "short_term_context": json.dumps([a.model_dump() for a in short_term]),
                "stock_predictions": json.dumps(truth.model_dump()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bundled news-stock OpenEnv dataset")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--out", default="data/dataset.csv", help="Output CSV path")
    args = parser.parse_args()

    rows = build_rows(args.start, args.end)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
