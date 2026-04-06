# OpenENV-Scalar

OpenENV-Scalar is a real-world OpenEnv environment for news-driven market signal prediction. The agent receives bundled long-term and short-term news context for a date, then predicts ranked equity movers and commodity direction labels.

This implementation is designed for hackathon submission constraints:
- Real-world simulation (financial/news analysis)
- Typed OpenEnv models and interface methods
- Three difficulty tiers with deterministic graders
- Reward in 0.0-1.0 with partial progress and penalties
- Baseline inference script with reproducibility controls
- Dockerized runtime suitable for Hugging Face Spaces

## Environment Description

Each task represents one trading date and includes:
- Long-term context: multi-day news bundle (relevancy-sorted)
- Short-term context: same-day news bundle (popularity-sorted, all fetched articles in one context block)
- Ground truth: top gainers, top losers, and direction labels for gold/silver/oil

The agent must output:
- Top 5 gainers (ranked)
- Top 5 losers (ranked)
- Asset directions for gold/silver/oil using UP, DOWN, or NEUTRAL

## OpenEnv Spec Coverage

- Typed models: Observation, Action, Reward, TaskState in [news_stock_env/data_types.py](news_stock_env/data_types.py)
- Environment API: reset, step, state in [news_stock_env/env.py](news_stock_env/env.py)
- Metadata: [openenv.yaml](openenv.yaml)

## Action and Observation Spaces

Observation fields:
- task_id: string
- date: string
- difficulty: easy | medium | hard
- instruction: string
- long_term_context: list of article objects
- short_term_context: list of article objects

Action fields:
- gainers: list of 5 ticker symbols
- losers: list of 5 ticker symbols
- assets: object with keys gold/silver/oil and values UP|DOWN|NEUTRAL

Reward fields:
- value: normalized total reward in [0.0, 1.0]
- progress: normalized pre-penalty score in [0.0, 1.0]
- penalty: non-negative penalty term
- details: component scores for diagnostics

## Difficulty Tasks and Grader

Current version implements only Approach 1 (signal density):
- easy: 6+ short-term articles reference movers/assets
- medium: 3-5 references
- hard: 0-2 references

Imbalance monitor:
- generate label distribution report with [news_stock_env/difficulty.py](news_stock_env/difficulty.py)
- if any class share is less than 15% or greater than 60%, flag recommendation to move to hybrid grading in next iteration

## Reward Design

Wordle-style positional reward:
- Ranked gainers/losers: +2 for correct position, +1 if present wrong position
- Assets: +1 for each correct direction

Normalization and penalties:
- Maximum raw score is 23
- value = clamp(progress - penalty, 0, 1)
- repeat/no-improvement actions are penalized incrementally

## Setup

1. Create virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configure environment variables using [.env.example](.env.example):
- NEWS_API_KEY
- GEMINI_API_KEY (preferred for this project)
- OPENAI_API_KEY (fallback support for baseline client)

## Usage

Build dataset:

```powershell
python -m news_stock_env.build_dataset --start 2024-01-01 --end 2024-01-31 --out data/dataset.csv
```

Validate the OpenEnv spec and deterministic fixture dataset:

```powershell
python -m news_stock_env.validate_openenv --dataset tests/fixtures/fixture_dataset.csv
```

Run baseline inference:

```powershell
python -m news_stock_env.inference --dataset data/dataset.csv --episodes 30 --out data/results.json
```

Run tests:

```powershell
python -m pytest -q
```

## Baseline Scores

Baseline output is stored in [data/results.json](data/results.json) when inference is run.
Because this depends on external APIs and date windows, commit a generated results file for reproducible submission evidence.
The baseline output now includes an Easy/Medium/Hard difficulty summary so you can audit task balance per run.

## Docker and Hugging Face Spaces

Local Docker build and run:

```powershell
docker build -t openenv-scalar .
docker run -p 7860:7860 --env NEWS_API_KEY=$env:NEWS_API_KEY --env GEMINI_API_KEY=$env:GEMINI_API_KEY openenv-scalar
```

Service endpoint:
- GET /health
- POST /evaluate

Files used:
- [Dockerfile](Dockerfile)
- [hf_space.yaml](hf_space.yaml)
- [news_stock_env/space_app.py](news_stock_env/space_app.py)

## Repository Layout

- [news_stock_env/config.py](news_stock_env/config.py)
- [news_stock_env/data_types.py](news_stock_env/data_types.py)
- [news_stock_env/news_fetcher.py](news_stock_env/news_fetcher.py)
- [news_stock_env/stock_fetcher.py](news_stock_env/stock_fetcher.py)
- [news_stock_env/difficulty.py](news_stock_env/difficulty.py)
- [news_stock_env/reward.py](news_stock_env/reward.py)
- [news_stock_env/env.py](news_stock_env/env.py)
- [news_stock_env/build_dataset.py](news_stock_env/build_dataset.py)
- [news_stock_env/inference.py](news_stock_env/inference.py)
- [openenv.yaml](openenv.yaml)

## Notes

- This version intentionally avoids local LLM runtime.
- Gemini is used through an OpenAI-compatible client path.
- Hybrid difficulty grading is not implemented yet; only imbalance detection is implemented in this release.