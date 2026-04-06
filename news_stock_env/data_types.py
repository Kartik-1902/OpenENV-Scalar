from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field, field_validator


Direction = Literal["UP", "DOWN", "NEUTRAL"]
DifficultyLevel = Literal["easy", "medium", "hard"]


class NewsArticle(BaseModel):
    title: str = ""
    description: str = ""
    source: str = ""
    url: str = ""
    published_at: str = ""


class StockPrediction(BaseModel):
    gainers: List[str] = Field(default_factory=list, min_length=5, max_length=5)
    losers: List[str] = Field(default_factory=list, min_length=5, max_length=5)
    assets: Dict[str, Direction] = Field(default_factory=dict)

    @field_validator("assets")
    @classmethod
    def validate_assets(cls, value: Dict[str, Direction]) -> Dict[str, Direction]:
        expected = {"gold", "silver", "oil"}
        missing = expected - set(value)
        if missing:
            raise ValueError(f"assets missing keys: {sorted(missing)}")
        return value


class Observation(BaseModel):
    task_id: str
    date: str
    difficulty: DifficultyLevel
    instruction: str
    long_term_context: List[NewsArticle] = Field(default_factory=list)
    short_term_context: List[NewsArticle] = Field(default_factory=list)


class Action(BaseModel):
    gainers: List[str] = Field(default_factory=list, min_length=5, max_length=5)
    losers: List[str] = Field(default_factory=list, min_length=5, max_length=5)
    assets: Dict[str, Direction] = Field(default_factory=dict)


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    progress: float = Field(ge=0.0, le=1.0)
    penalty: float = Field(ge=0.0)
    details: Dict[str, float] = Field(default_factory=dict)


class TaskState(BaseModel):
    episode_id: str
    step_index: int = 0
    task_id: str
    date: str
    difficulty: DifficultyLevel
    done: bool = False


class DatasetRow(BaseModel):
    id: str
    date: str
    difficulty: DifficultyLevel
    long_term_context: List[NewsArticle]
    short_term_context: List[NewsArticle]
    stock_predictions: StockPrediction


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, str | float | int | bool]
