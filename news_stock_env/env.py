from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import List

from .data_types import Action, DatasetRow, Observation, StepResult, TaskState
from .reward import compute_reward


@dataclass
class NewsSignalEnv:
    """OpenEnv-style environment for bundled news-to-signal prediction."""

    dataset: List[DatasetRow]

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("dataset cannot be empty")

        self._cursor = 0
        self._state: TaskState | None = None
        self._repeat_count = 0
        self._last_action: Action | None = None

    def reset(self) -> Observation:
        row = self.dataset[self._cursor]
        self._cursor = (self._cursor + 1) % len(self.dataset)

        self._state = TaskState(
            episode_id=str(uuid.uuid4()),
            step_index=0,
            task_id=row.id,
            date=row.date,
            difficulty=row.difficulty,
            done=False,
        )
        self._repeat_count = 0
        self._last_action = None
        return self._build_observation(row)

    def step(self, action: Action) -> StepResult:
        if self._state is None:
            raise RuntimeError("reset() must be called before step().")
        if self._state.done:
            raise RuntimeError("Episode already done. Call reset().")

        row = self._get_active_row()
        if self._last_action is not None and action.model_dump() == self._last_action.model_dump():
            self._repeat_count += 1
        else:
            self._repeat_count = 0

        reward = compute_reward(action, row.stock_predictions, repeat_count=self._repeat_count)

        self._state.step_index += 1
        self._state.done = True
        self._last_action = deepcopy(action)

        observation = self._build_observation(row)
        info = {
            "episode_id": self._state.episode_id,
            "task_id": self._state.task_id,
            "difficulty": self._state.difficulty,
            "repeat_count": self._repeat_count,
        }

        return StepResult(observation=observation, reward=reward, done=True, info=info)

    def state(self) -> TaskState:
        if self._state is None:
            raise RuntimeError("Environment has no state yet. Call reset().")
        return self._state

    def _get_active_row(self) -> DatasetRow:
        if self._state is None:
            raise RuntimeError("Environment has no state yet. Call reset().")

        for row in self.dataset:
            if row.id == self._state.task_id:
                return row
        raise RuntimeError("Current task is not present in dataset.")

    def _build_observation(self, row: DatasetRow) -> Observation:
        return Observation(
            task_id=row.id,
            date=row.date,
            difficulty=row.difficulty,
            instruction=(
                "Predict top-5 gainers, top-5 losers, and UP/DOWN/NEUTRAL for gold, silver, oil. "
                "Return strictly structured output."
            ),
            long_term_context=row.long_term_context,
            short_term_context=row.short_term_context,
        )
