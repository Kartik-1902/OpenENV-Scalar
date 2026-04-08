from __future__ import annotations

import json
import os
import uuid

import requests
from openai import OpenAI


API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
HF_TOKEN = os.environ["HF_TOKEN"]
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

SYSTEM_PROMPT = (
    "You are a financial news analyst. Given the market context and today's articles, "
    "predict which assets will be affected. Output ONLY valid JSON with keys: "
    "'gainers' (list of up to 5 tickers), 'losers' (list of up to 5 tickers), "
    "'assets' (dict with gold/silver/oil each set to UP, DOWN, or NEUTRAL). "
    "No explanation. No numbers. JSON only."
)

DIFFICULTIES = ["Easy", "Medium", "Hard"]
MAX_STEPS = 3
REQUEST_TIMEOUT = 20


def _json_line(prefix: str, payload: dict) -> None:
    print(f"{prefix} {json.dumps(payload)}", flush=True)


def _extract_reward(step_payload: dict) -> float:
    reward = step_payload.get("reward", 0.0)
    if isinstance(reward, dict):
        return float(reward.get("value", 0.0))
    try:
        return float(reward)
    except Exception:
        return 0.0


def _build_prompt(observation: dict) -> str:
    return json.dumps(observation, ensure_ascii=False)


def _llm_action(client: OpenAI, observation: dict) -> str:
    prompt = _build_prompt(observation)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0.1,
    )
    return (response.choices[0].message.content or "").strip()


def _run_task(client: OpenAI, difficulty: str) -> float:
    task_id = f"{difficulty.lower()}_{uuid.uuid4().hex[:8]}"

    try:
        reset_response = requests.post(
            f"{ENV_URL}/reset",
            json={"difficulty": difficulty},
            timeout=REQUEST_TIMEOUT,
        )
        reset_response.raise_for_status()
        observation = reset_response.json()

        _json_line("[START]", {"task_id": task_id, "difficulty": difficulty})

        total_reward = 0.0
        steps_taken = 0

        for step_idx in range(1, MAX_STEPS + 1):
            action = _llm_action(client, observation)
            step_response = requests.post(
                f"{ENV_URL}/step",
                json={"action": action},
                timeout=REQUEST_TIMEOUT,
            )
            step_response.raise_for_status()
            step_payload = step_response.json()

            reward_value = _extract_reward(step_payload)
            done = bool(step_payload.get("done", False))
            observation = step_payload.get("observation", observation)

            total_reward += reward_value
            steps_taken = step_idx

            _json_line(
                "[STEP]",
                {
                    "step": step_idx,
                    "action": action,
                    "reward": reward_value,
                    "done": done,
                },
            )

            if done:
                break

        _json_line(
            "[END]",
            {
                "task_id": task_id,
                "total_reward": total_reward,
                "steps": steps_taken,
            },
        )
        return total_reward
    except Exception:
        _json_line(
            "[END]",
            {
                "task_id": task_id,
                "total_reward": 0.0,
                "steps": 0,
            },
        )
        return 0.0


def main() -> None:
    health_response = requests.get(f"{ENV_URL}/health", timeout=REQUEST_TIMEOUT)
    health_response.raise_for_status()
    health_payload = health_response.json()
    if health_payload.get("status") != "ok":
        raise RuntimeError("Environment health check failed")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    rewards: dict[str, float] = {}
    for difficulty in DIFFICULTIES:
        rewards[difficulty] = _run_task(client, difficulty)

    easy_reward = rewards.get("Easy", 0.0)
    medium_reward = rewards.get("Medium", 0.0)
    hard_reward = rewards.get("Hard", 0.0)
    print(f"SUMMARY: Easy={easy_reward:.4f} Medium={medium_reward:.4f} Hard={hard_reward:.4f}", flush=True)


if __name__ == "__main__":
    main()
