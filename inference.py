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
REQUEST_TIMEOUT = 30


def _json_line(prefix: str, payload: dict) -> None:
    print(f"{prefix} {json.dumps(payload)}", flush=True)


def _build_prompt(state_payload: dict) -> str:
    # Keep context compact so router models with smaller limits can respond.
    state_obj = state_payload.get("observation", state_payload)

    def _compact_articles(name: str, articles: list, limit: int = 8) -> dict:
        compact = []
        for article in (articles or [])[:limit]:
            if isinstance(article, dict):
                compact.append(
                    {
                        "title": str(article.get("title", ""))[:140],
                        "source": str(article.get("source", ""))[:50],
                    }
                )
            else:
                compact.append(str(article)[:140])
        return {name: compact}

    payload = {
        "task_id": state_obj.get("task_id", state_payload.get("task_id")),
        "date": state_obj.get("date", state_payload.get("date")),
        "difficulty": state_obj.get("difficulty", state_payload.get("difficulty")),
    }
    payload.update(_compact_articles("long_term_context", state_obj.get("long_term_context", []), limit=8))
    payload.update(_compact_articles("short_term_context", state_obj.get("short_term_context", []), limit=8))

    text = json.dumps(payload, ensure_ascii=False)
    # Hard cap prompt length to protect against token limit errors.
    return text[:4500]


def _safe_action_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return '{"gainers":[],"losers":[],"assets":{"gold":"NEUTRAL","silver":"NEUTRAL","oil":"NEUTRAL"}}'

    # Strip markdown fences if model wraps JSON in code blocks.
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last + 1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    return '{"gainers":[],"losers":[],"assets":{"gold":"NEUTRAL","silver":"NEUTRAL","oil":"NEUTRAL"}}'


def _llm_action(client: OpenAI, observation: dict) -> str:
    prompt = _build_prompt(observation)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.1,
    )
    return _safe_action_text(response.choices[0].message.content or "")


def _run_task(client: OpenAI, difficulty: str) -> float:
    generated_task_id = f"{difficulty.lower()}_{uuid.uuid4().hex[:8]}"

    try:
        reset_response = requests.post(
            f"{ENV_URL}/reset",
            json={"difficulty": difficulty},
            timeout=REQUEST_TIMEOUT,
        )
        reset_response.raise_for_status()
        reset_data = reset_response.json()
        task_id = reset_data.get("task_id", generated_task_id)

        _json_line("[START]", {"task_id": task_id, "difficulty": difficulty})

        state_response = requests.get(f"{ENV_URL}/state", timeout=REQUEST_TIMEOUT)
        state_response.raise_for_status()
        state_data = state_response.json()
        print(f"DEBUG state keys: {list(state_data.keys())}", flush=True)

        total_reward = 0.0
        steps_taken = 0
        done = False

        for step_idx in range(1, MAX_STEPS + 1):
            if done:
                break

            action = _llm_action(client, state_data)
            step_response = requests.post(
                f"{ENV_URL}/step",
                json={"action": action},
                timeout=REQUEST_TIMEOUT,
            )
            step_response.raise_for_status()
            step_data = step_response.json()

            reward_value = float(step_data.get("reward", 0.0))
            done = bool(step_data.get("done", True))

            total_reward += reward_value
            steps_taken = step_idx

            _json_line(
                "[STEP]",
                {
                    "step": step_idx,
                    "action": action[:80],
                    "reward": reward_value,
                    "done": done,
                },
            )

            if done:
                break

            state_response = requests.get(f"{ENV_URL}/state", timeout=REQUEST_TIMEOUT)
            state_response.raise_for_status()
            state_data = state_response.json()
            print(f"DEBUG state keys: {list(state_data.keys())}", flush=True)

        _json_line(
            "[END]",
            {
                "task_id": task_id,
                "total_reward": total_reward,
                "steps": steps_taken,
            },
        )
        return total_reward
    except Exception as exc:
        print(f"ERROR run_task({difficulty}): {exc}", flush=True)
        _json_line(
            "[END]",
            {
                "task_id": generated_task_id,
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
