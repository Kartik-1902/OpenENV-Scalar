from __future__ import annotations

import logging
import os

from openai import OpenAI


logger = logging.getLogger(__name__)


def call_llm(messages: list[dict], system_prompt: str = None) -> str:
    """Call an OpenAI-compatible chat completion endpoint and return text output."""
    try:
        api_base_url = os.environ["API_BASE_URL"]
        model_name = os.environ["MODEL_NAME"]
        hf_token = os.environ["HF_TOKEN"]
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(
            f"Missing required environment variable: {missing}. "
            "Required: API_BASE_URL, MODEL_NAME, HF_TOKEN."
        ) from exc

    if not api_base_url or not model_name or not hf_token:
        raise ValueError(
            "Environment variables API_BASE_URL, MODEL_NAME, and HF_TOKEN must all be non-empty."
        )

    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    full_messages: list[dict] = []
    if system_prompt is not None:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=full_messages,
            max_tokens=1000,
            temperature=0.1,
        )
        content = response.choices[0].message.content
        return (content or "").strip()
    except Exception:
        logger.exception("LLM call failed")
        return ""


if __name__ == "__main__":
    result = call_llm([
        {"role": "user", "content": "Say hello in one word."}
    ])
    print(result)
