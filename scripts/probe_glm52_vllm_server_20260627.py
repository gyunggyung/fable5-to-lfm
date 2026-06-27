#!/usr/bin/env python3
"""Small OpenAI-compatible probe for a running GLM-5.2-FP8 vLLM server."""

from __future__ import annotations

import argparse
import json
import time
from urllib import request


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="glm-5.2-fp8")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    prompts = [
        "Return a JSON object with a shell command to list changed git files. Keep it concise.",
        "You are a terminal coding agent. Diagnose why a Python test may fail after a config rename.",
        "Choose the right tool call for checking a telecom account's current plan and explain briefly.",
    ]

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    for idx, prompt in enumerate(prompts, start=1):
        started = time.time()
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": args.max_tokens,
        }
        result = post_json(url, payload)
        elapsed = time.time() - started
        content = result["choices"][0]["message"].get("content", "")
        print(f"probe={idx} elapsed={elapsed:.2f}s chars={len(content)}")
        print(content[:1200])
        print("---")


if __name__ == "__main__":
    main()
