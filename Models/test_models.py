from __future__ import annotations

"""
Smoke test for the RunPod-hosted Shaer/Yehia endpoints.

Usage (PowerShell):
  $env:RUNPOD_API_KEY="..."
  $env:YEHIA_ENDPOINT_ID="..."
  $env:SHAER_ENDPOINT_ID="..."
  python models/test_models.py

If the OpenAI-compatible call 500s, the script falls back to RunPod's /runsync
REST call (same payload shape you used in the RunPod UI).
"""

import os
from pathlib import Path
from typing import Dict, List

import requests
from dotenv import load_dotenv
from openai import OpenAI


def build_client(endpoint_id: str, api_key: str) -> OpenAI:
    base_url = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"
    return OpenAI(api_key=api_key, base_url=base_url)


def openai_chat(client: OpenAI, model: str) -> str:
    messages = [
        {"role": "system", "content": "أنت شاعر عربي متمكن. أعطِ بيت شعر واحد فقط."},
        {"role": "user", "content": "اكتب بيت شعر عن الأمل بعد الشدة."},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=64,
        temperature=0.7,
        # OpenAI-compatible RunPod calls can be slow; allow ample time.
        timeout=180,
    )
    return resp.choices[0].message.content


def runpod_runsync(endpoint_id: str, api_key: str) -> str:
    """
    Call RunPod's /runsync API directly with the payload format shown in the UI.
    """
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    payload: Dict[str, object] = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "أنت شاعر عربي متمكن من مختلف البحور والأغراض الشعرية.",
                },
                {
                    "role": "user",
                    "content": "اكتب بيتًا شعريًا واحدًا عن الأمل بعد الشدّة.",
                },
            ],
            "sampling_params": {
                "temperature": 0.7,
                "max_tokens": 64,
            },
        }
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # End-to-end can take ~1–2 minutes; allow a generous timeout.
    resp = requests.post(url, json=payload, headers=headers, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    output = data.get("output") or []
    if output and isinstance(output, list):
        first = output[0]
        choices = first.get("choices") or []
        if choices and "tokens" in choices[0]:
            tokens: List[str] = choices[0]["tokens"]
            return "".join(tokens)
    return str(data)


if __name__ == "__main__":
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    load_dotenv(Path(__file__).with_name(".env"))

    api_key = os.getenv("RUNPOD_API_KEY")
    yehia_id = os.getenv("YEHIA_ENDPOINT_ID")
    shaer_id = os.getenv("SHAER_ENDPOINT_ID")
    model_name = os.getenv("MODEL_NAME", "shaer-7b-grpo")

    if not api_key or not yehia_id or not shaer_id:
        raise SystemExit(
            "Missing env vars. Set RUNPOD_API_KEY, YEHIA_ENDPOINT_ID, SHAER_ENDPOINT_ID."
        )

    # Shaer
    print("== Shaer endpoint ==")
    try:
        shaer_client = build_client(shaer_id, api_key)
        print(openai_chat(shaer_client, model_name))
    except Exception as e:
        print("OpenAI-compatible call failed:", e)
        print("Trying RunPod /runsync instead...")
        print(runpod_runsync(shaer_id, api_key))

    # Yehia
    print("\n== Yehia endpoint ==")
    try:
        yehia_client = build_client(yehia_id, api_key)
        print(openai_chat(yehia_client, model_name))
    except Exception as e:
        print("OpenAI-compatible call failed:", e)
        print("Trying RunPod /runsync instead...")
        print(runpod_runsync(yehia_id, api_key))
