#!/usr/bin/env python3
"""Small checkpoint-specific vision, tools, and reasoning acceptance battery."""

import json
import os
import sys
import urllib.request

PORT = os.environ.get("PORT", "18020")
KEY = os.environ["VLLM_API_KEY"]
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
TEXT_ONLY = os.environ.get("TEXT_ONLY", "0").lower() in {"1", "true", "yes"}


def chat(payload):
    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def require(name, condition, detail):
    if not condition:
        print(f"FAIL {name}: {detail}", file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS {name}: {detail}")


base = {
    "model": "qwen3.8-27b",
    "temperature": 0,
    "max_tokens": 128,
}

plain = chat(
    base
    | {
        "messages": [{"role": "user", "content": "Reply with exactly READY."}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
)
plain_text = plain["choices"][0]["message"].get("content") or ""
require("text", "READY" in plain_text.upper(), plain_text[:100])

thinking = chat(
    base
    | {
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "What is 37 * 19? Answer briefly."}],
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
    }
)
thinking_message = thinking["choices"][0]["message"]
reasoning = thinking_message.get("reasoning_content") or thinking_message.get("reasoning") or ""
require("reasoning", "703" in (thinking_message.get("content") or "") and bool(reasoning), "37 * 19 = 703")

tool = chat(
    base
    | {
        "messages": [
            {
                "role": "user",
                "content": "You must call get_weather for Rome, Italy. Do not answer directly.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "chat_template_kwargs": {"enable_thinking": False},
    }
)
calls = tool["choices"][0]["message"].get("tool_calls") or []
require("tools", bool(calls) and calls[0]["function"]["name"] == "get_weather", str(calls)[:180])

if TEXT_ONLY:
    print("SKIP vision: text-only server profile")
else:
    # One opaque red PNG. The gate verifies that the vision path is loaded and
    # returns a normal assistant response; detailed visual accuracy is benchmarked separately.
    red_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nGQAAAAASUVORK5CYII="
    vision = chat(
        base
        | {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Briefly describe this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{red_png}"}},
                    ],
                }
            ],
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    vision_text = vision["choices"][0]["message"].get("content") or ""
    require("vision", bool(vision_text.strip()), vision_text[:120])
