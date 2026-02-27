"""Fake inference runner for E2E tests.

Connects to the backend via WebSocket (runner protocol) and serves
HTTP /v1/chat/completions with canned responses. Configurable via env vars.
"""

import asyncio
import json
import os
import time
import uuid

import aiohttp
from aiohttp import web
import websockets

# Config from env vars
RUNNER_ID = os.environ.get("RUNNER_ID", "fake-runner-1")
RUNNER_NAME = os.environ.get("RUNNER_NAME", "Fake Runner")
MACHINE_TYPE = os.environ.get("MACHINE_TYPE", "gpu-server")
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8080"))
BACKEND_WS_URL = os.environ.get("BACKEND_WS_URL", "ws://backend:8080/ws/runners")
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "test-runner-secret")
MODELS = [m.strip() for m in os.environ.get("MODELS", "llama3:8b").split(",") if m.strip()]
MAC_ADDRESS = os.environ.get("MAC_ADDRESS", "AA:BB:CC:DD:EE:FF")
STARTUP_DELAY = float(os.environ.get("STARTUP_DELAY", "0"))
RESPONSE_DELAY = float(os.environ.get("RESPONSE_DELAY", "0"))
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", "30"))


def build_status():
    """Build RunnerStatus matching protocol.rs RunnerStatus struct."""
    return {
        "health": "healthy",
        "capabilities": [],
        "engines": [
            {
                "engine_type": "ollama",
                "is_healthy": True,
                "version": None,
                "loaded_models": MODELS,
                "available_models": [
                    {"id": m, "name": m, "size_bytes": None, "parameter_count": None}
                    for m in MODELS
                ],
                "error": None,
                "batch_size": 1,
            }
        ],
        "metrics": None,
        "model_aliases": {},
    }


def build_register_message():
    """Build RunnerMessage::Register matching protocol.rs."""
    return {
        "type": "register",
        "runner_id": RUNNER_ID,
        "runner_name": RUNNER_NAME,
        "machine_type": MACHINE_TYPE,
        "http_port": HTTP_PORT,
        "protocol_version": 1,
        "auth_token": AUTH_TOKEN,
        "mac_address": MAC_ADDRESS,
        "status": build_status(),
    }


async def websocket_client():
    """Connect to backend WebSocket and maintain registration."""
    if STARTUP_DELAY > 0:
        print(f"[ws] Delaying startup by {STARTUP_DELAY}s")
        await asyncio.sleep(STARTUP_DELAY)

    while True:
        try:
            print(f"[ws] Connecting to {BACKEND_WS_URL}")
            async with websockets.connect(BACKEND_WS_URL) as ws:
                # Send registration
                reg = build_register_message()
                await ws.send(json.dumps(reg))
                print(f"[ws] Sent register as {RUNNER_ID} ({MACHINE_TYPE})")

                # Wait for RegisterAck
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                msg = json.loads(raw)
                if msg.get("type") == "register_ack":
                    print(f"[ws] Registered successfully: {msg}")
                elif msg.get("type") == "error":
                    print(f"[ws] Registration error: {msg}")
                    await asyncio.sleep(5)
                    continue
                else:
                    print(f"[ws] Unexpected message: {msg}")

                # Main loop: heartbeats + handle gateway messages
                async def send_heartbeats():
                    while True:
                        await asyncio.sleep(HEARTBEAT_INTERVAL)
                        hb = {"type": "heartbeat", **build_status()}
                        await ws.send(json.dumps(hb))
                        print("[ws] Sent heartbeat")

                async def handle_messages():
                    async for raw_msg in ws:
                        msg = json.loads(raw_msg)
                        msg_type = msg.get("type")
                        if msg_type == "ping":
                            print(f"[ws] Received ping: {msg.get('timestamp')}")
                        elif msg_type == "request_status":
                            resp = {
                                "type": "command_response",
                                "request_id": msg["request_id"],
                                "success": True,
                                "error": None,
                                "status": build_status(),
                            }
                            await ws.send(json.dumps(resp))
                        elif msg_type == "load_model":
                            print(f"[ws] Load model request: {msg.get('model_id')}")
                            resp = {
                                "type": "command_response",
                                "request_id": msg["request_id"],
                                "success": True,
                                "error": None,
                                "status": build_status(),
                            }
                            await ws.send(json.dumps(resp))
                        else:
                            print(f"[ws] Received: {msg_type}")

                hb_task = asyncio.create_task(send_heartbeats())
                try:
                    await handle_messages()
                finally:
                    hb_task.cancel()

        except (websockets.ConnectionClosed, ConnectionRefusedError, OSError) as e:
            print(f"[ws] Connection lost/refused: {e}. Reconnecting in 3s...")
            await asyncio.sleep(3)
        except asyncio.TimeoutError:
            print("[ws] Timeout waiting for RegisterAck. Retrying in 3s...")
            await asyncio.sleep(3)
        except Exception as e:
            print(f"[ws] Unexpected error: {e}. Retrying in 5s...")
            await asyncio.sleep(5)


async def handle_chat_completions(request):
    """Handle POST /v1/chat/completions with canned response."""
    body = await request.json()
    model = body.get("model", MODELS[0] if MODELS else "unknown")
    messages = body.get("messages", [])

    if RESPONSE_DELAY > 0:
        await asyncio.sleep(RESPONSE_DELAY)

    last_content = ""
    if messages:
        last_content = messages[-1].get("content", "")

    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Fake response from {RUNNER_ID} ({MACHINE_TYPE}) for model {model}",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(last_content.split()),
            "completion_tokens": 10,
            "total_tokens": len(last_content.split()) + 10,
        },
    }

    return web.json_response(response)


async def handle_health(request):
    """Health check endpoint."""
    return web.json_response({
        "status": "ok",
        "runner_id": RUNNER_ID,
        "machine_type": MACHINE_TYPE,
        "models": MODELS,
    })


async def run_http_server():
    """Run the HTTP server for inference requests."""
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/health", handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", HTTP_PORT)
    await site.start()
    print(f"[http] Listening on port {HTTP_PORT}")

    # Keep running forever
    while True:
        await asyncio.sleep(3600)


async def main():
    print(f"[main] Starting fake-runner: id={RUNNER_ID} type={MACHINE_TYPE} models={MODELS}")
    await asyncio.gather(
        websocket_client(),
        run_http_server(),
    )


if __name__ == "__main__":
    asyncio.run(main())
