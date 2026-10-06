"""Exercise the patched upstream serving/cleanup paths without a GPU.

Run inside the overlay image. Only tokenization and GPU generation are faked;
admission, upstream response serialization, abort and release paths are real.
"""
import asyncio
import inspect
import sys
import types
import unittest
from unittest.mock import AsyncMock

sys.path.insert(0, "/halogen/tools")
import serve_api


class Tokenizer:
    eos_token_id = 0

    def convert_tokens_to_ids(self, token):
        return None

    def __call__(self, text, **kwargs):
        return {"input_ids": [1]}


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = types.SimpleNamespace(
            info={"ctx": 262144, "kv_pool": 262144, "kv_slots": 4},
            waiting=0, slots=asyncio.Semaphore(4), inflight={}, reserved={},
            abort=AsyncMock(), close=AsyncMock())
        app = serve_api.build_app(Tokenizer(), self.engine, 262144, max_cap=65536)
        endpoint = next(r.endpoint for r in app.routes if r.path == "/v1/chat/completions")
        original = inspect.getclosurevars(endpoint).nonlocals["serve"]
        bindings = inspect.getclosurevars(original).nonlocals
        self.gate = bindings["kv_admission"]
        self.finish = asyncio.Event()
        self.fail = False

        async def generation(*args, **kwargs):
            yield "hello", None
            await self.finish.wait()
            if self.fail:
                raise serve_api.HTTPException(502, "synthetic engine failure")
            yield None, {"reason": "stop", "n_gen": 1, "n_prompt": len(args[0])}

        def cell(value):
            return (lambda: value).__closure__[0]

        self.serve = types.FunctionType(
            original.__code__, original.__globals__, original.__name__,
            original.__defaults__, tuple(cell(generation if name == "run" else bindings[name])
                                         for name in original.__code__.co_freevars))

    async def start(self, prompt=132156, streaming=True, http_request=None):
        return await self.serve([1] * prompt, 32000, [], streaming, True,
                                "chatcmpl", thinking=False, http_request=http_request)

    def assert_released(self):
        self.assertEqual(self.gate.used, 0)
        self.assertEqual(self.engine.reserved, {})
        self.assertEqual(self.engine.inflight, {})
        self.assertEqual(self.engine.slots._value, 4)

    async def test_stream_drop_releases_only_after_abort(self):
        response = await self.start()
        await anext(response.body_iterator)
        waiting = asyncio.create_task(self.start(68399))
        await asyncio.sleep(0.01)
        self.assertFalse(waiting.done())
        self.assertEqual(len(self.engine.inflight), 1)
        drain = asyncio.Event()
        self.engine.abort.side_effect = lambda: None

        async def abort():
            await drain.wait()

        self.engine.abort.side_effect = abort
        close = asyncio.create_task(response.body_iterator.aclose())
        await asyncio.sleep(0.01)
        self.assertFalse(waiting.done())
        drain.set()
        await close
        response2 = await asyncio.wait_for(waiting, 0.5)
        await anext(response2.body_iterator)
        await response2.body_iterator.aclose()
        self.assert_released()

    async def test_stream_normal_completion_and_error_release(self):
        for fail in (False, True):
            self.fail = fail
            self.finish.set()
            response = await self.start()
            events = [part async for part in response.body_iterator]
            self.assertTrue(events)
            self.assert_released()

    async def test_nonstream_completion_and_cancellation_release(self):
        self.finish.set()
        response = await self.start(streaming=False)
        self.assertEqual(response["choices"][0]["message"]["content"], "hello")
        self.assert_released()
        self.finish.clear()
        task = asyncio.create_task(self.start(streaming=False))
        await asyncio.sleep(0.01)
        self.assertEqual(self.gate.active, 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_released()

    async def test_queued_disconnect_does_not_enter_engine(self):
        first = await self.start()
        await anext(first.body_iterator)
        request = types.SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
        response = await self.start(68399, http_request=request)
        self.assertEqual(response.status_code, 499)
        self.assertEqual(len(self.engine.inflight), 1)
        self.assertEqual(len(self.gate.waiters), 0)
        await first.body_iterator.aclose()
        self.assert_released()


if __name__ == "__main__":
    unittest.main()
