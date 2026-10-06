import asyncio
import unittest

from kv_admission import ClientDisconnected, KvAdmission


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def acquire(self, gate, prompt, output=32000, **kwargs):
        return await gate.acquire(prompt, output, 262144, 4, 1, **kwargs)

    async def test_production_overflow_waits_without_holding_engine_capacity(self):
        gate = KvAdmission()
        first = await self.acquire(gate, 132156)
        second = asyncio.create_task(self.acquire(gate, 68399))
        await asyncio.sleep(0.01)
        self.assertFalse(second.done())
        self.assertEqual(gate.status(), {
            "enabled": True, "reserved_positions": 164608, "active": 1, "queued": 1})
        first.release()
        lease = await asyncio.wait_for(second, 0.2)
        self.assertEqual(gate.used, 100864)
        lease.release()
        lease.release()  # Cleanup is idempotent.
        self.assertEqual((gate.used, gate.active), (0, 0))

    async def test_fitting_requests_remain_concurrent(self):
        gate = KvAdmission()
        leases = await asyncio.gather(*(self.acquire(gate, 60000) for _ in range(2)))
        self.assertEqual(gate.active, 2)
        for lease in leases:
            lease.release()

    async def test_simultaneous_arrivals_cannot_overbook(self):
        gate = KvAdmission()
        tasks = [asyncio.create_task(self.acquire(gate, 132156)) for _ in range(8)]
        await asyncio.sleep(0.01)
        self.assertEqual(sum(task.done() for task in tasks), 1)
        for task in tasks:
            lease = await asyncio.wait_for(task, 0.2)
            self.assertEqual(gate.active, 1)
            self.assertLessEqual(gate.used, 262144)
            lease.release()
        self.assertEqual(gate.used, 0)

    async def test_slot_limit_and_fifo(self):
        gate = KvAdmission()
        leases = [await self.acquire(gate, 100, 100) for _ in range(4)]
        large = asyncio.create_task(self.acquire(gate, 230000))
        small = asyncio.create_task(self.acquire(gate, 100, 100))
        await asyncio.sleep(0.01)
        leases.pop().release()
        await asyncio.sleep(0.01)
        self.assertFalse(large.done())  # It still cannot fit.
        self.assertFalse(small.done())  # Must not starve the older large turn.
        for lease in leases:
            lease.release()
        large_lease = await asyncio.wait_for(large, 0.2)
        self.assertFalse(small.done())
        large_lease.release()
        (await asyncio.wait_for(small, 0.2)).release()

    async def test_queued_cancellation_disconnect_and_timeout_do_not_leak(self):
        gate = KvAdmission()
        first = await self.acquire(gate, 230000)
        waiting = asyncio.create_task(self.acquire(gate, 100))
        await asyncio.sleep(0.01)
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        disconnected = False

        async def gone():
            return disconnected

        waiting = asyncio.create_task(self.acquire(gate, 100, disconnected=gone))
        await asyncio.sleep(0.01)
        disconnected = True
        with self.assertRaises(ClientDisconnected):
            await asyncio.wait_for(waiting, 0.5)
        with self.assertRaises(asyncio.TimeoutError):
            await gate.acquire(100, 32000, 262144, 4, 0.01)
        self.assertEqual(len(gate.waiters), 0)
        self.assertEqual(gate.active, 1)
        first.release()
        (await self.acquire(gate, 100)).release()
        self.assertEqual(gate.used, 0)

    async def test_stream_completion_error_and_drop_release_reservation(self):
        gate = KvAdmission()

        async def stream(fail=False):
            lease = await self.acquire(gate, 132156)
            try:
                yield "first token"
                if fail:
                    raise RuntimeError("upstream failed")
                yield "last token"
            finally:
                lease.release()

        for fail in (False, True):
            body = stream(fail)
            await anext(body)
            self.assertEqual(gate.active, 1)
            if fail:
                with self.assertRaises(RuntimeError):
                    await anext(body)
            else:
                self.assertEqual([part async for part in body], ["last token"])
            self.assertEqual(gate.used, 0)
        body = stream()
        await anext(body)
        await body.aclose()
        self.assertEqual(gate.used, 0)

    async def test_single_request_bounds_and_allocator_rounding(self):
        gate = KvAdmission()
        with self.assertRaises(ValueError):
            await self.acquire(gate, 250000)
        self.assertEqual(gate.used, 0)
        self.assertEqual(KvAdmission.positions(132156, 32000, 262144), 164608)
        self.assertEqual(KvAdmission.positions(68399, 32000, 262144), 100864)
        lease = await self.acquire(gate, 230144)
        self.assertEqual(gate.used, 262144)
        lease.release()


if __name__ == "__main__":
    unittest.main()
