"""Token-aware admission for the pinned Halogen API (one asyncio event loop).

The disk/prefix cache does not reduce a live request's KV reservation. This
gate runs after templating/tokenization and before handing work to the engine.
"""
import asyncio
from collections import deque
import time


class ClientDisconnected(Exception):
    pass


class KvLease:
    def __init__(self, gate, positions):
        self.gate = gate
        self.positions = positions
        self.released = False

    def release(self):
        if not self.released:
            self.released = True
            self.gate.used -= self.positions
            self.gate.active -= 1
            self.gate.changed.set()


class KvAdmission:
    def __init__(self):
        self.used = 0
        self.active = 0
        self.waiters = deque()
        self.changed = asyncio.Event()

    @staticmethod
    def positions(prompt_tokens, max_tokens, capacity):
        required = prompt_tokens + max_tokens
        if required <= 0 or required > capacity:
            raise ValueError(f"request needs {required} KV positions; pool has {capacity}")
        # The pinned allocator uses 256-position regions, with a guard block.
        # At the full-context boundary the engine clamps to its own limit.
        return min(capacity, ((required + 255) // 256 + 1) * 256)

    async def acquire(self, prompt_tokens, max_tokens, capacity, slots,
                      timeout, disconnected=None):
        if capacity <= 0 or slots <= 0:
            raise ValueError("Halogen did not report a positive KV pool and slot count")
        positions = self.positions(prompt_tokens, max_tokens, capacity)
        ticket = object()
        self.waiters.append(ticket)
        deadline = time.monotonic() + timeout
        try:
            while True:
                if disconnected is not None and await disconnected():
                    raise ClientDisconnected()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                # No await between checking and reserving: concurrent arrivals
                # cannot both observe the same free capacity. FIFO prevents a
                # stream of small requests from starving a long-context turn.
                if (self.waiters[0] is ticket and self.active < slots
                        and self.used + positions <= capacity):
                    self.used += positions
                    self.active += 1
                    return KvLease(self, positions)
                self.changed.clear()
                try:
                    await asyncio.wait_for(self.changed.wait(), min(remaining, 0.25))
                except asyncio.TimeoutError:
                    pass
        finally:
            self.waiters.remove(ticket)
            self.changed.set()

    def status(self):
        return {"enabled": True, "reserved_positions": self.used,
                "active": self.active, "queued": len(self.waiters)}
