"""Apply a fail-closed overlay to the pinned upstream image, without vendoring it."""
import hashlib
from pathlib import Path
import sys

SOURCE_SHA256 = "8607cc429448b7eaa51c1bad68fe24c5c59f4bf8662eb28bbf21b130236ac4f1"


def patch(source):
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Halogen API source changed; review the KV admission overlay before building")
    source = source.replace(
        "    engine.tokenizer = tok   # 0.8.0: the VOCAB line is built from it",
        "    from kv_admission import KvAdmission, ClientDisconnected\n"
        "    kv_admission = KvAdmission()\n"
        "    engine.tokenizer = tok   # 0.8.0: the VOCAB line is built from it")
    old = """        engine.waiting += 1
        try:
            await asyncio.wait_for(engine.slots.acquire(),
                                   timeout=queue_timeout)
        except asyncio.TimeoutError:
            raise EngineBusy()
        finally:
            engine.waiting -= 1"""
    new = """        engine.waiting += 1
        kv_lease = None
        try:
            pool = int(engine.info.get("kv_pool") or 0)
            slots = int(engine.info.get("kv_slots") or 1)
            if not pool:
                pool = slots * int(engine.info.get("slot_ctx") or limit)
            kv_lease = await kv_admission.acquire(
                len(ids), max_tokens, pool, slots, queue_timeout,
                http_request.is_disconnected if http_request is not None else None)
            # Keep upstream's slot bookkeeping/health in sync. The gate already
            # limits active requests to this semaphore's capacity.
            await engine.slots.acquire()
        except ClientDisconnected:
            return Response(status_code=499)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except asyncio.TimeoutError:
            raise EngineBusy()
        except BaseException:
            if kv_lease is not None:
                kv_lease.release()
            raise
        finally:
            engine.waiting -= 1"""
    if source.count(old) != 1 or source.count("engine.slots.release()") != 2:
        raise ValueError("unexpected Halogen admission/release structure")
    source = source.replace(old, new)
    # Release only AFTER upstream abort/drain, including stream cancellation.
    source = source.replace("                        engine.slots.release()",
                            "                        engine.slots.release()\n"
                            "                        kv_lease.release()")
    source = source.replace("                engine.slots.release()\n        fin =",
                            "                engine.slots.release()\n"
                            "                kv_lease.release()\n        fin =")
    source = source.replace('"queued": engine.waiting,',
                            '"queued": engine.waiting,\n'
                            '                "kv_admission": kv_admission.status(),')
    if source.count("kv_lease.release()") != 3:
        raise ValueError("missing KV lease cleanup")
    compile(source, "serve_api.py", "exec")
    return source


if __name__ == "__main__":
    target = Path(sys.argv[1])
    target.write_text(patch(target.read_text()))
