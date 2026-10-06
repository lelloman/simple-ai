# Halogen on a Strix Halo runner

Deployed on **halo1 and halo2, 2026-10-06**. Both runners and Halogen services are
healthy, and authenticated gateway `code:smart` requests reached both hosts. Live
checks passed for plain/streaming chat, streamed tool arguments, a 32-token
thinking budget, Halogen → existing Gemma → Halogen GPU switching, and adoption
of the loaded service after a halo1 runner restart. Halo2 passed the same chat,
tool, reasoning and GPU switching checks. The runner test suite and the
additional Halogen capability test pass.

Rollback files on each host are in `~/simple-ai-backups/halogen-20261006/`.
Live host configs under `scripts/configs/` are intentionally gitignored; the
portable config template is in `inference-runner/config.example.toml`.
Saved synthetic live responses are under
`docs/evals/halogen-strix-halo/integration/` and `halo2/integration/` beneath
the same report directory.

The optional `[engines.halogen]` adapter manages one checkpoint through
`simple-ai-halogen.service`, a systemd **user** unit. It starts on demand, adopts
an already healthy server after runner restart, and stops through the normal
model-unload/resource-switch paths. Do not enable this unit at boot: the runner
must coordinate its memory with other GPU engines.

The deployed Halo configurations retain only the alias `code:smart`, mapped to
`qwen3.8-flash-next-uncensored-iq4xs`. Halogen, llama.cpp and JEV all use resource
group `gpu:0`; active streams retain the resource lease until their bodies close.
Other models remain callable by their raw IDs. CPU providers are retained.
The gateway maps `class:big` through `code:smart` (`models.big = ["code:smart"]`
and `routing.wake_preferred_classes = ["big"]`).

The subsequent RTX cleanup removes its smart-model route and disables its vLLM
engine. RTX uses `code:fast` → `Qwen3.6-35B-A3B-MXFP4_MOE`, which is also in
the gateway's `class:fast` list. Its llama.cpp catalog is
`~/simple-ai-fast-models`, containing symlinks to that model and the retained
Qwen3 embedding model. Specialist services remain enabled; other chat weights
are retained on disk but excluded from discovery. RTX's previous configuration
is backed up in `~/simple-ai-backups/fast-only-20261006/config.toml`.
Four Halogen slots share a 262,144-position KV pool; each request may use the full
context when capacity is available. Multiple cold long prompts can still pause
ongoing decoding. The model is currently text-only through this integration.

### KV capacity admission overlay

Activated on **halo1 and halo2, 2026-10-06**. Both report
`kv_admission.enabled = true` and passed a `code:smart` request through their
runner APIs after activation. All 11 capacity and upstream integration checks
passed in the image builds. Original service units are backed up on each host
under `~/simple-ai-backups/kv-admission-20261006/`.

The service unit now references `localhost/simple-ai-halogen:kv-admission-v1`.
Build it from this directory with `podman build --pull=never -t
localhost/simple-ai-halogen:kv-admission-v1 .` before installing the unit.
The fleet deployer does this automatically. This is a small overlay on the
pinned upstream image; `patch_api.py` checks the original API source hash and
refuses to build against changed upstream code. No model or GPU kernel changes.

Admission runs in Halogen's API after its actual chat template and tokenizer,
before engine dispatch, for streaming and non-streaming requests on all wires.
Each request reserves its full prompt (including cached tokens) plus maximum
output, rounded to 256 positions with one additional 256-position allowance.
Reservations are capped at the full pool for a request that fits alone. Both
the slot count and sum of live reservations must fit. FIFO waiters remain
outside the engine; disconnects and timeouts remove them. Capacity is released
after generation and abort/drain cleanup. Disk cache does not add live capacity.

`/health` includes `kv_admission` with `enabled`, `reserved_positions`, `active`
and `queued`. For example, 132,156 prompt + 32,000 output reserves 164,608;
68,399 + 32,000 reserves 100,864. Those two requests cannot run together in
262,144 positions. Two 60k prompts with 32k output can still run concurrently.

The image build runs capacity/race/cancellation tests and integration checks
against the patched upstream serving function, including stream completion,
error, disconnect and abort cleanup. Integration checks stub GPU generation;
they do not load model weights. Run the capacity tests locally with
`python3 -m unittest discover -s deploy/halogen -p 'test_*.py' -v`.

This prevents aggregate overcommit; it does not change gateway runner selection,
extend client deadlines, or guarantee that fragmented resident cache regions
can be placed without eviction. Applying the image to an already loaded service
requires a Halogen restart, which interrupts its active requests; drain first.

Before deployment, stage the pinned image and artifacts described in
[the benchmark report](../../docs/evals/halogen-strix-halo/README.md), with the BIOS
GPU reservation set to 2 GiB on halo1 and 512 MiB on halo2. Halo2 retained Fedora
43 and kernel 7.2.7-100.fc43; no OS upgrade was needed. Its weights, MTP head and
tokenizer were copied from halo1. The unit mounts
`~/halogen-benchmark/models` read-only and exposes port 8731 only on loopback.
No weights are downloaded by the production service. `scripts/deploy-fleet.sh`
installs the unit on hosts whose configuration contains `[engines.halogen]`.
The image supervisor exits with code 1 even on a requested clean stop. The unit
accepts that status and uses `Restart=always` for unexpected exits; explicit
runner-requested stops do not restart it. A halo2 GPU switch verified an inactive
unit with `Result=success` before reloading Halogen.

Requests support streaming, tools, reasoning effort and thinking budgets. An
omitted effort preserves the model's xhigh default; `none` or budget 0 disables
thinking, and budget -1 leaves it unrestricted. The adapter translates positive
budgets to Halogen's `max_thinking_tokens`. Timing metrics use Halogen's reported
new-token prefill rate, avoiding inflated rates from cached prompt tokens.

To inspect the service after deployment:

```sh
systemctl --user status simple-ai-halogen
journalctl --user -u simple-ai-halogen -n 80
curl http://127.0.0.1:8080/health
```

The runner loads the model when a request arrives. Stopping Halogen manually
while serving interrupts its requests; use the runner's normal lifecycle instead.
For rollback, restore the host's previous runner binary/configuration and stop the
Halogen unit before restarting the old runner.

## Persistent conversation cache

Both hosts are configured with a **50 GB decimal disk-cache budget each**:
`HALOGEN_CACHE_DISK_GIB=46.56612873`, reported by `/cache` as 49,999,999,999
bytes. This is a managed cache budget, not a preallocated partition or a
filesystem quota. It grows as conversations are saved and evicts older entries
at its budget; temporary writes and filesystem overhead are separate.

The unit mounts `~/.cache/simple-ai/halogen` read-write at `/cache`, on the host's
XFS filesystem. Create it with `install -d -m 700 ~/.cache/simple-ai/halogen`;
the fleet deployer now does this automatically. The existing RAM pool remains
262,144 positions. Disk state survives engine unload/restart, but must be
restored into that pool before generation; it does not add active GPU capacity.

`HALOGEN_CACHE_PRUNE_OLD=1` removes incompatible engine/configuration namespaces
at startup so old versions do not accumulate outside the current budget. A
rollback to a different engine build therefore starts with a cold cache.
No cross-machine cache replication is configured.

Allow cache flushing on shutdown: Podman gets 45 seconds and systemd 55 seconds,
both within the runner's existing 60-second timeout. The deployed config files
and example set `[engines.halogen].shutdown_timeout_secs` to **90** for additional
margin; a running runner picks up that value on its next restart. Cache deployment backups are in
`~/simple-ai-backups/disk-cache-20261006/` on each host.

Check `curl http://127.0.0.1:8731/cache` for `disk.on`, `budget_bytes`, persisted
records, hits and restored bytes. Restart/restore validation artifacts are in
`docs/evals/halogen-strix-halo/disk-cache/`.
