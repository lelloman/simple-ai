# Persistent disk cache — deployed 2026-10-06

Both machines now run with a 50 GB decimal managed cache budget:
`HALOGEN_CACHE_DISK_GIB=46.56612873`, reported as 49,999,999,999 bytes.
The host directory is `~/.cache/simple-ai/halogen`, mode 0700, on XFS,
mounted at `/cache`. Cache size grows on demand; this is not a preallocated
partition or hard filesystem quota. `HALOGEN_CACHE_PRUNE_OLD=1` prevents older,
incompatible configuration namespaces accumulating beyond the active budget.
The RAM pool remains 262,144 positions per host. Caches remain local to each host.

## Validation

Generated a synthetic 6,064-token prompt, saved its state, stopped and started
Halogen, and submitted the same conversation plus its answer and a follow-up.
Both hosts restored 6,016 tokens from disk and prefilled only 197 new tokens
(including the uncached tail). API `/cache` confirmed disk hits and the budget.

| Host | Cold prefill | Tokens restored from disk | Disk restore time | Follow-up prefill |
| --- | ---: | ---: | ---: | ---: |
| halo1 | 4.168 s | 6,016 | 92.2 ms | 576.6 ms |
| halo2 | 4.127 s | 6,016 | 106.3 ms | 770.0 ms |

These disk restore times exclude model loading, queueing, prefill and generation.
They test process restart with the same OS running, not a cold-disk power cycle.
JSON artifacts in `halo1/` and `halo2/` include synthetic requests, responses,
timings and the final cache counters.

Halo2 was updated in an idle window with its runner temporarily stopped for
validation. Halo1 had continuous production traffic; after waiting for a quiet
window, its reload and validation instead used the runner's normal GPU resource
lease: a small Gemma request waited for active Halogen streams to finish, stopped
Halogen, and the following code request loaded Halogen with the new unit. This
was repeated to test disk persistence. The runner process remained online on
halo1. Test wall times there include production queueing and engine startup,
so they are not standalone cache-latency benchmarks.

Production timing logs on halo1 after the validation reload also showed:

- 67,648 cached tokens restored from disk in **440 ms**; total input 70,194,
  remaining prefill 2.32 s, generation 40.97 tokens/s.
- 44,864 cached tokens restored in **301 ms**; total input 48,369,
  remaining prefill 2.76 s, generation 40.47 tokens/s.

No production prompt/response content was saved. Both runner and Halogen
services were checked healthy after rollout. The systemd units passed remote
verification, the fleet script passed shell syntax checking, and the diff passed
whitespace checks. No Rust code was changed for disk caching.

## Lifecycle and rollback

The unit allows Podman 45 seconds to stop, with a systemd timeout of 55 seconds,
both within the existing runner's 60-second timeout. Config files and the
example provide 90 seconds; halo2 loaded that value during its runner restart,
while halo1 will read it at its next runner restart. Persistence through normal
GPU switching was verified with halo1's existing timeout.

Previous unit/config files are backed up on each host under
`~/simple-ai-backups/disk-cache-20261006/`. Restoring them disables disk caching
on the next engine start; do not stop an engine with active requests. Cache
files can be retained for rollback to the same engine/configuration. Switching
engine builds with pruning enabled discards incompatible cached state.
