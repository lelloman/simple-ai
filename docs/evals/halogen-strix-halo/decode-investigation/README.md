# Decode and batching investigation — 2026-10-06

Halo2's earlier 34–36 tokens/s measurement was not a sustained ceiling. With a
100,922-token cold input, it generated 1,024 tokens at **43.80 tokens/s**, with
one active request observed throughout. A separate 6,066-token input generated
512 tokens at **49.67 tokens/s**. The original short 128-token sample did not
record host telemetry, so its exact difference from halo1 remains unexplained;
it is not evidence of a persistent 23% host disadvantage.

## Longer prompts under live traffic

All runs used the deployed runner on port 8080, the uncensored IQ4_XS model,
Halogen 0.16.2, MTP enabled, four slots, cache mode 2, temperature zero and
explicit `reasoning_effort: none`. Each generated 1,024 tokens. Reported cached
input was zero. There were no request errors.

| Host | Input tokens | Prefill t/s | Generation t/s | Peak observed active requests | Draft tokens proposed / accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| halo1 | 24,442 | 1,615 | 12.00 | 2 | 33 / 24 |
| halo2 | 24,442 | 1,580 | 22.54 | 2 | 0 / 0 |
| halo1 | 49,932 | 1,571 | 21.40 | 2 | 0 / 0 |
| halo2 | 49,932 | 1,562 | 22.37 | 2 | 0 / 0 |
| halo2 | 100,922 | 1,546 | 43.80 | 1 | 860 / 610 |

The 24k and 50k runs overlapped production conversations. Halo1's 24k generation
also paused for another large cold prefill. These rows measure live contention,
not isolated host performance. The similar 50k results do not reproduce the
earlier halo2 disadvantage. During overlap, the engine logs explicitly report
tokens generated with the draft head off. The 100k run was alone in all telemetry
samples and used speculation. No halo1 100k run was needed to establish that
halo2 can sustain more than 40 tokens/s at large context.

During halo2's 100k generation, median reported GPU clock was 2.609 GHz, PPT
85.0 W and edge temperature 77°C (75–79°C). There were no swap-in/out events or
compaction stalls in that measured decode interval. The engine read about
9.1 MB from storage. These readings show no obvious power or thermal collapse;
they do not establish the cause of the earlier short sample. Both machines use
Ryzen AI Max+ 395 processors, but halo1 reports NucBox EVO-X2 / BIOS 1.12 and
halo2 BeyondMax Series / BIOS 1.07, with different kernels.

## Batching on halo2

One sweep, with 256 synthetic code functions per input, distinct case markers,
512 output tokens per request, and the same serving configuration. Actual input
was 6,066 tokens for the single case and 6,071 per request for batches. The
observed active-request count never exceeded the submitted benchmark count,
and no queued requests were observed. These are single samples rather than
confidence intervals or a sustained-load capacity test.

| Concurrent requests | Reported per-stream generation t/s | First-token latency | Time until all finish | Total output t/s including prefill |
| ---: | ---: | --- | ---: | ---: |
| 1 | 49.67 | 5.02 s | 15.33 s | 33.41 |
| 2 | 19.44–23.23 | 4.76 / 9.10 s | 31.14 s | 32.88 |
| 4 | 11.85–16.78 | 5.00 / 9.29 / 13.64 / 18.00 s | 48.51 s | 42.22 |

Total throughput is total completion tokens divided by the slowest request's
elapsed time (requests launch through a thread barrier). Per-stream generation
timings include pauses for other prompts' prefills; they are not the throughput
of an uninterrupted batched decode phase. Do not add those rates to estimate
aggregate throughput.

Four-way batching delivered about 26% more end-to-end output throughput than
the single-request baseline in this workload; two-way batching delivered no
improvement. Per-user latency rose substantially. The last admitted stream had
little subsequent prefill interference and ran around 23 t/s at batch 2 and
17 t/s at batch 4, versus roughly 50 t/s alone with speculation.

[Upstream documents](https://github.com/peonist-ai/halogen-flash-server#what-this-release-is-not)
that multiple generating conversations use batched steps without MTP; speculation
resumes when a conversation is alone. Incoming prompt prefill is chunked with
decode ticks between chunks, rather than fully independent of generation.

## Routing implication

Four slots are admission capacity, not four independent 50-token/s workers.
Fresh conversations benefit from using both machines before increasing the
batch on one. Existing conversation cache affinity has a competing benefit:
moving a 100k conversation can incur about a minute of cold prefill.

The current `Router::estimated_wait_ms` in `backend/src/gateway/router.rs`
uses integer `active_requests / batch_size` times an average latency. With a
batch size of four, its estimate remains zero for one to three active requests.
That affinity comparison does not explicitly model losing MTP or prefill
interference. A future routing change should compare expected completion time,
including cache benefit, rather than assume a free slot has no performance cost.
No routing, firmware, power, service or model configuration was changed here.

## Reproduction and artifacts

`../investigate-decode.py` calls `../benchmark.py`, waits up to 60 seconds for
an idle admission point, then submits through the runner so GPU leases remain
coordinated. Idle admission does not prevent later production overlap. Tests
were run on one host at a time; production traffic remained enabled throughout.
Telemetry is sampled roughly once per second, so very brief overlap can be
missed. It includes engine counters, clocks, PPT, temperature and later-added
VM/I/O counters. The first halo2 24k case predates the VM/I/O additions.

Example on a host with both scripts staged beside each other:

```sh
python3 investigate-decode.py --rows 4096 --out decode-investigation
python3 investigate-decode.py --rows 256 --repeat 10 --concurrency 4 \
  --output-tokens 512 --out decode-investigation
```

Use new repeat values to avoid overwriting artifacts or unintentionally reusing
the exact prompt cache. `halo1/` and `halo2/` contain synthetic requests, outputs,
raw SSE chunks and telemetry. `summary.json` contains the derived measurements.
No production prompt or response contents were saved.
