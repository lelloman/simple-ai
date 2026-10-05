# Wake-on-demand deadline

`gateway.wake_timeout_secs` covers waking a sleeping host and waiting for its
runner to register. It is separate from `routing.model_prepare_timeout_secs`,
which covers loading a model after the runner is connected, and from inference
execution and the caller's overall deadline.

The default and example wake budget are 480 seconds. This allows idle-manager's
three 120-second wake attempts plus polling and startup overhead. It is bounded;
it does not guarantee success if the host never wakes. Existing installations
with an explicit shorter value must update that override. The configuration is
loaded at backend startup, so restart after changing it.

## Fast-class speculative preparation

With auto-wake enabled and nonempty speculative wake targets for `fast`, a
`class:fast` request also wakes offline matching targets when another runner is
already online. The online runner serves immediately while each target boots
and loads its fast model in the background. Concurrent requests share that
preparation. Once ready, the target participates in normal smart routing,
including machine preferences, queue load, and latency scoring.

When all targets are asleep, they wake and prepare concurrently; the first
ready model can serve without waiting for the other targets. Background failure
does not fail a request being served by an online runner. Wake and model-load
stages retain their separate configured deadlines. Disabling auto-wake or
speculative wake preserves ordinary online routing.

## Homelab incident and rollout, 2026-09-25

A `code:smart` request at 17:00:14 UTC returned HTTP 500 after 120 seconds while
idle-manager was still retrying. The second wake was sent at 17:02:57 and the
runner registered at 17:03:45 (211 seconds after the first request). The failure
occurred before model loading. The first wake packet's failure cause is unknown.

The live Homelab override was changed from 120 to 480 seconds and simple-ai was
restarted after verifying there were no recent unanswered inference requests.
The existing model preparation allowance remains 600 seconds. Talìa's report
has its own 1200-second overall deadline: stage budgets are ceilings, not a
promise that every stage can consume its maximum within that caller deadline.

The live config file is bind-mounted. It was backed up privately as
`config.toml.before-wake-480-20260925` and updated in place to preserve the mounted
inode. No credentials or unrelated settings were changed. The running binary
already honors the explicit override, so applying this fix does not require an
image rebuild. The source default/example ensure future deployments are aligned.

## Regression checks

`cargo test -p simple-ai-backend --test wake_deadline --offline` uses real local
HTTP for wake acceptance, then virtual time for the registry wait:

- Old 120-second budget reproduces expiry before the observed retry.
- Default 480-second budget accepts registration after 211 seconds.
- A runner that never registers still times out at the new finite deadline.

All 290 backend library tests also passed. Real cold-start preview evidence is
recorded after completion separately from the simulated retry test; a normal
first-attempt wake does not reproduce a dropped Wake-on-LAN packet.

Live unsent preview `report-217d83734071196822d8c5836891b03d` completed all fourteen
report steps with successful `code:smart` assessments after waking the sleeping
GPU host. The wake request at 17:52:03 UTC was followed by registration at
17:52:22 (~19 seconds). Runner logs show the model server ready at 17:53:02,
39.965 seconds after launch. Thus the existing 600-second model preparation
budget was ample in this measured cold start. No Telegram preview was sent.
This live wake succeeded on its first attempt; the 211-second retry is covered
by the virtual-time regression, not claimed as reproduced in production.
