# Dedicating a host to SimpleAI or to agents

A physical machine can be shared by a SimpleAI runner and a Crumbles agent
runner, but the two must not run large models at the same time. Choose the
owner with the two independent switches:

- **SimpleAI runner state** (this repository): `enabled`, `draining` or
  `disabled`, set per runner ID in the admin UI or admin API.
- **Crumbles host state** (Crumbles): `enabled`, `draining` or `disabled` for
  the agent runner on the same machine.

SimpleAI state is persisted in the backend database (`runners.state`) and is
loaded on restart. It remains in force while the host is offline.

## SimpleAI runner states

| State | Dispatch | Wake-on-LAN / speculative wake / idle-manager keepalive | Loaded models |
| --- | --- | --- | --- |
| `enabled` | Normal routing | Allowed | Normal |
| `draining` | No new requests; in-flight requests finish | No keepalive for new work | Unloaded after in-flight work finishes |
| `disabled` | No new requests | Never woken or kept awake | Kept unloaded, including after registration |

When every runner for a requested class is `disabled`/`draining`, the gateway
returns HTTP 503 with `No enabled runner for class <class>` instead of trying
to wake a host. Unloading happens while the gateway holds no active request on
that runner; a disabled/draining runner that reconnects with models loaded is
told to unload them.

The existing `/admin/runners/{id}/drain` endpoint still sends the runner-level
`Drain` command used for maintenance and shutdown. It is separate from the
persisted state above.

## Admin UI and API

In the web admin, open **Runners**. Each card shows the state badge, loaded
models and, when the runner reports them, GPU and system memory in use. Use
**Enable**, **Drain** or **Disable** on the card.

The same change over the admin API:

```bash
curl -X PUT "$SIMPLE_AI_URL/admin/runners/strix-halo-01/state" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"state":"disabled"}'
```

`GET /admin/runners` includes `state`, `metrics`, `loaded_models` and
`available_models` for each connected or offline runner. The admin WebSocket
and SSE streams report state changes as `runner_state_changed`.

## Procedure: dedicate halo1 to agents

Do not enable the Crumbles host until the SimpleAI model is really gone. That
means all of the following are visible for `strix-halo-01` in the admin UI
(the **Runners** page and the dashboard fleet):

- state is `draining` or `disabled`;
- active requests are zero;
- loaded models are empty;
- GPU and system memory have dropped back to the host baseline.

1. In SimpleAI, set `strix-halo-01` to `draining` to stop new dispatch while
   in-flight work finishes.
2. Wait for the conditions above. If a model stays loaded, keep waiting; do
   not hand the host over while it is resident. Then set `strix-halo-01` to
   `disabled` to prevent any later load or wake.
3. Only after step 2 is confirmed, in Crumbles set the halo1 agent host to
   `enabled` (or clear its draining state).
4. On halo1 keep the SimpleAI runner unit enabled at the systemd level so it
   is part of the declared configuration and can come back later:
   `systemctl --user enable simple-ai-runner`
   It may be left running with state `disabled` (the gateway keeps its models
   unloaded), or stopped to free its process memory:
   `systemctl --user stop simple-ai-runner`
   Do not disable the unit.

## Procedure: dedicate halo1 to SimpleAI

1. In Crumbles, set the halo1 agent host to `draining`, then `disabled` once
   its VMs have stopped. Wait until the host's memory has been released.
2. On halo1 ensure the SimpleAI runner unit is enabled and start it:
   `systemctl --user enable simple-ai-runner`
   `systemctl --user start simple-ai-runner`
   A disabled persisted state still lets it register, but it will not serve
   work until step 3.
3. Set `strix-halo-01` to `enabled` in SimpleAI. Routing and wake-on-LAN resume
   normally.

Use the same steps for halo2 with its runner ID and Crumbles host. To dedicate
both Halos to one system, change both pairs; there is no global switch.

## Replacing the 2026-10-20 interim setup

The operator's interim setup was: halo2 SimpleAI-only with its Crumbles host
draining; halo1 agents-only with `simple-ai-runner` stopped and disabled via
`systemctl --user`.

Replace it with the two switches:

1. On halo1, run `systemctl --user enable simple-ai-runner` but leave it
   stopped while agents own the host.
2. In SimpleAI, set `strix-halo-01` to `disabled`.
3. In Crumbles, set the halo2 host to `enabled` (or undo its draining state)
   when halo2 is meant for agents; keep `strix-halo-02` `enabled` in SimpleAI
   only while SimpleAI owns halo2.
4. Remove the manual halo1 stop/disable from runbooks. The unit stays enabled
   (stopped is fine while agents own the host); the persisted SimpleAI
   `disabled` state is the switch that keeps it out of routing.

## Rollout order

1. Deploy the backend image and restart the gateway. Existing runners keep
   their previous behaviour (`enabled`) through the database migration.
2. Apply the intended state to each runner through the admin UI/API.
3. No inference-runner upgrade is required for these states; the gateway sends
   the existing `UnloadModel` command to keep models unloaded.
4. If a host is dedicated to agents, first set its SimpleAI state to
   `disabled` and confirm that no models or requests remain, then optionally
   stop the runner unit while leaving it enabled as described above. Never
   disable the unit. Merging the backend change does not deploy or restart
   anything.
