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

1. Wait for SimpleAI work on halo1 to finish, or accept interruption.
2. In SimpleAI, set `strix-halo-01` to `disabled`. This stops dispatch
   immediately, prevents any wake, and unloads its models when idle.
   (`draining` first is optional if you want a graceful hand-off.)
3. In Crumbles, set the halo1 agent host to `enabled` (or clear its draining
   state).
4. Re-enable the SimpleAI runner unit at the systemd level on halo1, so it
   comes back after a reboot and stays available when the machine is dedicated
   to SimpleAI again:
   `systemctl --user enable simple-ai-runner`
   Do not start it while the host is dedicated to agents.

## Procedure: dedicate halo1 to SimpleAI

1. In Crumbles, set the halo1 agent host to `draining`, then `disabled` once
   its VMs have stopped so the machine's memory is free.
2. In SimpleAI, confirm that the halo1 runner unit is enabled at the systemd
   level (`systemctl --user enable simple-ai-runner`) and can register.
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
4. Remove the manual halo1 stop/disable from runbooks; the persisted SimpleAI
   state is now the switch that keeps it out of routing.

## Rollout order

1. Deploy the backend image and restart the gateway. Existing runners keep
   their previous behaviour (`enabled`) through the database migration.
2. Apply the intended state to each runner through the admin UI/API.
3. No inference-runner upgrade is required for these states; the gateway sends
   the existing `UnloadModel` command to keep models unloaded.
4. If a host is dedicated to agents, stop/disable its SimpleAI runner unit as
   described above. Merging the backend change does not deploy or restart
   anything.
