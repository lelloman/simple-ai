# Step 04c: CORS canary

Reviewed shared source: `3aa933295860a9ed08b51ae882c8297f17e7eac2`.
Only the affected HTTP packages enable the optional `cors` feature.

Both backend and inference-runner call a production `cors_policy()` factory
using `simple_server::cors::CorsConfig`. Both allow any origin/method/request
header without credentials. The backend still exposes no extra response headers;
the runner still exposes all response headers. Existing layer placement, routes,
state, inference streams and auth behavior remain unchanged.

Direct tower-http dependencies are removed; the implementation is now private to
simple-server. The existing resolved tower-http 0.6.8 is retained with no
package version upgrades. `simple-server.rev` records the reviewed implementation.

## Verification

Baseline: **399 passed, one existing ignore**. Final: **401 passed, one
existing ignore** across backend and inference-runner package suites. New tests
call each production policy factory and check unauthorized responses and
preflights for multiple origins, wildcard values, absent credentials, Vary and
the backend/runner exposed-header distinction. Both new tests passed against the
original middleware before replacement.
All-target Clippy completes with existing capped warnings (backend 12, common 3,
runner 5); strict warning-free lint is not claimed. Changed-file formatting and
whitespace checks pass.

Baseline used `--locked`; after dependency-edge changes the final test command was:

```
cargo test --offline -p simple-ai-backend -p inference-runner -q -j 2 --target-dir /tmp/03c-ai-target --config profile.dev.debug=0 --config profile.test.debug=0
cargo clippy --locked -p simple-ai-backend -p inference-runner --all-targets -j 2 --target-dir /tmp/03c-ai-target --config profile.dev.debug=0 -- --cap-lints warn
```

The ignored scripts/configs/rtx.toml fixture was copied unchanged for existing
runner tests. Real GPU/model inference, external providers, browser/Android E2E
and containers were not rerun. Original semantic-evaluation work is unrelated
and must remain unchanged during integration.

## Integration

Started from active master `b5fa60aee57cd4e6b9c3a34e3c760d60a8885f60` using isolated sibling worktree
and branch `migration/step04c-cors`. Commit there, rebase master onto the migration,
verify ancestry and tested tree plus unrelated working files, then remove the
migration worktree and branch. Final integration evidence and commit IDs are
recorded in simple-server's matching HTML and Markdown migration trackers.
Nothing is pushed or deployed.
