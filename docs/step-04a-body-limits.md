# Step 04a: extractor body limits

Both production components: backend OCR/extract keep 25 MiB, backend audio 200 MiB; runner OCR 100 MiB and runner audio 200 MiB. All five declarations preserve placement. Domain file validation, inference scheduling, SSE and WebSockets remain local.

Uses optional simple-server body-limit feature at reviewed revision
`0b945750b6b97a9e18c531d1cf1137d4ed4b69c9`. Raw readers and custom domain limits
are not replaced. Shared differential tests qualify JSON/bytes/multipart
rejections, unknown lengths, route overrides and lazy response pass-through.

## Verification

Baseline and final: backend 312 passed, one ignored doctest; inference runner 87 passed. Combined final 399 passed, one ignored. Runner compilation initially required the existing ignored scripts/configs/rtx.toml fixture, copied unchanged into the isolated worktree. That local fixture is not committed.

Primary command: `cargo test --locked -p simple-ai-backend -p inference-runner` in the affected package/workspace. Runs use
two jobs, `/tmp/03c-ai-target` and dev/test debug=0. Lockfiles add only the
shared library's tower-layer dependency. All-target Clippy completes with capped warnings: existing 12 backend, three common and five runner warnings remain. Strict warning-free lint is not claimed. Changed-file rustfmt and
whitespace checks pass; repository-wide formatting is not claimed.

GPU/model/provider, Android/browser and container/release workflows were not rerun. Original semantic-evaluation working files are preserved.

## Integration

Started from active master `2b9c6a61`
in a dedicated `migration/step04a-body-limits` branch and sibling worktree.
Commit there, rebase the original development branch onto the migration, verify
ancestry/tested tree and preserve unrelated work before removing the temporary
worktree/branch. Exact integration revision and any concurrent changes are
recorded in the central simple-server trackers. No push or deployment.
