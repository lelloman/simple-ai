# Step 04b: response header policies

Reviewed shared source: `b88b908421db2552ff1e81966c56958925741e27`.
The optional response-headers feature is enabled only in the affected HTTP
package(s). Production calls use the shared header operations, preserving
existing values, placement, status/body/extension behavior and application policy.
No new global defaults or response buffering are introduced.

Five production streaming branches use shared replacement for their existing no-cache headers: backend chat, Responses and speech, plus runner chat and speech. Stream construction, reservation/cancellation/accounting, content-type, keep-alive and placement stay local; no body polling/buffering is introduced.

## Verification

Baseline and final: **399 passed, 0 failed, 1 ignored** across the
selected suites. Backend and inference-runner package suites run before/after. The existing ignored scripts/configs/rtx.toml test fixture was copied unchanged into the worktree; it is not committed. Shared module contracts separately verify lazy data/trailer/error frames.

All-target Clippy completes with capped warnings; existing lint debt remains, so strict warning-free lint is not claimed. Changed-file formatting and whitespace checks pass. The lockfile
changes add only the shared crate's direct http dependency, with no package
version changes. Applicable CI/readme/source pins reference the reviewed source.

Final test commands (baseline used --locked before the dependency-edge update;
final tests used --offline, then lint verified --locked):

- In `.`: `cargo test --offline -p simple-ai-backend -p inference-runner -q -j 2 --target-dir /tmp/03c-ai-target --config profile.dev.debug=0 --config profile.test.debug=0`

Lint commands:

- In `.`: `cargo clippy --locked -p simple-ai-backend -p inference-runner --all-targets -j 2 --target-dir /tmp/03c-ai-target --config profile.dev.debug=0 -- --cap-lints warn`

Real GPU/model inference, external providers, Android/browser and container/release workflows were not rerun. Existing semantic-evaluation working files are preserved.

## Integration

Started from active master `f5115239` in dedicated branch
`migration/step04b-response-headers` and a sibling worktree. Integration follows
the shared workflow: commit there, rebase the original development branch onto
the migration, verify ancestry/tested tree and preserve concurrent work, then
remove only the temporary migration worktree/branch. The central simple-server
trackers record the final commit, integration outcome and any concurrent changes.
No push or deployment.
