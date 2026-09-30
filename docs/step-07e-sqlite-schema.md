# Step 07e: audit bootstrap creation

All eight audit bootstrap tables and three named indexes now use shared
`TableSpec` descriptions and `create_plan`. AuditLogger's production startup
executes each plan at the existing point in its bootstrap sequence. The adapter
adds its existing IF NOT EXISTS policy; legacy ALTER statements, ignored duplicate
column errors, data operations and transaction behavior remain application-owned.
Creation adoption adds no new schema validation/rejection contract.

Generated indexes execute with SQLite's DQS_DDL compatibility flag temporarily
disabled. Otherwise a missing quoted column can become a string-literal index
instead of the old missing-column failure. The previous flag is restored on both
success and error; existing named indexes retain IF NOT EXISTS acceptance.

Reviewed dependency: published `lelloman-simple-server = 0.1.0`, exact version
with `database-sqlite-schema` enabled. Cargo.lock retains crates.io checksum
`1f3187c81c94701cd041df7ab17ce968b73c967db77c551170e3c24960b6c77a`.
No sibling path dependency, git checkout pin, push or deployment is introduced.

Verification (2026-09-30): focused baseline 64 audit tests passed. Final backend
library suite: **299 passed** with loopback permission. Four new tests compare
actual column/type/nullability/default/key/FK/index metadata against a frozen
old bootstrap, exercise defaults and unique failures, reopen legacy and newly
created files with retained records/markers, compare partial state after failed
bootstrap, and verify DQS restoration/existing-index acceptance.

Changed Rust files pass rustfmt and git diff checks. Production Clippy with
`--no-deps` completes with 11 existing warnings outside the new schema adapter;
strict workspace dependency Clippy fails on four pre-existing warnings in
simple-ai-common (collapsible_match/derivable_impls). Whole-workspace formatting
has existing debt; unrelated formatter changes were discarded. No full runner,
Docker, model/GPU or deployment qualification is claimed.

Commands:

```sh
cargo test --locked -p simple-ai-backend audit::sqlite::tests
cargo test --locked -p simple-ai-backend --lib
cargo clippy --locked -p simple-ai-backend --lib --bin simple-ai-backend --no-deps
rustfmt --edition 2021 --check backend/src/audit/mod.rs backend/src/audit/sqlite.rs backend/src/audit/schema.rs backend/src/audit/legacy_bootstrap_tests.rs
```
