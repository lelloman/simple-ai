# Step 08/09: shared authentication and authorization

The backend uses `simple_server::auth` from reviewed revision
`0a629da7b5eb5aeeb0ed64aac2c5f96cd4d9717b`. Its production
`authenticate_request` and `authenticate_inference_request` paths evaluate a
shared `AsyncAccess` flow. The application verifier still selects API key,
OIDC JWT, or the temporary LAN identity, and the shared check rejects disabled
accounts after the application loads the user record. The first Authorization
header is still selected; only exact `Bearer ` is recognized for API keys. A
supplied Authorization header, even empty or invalid text, prevents LAN access.
The shared header extractor is configured for those existing compatibility rules.

The backend's admin middleware and the query-token SSE and message-token
WebSocket paths all apply a shared `Access` check to an already validated
`AuthUser`. The application still decides admin status from its configured role
or explicit user list, and retains its existing HTML, SSE, and WebSocket error
responses. The OIDC/JWKS implementation, API-key database, trusted-network
policy, account state, model/resource permissions, and all route placement
remain application-owned. Model permission and TTS exception decisions remain
in `can_request_model` and `can_request_tts_model` because they depend on the
requested model and service policy.

The inference runner has no inbound user authentication or authorization path.
It sends a configured token to the gateway over WebSocket and an API key to an
upstream vLLM server, so the backend HTTP/user auth migration does not apply to
that binary. The backend verifies runner registration tokens inside the
WebSocket registration message, separately from HTTP user identity. That
application protocol check stays local, including its existing error handling.

Verification: before implementation, the selected LAN/admin HTTP tests passed
(3/3) and a new Authorization compatibility matrix passed on the old path
(1/1). The sandbox prevented localhost mock-server binding on an initial run;
the same tests passed with localhost binding allowed. After implementation,
`cargo test -p simple-ai-backend --all-targets` passed all 317 tests, including
admin role/list checks, admin HTTP access, the compatibility matrix, LAN
fallback, disabled accounts, and inference HTTP routes. Normal all-target
Clippy completed with existing warnings in untouched files and no warnings in
the changed auth code. Strict `-D warnings` stopped at three pre-existing
`derivable_impls` warnings in `simple-ai-common` before checking the backend.
The changed auth source passes `rustfmt --check`; `git diff --check` passes.
No external OIDC provider, production database, deployment, or browser run was
part of this local migration check.
