# SimpleAI Backend

OpenAI-compatible API gateway that proxies requests to Ollama with OIDC authentication and audit logging.

## Features

- OpenAI-compatible `/v1/chat/completions` endpoint
- Language detection via `/v1/detect-language` endpoint (FastText ML model, 176 languages)
- OIDC JWT authentication via JWKS
- **Gateway mode**: Route requests to a fleet of inference runners
- **Model classification**: Big/fast model tiers with permission-based access
- **Wake-on-demand**: Auto-wake offline runners when no runners available
- Full audit logging to SQLite

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `SIMPLEAI__HOST` | Server host | `0.0.0.0` |
| `SIMPLEAI__PORT` | Server port | `8080` |
| `SIMPLEAI__OLLAMA__BASE_URL` | Ollama API URL | `http://localhost:11434` |
| `SIMPLEAI__OLLAMA__MODEL` | Default model for model:specific users | `llama3.2` |
| `SIMPLEAI__OIDC__ISSUER` | OIDC issuer URL | Required |
| `SIMPLEAI__OIDC__AUDIENCE` | OIDC audience/client ID | Required |
| `SIMPLEAI__DATABASE__URL` | SQLite database path | `sqlite:./data/audit.db` |
| `SIMPLEAI__LOGGING__LEVEL` | Log level | `info` |
| `SIMPLEAI__CORS__ORIGINS` | CORS allowed origins | `*` |
| `SIMPLEAI__LANGUAGE__MODEL_PATH` | FastText model path | `/data/lid.176.ftz` |
| `SIMPLEAI__GATEWAY__ENABLED` | Enable gateway mode | `false` |
| `SIMPLEAI__GATEWAY__AUTH_TOKEN` | Token for runner authentication | Required if gateway enabled |
| `SIMPLEAI__GATEWAY__AUTO_WAKE_ENABLED` | Enable wake-on-demand | `false` |
| `SIMPLEAI__GATEWAY__WAKE_TIMEOUT_SECS` | Wake timeout | `90` |

## Running Locally

```bash
export SIMPLEAI__OIDC__ISSUER=https://auth.example.com
export SIMPLEAI__OIDC__AUDIENCE=simple-ai
export SIMPLEAI__OLLAMA__BASE_URL=http://localhost:11434

cargo run
```

## Docker

```bash
docker build -t simple-ai-backend .

docker run -p 8080:8080 \
  -e SIMPLEAI__OIDC__ISSUER=https://auth.example.com \
  -e SIMPLEAI__OIDC__AUDIENCE=simple-ai \
  -e SIMPLEAI__OLLAMA__BASE_URL=http://ollama:11434 \
  -v ./data:/data \
  simple-ai-backend
```

## API

### POST /v1/chat/completions

OpenAI-compatible chat completion endpoint.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "model": "gpt-oss:20b"
}
```

**Response:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-oss:20b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hi there!"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  }
}
```

### POST /v1/detect-language

Detect the language of a text. Requires authentication.

**Request:**
```json
{
  "text": "Ciao, come stai?"
}
```

**Response:**
```json
{
  "code": "it",
  "confidence": 0.99
}
```

Language codes are ISO 639-1 (e.g., `en`, `it`, `fr`, `de`, `es`).

### GET /health

Health check endpoint (no authentication required).

### GET /metrics

Prometheus-compatible metrics endpoint (no authentication required).

## Gateway Mode

When `gateway.enabled = true`, the backend routes requests to connected inference runners instead of a single Ollama instance. Runners connect via WebSocket at `/ws/runners`.

### Model Classification

Models are classified into tiers in `config.toml`:

```toml
[models]
big = ["llama3:70b", "qwen2:72b"]
fast = ["llama3:8b", "mistral:7b"]
```

### Permission Roles

Access is controlled via JWT roles or roles assigned to individual API keys in
the admin dashboard:

| Role | Can Request |
|------|-------------|
| `model:specific` | Any model by exact ID (e.g., `"model": "llama3:70b"`) |
| (default) | Only classes (e.g., `"model": "class:fast"` or `"model": "class:big"`) |

Users without `model:specific` role default to `class:fast` if no model is specified.
Existing and newly created API keys are class-only by default. Administrators
can grant or revoke `model:specific` independently for each active key.

### Wake-on-Demand

When `gateway.auto_wake_enabled = true` and no runners are available:

1. Backend finds an offline runner matching the request (by model or class)
2. Sends Wake-on-LAN packet (via bouncer or idle-manager if configured)
3. Waits up to `wake_timeout_secs` for the runner to connect
4. Retries the request once connected

### Personal Android gateway

Users sign in inside SimpleAI Android, then approve any installed apps they wish
to use it. Calling apps do not need OAuth clients or accounts. The gateway uses
its own access token and reports caller package names as statistical metadata.
`X-SimpleAI-Source-App` is stored in request history and never grants permissions.

Configure `[oidc].android_client_id` with the gateway’s public OAuth client ID.
Register `com.lelloman.simpleai:/oauth2redirect` with that client and enable
public authorization code + PKCE and refresh tokens. The backend publishes only
its issuer and this public client ID at `/.well-known/simple-ai`, and trusts the
gateway audience alongside its primary audience. Do not add individual calling
apps such as Pezzottify to `additional_audiences` for this integration.

### Temporary LAN-local access

In the admin dashboard, use **Temporary LAN-local access** to enter your private
LAN CIDR (for example `192.168.1.0/24`), select a timeout, and enable access.
Use `/32` for a single IPv4 device or `/128` for a single IPv6 device. IPv6 unique
local networks are supported too. You can renew or disable access at any time;
the maximum duration is 24 hours and restarting the server disables it.

During this window, tokenless inference requests from the selected network use
the shared `lan-local` user, with permission to select specific models and existing
request tracking, but no admin privileges. Disabling that user also blocks this access. Any supplied Authorization
header goes through normal authentication, including rejection of invalid tokens.
Admin APIs always require an authenticated administrator. Expiry prevents new
requests; work already accepted can finish.

Connect directly to the server's LAN IP and port. The network check uses the TCP
peer address. Requests with `Forwarded`, `X-Forwarded-For`, or `X-Real-IP` headers
are ineligible. Do not route external traffic through a proxy that hides its
origin behind an allowed LAN address without forwarding headers.

The admin API exposes `GET /admin/api/lan-local` and `PUT /admin/api/lan-local`.
To enable, send `{"enabled":true,"network":"192.168.1.0/24","duration_seconds":3600}`;
to disable, send `{"enabled":false}`. Status includes `enabled`, `network`,
`expires_at`, and `remaining_seconds`. This setting is local to each server process.
