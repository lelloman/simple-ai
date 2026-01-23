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

Access is controlled via JWT roles:

| Role | Can Request |
|------|-------------|
| `model:specific` | Any model by exact ID (e.g., `"model": "llama3:70b"`) |
| (default) | Only classes (e.g., `"model": "class:fast"` or `"model": "class:big"`) |

Users without `model:specific` role default to `class:fast` if no model is specified.

### Wake-on-Demand

When `gateway.auto_wake_enabled = true` and no runners are available:

1. Backend finds an offline runner matching the request (by model or class)
2. Sends Wake-on-LAN packet (via bouncer or idle-manager if configured)
3. Waits up to `wake_timeout_secs` for the runner to connect
4. Retries the request once connected
