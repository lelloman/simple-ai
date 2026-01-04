# SimpleAI Backend

OpenAI-compatible API gateway that proxies requests to Ollama with OIDC authentication and audit logging.

## Features

- OpenAI-compatible `/chat/completions` endpoint
- Language detection via `/detect-language` endpoint (FastText ML model, 176 languages)
- OIDC JWT authentication via JWKS
- Proxies to local Ollama instance
- Full audit logging to SQLite

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8080` |
| `OLLAMA_BASE_URL` | Ollama API URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default model | `llama3.2` |
| `OIDC_ISSUER` | OIDC issuer URL | Required |
| `OIDC_AUDIENCE` | OIDC audience/client ID | Required |
| `DATABASE_URL` | SQLite database path | `sqlite:./data/audit.db` |
| `LOG_LEVEL` | Log level | `info` |
| `CORS_ORIGINS` | CORS allowed origins | `*` |
| `SIMPLEAI__LANGUAGE__MODEL_PATH` | FastText model path | `/data/lid.176.ftz` |

## Running Locally

```bash
export OIDC_ISSUER=https://auth.example.com
export OIDC_AUDIENCE=simple-ai
export OLLAMA_BASE_URL=http://localhost:11434

cargo run
```

## Docker

```bash
docker build -t simple-ai-backend .

docker run -p 8080:8080 \
  -e OIDC_ISSUER=https://auth.example.com \
  -e OIDC_AUDIENCE=simple-ai \
  -e OLLAMA_BASE_URL=http://ollama:11434 \
  -v ./data:/data \
  simple-ai-backend
```

## API

### POST /chat/completions

OpenAI-compatible chat completion endpoint.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "model": "llama3.2"
}
```

**Response:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama3.2",
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

### POST /detect-language

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
