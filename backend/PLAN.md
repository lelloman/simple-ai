# Backend Implementation Plan: SimpleAI API Gateway

## Overview

Build a Rust/Axum backend that:
1. Exposes an OpenAI-compatible `/chat/completions` endpoint
2. Validates OIDC JWT tokens via JWKS
3. Proxies requests to a local Ollama instance
4. Stores full audit logs in SQLite

## Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│ Android App │────▶│  SimpleAI API   │────▶│   Ollama    │
│             │     │  (Rust/Axum)    │     │  (local)    │
└─────────────┘     └────────┬────────┘     └─────────────┘
                             │
                    ┌────────▼────────┐
                    │     SQLite      │
                    │  (audit logs)   │
                    └─────────────────┘
```

## Project Structure

```
backend/
├── Cargo.toml
├── Dockerfile
├── src/
│   ├── main.rs              # Entry point, server setup
│   ├── config.rs            # Configuration from env vars
│   ├── routes/
│   │   ├── mod.rs
│   │   ├── chat.rs          # POST /chat/completions
│   │   └── health.rs        # GET /health
│   ├── auth/
│   │   ├── mod.rs
│   │   ├── jwks.rs          # JWKS fetching and caching
│   │   └── middleware.rs    # Auth middleware
│   ├── llm/
│   │   ├── mod.rs
│   │   └── ollama.rs        # Ollama client
│   ├── audit/
│   │   ├── mod.rs
│   │   └── sqlite.rs        # SQLite audit logger
│   └── models/
│       ├── mod.rs
│       ├── chat.rs          # OpenAI-compatible request/response
│       └── audit.rs         # Audit log entry
├── migrations/
│   └── 001_create_audit_logs.sql
└── README.md
```

## Dependencies (Cargo.toml)

```toml
[dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
reqwest = { version = "0.12", features = ["json"] }
jsonwebtoken = "9"
sqlx = { version = "0.8", features = ["runtime-tokio", "sqlite"] }
tower-http = { version = "0.5", features = ["cors", "trace"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1", features = ["v4", "serde"] }
thiserror = "1"
```

## Implementation Steps

### Phase 1: Project Setup
1. Create `backend/` directory with Cargo project
2. Set up basic Axum server with health endpoint
3. Add configuration module (env vars)
4. Create Dockerfile (multi-stage build)

### Phase 2: OpenAI-Compatible API
1. Define request/response models matching OpenAI spec
2. Implement `POST /chat/completions` route
3. Non-streaming responses only (streaming deferred to future version)
4. Map Ollama response format to OpenAI format

### Phase 3: Ollama Integration
1. Create Ollama client with configurable base URL
2. Forward chat requests to Ollama's `/api/chat` endpoint
3. Transform between OpenAI and Ollama formats
4. Handle errors gracefully

### Phase 4: OIDC Authentication
1. Implement JWKS fetcher with caching (refresh every hour)
2. Create auth middleware that:
   - Extracts Bearer token from Authorization header
   - Validates JWT signature using cached JWKS
   - Extracts user info (sub, email) for audit logs
3. Return 401 for invalid/expired tokens

### Phase 5: Audit Logging
1. Create SQLite database with migrations
2. Define audit log schema:
   - id (UUID)
   - timestamp
   - user_id (from JWT sub)
   - user_email (from JWT)
   - request_path
   - request_body (JSON)
   - response_status
   - response_body (JSON, truncated if large)
   - latency_ms
   - model_used
   - tokens_prompt
   - tokens_completion
3. Log every request/response pair
4. Add background task for log cleanup (optional, configurable retention)

### Phase 6: Error Handling & Polish
1. Consistent error responses
2. Request ID tracking
3. Structured logging with tracing
4. CORS configuration for future web frontend
5. Graceful shutdown

## Configuration (Environment Variables)

```bash
# Server
PORT=8080
HOST=0.0.0.0

# Ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2

# OIDC
OIDC_ISSUER=https://auth.example.com
OIDC_AUDIENCE=simple-ai

# Database
DATABASE_URL=sqlite:./data/audit.db

# Optional
LOG_LEVEL=info
CORS_ORIGINS=*
```

## Dockerfile

```dockerfile
FROM rust:1.75-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/simple-ai-backend /usr/local/bin/
EXPOSE 8080
CMD ["simple-ai-backend"]
```

## API Compatibility Notes

The Android client (CloudLLMClient.kt) expects:
- `POST /chat/completions`
- Request: `{"messages": [...], "tools": [...]}`
- Response: `{"choices": [{"message": {...}, "finish_reason": ...}], "usage": {...}}`
- Auth: `Authorization: Bearer <token>`

Ollama's API differs:
- `POST /api/chat`
- Request: `{"model": "...", "messages": [...], "stream": false}`
- Response: `{"message": {...}, "done": true, ...}`

The backend translates between these formats.

## Testing Strategy

1. Unit tests for format translation
2. Integration tests with mock Ollama
3. Auth tests with test JWTs
4. Manual testing with Android app

## Future Considerations

- Streaming responses (SSE) - next priority
- Rate limiting per user
- Token usage quotas
- Multiple LLM provider support (OpenAI, Anthropic)
- Web frontend API extensions
