# SimpleAI

A distributed AI inference system providing OpenAI-compatible APIs, on-device capabilities, and multi-engine inference support.

## Overview

SimpleAI is a modular AI platform consisting of:

- **Backend** - Rust-based API gateway with OpenAI-compatible endpoints, OIDC authentication, and multi-runner orchestration
- **Inference Runner** - Standalone LLM runner that connects to the gateway via WebSocket, supporting Ollama and llama.cpp engines
- **Android** - Android service providing AI capabilities (NLU, translation, cloud/local LLM) to other apps via AIDL
- **simple-ai-common** - Shared protocol definitions for gateway-runner communication

## License

Apache 2.0

---

## Architecture

```
                                    ┌─────────────────────────────────────┐
                                    │         Client Applications          │
                                    │  (HTTP clients, Android apps, etc.)  │
                                    └─────────────────┬───────────────────┘
                                                      │ OpenAI-compatible API
                                                      │ (OIDC authenticated)
┌─────────────────────────────────────────────────────▼─────────────────────────────┐
│                            SimpleAI Backend                                       │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │   OIDC/JWKS     │  │   Audit Logger   │  │      Language Detection           │ │
│  │  Authentication │  │    (SQLite)      │  │       (FastText 176 langs)       │ │
│  └─────────────────┘  └──────────────────┘  └──────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │                     OpenAI-Compatible API Layer                               │ │
│  │   /v1/chat/completions  |  /v1/detect-language  |  /health  |  /metrics      │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │                        Runner Registry & Load Balancer                        │ │
│  │                   (WebSocket connections to runners)                          │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────┘
                                 │ WebSocket (protocol messages)
                                 │ (registration, heartbeat, commands)
                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                        Inference Runners (Horizontal Scale)                       │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐    │
│  │   Runner 1 (GPU)     │  │   Runner 2 (CPU)     │  │   Runner N (...)      │    │
│  │  ┌────────────────┐  │  │  ┌────────────────┐  │  │  ┌────────────────┐  │    │
│  │  │ Ollama Engine  │  │  │  │llama.cpp Engine│  │  │  │     ...        │  │    │
│  │  └────────────────┘  │  │  └────────────────┘  │  │  └────────────────┘  │    │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────────┘    │
└────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────────┐
│                            Android App (Optional)                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │Voice Commands│  │ Translation  │  │  Cloud AI    │  │      Local AI        │  │
│  │  (ONNX NLU   │  │  (ML Kit)    │  │  (Proxy)     │  │   (llama.cpp)        │  │
│  │   + LoRA)    │  │              │  │              │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                                  │
│  AIDL Interface (ISimpleAI) - Provides capabilities to other Android apps          │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### Backend

Rust-based API gateway that provides OpenAI-compatible endpoints with authentication and orchestration.

**Key Features:**
- OpenAI-compatible `/v1/chat/completions` endpoint
- OIDC JWT authentication via JWKS
- Language detection (176 languages via FastText)
- Full audit logging to SQLite
- Multi-runner orchestration via WebSocket
- Prometheus metrics endpoint

**Tech Stack:** Rust, Axum, Tokio, Reqwest, SQLite

See [backend/README.md](backend/README.md) for details.

---

### Inference Runner

Standalone LLM runner that abstracts local inference engines and connects to the gateway.

**Key Features:**
- OpenAI-compatible HTTP API (`/v1/chat/completions`)
- WebSocket gateway client for registration and control
- Support for multiple inference engines:
  - **Ollama** - HTTP-based inference server
  - **llama.cpp** - Direct process management (in development)
- Dynamic model loading/unloading via gateway commands
- Health reporting and capability discovery

**Tech Stack:** Rust, Axum, Tokio, WebSocket

See [inference-runner/PHASE4.md](inference-runner/PHASE4.md) for architecture details.

---

### Android

Android service providing AI capabilities to other apps via AIDL.

**Capabilities:**

| Capability | Description | Download Size |
|------------|-------------|---------------|
| Voice Commands | NLU intent classification + entity extraction | ~120 MB (XLM-RoBERTa) |
| Translation | On-device translation via ML Kit | ~30 MB per language |
| Cloud AI | Proxy to cloud LLM endpoint | None (requires auth) |
| Local AI | On-device LLM inference | ~1.3 GB (Qwen 3 1.7B) |

**Tech Stack:** Kotlin, Jetpack Compose, llama.cpp, ONNX Runtime, ML Kit

See [android/README.md](android/README.md) for API reference and integration guide.

---

### simple-ai-common

Shared protocol definitions and types for gateway-runner communication.

**Contents:**
- Chat message types (OpenAI-compatible)
- Capability descriptors
- WebSocket protocol messages (`RunnerToGateway`, `GatewayToRunner`)
- Runner status and health reporting types

---

## Quick Start

### Prerequisites

- **Rust** 1.70+ (for backend and inference-runner)
- **Android Studio** or SDK (for Android app)
- **Ollama** running locally or accessible via network
- **OIDC Provider** (e.g., Auth0, Keycloak) for authentication

### Running the Full Stack

#### 1. Start Ollama

```bash
ollama serve
ollama pull llama2
```

#### 2. Start the Backend

```bash
cd backend

export SIMPLEAI__OIDC__ISSUER="https://your-auth-provider.com"
export SIMPLEAI__OIDC__AUDIENCE="simple-ai"
export SIMPLEAI__OLLAMA__BASE_URL="http://localhost:11434"

cargo run
```

The backend will be available at `http://localhost:8080`

#### 3. Start an Inference Runner

```bash
cd inference-runner

# Configure gateway connection
export GATEWAY_URL="ws://localhost:8080/ws/runners"
export RUNNER_ID="runner-1"
export RUNNER_AUTH_TOKEN="your-secret-token"

# Configure engine
export ENGINE_TYPE="ollama"
export OLLAMA_BASE_URL="http://localhost:11434"

cargo run
```

#### 4. Test the API

```bash
# Health check
curl http://localhost:8080/health

# Chat completion (with auth token from your OIDC provider)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "model": "llama2"
  }'
```

---

## Configuration

### Backend Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `SIMPLEAI__HOST` | Server host | `0.0.0.0` |
| `SIMPLEAI__PORT` | Server port | `8080` |
| `SIMPLEAI__OLLAMA__BASE_URL` | Ollama API URL | `http://localhost:11434` |
| `SIMPLEAI__OIDC__ISSUER` | OIDC issuer URL | Required |
| `SIMPLEAI__OIDC__AUDIENCE` | OIDC audience/client ID | Required |
| `SIMPLEAI__DATABASE__URL` | SQLite database path | `sqlite:./data/audit.db` |
| `SIMPLEAI__CORS__ORIGINS` | CORS allowed origins | `*` |
| `SIMPLEAI__LANGUAGE__MODEL_PATH` | FastText model path | `/data/lid.176.ftz` |

### Inference Runner Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `GATEWAY_URL` | Gateway WebSocket URL | Required |
| `RUNNER_ID` | Unique runner identifier | Required |
| `RUNNER_AUTH_TOKEN` | Authentication token | Required |
| `ENGINE_TYPE` | `ollama` or `llamacpp` | `ollama` |
| `OLLAMA_BASE_URL` | Ollama API URL | `http://localhost:11434` |
| `RUNNER_HOST` | Runner HTTP host | `0.0.0.0` |
| `RUNNER_PORT` | Runner HTTP port | `8081` |

### Android Setup

1. Build and install the app:
   ```bash
   cd android
   ./gradlew assembleDebug
   ./gradlew installDebug
   ```

2. Open the SimpleAI app to enable and download capabilities

3. Integrate with your app using the AIDL interface (see [android/README.md](android/README.md))

---

## Development

### Project Structure

```
simple-ai/
├── backend/               # Rust API gateway
│   ├── src/
│   │   ├── main.rs        # Entry point
│   │   ├── api/           # HTTP handlers
│   │   ├── auth/          # OIDC/JWKS
│   │   ├── db/            # Audit logging
│   │   └── gateway/       # Runner orchestration
│   └── Cargo.toml
├── inference-runner/      # Standalone LLM runner
│   ├── src/
│   │   ├── main.rs
│   │   ├── gateway/       # WebSocket client
│   │   ├── engines/       # Engine abstractions
│   │   └── api/           # HTTP endpoints
│   └── Cargo.toml
├── simple-ai-common/      # Shared types
│   ├── src/
│   │   ├── protocol.rs    # Gateway-runner protocol
│   │   ├── chat.rs        # Chat types
│   │   └── capability.rs  # Capability types
│   └── Cargo.toml
├── android/               # Android app
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── aidl/      # AIDL interface
│   │   │   ├── java/      # Kotlin source
│   │   │   └── res/       # Resources
│   ├── build.gradle.kts
│   └── README.md
├── scripts/               # Utility scripts
├── PLAN.md                # Project roadmap
└── Cargo.toml             # Workspace config
```

### Building

```bash
# Build all Rust components
cargo build --release

# Build specific component
cargo build -p backend
cargo build -p inference-runner

# Build Android app
cd android && ./gradlew assembleDebug
```

### Testing

```bash
# Run all tests
cargo test

# Run backend tests
cargo test -p backend

# Run Android tests
cd android && ./gradlew test
```

---

## Deployment

### Docker

#### Backend

```bash
docker build -t simple-ai-backend backend/
docker run -p 8080:8080 \
  -e SIMPLEAI__OIDC__ISSUER=https://auth.example.com \
  -e SIMPLEAI__OIDC__AUDIENCE=simple-ai \
  -e SIMPLEAI__OLLAMA__BASE_URL=http://ollama:11434 \
  -v ./data:/data \
  simple-ai-backend
```

#### Inference Runner

```bash
docker build -t simple-ai-runner inference-runner/
docker run \
  -e GATEWAY_URL=ws://gateway:8080/ws/runners \
  -e RUNNER_ID=runner-1 \
  -e RUNNER_AUTH_TOKEN=secret \
  simple-ai-runner
```

### Production Considerations

- **Authentication**: Use a secure OIDC provider with proper key rotation
- **Database**: Configure PostgreSQL instead of SQLite for production
- **Load Balancing**: Deploy multiple runners behind the gateway
- **Monitoring**: Enable Prometheus metrics scraping
- **Logging**: Configure centralized log aggregation
- **Rate Limiting**: Implement per-user rate limits on the gateway
- **Circuit Breaker**: Enable circuit breaker for Ollama proxy calls

---

## API Endpoints

### Backend API

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check |
| `/metrics` | GET | No | Prometheus metrics |
| `/v1/chat/completions` | POST | Yes | OpenAI-compatible chat |
| `/v1/detect-language` | POST | Yes | Language detection |
| `/ws/runners` | WebSocket | Token | Runner registration |
| `/admin/runners` | GET | Yes | List connected runners |

### Inference Runner API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Runner health |
| `/v1/chat/completions` | POST | Chat completion |

---

## Roadmap

See [PLAN.md](PLAN.md) for the full project roadmap including:

- **P0** - Critical: Rate limiting, circuit breaker, security audit
- **P1** - High Priority: Request queuing, load balancing, dark mode
- **P2** - Medium Priority: Distributed tracing, CI/CD improvements
- **P3** - Future: Admin panel, model hub, advanced analytics

---

## Contributing

Contributions are welcome! Please:

1. Check existing issues and [PLAN.md](PLAN.md)
2. Follow the existing code style
3. Add tests for new features
4. Update documentation as needed

---

## Support

- **Issues**: Report bugs on GitHub
- **Documentation**: See component-specific READMEs
- **Android API**: See [android/README.md](android/README.md)
