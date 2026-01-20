# Phase 4: Gateway Server

## Overview

Build the central Gateway/Orchestration service that:
- Accepts WebSocket connections from multiple inference runners
- Maintains runner registry with health/capability tracking
- Routes inference requests to appropriate runners
- Provides a unified OpenAI-compatible API for clients

## Prerequisites

- [ ] Test inference-runner standalone (Phase 1-3 complete)
- [ ] Verify GatewayClient connects and reports status correctly

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Gateway Server                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │   Runner    │  │    Load      │  │   OpenAI-compat   │  │
│  │  Registry   │  │  Balancer    │  │       API         │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         ▲                                    ▲
         │ WebSocket                          │ HTTP
         │ (status, commands)                 │ (inference)
         ▼                                    │
┌─────────────────┐                          │
│ Inference Runner│◄─────────────────────────┘
│  (GatewayClient)│   HTTP proxy to runner's /v1/...
└─────────────────┘
```

## Components

### 1. WebSocket Server
- Accept runner connections at `/ws/runners`
- Authenticate via `auth_token` in config
- Handle protocol messages from `simple-ai-common/src/protocol.rs`

### 2. Runner Registry
- Track connected runners by ID
- Store capabilities, loaded models, health status
- Detect disconnections/timeouts
- Support dynamic runner addition/removal

### 3. Load Balancer / Router
- Select runner for inference based on:
  - Model availability
  - Machine type (GPU vs CPU)
  - Current load/health
- Strategy: round-robin, least-loaded, or capability-based

### 4. API Layer
- `/v1/chat/completions` - route to appropriate runner
- `/v1/models` - aggregate models from all runners
- `/health` - gateway health
- `/runners` - admin endpoint listing connected runners

## Key Design Decisions

TBD after testing inference-runner standalone:
- WebSocket vs HTTP for inference request forwarding
- Where to place the gateway crate (new crate vs extend backend/)
- Authentication strategy for external clients

## Files to Create

| File | Description |
|------|-------------|
| `gateway/Cargo.toml` | New crate dependencies |
| `gateway/src/main.rs` | Server entry point |
| `gateway/src/config.rs` | Gateway configuration |
| `gateway/src/ws/mod.rs` | WebSocket handler |
| `gateway/src/registry.rs` | Runner registry |
| `gateway/src/router.rs` | Request routing logic |
| `gateway/src/api/mod.rs` | HTTP API routes |

## Protocol Reference

See `simple-ai-common/src/protocol.rs` for:
- `RunnerToGateway`: Registration, Heartbeat, CommandResponse
- `GatewayToRunner`: Welcome, LoadModel, UnloadModel, RequestStatus
- `RunnerStatus`: Capabilities, loaded models, metrics
