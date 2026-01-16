# PLAN.md

## P0 - Critical (Production Ready)

### Backend
- [ ] Address compiler warnings (unused imports: `std::sync::Arc`, unused variables)
  - Fix warnings in `backend/src/` to ensure clean build output
- [ ] Add rate limiting for API gateway
  - Prevent abuse by limiting requests per user/time window
- [ ] Implement circuit breaker for Ollama proxy
  - Prevent cascading failures when Ollama is unavailable

### Android
- [ ] Fix JSON null handling edge cases in CloudLLmClient
  - Ensure graceful handling of null responses from cloud API
- [ ] Add null safety checks for NLU engine loading
  - Prevent crashes when NLU models fail to load

### Both
- [ ] Complete E2E test coverage for critical paths
  - Achieve 90%+ coverage for core user journeys
- [ ] Security audit for OIDC implementation
  - Review JWT validation, token refresh, and scope handling

---

## P1 - High Priority (Core Improvements)

### Backend
- [ ] Request queuing system for Ollama proxy
  - Handle burst traffic and prevent Ollama overload
- [ ] Add support for multiple Ollama instances (load balancing)
  - Distribute load across multiple inference servers
- [ ] Database connection pooling optimizations
  - Improve throughput under high concurrency
- [ ] Enhanced metrics endpoint
  - Add detailed performance and usage metrics
- [ ] Audit log archival/rotation
  - Prevent disk fills and enable log analysis

### Android
- [ ] Dark mode support (Material 3 theme)
  - Implement system-wide dark theme following Material 3 guidelines
- [ ] Onboarding flow for first-time users
  - Guide new users through initial setup and model selection
- [ ] Progress indicators for model downloads
  - Show download progress, ETA, and status for all model operations
- [ ] Background download queue with priority
  - Allow downloads to continue when app is minimized
- [ ] Storage management utilities
  - Help users manage disk space used by models

### NLU
- [ ] Multi-intent classification support
  - Allow users to express multiple intents in a single utterance
- [ ] Custom adapter management UI
  - Allow users to upload and manage custom NLU adapters
- [ ] Enhanced entity extraction accuracy
  - Improve recognition of named entities and types

---

## P2 - Medium Priority (Enhancements)

### Backend
- [ ] Distributed tracing integration
  - Track requests across services for debugging
- [ ] Log aggregation setup
  - Centralize logs for production monitoring
- [ ] Multi-architecture Docker builds
  - Support x86_64 in addition to ARM
- [ ] OpenAPI/Swagger documentation
  - Auto-generate API docs from code

### Android
- [ ] Model version management UI
  - Show installed model versions and enable updates
- [ ] Efficient model update mechanisms
  - Implement delta updates to reduce download sizes
- [ ] Battery impact optimization
  - Reduce background processing when on battery
- [ ] Cold start performance improvements
  - Reduce app launch time
- [ ] Memory optimization for large models
  - Prevent OOM crashes with large language models

### User Experience
- [ ] Offline scenario handling improvements
  - Better UX when network is unavailable
- [ ] Privacy dashboard
  - Show users what data is processed locally vs cloud
- [ ] Voice input/output integration (TTS)
  - Add text-to-speech for voice command responses
- [ ] Multi-language NLU support
  - Support languages beyond English

### Infrastructure
- [ ] CI/CD security scanning
  - Add SAST/DAST tools to pipeline
- [ ] Automatic release pipeline
  - Streamline builds and deployments
- [ ] Load testing suite
  - Benchmark performance under expected load

---

## P3 - Nice to Have (Future)

- [ ] Web-based admin panel
  - Browser interface for backend administration
- [ ] Model hub for pre-trained adapters
  - Repository of community-shared NLU adapters
- [ ] Client integration examples/docs
  - Help developers integrate with SimpleAI service
- [ ] Performance benchmark suite
  - Track performance metrics over time
- [ ] Gamification/achievement system
  - Encourage exploration of AI features
- [ ] Team collaboration features
  - Shared models and configurations
- [ ] Advanced analytics dashboard
  - Visualize usage patterns and insights

---

## Completed

This section tracks completed items for reference.

- [ ] Initial project setup (Android service + Rust backend)
- [ ] AIDL interface definition and versioning
- [ ] Ollama proxy with OIDC authentication
- [ ] Audit logging system
- [ ] NLU with ONNX Runtime and XLM-RoBERTa
- [ ] Local LLM inference with llama.cpp
- [ ] Translation via ML Kit
- [ ] Cloud AI integration
- [ ] Admin UI with role-based access control
- [ ] Metrics endpoint
- [ ] Language detection with FastText (176 languages)
- [ ] Tool calling support for Ollama
- [ ] E2E smoke tests (136 tests)
- [ ] Jetpack Compose UI with Material 3
- [ ] Jetpack Navigation integration
