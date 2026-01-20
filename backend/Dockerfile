FROM rust:1.91-bookworm AS builder

WORKDIR /app

# Copy workspace manifests and Cargo.lock
COPY Cargo.toml Cargo.lock ./
COPY simple-ai-common/Cargo.toml ./simple-ai-common/
COPY backend/Cargo.toml ./backend/
COPY inference-runner/Cargo.toml ./inference-runner/

# Create dummy sources to cache dependencies
RUN mkdir -p simple-ai-common/src backend/src inference-runner/src && \
    echo "pub fn dummy() {}" > simple-ai-common/src/lib.rs && \
    echo "fn main() {}" > backend/src/main.rs && \
    echo "fn main() {}" > inference-runner/src/main.rs
RUN cargo build --release -p simple-ai-backend
RUN rm -rf simple-ai-common/src backend/src inference-runner/src

# Copy actual source
COPY simple-ai-common/src ./simple-ai-common/src
COPY backend/src ./backend/src
COPY backend/templates ./backend/templates
COPY backend/static ./backend/static

# Build for release (touch both to ensure cargo rebuilds with actual sources)
RUN touch simple-ai-common/src/lib.rs backend/src/main.rs && cargo build --release -p simple-ai-backend

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/simple-ai-backend /usr/local/bin/

# Create data directory for SQLite and download language detection model
RUN mkdir -p /data
ENV DATABASE_URL=sqlite:/data/audit.db

# Download FastText language detection model
ADD https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz /data/lid.176.ftz
ENV SIMPLEAI__LANGUAGE__MODEL_PATH=/data/lid.176.ftz

EXPOSE 8080 8081

CMD ["simple-ai-backend"]
