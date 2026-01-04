FROM rust:1.91-bookworm AS builder

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock* ./

# Create dummy source to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# Copy actual source
COPY src ./src
COPY templates ./templates

# Build for release
RUN touch src/main.rs && cargo build --release

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

EXPOSE 8080

CMD ["simple-ai-backend"]
