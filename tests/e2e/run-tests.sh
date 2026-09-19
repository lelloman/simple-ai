#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Tear down this Compose project even if a build or test fails.
trap 'docker compose down -v' EXIT

echo "=== Building E2E test containers ==="
docker compose build

echo "=== Starting infrastructure ==="
docker compose up -d mock-oidc backend fake-runner-gpu fake-runner-halo

echo "=== Waiting for runners to register ==="
sleep 5

echo "=== Running tests ==="
docker compose run --rm test-runner
