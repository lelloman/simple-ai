#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Building E2E test containers ==="
docker compose build

echo "=== Starting infrastructure ==="
docker compose up -d mock-oidc backend fake-runner-gpu fake-runner-halo

echo "=== Waiting for runners to register ==="
sleep 5

echo "=== Running tests ==="
docker compose run --rm test-runner
EXIT_CODE=$?

echo "=== Tearing down ==="
docker compose down -v

exit $EXIT_CODE
