#!/bin/bash
# Inner test runner script — executed inside the test-runner container.
set -e

TOKEN_BINARY="/app/e2e/get-token.sh"
ARGS="--url http://backend:8080 --token-binary $TOKEN_BINARY --no-verify-ssl -v"

echo "=== Running auth tests ==="
python test_auth.py $ARGS

echo "=== Running model routing tests ==="
python test_model_routing.py $ARGS --model class:fast

echo "=== Running workload routing tests ==="
python test_workload_routing.py $ARGS --requests 6

echo "=== Running speculative wake priority tests ==="
python test_speculative_wake_priority.py $ARGS --preferred-model llama3:8b --preferred-wait 30

echo "=== All tests passed ==="
