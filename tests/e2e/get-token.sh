#!/bin/bash
# Token binary shim for E2E tests.
# Mimics the token binary interface: get-token.sh token [--user USER] [--role ROLE]
shift  # skip "token" subcommand

USER="test-user"
ROLES="[]"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) ROLES="[\"$2\"]"; shift 2;;
    --user) USER="$2"; shift 2;;
    *) shift;;
  esac
done

curl -s -X POST "${MOCK_OIDC_URL:-http://mock-oidc:9090}/token" \
  -H "Content-Type: application/json" \
  -d "{\"sub\":\"$USER\",\"roles\":$ROLES}" | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])"
