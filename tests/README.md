# Backend Inference Testing Scripts

This directory contains test scripts for validating the inference capabilities of the backend at `ai.lelloman.com`. The scripts use the OpenAI-compatible `/v1/chat/completions` endpoint.

## Prerequisites

### Python Requirements

Install the required Python packages:

```bash
pip install requests python-dotenv
```

Or using requirements.txt:

```bash
pip install -r requirements.txt
```

### Authentication

The test scripts require JWT authentication. You can provide tokens in three ways:

1. **Direct token**: Use `--token YOUR_JWT_TOKEN`
2. **Token binary**: Use `--token-binary /path/to/token-binary` to invoke a binary that generates tokens
3. **Environment variable**: Set `SIMPLEAI_TEST_TOKEN` environment variable

## Test Scripts

### 1. `test_wol.py` - Wake-on-LAN Testing

Verifies WOL (Wake-on-LAN) functionality wakes runners when needed.

**Usage:**

```bash
# Basic test
python test_wol.py --token YOUR_JWT_TOKEN

# Test with specific model
python test_wol.py --token-binary ./get-token --model llama3:8b

# Timing-focused test
python test_wol.py --timing --token YOUR_JWT_TOKEN

# Verbose output
python test_wol.py -v --token YOUR_JWT_TOKEN
```

**What it tests:**
- Checks initial runner status via `/v1/models`
- Sends inference request that may trigger auto-wake
- Measures time until runner responds
- Verifies request completes successfully

**Options:**
- `--url BASE_URL` - Backend URL (default: https://ai.lelloman.com)
- `--token TOKEN` - JWT token for authentication
- `--token-binary PATH` - Path to token generation binary
- `--timeout SECONDS` - Request timeout (default: 120)
- `--model MODEL` - Model to request (default: class:fast)
- `--timing` - Run timing-focused test
- `--no-verify-ssl` - Disable SSL verification
- `-v, --verbose` - Enable verbose output
- `--debug` - Enable debug output

---

### 2. `test_model_routing.py` - Model-Based Routing

Tests that requests are routed correctly based on model specification.

**Usage:**

```bash
# Test class:fast routing
python test_model_routing.py --model class:fast

# Test specific model (requires model:specific role)
python test_model_routing.py --model llama3:8b --token-binary ./get-token

# Test multiple models
python test_model_routing.py --models class:fast class:big

# Run permission tests
python test_model_routing.py --test-permissions
```

**What it tests:**
- **Specific model request**: `model: "llama3:8b"` (requires `model:specific` role)
- **Fast class request**: `model: "class:fast"` (any authenticated user)
- **Big class request**: `model: "class:big"` (any authenticated user)
- Permission checks for different user roles

**Options:**
- `--url BASE_URL` - Backend URL
- `--token TOKEN` - JWT token
- `--token-binary PATH` - Path to token binary
- `--model MODEL` - Single model to test
- `--models MODEL1 MODEL2 ...` - Multiple models to test
- `--test-permissions` - Run permission-based access tests
- `--timeout SECONDS` - Request timeout (default: 60)

---

### 3. `test_workload_routing.py` - Load Balancing

Verifies requests are distributed across available runners.

**Usage:**

```bash
# Basic test with 10 concurrent requests
python test_workload_routing.py --requests 10

# Test with specific model
python test_workload_routing.py --model class:fast --requests 20

# Sequential test for round-robin
python test_workload_routing.py --sequential --requests 15

# Timing analysis
python test_workload_routing.py --analyze-timing
```

**What it tests:**
- Sends concurrent requests (configurable count)
- Tracks which runner/model handles each request
- Verifies distribution across runners (not all to same runner)
- Analyzes response times for load balancing patterns

**Options:**
- `--url BASE_URL` - Backend URL
- `--token TOKEN` - JWT token
- `--token-binary PATH` - Path to token binary
- `--model MODEL` - Model to request (default: class:fast)
- `--requests N` - Number of requests (default: 10)
- `--workers N` - Max concurrent workers (default: 10)
- `--sequential` - Run requests sequentially instead of concurrently
- `--analyze-timing` - Run timing analysis
- `--timeout SECONDS` - Request timeout (default: 60)

---

### 4. `test_auth.py` - Authentication & Authorization

Tests authentication with different permission levels.

**Usage:**

```bash
# Basic test with token
python test_auth.py --token YOUR_JWT_TOKEN

# Full test suite with token binary (tests all roles)
python test_auth.py --token-binary ./get-token

# Verbose output
python test_auth.py -v --token YOUR_JWT_TOKEN
```

**What it tests:**

1. **No authentication**: Request without JWT → 401/403 expected
2. **Basic user** (no special roles):
   - Can use `class:fast`/`class:big`
   - Cannot use specific models
3. **model:specific role**: Can use any specific model by name
4. **admin role**: Full access

**Options:**
- `--url BASE_URL` - Backend URL
- `--token TOKEN` - JWT token
- `--token-binary PATH` - Path to token binary
- `--timeout SECONDS` - Request timeout (default: 30)

---

## Common Options

All scripts support these options:

| Option | Description |
|--------|-------------|
| `--url BASE_URL` | Backend URL (default: https://ai.lelloman.com) |
| `--token TOKEN` | JWT token for authentication |
| `--token-binary PATH` | Path to binary that generates JWT tokens |
| `--timeout SECONDS` | Request timeout in seconds |
| `--no-verify-ssl` | Disable SSL verification (for self-signed certs) |
| `-v, --verbose` | Enable verbose output |
| `--debug` | Enable debug output (more detailed) |
| `-h, --help` | Show help message |

---

## Token Binary Interface

When using `--token-binary`, the script invokes the binary with optional arguments:

```bash
# Get default user token
./get-token

# Get token for specific user
./get-token --user myuser

# Get token for specific role
./get-token --role model:specific

# Combined
./get-token --user admin --role admin
```

The binary should output only the JWT token to stdout.

---

## Exit Codes

- `0` - All tests passed
- `1` - Some tests failed
- `130` - Interrupted by user (Ctrl+C)

---

## Examples

### Complete Test Suite

Run all tests with a token binary:

```bash
# WOL test
python test_wol.py --token-binary ./get-token

# Model routing test
python test_model_routing.py --token-binary ./get-token --models class:fast class:big

# Workload distribution test
python test_workload_routing.py --token-binary ./get-token --requests 20

# Auth test
python test_auth.py --token-binary ./get-token
```

### Quick Smoke Test

Test basic connectivity and authentication:

```bash
python test_model_routing.py --model class:fast --token YOUR_TOKEN
```

### Debug Failed Tests

Run with debug output to see detailed information:

```bash
python test_wol.py --debug --token YOUR_TOKEN
```

---

## Troubleshooting

### "No JWT token available"

Provide a token via one of the methods:
- `--token YOUR_JWT_TOKEN`
- `--token-binary /path/to/get-token`
- Environment variable `SIMPLEAI_TEST_TOKEN`

### "Request timed out"

Increase timeout with `--timeout 180` (or higher for slow WOL).

### SSL Verification Errors

Disable SSL verification for testing with self-signed certificates:
```bash
python test_wol.py --no-verify-ssl
```

### "Permission denied" errors

These are expected when testing without proper roles. Use `--token-binary` with role support to test different permission levels.

---

## Backend Reference

Key backend files referenced by tests:

| Component | Path |
|-----------|------|
| Chat endpoint | `backend/src/routes/chat.rs` |
| Router/load balancer | `backend/src/gateway/router.rs` |
| Model classification | `backend/src/gateway/model_class.rs` |
| WOL service | `backend/src/wol.rs` |
| Authentication | `backend/src/auth/jwks.rs` |
| Configuration | `backend/src/config.rs` |
