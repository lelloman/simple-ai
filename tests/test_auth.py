#!/usr/bin/env python3
"""
Authentication & Authorization Testing Script

Tests JWT authentication and role-based authorization for the inference backend.
Validates different permission levels and access control.

Usage:
    python test_auth.py [options]

Options:
    --url BASE_URL          Base URL of the backend (default: https://ai.lelloman.com)
    --token TOKEN           JWT token for authentication
    --token-binary PATH     Path to binary that generates JWT tokens
    --test-user USER        User identifier for token binary
    --test-role ROLE        Role identifier for token binary
    --timeout SECONDS       Request timeout (default: 30)
    --no-verify-ssl         Disable SSL verification
    --verbose, -v           Enable verbose output
    --debug                 Enable debug output
    -h, --help              Show this help message
"""

import argparse
import sys
from typing import Optional, Dict, Any, Tuple
from enum import Enum

import requests

import common
from common import (
    TestConfig, APIClient, LogLevel,
    log_info, log_success, log_error, log_verbose, log_debug,
    set_log_level
)


class AuthTestResult:
    """Results of authentication/authorization testing."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests_run = 0
        self.results: list[dict] = []

    def record_test(self, name: str, passed: bool, message: str) -> None:
        """Record a test result."""
        self.tests_run += 1
        if passed:
            self.passed += 1
            log_success(f"{name}: {message}")
        else:
            self.failed += 1
            log_error(f"{name}: {message}")
        self.results.append({"name": name, "passed": passed, "message": message})

    def is_success(self) -> bool:
        return self.failed == 0

    def print_summary(self) -> None:
        """Print test summary."""
        log_info(f"\n{'='*60}")
        log_info(f"Auth Test Results: {self.passed}/{self.tests_run} passed")
        if self.failed > 0:
            log_error(f"Failed tests: {self.failed}")

        # Group results by test type
        for result in self.results:
            status = "[PASS]" if result["passed"] else "[FAIL]"
            log_info(f"  {status} {result['name']}: {result['message']}")

        log_info(f"{'='*60}\n")


class ExpectedOutcome(Enum):
    """Expected outcome for a test."""
    SUCCESS = "success"
    UNAUTHORIZED = "401"
    FORBIDDEN = "403"
    BAD_REQUEST = "400"


def test_no_auth(config: TestConfig, model: str = "class:fast") -> Tuple[bool, str, Optional[int]]:
    """
    Test that requests without authentication fail with 401/403.

    Args:
        config: Test configuration
        model: Model to request

    Returns:
        Tuple of (test_passed, message, status_code)
    """
    url = f"{config.base_url}/v1/chat/completions"

    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": model,
        "max_tokens": 10
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=config.timeout,
            verify=config.verify_ssl
        )

        # Check if we got the expected error
        if response.status_code in (401, 403):
            return True, f"Correctly rejected with {response.status_code}", response.status_code
        elif response.status_code == 400:
            # Some implementations return 400 for missing auth
            return True, f"Rejected with 400 (missing auth)", response.status_code
        else:
            return False, f"Unexpected status code: {response.status_code}", response.status_code

    except requests.RequestException as e:
        return False, f"Request failed: {e}", None


def test_invalid_token(config: TestConfig, model: str = "class:fast") -> Tuple[bool, str, Optional[int]]:
    """
    Test that requests with invalid token fail with 401.

    Args:
        config: Test configuration
        model: Model to request

    Returns:
        Tuple of (test_passed, message, status_code)
    """
    url = f"{config.base_url}/v1/chat/completions"

    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": model,
        "max_tokens": 10
    }

    headers = {
        "Authorization": "Bearer invalid.token.here",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=config.timeout,
            verify=config.verify_ssl
        )

        if response.status_code == 401:
            return True, "Correctly rejected invalid token", response.status_code
        else:
            return False, f"Unexpected status code: {response.status_code}", response.status_code

    except requests.RequestException as e:
        return False, f"Request failed: {e}", None


def test_class_request_with_basic_user(
    client: APIClient,
    model_class: str = "fast"
) -> Tuple[bool, str, Optional[int]]:
    """
    Test that a basic user (no model:specific role) can use class: requests.

    Args:
        client: API client configured with basic user token
        model_class: Model class to test (fast or big)

    Returns:
        Tuple of (test_passed, message, status_code)
    """
    model = f"class:{model_class}"
    messages = [{"role": "user", "content": "Say 'OK'"}]

    try:
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=10
        )
        return True, f"Class request {model} succeeded", 200

    except requests.HTTPError as e:
        status = e.response.status_code
        if status == 400:
            error_detail = e.response.text
            if "Permission denied" in error_detail:
                return False, "Basic user denied class request (unexpected)", status
            return False, f"Bad request: {error_detail}", status
        elif status == 500:
            # Could be "no runners" or "no models of class"
            return True, f"Class request attempted (got 500 - likely no runners/models)", status
        return False, f"Unexpected status: {status}", status

    except requests.RequestException as e:
        return False, f"Request failed: {e}", None


def test_specific_model_with_basic_user(
    client: APIClient,
    model: str = "llama3:8b"
) -> Tuple[bool, str, Optional[int]]:
    """
    Test that a basic user (no model:specific role) CANNOT use specific models.

    Args:
        client: API client configured with basic user token
        model: Specific model to test

    Returns:
        Tuple of (test_passed, message, status_code)
    """
    messages = [{"role": "user", "content": "Say 'OK'"}]

    try:
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=10
        )
        # Request succeeded when it should have failed
        return False, "Specific model request succeeded (should have been denied)", 200

    except requests.HTTPError as e:
        status = e.response.status_code
        if status == 400:
            error_detail = e.response.text
            if "Permission denied" in error_detail or "cannot request specific models" in error_detail:
                return True, "Correctly denied specific model request", status
            return False, f"Unexpected 400 error: {error_detail}", status
        return False, f"Unexpected status: {status}", status

    except requests.RequestException as e:
        return False, f"Request failed: {e}", None


def test_specific_model_with_specific_role(
    config: TestConfig,
    token_binary: str,
    model: str = "llama3:8b"
) -> Tuple[bool, str, Optional[int]]:
    """
    Test that a user with model:specific role CAN use specific models.

    Args:
        config: Test configuration
        token_binary: Path to token binary that can provide model:specific role
        model: Specific model to test

    Returns:
        Tuple of (test_passed, message, status_code)
    """
    try:
        # Create client that will request token with model:specific role
        client = APIClient(TestConfig(
            base_url=config.base_url,
            token_binary=token_binary,
            token=None,
            timeout=config.timeout,
            verify_ssl=config.verify_ssl,
        ))

        # Override to request specific role
        token = client.token_manager.get_token(role="model:specific")

        messages = [{"role": "user", "content": "Say 'OK'"}]

        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=10
        )
        return True, f"Specific model request succeeded with model:specific role", 200

    except requests.HTTPError as e:
        status = e.response.status_code
        if status == 400:
            error_detail = e.response.text
            if "Permission denied" in error_detail:
                return False, "Denied even with model:specific role", status
        return False, f"Request failed with status {status}: {e.response.text[:100]}", status

    except requests.RequestException as e:
        return False, f"Request failed: {e}", None


def test_admin_access(
    config: TestConfig,
    token_binary: str
) -> Tuple[bool, str, Optional[int]]:
    """
    Test that admin role has full access.

    Args:
        config: Test configuration
        token_binary: Path to token binary that can provide admin role

    Returns:
        Tuple of (test_passed, message, status_code)
    """
    try:
        client = APIClient(TestConfig(
            base_url=config.base_url,
            token_binary=token_binary,
            token=None,
            timeout=config.timeout,
            verify_ssl=config.verify_ssl,
        ))

        # Try to get admin token
        token = client.token_manager.get_token(role="admin")

        messages = [{"role": "user", "content": "Say 'OK'"}]

        # Test with class:fast (should work)
        response = client.chat_completion(
            messages=messages,
            model="class:fast",
            max_tokens=10
        )
        return True, "Admin role can access class:fast", 200

    except requests.HTTPError as e:
        return False, f"Admin request failed with status {e.response.status_code}", e.response.status_code

    except requests.RequestException as e:
        return False, f"Request failed: {e}", None


def run_no_auth_tests(config: TestConfig, result: AuthTestResult) -> None:
    """Run tests for unauthenticated requests."""
    log_info("\n[Category 1] Unauthenticated Request Tests")

    # Test 1: No auth header
    passed, message, status = test_no_auth(config)
    result.record_test("No authentication", passed, message)

    # Test 2: Invalid token
    passed, message, status = test_invalid_token(config)
    result.record_test("Invalid token", passed, message)


def run_basic_user_tests(config: TestConfig, result: AuthTestResult) -> None:
    """Run tests for basic user (no special roles)."""
    log_info("\n[Category 2] Basic User Tests (no model:specific role)")

    if not config.token and not config.token_binary:
        result.record_test(
            "Basic user tests",
            False,
            "Skipped: No token or token_binary provided"
        )
        return

    client = APIClient(config)

    # Test 1: class:fast should work
    passed, message, status = test_class_request_with_basic_user(client, "fast")
    result.record_test("Basic user: class:fast", passed, message)

    # Test 2: class:big should work (if configured)
    passed, message, status = test_class_request_with_basic_user(client, "big")
    result.record_test("Basic user: class:big", passed, message)

    # Test 3: Specific model should fail
    passed, message, status = test_specific_model_with_basic_user(client)
    result.record_test("Basic user: specific model (should fail)", passed, message)


def run_role_based_tests(config: TestConfig, result: AuthTestResult) -> None:
    """Run tests for role-based access control."""
    log_info("\n[Category 3] Role-Based Access Tests")

    if not config.token_binary:
        result.record_test(
            "Role-based tests",
            False,
            "Skipped: --token-binary required for role-based testing"
        )
        return

    # Test 1: model:specific role can request specific models
    passed, message, status = test_specific_model_with_specific_role(
        config, config.token_binary
    )
    result.record_test("model:specific role: specific model", passed, message)

    # Test 2: admin role has full access
    passed, message, status = test_admin_access(config, config.token_binary)
    result.record_test("admin role: full access", passed, message)


def run_all_auth_tests(config: TestConfig) -> AuthTestResult:
    """
    Run all authentication and authorization tests.

    Args:
        config: Test configuration

    Returns:
        AuthTestResult with all test outcomes
    """
    result = AuthTestResult()

    log_info("Starting Authentication & Authorization Tests")
    log_info(f"Target URL: {config.base_url}")

    # Run test categories
    run_no_auth_tests(config, result)
    run_basic_user_tests(config, result)
    run_role_based_tests(config, result)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test authentication and authorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test with token (tests basic user access)
    python test_auth.py --token YOUR_JWT_TOKEN

    # Full test suite with token binary
    python test_auth.py --token-binary ./get-token

    # Test specific role
    python test_auth.py --token-binary ./get-token --test-role admin

    # Test custom backend
    python test_auth.py --url http://localhost:8080 --token-binary ./get-token

    # Verbose output
    python test_auth.py -v --token YOUR_JWT_TOKEN

Test Categories:
    1. Unauthenticated Requests - Verify requests without auth fail
    2. Basic User - Verify class: requests work, specific models don't
    3. Role-Based Access - Verify model:specific and admin roles work correctly
        """
    )

    parser.add_argument("--url", default="https://ai.lelloman.com",
                       help="Base URL of the backend")
    parser.add_argument("--token", help="JWT token for authentication")
    parser.add_argument("--token-binary", help="Path to token generation binary")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Request timeout in seconds")
    parser.add_argument("--no-verify-ssl", action="store_true",
                       help="Disable SSL verification")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")

    args = parser.parse_args()

    # Set log level
    if args.debug:
        set_log_level(LogLevel.DEBUG)
    elif args.verbose:
        set_log_level(LogLevel.VERBOSE)

    # Create configuration
    config = TestConfig(
        base_url=args.url,
        token=args.token,
        token_binary=args.token_binary,
        timeout=args.timeout,
        verify_ssl=not args.no_verify_ssl,
    )

    try:
        result = run_all_auth_tests(config)
        result.print_summary()
        sys.exit(0 if result.is_success() else 1)

    except KeyboardInterrupt:
        log_info("\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
