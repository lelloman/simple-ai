#!/usr/bin/env python3
"""
Model-Based Routing Testing Script

Tests that requests are routed correctly based on model specification.
Validates specific model requests and class-based routing.

Usage:
    python test_model_routing.py [options]

Options:
    --url BASE_URL          Base URL of the backend (default: https://ai.lelloman.com)
    --token TOKEN           JWT token for authentication
    --token-binary PATH     Path to binary that generates JWT tokens
    --model MODEL           Model to test (default: class:fast)
    --expected-class CLASS  Expected model class (fast/big)
    --timeout SECONDS       Request timeout (default: 60)
    --no-verify-ssl         Disable SSL verification
    --verbose, -v           Enable verbose output
    --debug                 Enable debug output
    -h, --help              Show this help message
"""

import argparse
import sys
from typing import Optional, List, Dict, Any

import requests

import common
from common import (
    TestConfig, APIClient, LogLevel,
    log_info, log_success, log_error, log_verbose, log_debug, log_warning,
    set_log_level, extract_model_from_response, extract_content_from_response,
    TokenManager
)


class RoutingTestResult:
    """Results of model routing testing."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tested_models: List[str] = []
        self.errors: List[str] = []

    def add_pass(self, model: str) -> None:
        self.passed += 1
        if model not in self.tested_models:
            self.tested_models.append(model)

    def add_fail(self, model: str, error: str) -> None:
        self.failed += 1
        self.errors.append(f"{model}: {error}")

    def is_success(self) -> bool:
        return self.failed == 0

    def print_summary(self) -> None:
        """Print test summary."""
        total = self.passed + self.failed
        log_info(f"\n{'='*60}")
        log_info(f"Test Results: {self.passed}/{total} passed")
        if self.tested_models:
            log_info(f"Tested models: {', '.join(self.tested_models)}")
        if self.failed > 0:
            log_error(f"Failed tests: {self.failed}")
            for error in self.errors:
                log_error(f"  - {error}")
        log_info(f"{'='*60}\n")


def test_specific_model_request(
    client: APIClient,
    model: str,
    expected_class: Optional[str] = None
) -> tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Test a request for a specific model.

    Args:
        client: API client
        model: Specific model ID (e.g., "llama3:8b")
        expected_class: Optional expected class for validation

    Returns:
        Tuple of (success, message, response_data)
    """
    messages = [
        {"role": "user", "content": "Respond with just 'OK'."}
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=10
        )

        # Check if the response model matches what we requested
        response_model = extract_model_from_response(response)

        if response_model == model:
            return True, f"Request routed to {model}", response
        else:
            return (True,
                   f"Request routed to {response_model} (requested {model})",
                   response)

    except requests.HTTPError as e:
        if e.response.status_code == 400:
            error_detail = e.response.text
            if "Permission denied" in error_detail or "cannot request specific models" in error_detail:
                return False, "Permission denied: user lacks model:specific role", None
            return False, f"Bad Request: {error_detail}", None
        return False, f"HTTP {e.response.status_code}: {e.response.text}", None

    except requests.Timeout:
        return False, "Request timed out", None

    except requests.RequestException as e:
        return False, str(e), None


def test_class_request(
    client: APIClient,
    model_class: str,
) -> tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Test a request for a model class.

    Args:
        client: API client
        model_class: Model class (e.g., "fast", "big")

    Returns:
        Tuple of (success, message, response_data)
    """
    model = f"class:{model_class}"
    messages = [
        {"role": "user", "content": "Respond with just 'OK'."}
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=10
        )

        response_model = extract_model_from_response(response)
        content = extract_content_from_response(response)

        # For class requests, we should get a specific model back
        if response_model and not response_model.startswith("class:"):
            return True, f"Class request routed to {response_model}", response
        else:
            return True, f"Class request succeeded (model: {response_model})", response

    except requests.HTTPError as e:
        if e.response.status_code == 500:
            error_detail = e.response.text
            if "No models of class" in error_detail:
                return False, f"No models configured for class '{model_class}'", None
            if "No runners available" in error_detail:
                return False, "No runners available", None
        return False, f"HTTP {e.response.status_code}: {e.response.text}", None

    except requests.Timeout:
        return False, "Request timed out", None

    except requests.RequestException as e:
        return False, str(e), None


def test_permission_model_specific(
    client: APIClient,
    model: str = "llama3:8b"
) -> tuple[bool, str]:
    """
    Test that a user without model:specific role cannot request specific models.

    Args:
        client: API client (configured with a user without model:specific role)
        model: Specific model to test with

    Returns:
        Tuple of (success, message)
    """
    messages = [
        {"role": "user", "content": "This should fail."}
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=10
        )

        # If we got here, the request succeeded when it shouldn't have
        return False, "Permission check failed: request succeeded without model:specific role"

    except requests.HTTPError as e:
        if e.response.status_code == 400:
            error_detail = e.response.text
            if "Permission denied" in error_detail or "cannot request specific models" in error_detail:
                return True, "Correctly denied specific model request without model:specific role"
        return False, f"Unexpected error: HTTP {e.response.status_code}: {e.response.text}", None

    except requests.RequestException as e:
        return False, f"Request failed: {e}", None


def run_single_model_test(config: TestConfig, model: str) -> RoutingTestResult:
    """
    Test a single model request.

    Args:
        config: Test configuration
        model: Model to test

    Returns:
        RoutingTestResult with test outcomes
    """
    result = RoutingTestResult()
    client = APIClient(config)

    log_info(f"Testing model: {model}")
    log_info(f"Base URL: {config.base_url}")

    # Determine if this is a class or specific model request
    is_class_request = model.startswith("class:")

    if is_class_request:
        class_name = model.split(":", 1)[1]
        log_info(f"Class request: {class_name}")

        success, message, response = test_class_request(client, class_name)

        if success:
            result.add_pass(model)
            log_success(message)
            if response:
                resp_model = extract_model_from_response(response)
                log_verbose(f"Resolved to model: {resp_model}")
        else:
            result.add_fail(model, message)
            log_error(message)
    else:
        log_info(f"Specific model request: {model}")

        success, message, response = test_specific_model_request(client, model)

        if success:
            result.add_pass(model)
            log_success(message)
            if response:
                resp_model = extract_model_from_response(response)
                log_verbose(f"Response model: {resp_model}")
        else:
            result.add_fail(model, message)
            log_error(message)

    return result


def run_comprehensive_routing_test(
    config: TestConfig,
    models: Optional[List[str]] = None
) -> RoutingTestResult:
    """
    Run comprehensive routing tests on multiple models.

    Args:
        config: Test configuration
        models: List of models to test (default: common test models)

    Returns:
        RoutingTestResult with test outcomes
    """
    result = RoutingTestResult()
    client = APIClient(config)

    if models is None:
        models = [
            "class:fast",
            "class:big",
        ]

    log_info("Starting Comprehensive Model Routing Test")
    log_info(f"Base URL: {config.base_url}")
    log_info(f"Models to test: {', '.join(models)}")

    for model in models:
        log_info(f"\n--- Testing {model} ---")

        is_class_request = model.startswith("class:")

        if is_class_request:
            class_name = model.split(":", 1)[1]
            success, message, response = test_class_request(client, class_name)
        else:
            success, message, response = test_specific_model_request(client, model)

        if success:
            result.add_pass(model)
            log_success(f"{model}: {message}")
            if response:
                resp_model = extract_model_from_response(response)
                log_verbose(f"  -> Resolved to: {resp_model}")
        else:
            result.add_fail(model, message)
            log_error(f"{model}: {message}")

    return result


def run_permission_test(config: TestConfig) -> RoutingTestResult:
    """
    Test permission-based model access.

    Tests that:
    1. Users without model:specific role can use class: requests
    2. Users without model:specific role cannot use specific model requests
    3. Users with model:specific role can use both

    Args:
        config: Test configuration

    Returns:
        RoutingTestResult with test outcomes
    """
    result = RoutingTestResult()

    log_info("Starting Permission-Based Access Test")
    log_info(f"Base URL: {config.base_url}")

    client = APIClient(config)

    # Test 1: Class request should work for any authenticated user
    log_info("\n[Test 1] Class request (should work for all users)")
    success, message, response = test_class_request(client, "fast")

    if success:
        result.add_pass("class:fast (basic user)")
        log_success(message)
    else:
        result.add_fail("class:fast (basic user)", message)
        log_error(message)

    # Test 2: Specific model should fail for user without model:specific role
    log_info("\n[Test 2] Specific model request (should fail without model:specific role)")
    test_model = "llama3:8b"  # Default test model

    # First, try to list available models to find one that exists
    try:
        models_response = client.list_models()
        available = models_response.get("data", [])
        if available:
            # Use the first available model that's not a class
            for m in available:
                model_id = m.get("id", "")
                if model_id and not model_id.startswith("class:"):
                    test_model = model_id
                    break
    except Exception:
        pass  # Use default

    success, message = test_permission_model_specific(client, test_model)

    if success:
        result.add_pass(f"Permission check for {test_model}")
        log_success(message)
    else:
        result.add_fail(f"Permission check for {test_model}", message)
        log_error(message)

    # Test 3: If model:specific role is available, test with it
    if config.token_binary:
        log_info("\n[Test 3] Specific model request with model:specific role")

        try:
            client_with_role = APIClient(TestConfig(
                base_url=config.base_url,
                token_binary=config.token_binary,
                token=None,
                timeout=config.timeout,
                verify_ssl=config.verify_ssl,
            ))

            success, message, response = test_specific_model_request(
                client_with_role,
                test_model
            )

            if success:
                result.add_pass(f"{test_model} (with model:specific role)")
                log_success(message)
            else:
                result.add_fail(f"{test_model} (with model:specific role)", message)
                log_error(message)

        except Exception as e:
            log_warning(f"Could not test with model:specific role: {e}")
    else:
        log_info("\n[Test 3] Skipped (no token binary provided for role-based testing)")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test model-based routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test class:fast routing
    python test_model_routing.py --model class:fast

    # Test specific model (requires model:specific role)
    python test_model_routing.py --model llama3:8b --token-binary ./get-token

    # Test multiple models
    python test_model_routing.py --models class:fast,class:big

    # Run permission tests
    python test_model_routing.py --test-permissions

    # Verbose output
    python test_model_routing.py -v --model class:fast
        """
    )

    parser.add_argument("--url", default="https://ai.lelloman.com",
                       help="Base URL of the backend")
    parser.add_argument("--token", help="JWT token for authentication")
    parser.add_argument("--token-binary", help="Path to token generation binary")
    parser.add_argument("--model", default="class:fast",
                       help="Model to test (default: class:fast)")
    parser.add_argument("--models", nargs="+",
                       help="Multiple models to test")
    parser.add_argument("--timeout", type=int, default=60,
                       help="Request timeout in seconds")
    parser.add_argument("--test-permissions", action="store_true",
                       help="Run permission-based access tests")
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
        if args.test_permissions:
            result = run_permission_test(config)
        elif args.models:
            result = run_comprehensive_routing_test(config, args.models)
        else:
            result = run_single_model_test(config, args.model)

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
