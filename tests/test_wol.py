#!/usr/bin/env python3
"""
Wake-on-LAN Testing Script

Tests the WOL (Wake-on-LAN) functionality of the backend.
Verifies that offline runners are automatically woken when inference requests arrive.

Usage:
    python test_wol.py [options]

Options:
    --url BASE_URL          Base URL of the backend (default: https://ai.lelloman.com)
    --token TOKEN           JWT token for authentication
    --token-binary PATH     Path to binary that generates JWT tokens
    --timeout SECONDS       Request timeout (default: 120)
    --wol-timeout SECONDS   Max time to wait for WOL (default: 90)
    --no-verify-ssl         Disable SSL verification
    --verbose, -v           Enable verbose output
    --debug                 Enable debug output
    -h, --help              Show this help message
"""

import argparse
import sys
import time
from typing import Optional, Dict, Any

import requests

import common
from common import (
    TestConfig, APIClient, LogLevel,
    log_info, log_success, log_error, log_verbose, log_debug,
    set_log_level, extract_content_from_response
)


class WOLTestResult:
    """Results of WOL testing."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.wake_duration: Optional[float] = None
        self.runner_id: Optional[str] = None
        self.errors = []

    def add_pass(self) -> None:
        self.passed += 1

    def add_fail(self, error: str) -> None:
        self.failed += 1
        self.errors.append(error)

    def is_success(self) -> bool:
        return self.failed == 0

    def print_summary(self) -> None:
        """Print test summary."""
        total = self.passed + self.failed
        log_info(f"\n{'='*60}")
        log_info(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            log_error(f"Failed tests: {self.failed}")
            for error in self.errors:
                log_error(f"  - {error}")
        if self.wake_duration:
            log_info(f"Wake time: {common.format_duration(self.wake_duration)}")
        if self.runner_id:
            log_info(f"Runner ID: {self.runner_id}")
        log_info(f"{'='*60}\n")


def check_runner_status(client: APIClient) -> tuple[int, list]:
    """
    Check current runner status via /v1/models endpoint.

    Returns:
        Tuple of (runner_count, model_list)
    """
    try:
        response = client.list_models()
        models = response.get("data", [])
        return len(models), models
    except requests.RequestException as e:
        log_error(f"Failed to check runner status: {e}")
        return 0, []


def attempt_inference_request(
    client: APIClient,
    model: str = "class:fast",
    timeout: int = 90
) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Attempt to send an inference request that should trigger auto-wake.

    Args:
        client: API client
        model: Model to request
        timeout: Time to wait for response

    Returns:
        Tuple of (success, error_message, response_data)
    """
    messages = [
        {"role": "user", "content": "Say 'WOL test successful' in exactly those words."}
    ]

    start_time = time.time()

    try:
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=50,
            timeout=timeout
        )

        duration = time.time() - start_time
        log_verbose(f"Request completed in {common.format_duration(duration)}")

        content = extract_content_from_response(response)
        if content:
            log_verbose(f"Response content: {content[:100]}...")

        return True, None, response

    except requests.Timeout:
        duration = time.time() - start_time
        return False, f"Request timed out after {common.format_duration(duration)}", None

    except requests.HTTPError as e:
        duration = time.time() - start_time
        error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
        return False, error_msg, None

    except requests.RequestException as e:
        duration = time.time() - start_time
        return False, str(e), None


def run_wol_test(config: TestConfig, model: str = "class:fast") -> WOLTestResult:
    """
    Run the WOL test sequence.

    Args:
        config: Test configuration
        model: Model to request

    Returns:
        WOLTestResult with test outcomes
    """
    result = WOLTestResult()
    client = APIClient(config)

    log_info("Starting Wake-on-LAN Test")
    log_info(f"Target model: {model}")
    log_info(f"Target URL: {config.base_url}")

    # Step 1: Check initial runner status
    log_info("\n[Step 1] Checking initial runner status...")
    initial_count, initial_models = check_runner_status(client)

    if initial_count > 0:
        log_info(f"Found {initial_count} models on {len(set(m.get('id', '') for m in initial_models))} runner(s)")
        for model_entry in initial_models[:5]:  # Show first 5
            log_verbose(f"  - {model_entry.get('id', 'unknown')}")
        if len(initial_models) > 5:
            log_verbose(f"  ... and {len(initial_models) - 5} more")
        result.add_pass()
    else:
        log_info("No runners currently available")
        result.add_pass()

    # Step 2: Send inference request (may trigger WOL)
    log_info("\n[Step 2] Sending inference request...")
    start_time = time.time()

    success, error, response = attempt_inference_request(
        client,
        model=model,
        timeout=config.timeout
    )

    wake_duration = time.time() - start_time

    if success:
        result.wake_duration = wake_duration
        result.add_pass()
        log_success(f"Request succeeded after {common.format_duration(wake_duration)}")

        if response:
            result.runner_id = response.get("model")
            log_verbose(f"Model used: {result.runner_id}")

            # Verify response contains expected content
            content = extract_content_from_response(response)
            if content:
                log_verbose(f"Assistant response: {content[:200]}")
    else:
        result.add_fail(f"Inference request failed: {error}")
        log_error(error)
        return result

    # Step 3: Verify runner is now available
    log_info("\n[Step 3] Verifying runner availability...")
    final_count, final_models = check_runner_status(client)

    if final_count > 0:
        log_success(f"Runner(s) now available: {final_count} models")
        result.add_pass()

        # Check if we have more models than before (runner woke up)
        if initial_count == 0 and final_count > 0:
            log_success(f"WOL appears successful: runner came online")
        elif final_count > initial_count:
            log_success(f"WOL appears successful: additional runner came online")
    else:
        result.add_fail("No runners available after request")
        log_error("Expected runner to be available after WOL")

    return result


def run_wol_timing_test(config: TestConfig, model: str = "class:fast") -> WOLTestResult:
    """
    Run a WOL timing test to measure wake duration.

    This test is designed to run when runners are offline to measure
    the time from request to successful response.
    """
    result = WOLTestResult()
    client = APIClient(config)

    log_info("Starting WOL Timing Test")
    log_info(f"Target model: {model}")
    log_info(f"Wake timeout: {config.timeout}s")

    # Check initial state
    log_info("\n[Step 1] Checking if runners are offline...")
    initial_count, _ = check_runner_status(client)

    if initial_count > 0:
        log_warning(f"Runners already online ({initial_count} models). "
                   "For accurate timing, ensure runners are offline first.")
        log_info("Continuing with test anyway...")

    # Send request and time it
    log_info("\n[Step 2] Sending request and measuring wake time...")
    start_time = time.time()

    success, error, response = attempt_inference_request(
        client,
        model=model,
        timeout=config.timeout
    )

    duration = time.time() - start_time

    if success:
        result.wake_duration = duration
        result.add_pass()
        log_success(f"Request completed in {common.format_duration(duration)}")

        if duration < 30:
            log_success("Fast wake (< 30s)")
        elif duration < 60:
            log_info("Moderate wake time (30-60s)")
        else:
            log_warning(f"Slow wake (> 60s): {common.format_duration(duration)}")

        if response:
            result.runner_id = response.get("model")
    else:
        result.add_fail(f"Request failed: {error}")
        log_error(error)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test Wake-on-LAN functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test with token
    python test_wol.py --token YOUR_JWT_TOKEN

    # Test with token binary
    python test_wol.py --token-binary ./get-token

    # Test specific model
    python test_wol.py --model llama3:8b

    # Verbose output
    python test_wol.py -v --token YOUR_JWT_TOKEN

    # Custom backend URL
    python test_wol.py --url http://localhost:8080 --token YOUR_JWT_TOKEN
        """
    )

    parser.add_argument("--url", default="https://ai.lelloman.com",
                       help="Base URL of the backend")
    parser.add_argument("--token", help="JWT token for authentication")
    parser.add_argument("--token-binary", help="Path to token generation binary")
    parser.add_argument("--timeout", type=int, default=120,
                       help="Request timeout in seconds")
    parser.add_argument("--model", default="class:fast",
                       help="Model to request (default: class:fast)")
    parser.add_argument("--no-verify-ssl", action="store_true",
                       help="Disable SSL verification")
    parser.add_argument("--timing", action="store_true",
                       help="Run timing-focused test")
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

    # Run appropriate test
    try:
        if args.timing:
            result = run_wol_timing_test(config, model=args.model)
        else:
            result = run_wol_test(config, model=args.model)

        result.print_summary()

        # Exit with appropriate code
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
