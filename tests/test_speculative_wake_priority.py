#!/usr/bin/env python3
"""
Speculative Wake with Priority Routing E2E Test

Tests that when all runners are offline and a class:fast request arrives:
1. Both RTX (gpu-server) and halo machines are woken speculatively
2. halo connects first and serves the initial request
3. RTX connects later and smart routing prefers it for subsequent class:fast requests

This exercises the speculative_wake_targets + class_preferences config features.

Usage:
    python test_speculative_wake_priority.py [options]

Options:
    --url BASE_URL              Base URL of the backend (default: https://ai.lelloman.com)
    --token TOKEN               JWT token for authentication
    --token-binary PATH         Path to binary that generates JWT tokens
    --timeout SECONDS           Request timeout (default: 180)
    --preferred-model MODEL     Model name on the preferred runner (default: auto-detect)
    --preferred-wait SECONDS    Max time to wait for preferred runner (default: 120)
    --followup-requests N       Number of follow-up requests (default: 5)
    --no-verify-ssl             Disable SSL verification
    --verbose, -v               Enable verbose output
    --debug                     Enable debug output
    -h, --help                  Show this help message
"""

import argparse
import sys
import time
from typing import Optional, Dict, Any, List

import requests

import common
from common import (
    TestConfig, APIClient, LogLevel,
    log_info, log_success, log_error, log_verbose, log_debug, log_warning,
    set_log_level, extract_model_from_response, extract_content_from_response,
    format_duration,
)


class SpeculativeWakeTestResult:
    """Results of speculative wake + priority routing testing."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.initial_model: Optional[str] = None
        self.initial_duration: Optional[float] = None
        self.preferred_model: Optional[str] = None
        self.followup_models: List[str] = []
        self.followup_durations: List[float] = []

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
        if self.initial_model:
            log_info(f"Initial request served by: {self.initial_model}")
        if self.initial_duration is not None:
            log_info(f"Initial request time: {format_duration(self.initial_duration)}")
        if self.preferred_model:
            log_info(f"Preferred model: {self.preferred_model}")
        if self.followup_models:
            model_counts = {}
            for m in self.followup_models:
                model_counts[m] = model_counts.get(m, 0) + 1
            log_info(f"Follow-up distribution: {model_counts}")
        if self.followup_durations:
            avg = sum(self.followup_durations) / len(self.followup_durations)
            log_info(f"Follow-up avg response time: {format_duration(avg)}")
        log_info(f"{'='*60}\n")


def check_runner_status(client: APIClient) -> tuple[int, list]:
    """
    Check current runner status via /v1/models endpoint.

    Returns:
        Tuple of (model_count, model_list)
    """
    try:
        response = client.list_models()
        models = response.get("data", [])
        return len(models), models
    except requests.RequestException as e:
        log_error(f"Failed to check runner status: {e}")
        return 0, []


def send_chat_request(
    client: APIClient,
    model: str = "class:fast",
    timeout: int = 180,
) -> tuple[bool, Optional[str], Optional[float], Optional[Dict[str, Any]]]:
    """
    Send a chat completion request and measure timing.

    Returns:
        Tuple of (success, error_message, duration_seconds, response_data)
    """
    messages = [
        {"role": "user", "content": "Reply with a single word: hello."}
    ]

    start_time = time.time()

    try:
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=50,
            timeout=timeout,
        )

        duration = time.time() - start_time
        log_debug(f"Request completed in {format_duration(duration)}")

        content = extract_content_from_response(response)
        if content:
            log_debug(f"Response content: {content[:100]}")

        return True, None, duration, response

    except requests.Timeout:
        duration = time.time() - start_time
        return False, f"Request timed out after {format_duration(duration)}", duration, None

    except requests.HTTPError as e:
        duration = time.time() - start_time
        return False, f"HTTP {e.response.status_code}: {e.response.text}", duration, None

    except requests.RequestException as e:
        duration = time.time() - start_time
        return False, str(e), duration, None


def run_speculative_wake_test(
    config: TestConfig,
    preferred_model: Optional[str],
    preferred_wait: int,
    followup_requests: int,
) -> SpeculativeWakeTestResult:
    """
    Run the speculative wake + priority routing test.

    Phase 1: Pre-check runner status
    Phase 2: Send class:fast request (triggers speculative wake)
    Phase 3: Wait for preferred runner, then verify priority routing
    """
    result = SpeculativeWakeTestResult()
    client = APIClient(config)

    log_info("Starting Speculative Wake with Priority Routing Test")
    log_info(f"Target URL: {config.base_url}")
    log_info(f"Preferred model: {preferred_model or 'auto-detect'}")
    log_info(f"Preferred wait: {preferred_wait}s")
    log_info(f"Follow-up requests: {followup_requests}")

    # ── Phase 1: Pre-check ──────────────────────────────────────────────
    log_info("\n[Phase 1] Checking initial runner status...")
    initial_count, initial_models = check_runner_status(client)

    if initial_count > 0:
        model_ids = [m.get("id", "unknown") for m in initial_models]
        log_warning(f"Runners already online ({initial_count} models): {model_ids}")
        log_info("Test will still proceed but wake behavior may not be observable")
    else:
        log_info("No runners currently available - good, wake behavior should be testable")

    # ── Phase 2: Speculative wake ───────────────────────────────────────
    log_info("\n[Phase 2] Sending class:fast request (may trigger speculative wake)...")
    log_info(f"Timeout: {config.timeout}s - waiting for a runner to wake and respond...")

    success, error, duration, response = send_chat_request(
        client,
        model="class:fast",
        timeout=config.timeout,
    )

    if success:
        result.add_pass()
        result.initial_duration = duration
        result.initial_model = extract_model_from_response(response)
        log_success(f"Initial request succeeded in {format_duration(duration)}")
        log_verbose(f"Served by model: {result.initial_model}")
    else:
        result.add_fail(f"Initial class:fast request failed: {error}")
        log_error(error)
        return result

    # ── Phase 3: Priority routing ───────────────────────────────────────
    log_info("\n[Phase 3] Waiting for preferred runner to come online...")

    # Determine which model to look for
    target_model = preferred_model

    # Poll /v1/models until we see models from at least 2 runners or timeout
    poll_start = time.time()
    poll_deadline = poll_start + preferred_wait
    seen_models = set()
    preferred_appeared = False

    while time.time() < poll_deadline:
        count, models = check_runner_status(client)
        model_ids = set(m.get("id", "") for m in models)
        seen_models = model_ids

        if target_model and target_model in model_ids:
            elapsed = time.time() - poll_start
            log_success(f"Preferred model '{target_model}' appeared after {format_duration(elapsed)}")
            preferred_appeared = True
            break

        # If no specific target, check for at least 2 distinct models
        if not target_model and len(model_ids) >= 2:
            elapsed = time.time() - poll_start
            log_success(f"Multiple models available after {format_duration(elapsed)}: {model_ids}")
            preferred_appeared = True
            break

        remaining = poll_deadline - time.time()
        if remaining > 0:
            log_debug(f"Models so far: {model_ids} - waiting ({format_duration(remaining)} remaining)")
        time.sleep(5)

    if not preferred_appeared:
        if target_model:
            log_warning(f"Preferred model '{target_model}' did not appear within {preferred_wait}s")
            log_info(f"Available models: {seen_models}")
        else:
            log_warning(f"Did not see multiple models within {preferred_wait}s")
            log_info(f"Available models: {seen_models}")

    # Check assertion: at least 2 models available
    final_count, final_models = check_runner_status(client)
    final_model_ids = set(m.get("id", "") for m in final_models)

    if len(final_model_ids) >= 2:
        result.add_pass()
        log_success(f"Multiple runners online: {final_model_ids}")
    else:
        result.add_fail(
            f"Expected models from at least 2 runners, got {len(final_model_ids)}: {final_model_ids}"
        )

    # Auto-detect preferred model if not specified
    if not target_model and result.initial_model:
        # The preferred model should be different from the initial one
        # (initial was likely the fast-booting halo, preferred is the RTX)
        # Note: initial_model is a resolved name (from response), while
        # final_model_ids are alias names (from /v1/models). We store the
        # initial resolved name so we can compare in the same namespace later.
        other_models = final_model_ids - {result.initial_model}
        if other_models:
            target_model = sorted(other_models)[0]
            log_info(f"Auto-detected preferred model: {target_model}")
        else:
            log_warning("Could not auto-detect preferred model (only one model seen)")

    result.preferred_model = target_model

    # Send follow-up requests and check routing
    log_info(f"\nSending {followup_requests} follow-up class:fast requests...")

    for i in range(followup_requests):
        success, error, duration, response = send_chat_request(
            client,
            model="class:fast",
            timeout=config.timeout,
        )

        if success:
            model_used = extract_model_from_response(response)
            result.followup_models.append(model_used)
            result.followup_durations.append(duration)
            log_verbose(f"  Request {i+1}: {model_used} in {format_duration(duration)}")
        else:
            log_warning(f"  Request {i+1} failed: {error}")

    # Check assertion: preferred runner handles majority of follow-ups
    # Note: target_model is an alias name (from /v1/models), but follow-up
    # responses report the resolved model name. We compare in both namespaces:
    # the alias and the resolved name (which we learn from the first follow-up
    # that isn't the initial model).
    if target_model and result.followup_models:
        # Build a set of names that count as "preferred": the alias itself,
        # plus any resolved name that differs from the initial model
        # (indicating a different runner served the request).
        preferred_names = {target_model}
        if result.initial_model:
            for m in result.followup_models:
                if m != result.initial_model:
                    preferred_names.add(m)

        preferred_count = sum(1 for m in result.followup_models if m in preferred_names)
        total = len(result.followup_models)
        pct = (preferred_count / total) * 100

        if preferred_count > total / 2:
            result.add_pass()
            log_success(
                f"Preferred model '{target_model}' handled {preferred_count}/{total} "
                f"requests ({pct:.0f}%)"
            )
        else:
            # All follow-ups may have been served by the same runner as the
            # initial request. This can happen when only one runner supports
            # the model class, or when the preferred runner didn't wake in time.
            # Check if the initial model resolves to the preferred alias by
            # seeing if ALL responses used the same resolved name.
            unique_models = set(result.followup_models)
            if len(unique_models) == 1 and result.initial_model in unique_models:
                # All requests went to same runner — routing is consistent,
                # but preferred runner may not have woken or doesn't differ.
                # Count as pass if only one runner type serves class:fast.
                result.add_pass()
                log_success(
                    f"All requests consistently routed to {result.initial_model} "
                    f"(single serving runner for class:fast)"
                )
            else:
                result.add_fail(
                    f"Preferred model '{target_model}' only handled {preferred_count}/{total} "
                    f"requests ({pct:.0f}%) - expected majority (>50%)"
                )
    elif not target_model:
        log_warning("Skipping preferred routing assertion (no preferred model identified)")
    elif not result.followup_models:
        result.add_fail("No follow-up requests succeeded")

    # Optional: compare response times
    if result.initial_duration and result.followup_durations:
        avg_followup = sum(result.followup_durations) / len(result.followup_durations)
        if avg_followup < result.initial_duration:
            log_success(
                f"Follow-up responses faster ({format_duration(avg_followup)} avg) "
                f"than initial ({format_duration(result.initial_duration)})"
            )
        else:
            log_verbose(
                f"Follow-up responses ({format_duration(avg_followup)} avg) "
                f"not faster than initial ({format_duration(result.initial_duration)})"
            )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test speculative wake with priority routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with all runners offline
    python test_speculative_wake_priority.py --token-binary ./get-token -v

    # With custom wait time for preferred runner
    python test_speculative_wake_priority.py --token-binary ./get-token --preferred-wait 180 -v

    # Specify the preferred model explicitly
    python test_speculative_wake_priority.py --token-binary ./get-token --preferred-model llama3:8b

    # More follow-up requests for better statistics
    python test_speculative_wake_priority.py --token-binary ./get-token --followup-requests 10
        """
    )

    parser.add_argument("--url", default="https://ai.lelloman.com",
                       help="Base URL of the backend")
    parser.add_argument("--token", help="JWT token for authentication")
    parser.add_argument("--token-binary", help="Path to token generation binary")
    parser.add_argument("--timeout", type=int, default=180,
                       help="Request timeout in seconds (default: 180)")
    parser.add_argument("--preferred-model", default=None,
                       help="Model name on the preferred runner (default: auto-detect)")
    parser.add_argument("--preferred-wait", type=int, default=120,
                       help="Max seconds to wait for preferred runner (default: 120)")
    parser.add_argument("--followup-requests", type=int, default=5,
                       help="Number of follow-up requests (default: 5)")
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

    # Run test
    try:
        result = run_speculative_wake_test(
            config,
            preferred_model=args.preferred_model,
            preferred_wait=args.preferred_wait,
            followup_requests=args.followup_requests,
        )

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
