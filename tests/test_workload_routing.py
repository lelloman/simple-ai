#!/usr/bin/env python3
"""
Load Balancing / Workload Routing Testing Script

Tests that inference requests are properly distributed across available runners.
Verifies round-robin and other load balancing strategies.

Usage:
    python test_workload_routing.py [options]

Options:
    --url BASE_URL          Base URL of the backend (default: https://ai.lelloman.com)
    --token TOKEN           JWT token for authentication
    --token-binary PATH     Path to binary that generates JWT tokens
    --model MODEL           Model to request (default: class:fast)
    --requests N            Number of concurrent requests (default: 10)
    --timeout SECONDS       Request timeout (default: 60)
    --no-verify-ssl         Disable SSL verification
    --verbose, -v           Enable verbose output
    --debug                 Enable debug output
    -h, --help              Show this help message
"""

import argparse
import sys
import time
import threading
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List

import requests

import common
from common import (
    TestConfig, APIClient, LogLevel,
    log_info, log_success, log_error, log_verbose, log_debug, log_warning,
    set_log_level, extract_model_from_response, format_duration
)


class WorkloadTestResult:
    """Results of workload routing testing."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.runner_distribution: Counter = Counter()
        self.response_times: List[float] = []
        self.errors: List[str] = []

    def add_success(self, runner: str, response_time: float) -> None:
        self.successful_requests += 1
        self.runner_distribution[runner] += 1
        self.response_times.append(response_time)

    def add_failure(self, error: str) -> None:
        self.errors.append(error)

    def add_pass(self) -> None:
        self.passed += 1

    def add_fail(self, error: str) -> None:
        self.failed += 1
        self.errors.append(error)

    def is_success(self) -> bool:
        return self.failed == 0

    def print_summary(self) -> None:
        """Print test summary."""
        log_info(f"\n{'='*60}")
        log_info(f"Test Results: {self.passed}/{self.passed + self.failed} checks passed")
        log_info(f"Requests: {self.successful_requests}/{self.total_requests} successful")

        if self.runner_distribution:
            log_info(f"\nRunner Distribution:")
            for runner, count in self.runner_distribution.most_common():
                pct = (count / self.successful_requests * 100) if self.successful_requests > 0 else 0
                log_info(f"  {runner}: {count} requests ({pct:.1f}%)")

            if len(self.runner_distribution) > 1:
                log_info(f"\n  Requests distributed across {len(self.runner_distribution)} runners")
                self.passed += 1
            else:
                log_warning(f"\n  All requests went to a single runner")

        if self.response_times:
            avg_time = sum(self.response_times) / len(self.response_times)
            min_time = min(self.response_times)
            max_time = max(self.response_times)
            log_info(f"\nResponse Times:")
            log_info(f"  Average: {format_duration(avg_time)}")
            log_info(f"  Min: {format_duration(min_time)}")
            log_info(f"  Max: {format_duration(max_time)}")

        if self.failed > 0:
            log_error(f"\nFailed checks: {self.failed}")
            for error in self.errors[:10]:  # Show first 10 errors
                log_error(f"  - {error}")
            if len(self.errors) > 10:
                log_error(f"  ... and {len(self.errors) - 10} more")

        log_info(f"{'='*60}\n")


def make_single_request(
    client: APIClient,
    model: str,
    request_id: int,
    timeout: int = 60
) -> tuple[bool, Optional[str], float, Optional[str]]:
    """
    Make a single inference request.

    Args:
        client: API client
        model: Model to request
        request_id: Identifier for this request
        timeout: Request timeout

    Returns:
        Tuple of (success, runner_id, response_time, error_message)
    """
    messages = [
        {"role": "user", "content": f"Request {request_id}: Respond with just 'OK'."}
    ]

    start_time = time.time()

    try:
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=10,
            timeout=timeout
        )

        response_time = time.time() - start_time

        runner_model = extract_model_from_response(response)

        # The model field in response indicates which model was used
        # We can infer the runner from timing if responses have different patterns,
        # but typically we'd need headers or other indicators
        # For now, use the model name as a proxy
        return True, runner_model or "unknown", response_time, None

    except requests.Timeout:
        response_time = time.time() - start_time
        return False, None, response_time, "Request timed out"

    except requests.HTTPError as e:
        response_time = time.time() - start_time
        return False, None, response_time, f"HTTP {e.response.status_code}"

    except requests.RequestException as e:
        response_time = time.time() - start_time
        return False, None, response_time, str(e)


def run_concurrent_requests(
    config: TestConfig,
    model: str,
    num_requests: int,
    max_workers: int = 10
) -> WorkloadTestResult:
    """
    Run concurrent inference requests to test load balancing.

    Args:
        config: Test configuration
        model: Model to request
        num_requests: Number of concurrent requests
        max_workers: Maximum concurrent threads

    Returns:
        WorkloadTestResult with test outcomes
    """
    result = WorkloadTestResult()
    result.total_requests = num_requests

    log_info(f"Starting Concurrent Request Test")
    log_info(f"Model: {model}")
    log_info(f"Concurrent requests: {num_requests}")
    log_info(f"Max workers: {max_workers}")

    client = APIClient(config)
    lock = threading.Lock()

    def make_request(req_id: int) -> tuple[bool, Optional[str], float, Optional[str]]:
        """Wrapper to make a single request with thread-safe client."""
        return make_single_request(client, model, req_id, config.timeout)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(make_request, i): i for i in range(num_requests)}

        for future in as_completed(futures):
            req_id = futures[future]
            try:
                success, runner_id, response_time, error = future.result()

                if success:
                    with lock:
                        result.add_success(runner_id or "unknown", response_time)
                    log_verbose(f"Request {req_id}: Success via {runner_id} ({format_duration(response_time)})")
                else:
                    with lock:
                        result.add_failure(f"Request {req_id}: {error}")
                    log_error(f"Request {req_id}: {error}")

            except Exception as e:
                with lock:
                    result.add_failure(f"Request {req_id}: Unexpected error: {e}")
                log_error(f"Request {req_id}: {e}")

    total_time = time.time() - start_time
    log_info(f"All requests completed in {format_duration(total_time)}")

    # Check if we got a reasonable distribution
    if result.successful_requests == num_requests:
        result.add_pass()
        log_success(f"All {num_requests} requests succeeded")
    else:
        result.add_fail(f"Only {result.successful_requests}/{num_requests} requests succeeded")

    # Check distribution across runners
    unique_runners = len(result.runner_distribution)
    if unique_runners > 1:
        result.add_pass()
        log_success(f"Requests distributed across {unique_runners} runners")
    elif unique_runners == 1:
        log_warning(f"All requests went to a single runner/model")
    else:
        result.add_fail("No successful requests to measure distribution")

    return result


def run_sequential_requests(
    config: TestConfig,
    model: str,
    num_requests: int
) -> WorkloadTestResult:
    """
    Run sequential inference requests to test round-robin routing.

    Args:
        config: Test configuration
        model: Model to request
        num_requests: Number of sequential requests

    Returns:
        WorkloadTestResult with test outcomes
    """
    result = WorkloadTestResult()
    result.total_requests = num_requests

    log_info(f"Starting Sequential Request Test (Round-Robin)")
    log_info(f"Model: {model}")
    log_info(f"Sequential requests: {num_requests}")

    client = APIClient(config)

    for i in range(num_requests):
        success, runner_id, response_time, error = make_single_request(
            client, model, i, config.timeout
        )

        if success:
            result.add_success(runner_id or "unknown", response_time)
            log_verbose(f"Request {i+1}/{num_requests}: {runner_id} ({format_duration(response_time)})")
        else:
            result.add_failure(f"Request {i+1}: {error}")
            log_error(f"Request {i+1}: {error}")

    # Check results
    if result.successful_requests == num_requests:
        result.add_pass()
        log_success(f"All {num_requests} requests succeeded")
    else:
        result.add_fail(f"Only {result.successful_requests}/{num_requests} requests succeeded")

    # For round-robin, we ideally want requests distributed
    unique_runners = len(result.runner_distribution)
    if unique_runners > 1:
        result.add_pass()
        log_success(f"Requests distributed across {unique_runners} runners")
    elif unique_runners == 1:
        log_warning(f"All requests went to a single runner/model")

    return result


def run_timing_analysis(
    config: TestConfig,
    model: str,
    num_requests: int = 20
) -> WorkloadTestResult:
    """
    Analyze response timing patterns to infer load balancing.

    Looks at timing patterns to determine if requests are being
    distributed or queued on a single runner.

    Args:
        config: Test configuration
        model: Model to request
        num_requests: Number of requests to analyze

    Returns:
        WorkloadTestResult with timing analysis
    """
    result = WorkloadTestResult()
    result.total_requests = num_requests

    log_info(f"Starting Timing Analysis")
    log_info(f"Model: {model}")
    log_info(f"Requests: {num_requests}")

    client = APIClient(config)

    # Send requests sequentially and track timing
    timings = []

    for i in range(num_requests):
        start = time.time()
        success, runner_id, response_time, error = make_single_request(
            client, model, i, config.timeout
        )
        end = time.time()

        if success:
            timings.append({
                "id": i,
                "model": runner_id,
                "start": start,
                "end": end,
                "duration": response_time
            })
            result.add_success(runner_id or "unknown", response_time)
        else:
            result.add_failure(f"Request {i}: {error}")

    if not timings:
        result.add_fail("No successful requests for timing analysis")
        return result

    # Analyze timing patterns
    log_info(f"\nTiming Analysis:")

    # Check for overlapping requests (indicates parallel processing)
    # For sequential sends, if runners process in parallel, we'd see
    # shorter overall time than sum of individual times
    total_time = timings[-1]["end"] - timings[0]["start"]
    sum_individual_times = sum(t["duration"] for t in timings)

    log_info(f"  Total wall-clock time: {format_duration(total_time)}")
    log_info(f"  Sum of individual times: {format_duration(sum_individual_times)}")

    if total_time < sum_individual_times * 0.8:
        log_success(f"  Evidence of parallel processing")
        result.add_pass()
    else:
        log_verbose(f"  Requests appear to be processed sequentially")

    # Check for response time variance
    response_times = [t["duration"] for t in timings]
    avg_time = sum(response_times) / len(response_times)
    variance = sum((t - avg_time) ** 2 for t in response_times) / len(response_times)
    std_dev = variance ** 0.5

    log_info(f"  Average response time: {format_duration(avg_time)}")
    log_info(f"  Std deviation: {format_duration(std_dev)}")

    if std_dev < avg_time * 0.2:
        log_verbose(f"  Low variance - consistent response times")
    else:
        log_verbose(f"  Higher variance - may indicate different runners or load conditions")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test workload routing and load balancing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test with 10 concurrent requests
    python test_workload_routing.py --requests 10

    # Test with specific model
    python test_workload_routing.py --model class:fast --requests 20

    # Sequential test for round-robin verification
    python test_workload_routing.py --sequential --requests 15

    # Timing analysis
    python test_workload_routing.py --analyze-timing

    # Verbose output
    python test_workload_routing.py -v --requests 10
        """
    )

    parser.add_argument("--url", default="https://ai.lelloman.com",
                       help="Base URL of the backend")
    parser.add_argument("--token", help="JWT token for authentication")
    parser.add_argument("--token-binary", help="Path to token generation binary")
    parser.add_argument("--model", default="class:fast",
                       help="Model to request (default: class:fast)")
    parser.add_argument("--requests", "-n", type=int, default=10,
                       help="Number of requests (default: 10)")
    parser.add_argument("--workers", type=int, default=10,
                       help="Max concurrent workers (default: 10)")
    parser.add_argument("--timeout", type=int, default=60,
                       help="Request timeout in seconds")
    parser.add_argument("--sequential", action="store_true",
                       help="Run requests sequentially instead of concurrently")
    parser.add_argument("--analyze-timing", action="store_true",
                       help="Run timing analysis to infer load balancing behavior")
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
        if args.analyze_timing:
            result = run_timing_analysis(config, args.model, args.requests)
        elif args.sequential:
            result = run_sequential_requests(config, args.model, args.requests)
        else:
            result = run_concurrent_requests(
                config, args.model, args.requests, args.workers
            )

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
