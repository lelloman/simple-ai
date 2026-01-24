"""
Common utilities for backend inference testing scripts.

Provides shared functionality for authentication, API requests,
and response handling across all test scripts.
"""

import os
import subprocess
import sys
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

import requests


class LogLevel(Enum):
    """Logging levels for test output."""
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


# Global log level
_log_level = LogLevel.NORMAL


def set_log_level(level: LogLevel) -> None:
    """Set the global log level."""
    global _log_level
    _log_level = level


def log(message: str, level: LogLevel = LogLevel.NORMAL) -> None:
    """Log a message if the level is enabled."""
    if level.value <= _log_level.value:
        print(message)


def log_debug(message: str) -> None:
    """Log a debug message."""
    log(f"[DEBUG] {message}", LogLevel.DEBUG)


def log_verbose(message: str) -> None:
    """Log a verbose message."""
    log(message, LogLevel.VERBOSE)


def log_info(message: str) -> None:
    """Log an info message."""
    log(message, LogLevel.NORMAL)


def log_success(message: str) -> None:
    """Log a success message."""
    log(f"[PASS] {message}", LogLevel.NORMAL)


def log_error(message: str) -> None:
    """Log an error message."""
    print(f"[FAIL] {message}", file=sys.stderr)


def log_warning(message: str) -> None:
    """Log a warning message."""
    print(f"[WARN] {message}", file=sys.stderr)


@dataclass
class TestConfig:
    """Configuration for test scripts."""
    base_url: str = "https://ai.lelloman.com"
    token_binary: Optional[str] = None
    token: Optional[str] = None
    timeout: int = 120
    verify_ssl: bool = True
    log_level: LogLevel = LogLevel.NORMAL

    def __post_init__(self):
        set_log_level(self.log_level)


class TokenManager:
    """Manages JWT token acquisition and caching."""

    def __init__(self, config: TestConfig):
        self.config = config
        self._cached_token: Optional[str] = None
        self._cached_token_user: Optional[str] = None
        self._cached_token_role: Optional[str] = None

    def get_token(self, user: Optional[str] = None, role: Optional[str] = None) -> str:
        """
        Get a JWT token for authentication.

        Args:
            user: Optional user identifier to pass to the token binary
            role: Optional role to pass to the token binary

        Returns:
            JWT token string

        Raises:
            RuntimeError: If token cannot be obtained
        """
        # Return cached token if requesting the same user/role
        if (self._cached_token and
            self._cached_token_user == user and
            self._cached_token_role == role):
            log_debug("Using cached token")
            return self._cached_token

        # If a token is directly provided in config, use it
        if self.config.token and not user and not role:
            log_debug("Using token from config")
            return self.config.token

        # Try to get token from binary
        if self.config.token_binary:
            return self._get_token_from_binary(user, role)

        # Try environment variable
        env_token = os.getenv("SIMPLEAI_TEST_TOKEN")
        if env_token and not user and not role:
            log_debug("Using token from environment variable")
            return env_token

        raise RuntimeError(
            "No JWT token available. Provide one via:\n"
            "  - --token argument\n"
            "  - SIMPLEAI_TEST_TOKEN environment variable\n"
            "  - --token-binary to specify the token acquisition binary"
        )

    def _get_token_from_binary(self, user: Optional[str], role: Optional[str]) -> str:
        """Get token by invoking the configured binary."""
        cmd = [self.config.token_binary, "token"]

        if user:
            cmd.extend(["--user", user])
        if role:
            cmd.extend(["--role", role])

        log_debug(f"Executing: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
                env={**os.environ, 'HOME': os.path.expanduser('~')}
            )
            token = result.stdout.strip()
            if not token:
                raise RuntimeError("Token binary returned empty output")

            # Cache the token
            self._cached_token = token
            self._cached_token_user = user
            self._cached_token_role = role

            log_debug("Token obtained successfully")
            return token

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Token binary failed with exit code {e.returncode}\n"
                f"stderr: {e.stderr}"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Token binary timed out after 30 seconds")
        except FileNotFoundError:
            raise RuntimeError(f"Token binary not found: {self.config.token_binary}")


class APIClient:
    """Client for making API requests to the inference backend."""

    def __init__(self, config: TestConfig):
        self.config = config
        self.token_manager = TokenManager(config)
        self.session = requests.Session()
        self.session.verify = config.verify_ssl

    def get_auth_headers(self, user: Optional[str] = None, role: Optional[str] = None) -> Dict[str, str]:
        """Get headers with JWT authentication."""
        token = self.token_manager.get_token(user, role)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def list_models(self, user: Optional[str] = None, role: Optional[str] = None) -> Dict[str, Any]:
        """
        List available models via /v1/models endpoint.

        Returns:
            Dictionary with 'data' key containing list of models
        """
        url = f"{self.config.base_url}/v1/models"
        headers = self.get_auth_headers(user, role)

        log_debug(f"GET {url}")
        response = self.session.get(url, headers=headers, timeout=self.config.timeout)

        response.raise_for_status()
        return response.json()

    def chat_completion(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 100,
        user: Optional[str] = None,
        role: Optional[str] = None,
        stream: bool = False,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier (e.g., "llama3:8b", "class:fast")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            user: Optional user for token acquisition
            role: Optional role for token acquisition
            stream: Whether to use streaming mode

        Returns:
            Response dictionary from the API
        """
        url = f"{self.config.base_url}/v1/chat/completions"
        headers = self.get_auth_headers(user, role)

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if model:
            payload["model"] = model

        log_debug(f"POST {url} with model={model}")
        log_debug(f"Request: {payload}")

        response = self.session.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout if timeout is not None else self.config.timeout
        )

        response.raise_for_status()
        return response.json()

    def chat_completion_raw(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 100,
        user: Optional[str] = None,
        role: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> requests.Response:
        """
        Send a chat completion request and return the raw response.

        Useful for inspecting headers and status codes.
        """
        url = f"{self.config.base_url}/v1/chat/completions"
        headers = self.get_auth_headers(user, role)

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if model:
            payload["model"] = model

        log_debug(f"POST {url} with model={model}")

        return self.session.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout if timeout is not None else self.config.timeout
        )


def extract_model_from_response(response: Dict[str, Any]) -> Optional[str]:
    """Extract the model name from a chat completion response."""
    return response.get("model")


def extract_content_from_response(response: Dict[str, Any]) -> Optional[str]:
    """Extract the assistant's content from a chat completion response."""
    choices = response.get("choices", [])
    if choices and len(choices) > 0:
        message = choices[0].get("message", {})
        return message.get("content")
    return None


def wait_for_condition(
    condition_func,
    timeout_seconds: int = 60,
    poll_interval: float = 0.5,
    timeout_message: str = "Timeout waiting for condition"
) -> bool:
    """
    Wait for a condition to become true.

    Args:
        condition_func: Callable that returns True when condition is met
        timeout_seconds: Maximum time to wait
        poll_interval: Seconds between checks
        timeout_message: Error message on timeout

    Returns:
        True if condition was met, False on timeout

    Raises:
        TimeoutError: If timeout is exceeded
    """
    start_time = time.time()
    deadline = start_time + timeout_seconds

    while time.time() < deadline:
        if condition_func():
            return True
        time.sleep(poll_interval)

    raise TimeoutError(timeout_message)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"
