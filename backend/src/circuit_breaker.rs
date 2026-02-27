use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Closed,
    Open { since: Instant },
    HalfOpen,
}

struct BreakerState {
    state: State,
    consecutive_failures: u32,
}

/// Circuit breaker for failing backends.
///
/// Tracks per-key failure counts and transitions through Closed → Open → HalfOpen → Closed.
/// When the circuit is Open, `is_available` returns false to fail fast.
pub struct CircuitBreaker {
    threshold: u32,
    recovery_timeout: Duration,
    states: Mutex<HashMap<String, BreakerState>>,
}

impl CircuitBreaker {
    pub fn new(threshold: u32, recovery_timeout_secs: u64) -> Self {
        Self {
            threshold,
            recovery_timeout: Duration::from_secs(recovery_timeout_secs),
            states: Mutex::new(HashMap::new()),
        }
    }

    /// Returns true if disabled (threshold == 0).
    fn is_disabled(&self) -> bool {
        self.threshold == 0
    }

    /// Check if a backend is available (not in Open state).
    ///
    /// Transitions Open → HalfOpen if recovery timeout has elapsed.
    pub fn is_available(&self, key: &str) -> bool {
        if self.is_disabled() {
            return true;
        }

        let mut states = self.states.lock().unwrap();
        let entry = match states.get_mut(key) {
            Some(e) => e,
            None => return true, // No state = Closed
        };

        match entry.state {
            State::Closed | State::HalfOpen => true,
            State::Open { since } => {
                if since.elapsed() >= self.recovery_timeout {
                    entry.state = State::HalfOpen;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Record a successful request. Resets the breaker to Closed.
    pub fn record_success(&self, key: &str) {
        if self.is_disabled() {
            return;
        }

        let mut states = self.states.lock().unwrap();
        if let Some(entry) = states.get_mut(key) {
            entry.state = State::Closed;
            entry.consecutive_failures = 0;
        }
    }

    /// Record a failed request. Opens the breaker if threshold is reached.
    pub fn record_failure(&self, key: &str) {
        if self.is_disabled() {
            return;
        }

        let mut states = self.states.lock().unwrap();
        let entry = states.entry(key.to_string()).or_insert(BreakerState {
            state: State::Closed,
            consecutive_failures: 0,
        });

        entry.consecutive_failures += 1;

        if entry.consecutive_failures >= self.threshold {
            entry.state = State::Open {
                since: Instant::now(),
            };
            tracing::warn!(
                "Circuit breaker opened for '{}' after {} consecutive failures",
                key,
                entry.consecutive_failures
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_disabled_when_threshold_zero() {
        let cb = CircuitBreaker::new(0, 30);
        // Always available even after failures
        cb.record_failure("test");
        cb.record_failure("test");
        cb.record_failure("test");
        assert!(cb.is_available("test"));
    }

    #[test]
    fn test_closed_to_open() {
        let cb = CircuitBreaker::new(3, 30);
        assert!(cb.is_available("runner-1"));

        cb.record_failure("runner-1");
        assert!(cb.is_available("runner-1"));

        cb.record_failure("runner-1");
        assert!(cb.is_available("runner-1"));

        cb.record_failure("runner-1"); // threshold reached
        assert!(!cb.is_available("runner-1"));
    }

    /// Helper: manually set the `since` instant for an Open breaker to simulate time passing.
    fn force_open_since(cb: &CircuitBreaker, key: &str, since: Instant) {
        let mut states = cb.states.lock().unwrap();
        if let Some(entry) = states.get_mut(key) {
            entry.state = State::Open { since };
        }
    }

    #[test]
    fn test_open_to_half_open_on_timeout() {
        let cb = CircuitBreaker::new(2, 60);
        cb.record_failure("key");
        cb.record_failure("key");
        assert!(!cb.is_available("key"));

        // Simulate recovery timeout elapsed
        force_open_since(&cb, "key", Instant::now() - Duration::from_secs(61));
        assert!(cb.is_available("key"));
    }

    #[test]
    fn test_half_open_to_closed_on_success() {
        let cb = CircuitBreaker::new(2, 60);
        cb.record_failure("key");
        cb.record_failure("key");

        // Simulate recovery timeout elapsed → HalfOpen
        force_open_since(&cb, "key", Instant::now() - Duration::from_secs(61));
        assert!(cb.is_available("key"));

        // Success resets to Closed
        cb.record_success("key");
        assert!(cb.is_available("key"));

        // Verify it's truly Closed (need full threshold to open again)
        cb.record_failure("key");
        assert!(cb.is_available("key")); // Only 1 failure, threshold is 2
    }

    #[test]
    fn test_half_open_to_open_on_failure() {
        let cb = CircuitBreaker::new(2, 60);
        cb.record_failure("key");
        cb.record_failure("key");

        // Simulate recovery timeout elapsed → HalfOpen
        force_open_since(&cb, "key", Instant::now() - Duration::from_secs(61));
        assert!(cb.is_available("key"));

        // Failure in HalfOpen should re-open (consecutive_failures already >= threshold)
        cb.record_failure("key");
        assert!(!cb.is_available("key"));
    }

    #[test]
    fn test_per_key_isolation() {
        let cb = CircuitBreaker::new(2, 30);

        cb.record_failure("runner-a");
        cb.record_failure("runner-a");
        assert!(!cb.is_available("runner-a"));

        // runner-b is unaffected
        assert!(cb.is_available("runner-b"));
    }

    #[test]
    fn test_success_resets_failure_count() {
        let cb = CircuitBreaker::new(3, 30);

        cb.record_failure("key");
        cb.record_failure("key");
        cb.record_success("key");

        // After reset, need 3 more failures to open
        cb.record_failure("key");
        assert!(cb.is_available("key"));
    }

    #[test]
    fn test_unknown_key_is_available() {
        let cb = CircuitBreaker::new(3, 30);
        assert!(cb.is_available("never-seen"));
    }
}
