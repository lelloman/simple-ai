//! Persisted administrative state for an inference runner.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Whether a runner may receive new work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunnerState {
    /// Normal operation: the runner is dispatched to and may be woken.
    #[default]
    Enabled,
    /// No new requests are dispatched. Accepted work finishes and the
    /// runner's models are then unloaded.
    Draining,
    /// No requests are dispatched, the runner is never woken, and its models
    /// are kept unloaded.
    Disabled,
}

impl RunnerState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Enabled => "enabled",
            Self::Draining => "draining",
            Self::Disabled => "disabled",
        }
    }
}

impl fmt::Display for RunnerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for RunnerState {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "enabled" => Ok(Self::Enabled),
            "draining" => Ok(Self::Draining),
            "disabled" => Ok(Self::Disabled),
            other => Err(format!("unknown runner state '{other}'")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_strings_round_trip() {
        for state in [
            RunnerState::Enabled,
            RunnerState::Draining,
            RunnerState::Disabled,
        ] {
            assert_eq!(RunnerState::from_str(state.as_str()).unwrap(), state);
            assert_eq!(state.to_string(), state.as_str());
        }
        assert_eq!(RunnerState::default(), RunnerState::Enabled);
        assert!(RunnerState::from_str("paused").is_err());
    }
}
