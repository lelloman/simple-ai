//! Capability types for the inference system.

use serde::{Deserialize, Serialize};

/// Abstract capabilities that runners can provide.
///
/// These are logical capability types, not specific models. Each runner
/// maps its available models to these capabilities via configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Fast chat completion - smaller models, quick responses
    FastChat,
    /// Large chat completion - bigger models, higher quality
    LargeChat,
    /// Vector embeddings generation
    Embeddings,
    /// Language translation
    Translation,
}

impl Capability {
    /// All capability variants for iteration.
    pub const ALL: [Capability; 4] = [
        Capability::FastChat,
        Capability::LargeChat,
        Capability::Embeddings,
        Capability::Translation,
    ];
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Capability::FastChat => write!(f, "fast_chat"),
            Capability::LargeChat => write!(f, "large_chat"),
            Capability::Embeddings => write!(f, "embeddings"),
            Capability::Translation => write!(f, "translation"),
        }
    }
}

/// Status of a capability on a runner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum CapabilityStatus {
    /// Model not loaded, can be loaded on demand
    Unloaded,
    /// Model is being loaded
    Loading {
        /// Progress as a fraction (0.0 to 1.0)
        #[serde(default)]
        progress: Option<f32>,
    },
    /// Model is loaded and ready for inference
    Loaded,
    /// Model is being unloaded
    Unloading,
    /// Model failed to load or encountered an error
    Error {
        /// Error message
        message: String,
    },
}

impl CapabilityStatus {
    pub fn is_ready(&self) -> bool {
        matches!(self, CapabilityStatus::Loaded)
    }

    pub fn is_available(&self) -> bool {
        matches!(
            self,
            CapabilityStatus::Loaded | CapabilityStatus::Unloaded
        )
    }
}

/// Information about a capability provided by a runner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityInfo {
    /// The capability type
    pub capability: Capability,
    /// Current status
    pub status: CapabilityStatus,
    /// The model providing this capability
    pub model_id: String,
    /// Current number of active requests (if loaded)
    #[serde(default)]
    pub active_requests: u32,
    /// Average latency in milliseconds (if available)
    #[serde(default)]
    pub avg_latency_ms: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_serialization() {
        let cap = Capability::FastChat;
        let json = serde_json::to_string(&cap).unwrap();
        assert_eq!(json, r#""fast_chat""#);

        let parsed: Capability = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, Capability::FastChat);
    }

    #[test]
    fn test_capability_status_serialization() {
        let status = CapabilityStatus::Loading {
            progress: Some(0.5),
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains(r#""status":"loading""#));
        assert!(json.contains(r#""progress":0.5"#));
    }

    #[test]
    fn test_capability_status_is_ready() {
        assert!(CapabilityStatus::Loaded.is_ready());
        assert!(!CapabilityStatus::Unloaded.is_ready());
        assert!(!CapabilityStatus::Loading { progress: None }.is_ready());
    }
}
