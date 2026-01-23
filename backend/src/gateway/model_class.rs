//! Model classification for routing and permissions.
//!
//! Models are classified into tiers for:
//! - Permission-based access (some users can only request classes, not specific models)
//! - Wake-on-demand routing (wake runners that have models of the requested class)

use serde::{Deserialize, Serialize};

use crate::config::ModelsConfig;

/// Model class/tier for routing decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelClass {
    /// Large models (70B+ parameters) - slower but more capable
    Big,
    /// Smaller models - faster inference
    Fast,
}

impl ModelClass {
    /// Parse a model class from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "big" => Some(Self::Big),
            "fast" => Some(Self::Fast),
            _ => None,
        }
    }

    /// Get the string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Big => "big",
            Self::Fast => "fast",
        }
    }
}

impl std::fmt::Display for ModelClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Classify a model ID into a ModelClass using configuration.
///
/// Returns None if the model is not configured in either list.
pub fn classify_model(model_id: &str, config: &ModelsConfig) -> Option<ModelClass> {
    match config.classify(model_id) {
        Some("big") => Some(ModelClass::Big),
        Some("fast") => Some(ModelClass::Fast),
        _ => None,
    }
}

/// Parsed model request - either a specific model or a class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRequest {
    /// Request for a specific model by ID
    Specific(String),
    /// Request for any model of a class
    Class(ModelClass),
}

impl ModelRequest {
    /// Parse a model field value.
    ///
    /// Syntax:
    /// - `class:fast` or `class:big` - request a class
    /// - anything else - request a specific model
    pub fn parse(model: &str) -> Self {
        if let Some(class_name) = model.strip_prefix("class:") {
            if let Some(class) = ModelClass::from_str(class_name) {
                return Self::Class(class);
            }
        }
        Self::Specific(model.to_string())
    }

    /// Check if this is a class request.
    pub fn is_class_request(&self) -> bool {
        matches!(self, Self::Class(_))
    }

    /// Get the class for this request.
    ///
    /// For class requests, returns the requested class.
    /// For specific model requests, returns the class from config (or None if not configured).
    pub fn effective_class(&self, config: &ModelsConfig) -> Option<ModelClass> {
        match self {
            Self::Class(class) => Some(*class),
            Self::Specific(model_id) => classify_model(model_id, config),
        }
    }
}

/// Permission roles for model selection.
pub mod roles {
    /// Role that allows requesting specific models by ID.
    pub const MODEL_SPECIFIC: &str = "model:specific";

    /// Role that allows requesting model classes only.
    /// This is the default/lower permission level.
    pub const MODEL_CLASS: &str = "model:class";
}

/// Check if a user can make the given model request.
///
/// - Users with `model:specific` role can request anything
/// - Users with `model:class` role (or no model role) can only request classes
pub fn can_request_model(user_roles: &[String], request: &ModelRequest) -> bool {
    // Users with model:specific can request anything
    if user_roles.iter().any(|r| r == roles::MODEL_SPECIFIC) {
        return true;
    }

    // Otherwise, can only request classes
    request.is_class_request()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_model_big() {
        let config = ModelsConfig {
            big: vec![
                "llama3:70b".to_string(),
                "qwen2:72b".to_string(),
            ],
            fast: vec![],
        };
        assert_eq!(classify_model("llama3:70b", &config), Some(ModelClass::Big));
        assert_eq!(classify_model("LLAMA3:70B", &config), Some(ModelClass::Big)); // case-insensitive
        assert_eq!(classify_model("qwen2:72b", &config), Some(ModelClass::Big));
    }

    #[test]
    fn test_classify_model_fast() {
        let config = ModelsConfig {
            big: vec![],
            fast: vec![
                "llama3:8b".to_string(),
                "mistral:7b".to_string(),
            ],
        };
        assert_eq!(classify_model("llama3:8b", &config), Some(ModelClass::Fast));
        assert_eq!(classify_model("mistral:7b", &config), Some(ModelClass::Fast));
    }

    #[test]
    fn test_classify_model_not_configured() {
        // With empty config, nothing is classified
        let config = ModelsConfig::default();
        assert_eq!(classify_model("llama3:8b", &config), None);
        assert_eq!(classify_model("unknown", &config), None);
    }

    #[test]
    fn test_classify_model_custom_config() {
        let config = ModelsConfig {
            big: vec!["custom-big-model".to_string()],
            fast: vec!["small-model".to_string()],
        };
        assert_eq!(classify_model("custom-big-model", &config), Some(ModelClass::Big));
        assert_eq!(classify_model("Custom-Big-Model", &config), Some(ModelClass::Big)); // case-insensitive
        assert_eq!(classify_model("small-model", &config), Some(ModelClass::Fast));
        assert_eq!(classify_model("unknown-model", &config), None); // not in config
    }

    #[test]
    fn test_model_class_from_str() {
        assert_eq!(ModelClass::from_str("big"), Some(ModelClass::Big));
        assert_eq!(ModelClass::from_str("Big"), Some(ModelClass::Big));
        assert_eq!(ModelClass::from_str("BIG"), Some(ModelClass::Big));
        assert_eq!(ModelClass::from_str("fast"), Some(ModelClass::Fast));
        assert_eq!(ModelClass::from_str("Fast"), Some(ModelClass::Fast));
        assert_eq!(ModelClass::from_str("unknown"), None);
    }

    #[test]
    fn test_model_class_as_str() {
        assert_eq!(ModelClass::Big.as_str(), "big");
        assert_eq!(ModelClass::Fast.as_str(), "fast");
    }

    #[test]
    fn test_model_request_parse_class() {
        assert_eq!(
            ModelRequest::parse("class:fast"),
            ModelRequest::Class(ModelClass::Fast)
        );
        assert_eq!(
            ModelRequest::parse("class:big"),
            ModelRequest::Class(ModelClass::Big)
        );
    }

    #[test]
    fn test_model_request_parse_specific() {
        assert_eq!(
            ModelRequest::parse("llama3:8b"),
            ModelRequest::Specific("llama3:8b".to_string())
        );
        assert_eq!(
            ModelRequest::parse("class:invalid"),
            ModelRequest::Specific("class:invalid".to_string())
        );
    }

    #[test]
    fn test_model_request_is_class_request() {
        assert!(ModelRequest::Class(ModelClass::Fast).is_class_request());
        assert!(!ModelRequest::Specific("llama3:8b".to_string()).is_class_request());
    }

    #[test]
    fn test_model_request_effective_class() {
        let config = ModelsConfig {
            big: vec!["llama3:70b".to_string()],
            fast: vec!["llama3:8b".to_string()],
        };

        // Class requests always return Some(class)
        assert_eq!(
            ModelRequest::Class(ModelClass::Big).effective_class(&config),
            Some(ModelClass::Big)
        );
        assert_eq!(
            ModelRequest::Class(ModelClass::Fast).effective_class(&config),
            Some(ModelClass::Fast)
        );

        // Specific model requests are classified according to config
        assert_eq!(
            ModelRequest::Specific("llama3:70b".to_string()).effective_class(&config),
            Some(ModelClass::Big)
        );
        assert_eq!(
            ModelRequest::Specific("llama3:8b".to_string()).effective_class(&config),
            Some(ModelClass::Fast)
        );

        // Unknown models return None
        assert_eq!(
            ModelRequest::Specific("unknown".to_string()).effective_class(&config),
            None
        );
    }

    #[test]
    fn test_can_request_model_with_specific_role() {
        let roles = vec!["model:specific".to_string()];

        // Can request specific models
        assert!(can_request_model(
            &roles,
            &ModelRequest::Specific("llama3:70b".to_string())
        ));

        // Can also request classes
        assert!(can_request_model(
            &roles,
            &ModelRequest::Class(ModelClass::Fast)
        ));
    }

    #[test]
    fn test_can_request_model_with_class_role() {
        let roles = vec!["model:class".to_string()];

        // Cannot request specific models
        assert!(!can_request_model(
            &roles,
            &ModelRequest::Specific("llama3:70b".to_string())
        ));

        // Can request classes
        assert!(can_request_model(
            &roles,
            &ModelRequest::Class(ModelClass::Fast)
        ));
    }

    #[test]
    fn test_can_request_model_with_no_model_roles() {
        let roles = vec!["admin".to_string()]; // No model roles

        // Cannot request specific models
        assert!(!can_request_model(
            &roles,
            &ModelRequest::Specific("llama3:70b".to_string())
        ));

        // Can request classes (default behavior)
        assert!(can_request_model(
            &roles,
            &ModelRequest::Class(ModelClass::Fast)
        ));
    }

    #[test]
    fn test_model_class_display() {
        assert_eq!(format!("{}", ModelClass::Big), "big");
        assert_eq!(format!("{}", ModelClass::Fast), "fast");
    }
}
