//! Bounded batch translation contract. Language tags follow the MADLAD model card.
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::OnceLock};

pub const TRANSLATION_MODEL: &str = "google/madlad400-3b-mt";
pub const TRANSLATION_REVISION: &str = "fa184c675da0b5c9e1c8694fccd4e12e2d422094";

pub fn translation_target(code: &str) -> Option<&'static str> {
    static LANGUAGES: OnceLock<HashMap<String, String>> = OnceLock::new();
    LANGUAGES
        .get_or_init(|| {
            serde_json::from_str(include_str!("translation-languages.json"))
                .expect("static language map")
        })
        .get(code)
        .map(String::as_str)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranslationRequest {
    pub model: String,
    pub input: Vec<String>,
    pub target: String,
    pub source: Option<String>,
    pub max_tokens: Option<u32>,
}
impl TranslationRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.input.is_empty() || self.input.len() > 256 {
            return Err("input must contain between 1 and 256 texts".into());
        }
        if self
            .input
            .iter()
            .any(|s| s.trim().is_empty() || s.chars().count() > 4000)
            || self.input.iter().map(|s| s.chars().count()).sum::<usize>() > 100_000
        {
            return Err(
                "input texts must be non-empty, at most 4000 characters each and 100000 total"
                    .into(),
            );
        }
        if translation_target(&self.target).is_none() {
            return Err("unknown target language".into());
        }
        if self
            .source
            .as_ref()
            .is_some_and(|s| translation_target(s).is_none())
        {
            return Err("unknown source language".into());
        }
        if self.max_tokens.is_some_and(|n| n == 0 || n > 512) {
            return Err("max_tokens must be between 1 and 512".into());
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResult {
    pub index: usize,
    pub translation: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResponse {
    pub data: Vec<TranslationResult>,
    pub model: String,
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validation_and_aliases() {
        let mut request: TranslationRequest =
            serde_json::from_str(r#"{"model":"class:translate","input":["Ciao"],"target":"en"}"#)
                .unwrap();
        assert!(request.validate().is_ok());
        assert_eq!(translation_target("eng"), Some("en"));
        assert_eq!(translation_target("deu"), Some("de"));
        assert_eq!(translation_target("zh-Hans"), Some("zh"));
        request.target = "unknown".into();
        assert!(request.validate().is_err());
        request.target = "en".into();
        request.input.clear();
        assert!(request.validate().is_err());
        request.input = vec!["a".into(); 257];
        assert!(request.validate().is_err());
        request.input = vec!["a".repeat(4001)];
        assert!(request.validate().is_err());
        request.input = vec!["a".repeat(4000); 26];
        assert!(request.validate().is_err());
        request.input = vec![" ".into()];
        assert!(request.validate().is_err());
        request.input = vec!["à".repeat(4000)];
        assert!(request.validate().is_ok());
        request.max_tokens = Some(0);
        assert!(request.validate().is_err());
        request.max_tokens = Some(513);
        assert!(request.validate().is_err());
        request.max_tokens = None;
        request.source = Some("bad".into());
        assert!(request.validate().is_err());
    }
}
