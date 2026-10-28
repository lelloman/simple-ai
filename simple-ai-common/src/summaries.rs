//! Bounded site summary contract; no changes to the runner WebSocket protocol.
use serde::{Deserialize, Serialize};
pub const SUMMARY_MODEL: &str = "Qwen/Qwen3-4B-Instruct-2507";
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SummaryRequest {
    pub model: String,
    pub input: Vec<String>,
    pub max_tokens: Option<u32>,
}
impl SummaryRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.input.is_empty()
            || self.input.len() > 64
            || self
                .input
                .iter()
                .any(|s| s.trim().is_empty() || s.chars().count() > 32768)
            || self.input.iter().map(|s| s.len()).sum::<usize>() > 1048576
        {
            return Err("input requires 1–64 nonblank texts, at most 32768 characters each and 1 MiB total UTF-8".into());
        }
        if self.max_tokens.is_some_and(|n| n == 0 || n > 60) {
            return Err("max_tokens must be between 1 and 60".into());
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryResult {
    pub index: usize,
    pub summary: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryResponse {
    pub data: Vec<SummaryResult>,
    pub model: String,
    pub usage: SummaryUsage,
}
impl SummaryResponse {
    pub fn validate(&self, count: usize) -> Result<(), String> {
        if self.model != SUMMARY_MODEL
            || self.data.len() != count
            || self
                .data
                .iter()
                .enumerate()
                .any(|(i, r)| r.index != i || r.summary.trim().is_empty())
            || self
                .usage
                .prompt_tokens
                .checked_add(self.usage.completion_tokens)
                != Some(self.usage.total_tokens)
        {
            return Err("invalid summary provider output".into());
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_bounds() {
        let mut r = SummaryRequest {
            model: "class:summarize".into(),
            input: vec!["hello".into(); 64],
            max_tokens: None,
        };
        assert!(r.validate().is_ok());
        for input in [
            vec![],
            vec![" ".into()],
            vec!["x".into(); 65],
            vec!["x".repeat(32769)],
            vec!["x".repeat(32768); 33],
        ] {
            r.input = input;
            assert!(r.validate().is_err());
        }
        r.input = vec!["hello".into()];
        for n in [0, 61] {
            r.max_tokens = Some(n);
            assert!(r.validate().is_err());
        }
    }
    #[test]
    fn malformed_output_is_rejected() {
        let mut reply = SummaryResponse {
            model: SUMMARY_MODEL.into(),
            data: vec![SummaryResult {
                index: 0,
                summary: "Example operates the site.".into(),
            }],
            usage: SummaryUsage {
                prompt_tokens: 2,
                completion_tokens: 3,
                total_tokens: 5,
            },
        };
        assert!(reply.validate(1).is_ok());
        assert!(reply.validate(2).is_err());
        reply.data[0].index = 1;
        assert!(reply.validate(1).is_err());
        reply.data[0].index = 0;
        reply.data[0].summary = " ".into();
        assert!(reply.validate(1).is_err());
        reply.data[0].summary = "hello".into();
        reply.usage.total_tokens = 0;
        assert!(reply.validate(1).is_err());
    }
}
