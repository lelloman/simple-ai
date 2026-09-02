//! Generic zero-shot text classification request and response types.

use serde::{Deserialize, Serialize};

/// One or more texts to classify.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ClassificationInput {
    Single(String),
    Multiple(Vec<String>),
}

impl ClassificationInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(value) => vec![value],
            Self::Multiple(values) => values,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Multiple(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Single(value) => value.trim().is_empty(),
            Self::Multiple(values) => values.is_empty(),
        }
    }
}

/// A stable caller-facing label and the hypothesis the NLI model evaluates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationLabel {
    pub label: String,
    pub hypothesis: String,
}

/// Request for independent NLI scores for every input/label pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationRequest {
    pub model: String,
    pub input: ClassificationInput,
    pub labels: Vec<ClassificationLabel>,
}

/// Normalized three-way NLI probabilities for one hypothesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationScore {
    pub label: String,
    pub entailment: f32,
    pub neutral: f32,
    pub contradiction: f32,
}

/// Scores for one input, in the same order as the requested labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub index: usize,
    pub scores: Vec<ClassificationScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationUsage {
    pub input_count: u32,
    pub pair_count: u32,
    pub prompt_tokens: u32,
}

/// Response for `/v1/classifications`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResponse {
    pub object: String,
    pub data: Vec<ClassificationResult>,
    pub model: String,
    pub usage: ClassificationUsage,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_accepts_single_and_multiple_inputs() {
        let single: ClassificationRequest = serde_json::from_str(
            r#"{"model":"nli","input":"hello","labels":[{"label":"greeting","hypothesis":"This is a greeting."}]}"#,
        )
        .unwrap();
        assert_eq!(single.input.len(), 1);

        let multiple: ClassificationRequest = serde_json::from_str(
            r#"{"model":"nli","input":["hello","bye"],"labels":[{"label":"greeting","hypothesis":"This is a greeting."}]}"#,
        )
        .unwrap();
        assert_eq!(multiple.input.len(), 2);
    }

    #[test]
    fn response_preserves_three_way_nli_scores() {
        let response = ClassificationResponse {
            object: "list".to_string(),
            data: vec![ClassificationResult {
                index: 0,
                scores: vec![ClassificationScore {
                    label: "leak".to_string(),
                    entailment: 0.8,
                    neutral: 0.15,
                    contradiction: 0.05,
                }],
            }],
            model: "nli".to_string(),
            usage: ClassificationUsage {
                input_count: 1,
                pair_count: 1,
                prompt_tokens: 12,
            },
        };
        let value = serde_json::to_value(response).unwrap();
        let entailment = value["data"][0]["scores"][0]["entailment"]
            .as_f64()
            .unwrap();
        assert!((entailment - 0.8).abs() < 1e-6);
        assert_eq!(value["usage"]["pair_count"], 1);
    }
}
