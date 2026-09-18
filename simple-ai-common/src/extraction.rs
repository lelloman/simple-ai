//! Schema-driven information extraction. Spans use Unicode code points, end exclusive.
pub use crate::classification::ClassificationInput as ExtractionInput;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExtractionLabels {
    Names(Vec<String>),
    Descriptions(BTreeMap<String, String>),
}
impl ExtractionLabels {
    fn validate(&self) -> Result<usize, String> {
        let names: Vec<&str> = match self {
            Self::Names(v) => v.iter().map(String::as_str).collect(),
            Self::Descriptions(m) => {
                if m.values()
                    .any(|s| s.trim().is_empty() || s.chars().count() > 512)
                {
                    return Err("label descriptions must contain 1–512 characters".into());
                }
                m.keys().map(String::as_str).collect()
            }
        };
        if names.is_empty()
            || names.len() > 32
            || names.iter().any(|s| !valid_name(s))
            || names.iter().collect::<HashSet<_>>().len() != names.len()
        {
            return Err(
                "labels must contain 1–32 unique non-empty names of at most 128 characters".into(),
            );
        }
        Ok(names.len())
    }
}
fn valid_name(s: &str) -> bool {
    !s.trim().is_empty() && s.chars().count() <= 128
}
fn threshold() -> f32 {
    0.5
}
fn chunk_size() -> u32 {
    256
}
fn chunk_overlap() -> u32 {
    64
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionClassification {
    pub task: String,
    pub labels: ExtractionLabels,
    #[serde(default)]
    pub multi_label: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionField {
    pub name: String,
    #[serde(default = "string_type")]
    pub dtype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cardinality: Option<String>,
}
fn string_type() -> String {
    "str".into()
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionStructure {
    pub fields: Vec<ExtractionField>,
    pub anchor: String,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionSchema {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entities: Option<ExtractionLabels>,
    #[serde(default)]
    pub classifications: Vec<ExtractionClassification>,
    #[serde(default)]
    pub structures: BTreeMap<String, ExtractionStructure>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relations: Option<ExtractionLabels>,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionOverlap {
    #[default]
    Flat,
    Nested,
    Allow,
    Longest,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionSplitter {
    #[default]
    Whitespace,
    Char,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionRequest {
    pub model: String,
    pub input: ExtractionInput,
    pub schema: ExtractionSchema,
    #[serde(default = "threshold")]
    pub threshold: f32,
    #[serde(default)]
    pub overlap: ExtractionOverlap,
    #[serde(default)]
    pub splitter: ExtractionSplitter,
    #[serde(default)]
    pub long_text: bool,
    #[serde(default = "chunk_size")]
    pub chunk_size: u32,
    #[serde(default = "chunk_overlap")]
    pub chunk_overlap: u32,
}
impl ExtractionRequest {
    pub fn validate(&self) -> Result<(), String> {
        if !valid_name(&self.model) {
            return Err("model must contain 1–128 characters".into());
        }
        let texts = self.input.clone().into_vec();
        let max_chars = if self.long_text { 100_000 } else { 1500 };
        if texts.is_empty()
            || texts.len() > 16
            || texts
                .iter()
                .any(|s| s.trim().is_empty() || s.chars().count() > max_chars)
            || texts.iter().map(|s| s.chars().count()).sum::<usize>() > 200_000
        {
            return Err(format!("input requires 1–16 non-empty texts, at most {max_chars} characters each and 200000 total; enable long_text for documents over 1500 characters"));
        }
        if !self.threshold.is_finite() || !(0.0..=1.0).contains(&self.threshold) {
            return Err("threshold must be between 0 and 1".into());
        }
        if !(64..=512).contains(&self.chunk_size) || self.chunk_overlap >= self.chunk_size {
            return Err(
                "chunk_size must be 64–512 and chunk_overlap smaller than chunk_size".into(),
            );
        }
        if self.long_text && self.chunk_size - self.chunk_overlap < 32 {
            return Err("chunks must advance by at least 32 words".into());
        }
        if serde_json::to_vec(&self.schema)
            .map_err(|e| e.to_string())?
            .len()
            > 8192
        {
            return Err("schema exceeds 8192 bytes".into());
        }
        let mut queries = 0;
        for labels in [&self.schema.entities, &self.schema.relations]
            .into_iter()
            .flatten()
        {
            queries += labels.validate()?;
        }
        let mut outputs: HashSet<&str> = ["entities", "relation_extraction"].into_iter().collect();
        for task in &self.schema.classifications {
            if !valid_name(&task.task) || !outputs.insert(&task.task) {
                return Err("classification tasks must have unique, non-reserved names".into());
            }
            queries += task.labels.validate()?;
        }
        for (name, record) in &self.schema.structures {
            if !valid_name(name)
                || !outputs.insert(name)
                || record.fields.is_empty()
                || record.fields.len() > 16
            {
                return Err("invalid or duplicate structure name/fields".into());
            }
            let mut fields = HashSet::new();
            for field in &record.fields {
                if !valid_name(&field.name)
                    || !fields.insert(&field.name)
                    || !["str", "list"].contains(&field.dtype.as_str())
                    || field
                        .description
                        .as_ref()
                        .is_some_and(|d| d.trim().is_empty() || d.chars().count() > 512)
                    || field.cardinality.as_ref().is_some_and(|c| {
                        ![
                            "optional_one",
                            "required_one",
                            "zero_or_more",
                            "one_or_more",
                        ]
                        .contains(&c.as_str())
                    })
                {
                    return Err("invalid or duplicate structure field".into());
                }
            }
            if !fields.contains(&record.anchor) {
                return Err("structure anchor must name a declared field".into());
            }
            queries += record.fields.len();
        }
        if queries == 0 || queries > 64 {
            return Err("schema must contain 1–64 labels/fields in total".into());
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub index: usize,
    pub result: serde_json::Value,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionUsage {
    pub input_count: u32,
    pub input_characters: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResponse {
    pub object: String,
    pub model: String,
    pub revision: String,
    pub data: Vec<ExtractionResult>,
    pub usage: ExtractionUsage,
    pub inference_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample() -> serde_json::Value {
        serde_json::json!({"model":"class:information_extraction", "input":"Zoë works in Zürich", "schema":{"entities":["person","location"]}})
    }
    #[test]
    fn accepts_batch_and_rejects_empty_member() {
        let mut v = sample();
        v["input"] = serde_json::json!(["ciao", "hello"]);
        assert!(serde_json::from_value::<ExtractionRequest>(v.clone())
            .unwrap()
            .validate()
            .is_ok());
        v["input"][1] = serde_json::json!(" ");
        assert!(serde_json::from_value::<ExtractionRequest>(v)
            .unwrap()
            .validate()
            .is_err());
    }
    #[test]
    fn bounds_work_and_rejects_output_collisions() {
        let mut v = sample();
        v["threshold"] = serde_json::json!(1.1);
        assert!(serde_json::from_value::<ExtractionRequest>(v)
            .unwrap()
            .validate()
            .is_err());
        let mut v = sample();
        v["schema"]["classifications"] =
            serde_json::json!([{"task":"entities","labels":["yes","no"]}]);
        assert!(serde_json::from_value::<ExtractionRequest>(v)
            .unwrap()
            .validate()
            .is_err());
        let mut v = sample();
        v["long_text"] = serde_json::json!(true);
        v["chunk_overlap"] = serde_json::json!(255);
        assert!(serde_json::from_value::<ExtractionRequest>(v)
            .unwrap()
            .validate()
            .is_err());
    }
    #[test]
    fn rejects_silent_truncation_and_unknown_options() {
        let mut v = sample();
        v["input"] = serde_json::json!("x".repeat(1501));
        assert!(serde_json::from_value::<ExtractionRequest>(v.clone())
            .unwrap()
            .validate()
            .is_err());
        v["long_text"] = serde_json::json!(true);
        assert!(serde_json::from_value::<ExtractionRequest>(v.clone())
            .unwrap()
            .validate()
            .is_ok());
        v["typo"] = serde_json::json!(true);
        assert!(serde_json::from_value::<ExtractionRequest>(v).is_err());
    }
}
