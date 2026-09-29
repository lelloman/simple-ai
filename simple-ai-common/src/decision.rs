//! Typed semantic decisions. These are distinct from three-way NLI scores.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

pub const DECISION_MODEL: &str = "autotrust/JEV-9B";
pub const DECISION_REVISION: &str = "4ab5dfb9331c4eb3a212742e1a1aa5446c1fda35";
pub const DECISION_BODY_LIMIT: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRequest {
    pub model: String,
    pub state: Value,
    pub questions: Vec<DecisionQuestion>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DecisionQuestion {
    Boolean {
        id: String,
        instruction: String,
    },
    Choice {
        id: String,
        instruction: String,
        options: Vec<DecisionOption>,
    },
    Rating {
        id: String,
        instruction: String,
        levels: Vec<String>,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionOption {
    pub id: String,
    pub description: String,
}
impl DecisionQuestion {
    pub fn id(&self) -> &str {
        match self {
            Self::Boolean { id, .. } | Self::Choice { id, .. } | Self::Rating { id, .. } => id,
        }
    }
    pub fn instruction(&self) -> &str {
        match self {
            Self::Boolean { instruction, .. }
            | Self::Choice { instruction, .. }
            | Self::Rating { instruction, .. } => instruction,
        }
    }
}
fn valid_id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}
fn valid_text(s: &str) -> bool {
    !s.trim().is_empty() && s.chars().count() <= 8192
}
impl DecisionRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.model.trim().is_empty() {
            return Err("model is required".into());
        }
        let state_ok = match &self.state {
            Value::String(s) => !s.trim().is_empty(),
            Value::Object(v) => !v.is_empty(),
            Value::Array(v) => !v.is_empty(),
            _ => false,
        };
        if !state_ok {
            return Err("state must be nonempty text, object or array".into());
        }
        if serde_json::to_vec(self).map_err(|e| e.to_string())?.len() > DECISION_BODY_LIMIT {
            return Err("request exceeds 2 MiB".into());
        }
        if !(1..=128).contains(&self.questions.len()) {
            return Err("questions requires 1..128 items".into());
        }
        let mut ids = HashSet::new();
        for q in &self.questions {
            if !valid_id(q.id()) || !ids.insert(q.id()) || !valid_text(q.instruction()) {
                return Err("questions require unique ASCII IDs (1..64) and nonempty instructions (at most 8192 characters)".into());
            }
            match q {
                DecisionQuestion::Choice { options, .. } => {
                    let mut keys = HashSet::new();
                    if !(2..=16).contains(&options.len())
                        || options.iter().any(|o| {
                            !valid_id(&o.id) || !keys.insert(&o.id) || !valid_text(&o.description)
                        })
                    {
                        return Err(
                            "choice requires 2..16 unique option IDs and nonempty descriptions"
                                .into(),
                        );
                    }
                }
                DecisionQuestion::Rating { levels, .. } => {
                    if levels.len() != 6 || levels.iter().any(|s| !valid_text(s)) {
                        return Err("rating requires exactly six nonempty levels".into());
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DecisionAnswer {
    Boolean {
        id: String,
        value: bool,
        probabilities: BTreeMap<String, f64>,
    },
    Choice {
        id: String,
        selected: String,
        probabilities: BTreeMap<String, f64>,
    },
    Rating {
        id: String,
        value: f64,
        probabilities: BTreeMap<String, f64>,
        levels: Vec<String>,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionUsage {
    pub question_count: usize,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: Option<u64>,
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DecisionTiming {
    pub prepare_ms: f64,
    pub inference_ms: f64,
    #[serde(default)]
    pub queue_ms: f64,
    #[serde(default)]
    pub total_ms: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionResponse {
    pub request_id: String,
    pub model: String,
    pub revision: String,
    pub answers: Vec<DecisionAnswer>,
    pub usage: DecisionUsage,
    pub timing: DecisionTiming,
    pub calibrated: bool,
    pub upstream_temperature_applied: bool,
}
impl DecisionResponse {
    pub fn validate_for(&self, request: &DecisionRequest, revision: &str) -> Result<(), String> {
        if self.model != request.model
            || self.request_id.is_empty()
            || [
                self.timing.prepare_ms,
                self.timing.inference_ms,
                self.timing.queue_ms,
                self.timing.total_ms,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
            || self.revision != revision
            || self.answers.len() != request.questions.len()
            || self.usage.question_count != request.questions.len()
            || self.calibrated
            || !self.upstream_temperature_applied
        {
            return Err("invalid decision response metadata".into());
        }
        for (q, a) in request.questions.iter().zip(&self.answers) {
            let (id, probabilities, keys): (&str, &BTreeMap<String, f64>, Vec<String>) =
                match (q, a) {
                    (
                        DecisionQuestion::Boolean { .. },
                        DecisionAnswer::Boolean {
                            id, probabilities, ..
                        },
                    ) => (id, probabilities, vec!["false".into(), "true".into()]),
                    (
                        DecisionQuestion::Choice { options, .. },
                        DecisionAnswer::Choice {
                            id, probabilities, ..
                        },
                    ) => (
                        id,
                        probabilities,
                        options.iter().map(|o| o.id.clone()).collect(),
                    ),
                    (
                        DecisionQuestion::Rating { levels, .. },
                        DecisionAnswer::Rating {
                            id,
                            value,
                            probabilities,
                            levels: got,
                        },
                    ) if levels == got && value.is_finite() && (0.0..=5.0).contains(value) => {
                        (id, probabilities, (0..6).map(|i| i.to_string()).collect())
                    }
                    _ => return Err("decision answer type mismatch".into()),
                };
            if id != q.id()
                || probabilities.len() != keys.len()
                || keys.iter().any(|k| !probabilities.contains_key(k))
                || probabilities
                    .values()
                    .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
                || (probabilities.values().sum::<f64>() - 1.0).abs() > 1e-5
            {
                return Err("invalid decision probabilities or IDs".into());
            }
            let best = keys
                .iter()
                .enumerate()
                .max_by(|(ia, a), (ib, b)| {
                    probabilities[*a]
                        .total_cmp(&probabilities[*b])
                        .then(ib.cmp(ia))
                })
                .unwrap()
                .1;
            match a {
                DecisionAnswer::Boolean { value, .. } if *value != (best == "true") => {
                    return Err("boolean selection mismatch".into())
                }
                DecisionAnswer::Choice { selected, .. } if selected != best => {
                    return Err("choice selection mismatch".into())
                }
                DecisionAnswer::Rating { value, .. }
                    if (value
                        - keys
                            .iter()
                            .enumerate()
                            .map(|(i, k)| i as f64 * probabilities[k])
                            .sum::<f64>())
                    .abs()
                        > 1e-5 =>
                {
                    return Err("rating expectation mismatch".into())
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> Value {
        serde_json::json!({"model":"class:semantic_decisions","state":{"text":"refund"},"questions":[{"id":"refund","type":"boolean","instruction":"Refund requested?"}]})
    }
    #[test]
    fn validation_and_strict_variants() {
        let value = request();
        serde_json::from_value::<DecisionRequest>(value.clone())
            .unwrap()
            .validate()
            .unwrap();
        let mut bad = value.clone();
        bad["questions"][0]["levels"] = serde_json::json!([]);
        assert!(serde_json::from_value::<DecisionRequest>(bad).is_err());
        let mut bad = value.clone();
        bad["questions"]
            .as_array_mut()
            .unwrap()
            .push(value["questions"][0].clone());
        assert!(serde_json::from_value::<DecisionRequest>(bad)
            .unwrap()
            .validate()
            .is_err());
        let mut bad = value;
        bad["state"] = Value::Bool(true);
        assert!(serde_json::from_value::<DecisionRequest>(bad)
            .unwrap()
            .validate()
            .is_err());
    }
    #[test]
    fn responses_reject_wrong_identity_probabilities_and_selection() {
        let mut input = request();
        input["model"] = DECISION_MODEL.into();
        let request: DecisionRequest = serde_json::from_value(input).unwrap();
        let value = serde_json::json!({
            "request_id":"test", "model":DECISION_MODEL, "revision":DECISION_REVISION,
            "answers":[{"id":"refund","type":"boolean","value":true,"probabilities":{"false":0.2,"true":0.8}}],
            "usage":{"question_count":1,"prompt_tokens":20,"completion_tokens":1,"cached_tokens":null},
            "timing":{"prepare_ms":1.0,"inference_ms":2.0},
            "calibrated":false,"upstream_temperature_applied":true
        });
        serde_json::from_value::<DecisionResponse>(value.clone())
            .unwrap()
            .validate_for(&request, DECISION_REVISION)
            .unwrap();
        for (pointer, replacement) in [
            ("/model", serde_json::json!("wrong")),
            ("/revision", serde_json::json!("wrong")),
            ("/answers/0/id", serde_json::json!("other")),
            ("/answers/0/value", serde_json::json!(false)),
            ("/answers/0/probabilities/true", serde_json::json!(0.3)),
            ("/calibrated", serde_json::json!(true)),
            ("/timing/inference_ms", serde_json::json!(-1.0)),
        ] {
            let mut bad = value.clone();
            *bad.pointer_mut(pointer).unwrap() = replacement;
            assert!(
                serde_json::from_value::<DecisionResponse>(bad)
                    .unwrap()
                    .validate_for(&request, DECISION_REVISION)
                    .is_err(),
                "{pointer}"
            );
        }
    }

    #[test]
    fn rating_has_six_levels() {
        let mut value = request();
        value["questions"] = serde_json::json!([{"id":"rate","type":"rating","instruction":"Urgency","levels":["0","1","2","3","4","5"]}]);
        serde_json::from_value::<DecisionRequest>(value.clone())
            .unwrap()
            .validate()
            .unwrap();
        value["questions"][0]["levels"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(serde_json::from_value::<DecisionRequest>(value)
            .unwrap()
            .validate()
            .is_err());
    }
}
