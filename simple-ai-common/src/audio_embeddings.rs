//! Audio embedding request options, provider metadata, and response types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Client options for a single audio embedding request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioEmbeddingOptions {
    /// Loadable model/provider id, for example "musicfm-msd" or "ast-audioset".
    pub model: String,
    /// Output namespace produced by the model, for example "musicfm.mean.v1".
    pub namespace: String,
    #[serde(default)]
    pub clip_offset_seconds: Option<f32>,
    #[serde(default)]
    pub clip_seconds: Option<f32>,
}

/// Namespace advertised by an audio embedding model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioEmbeddingNamespaceInfo {
    pub namespace: String,
    pub dim: u32,
    pub dtype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// A loadable audio embedding model and the namespaces it can produce.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioEmbeddingModelInfo {
    pub id: String,
    pub provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_version: Option<String>,
    pub namespaces: Vec<AudioEmbeddingNamespaceInfo>,
}

/// Audio embedding provider metadata advertised by runners.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioEmbeddingProviderInfo {
    pub provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_version: Option<String>,
    pub models: Vec<AudioEmbeddingModelInfo>,
    pub max_file_bytes: u64,
}

/// Response returned by audio embedding providers, runners, and the gateway.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioEmbeddingResponse {
    pub object: String,
    pub model: String,
    pub namespace: String,
    pub embedding: Vec<f32>,
    pub dim: u32,
    pub dtype: String,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default)]
    pub model_info: Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_embedding_options_camel_case() {
        let options: AudioEmbeddingOptions = serde_json::from_str(
            r#"{"model":"musicfm-msd","namespace":"musicfm.mean.v1","clipOffsetSeconds":30.0}"#,
        )
        .unwrap();
        assert_eq!(options.model, "musicfm-msd");
        assert_eq!(options.namespace, "musicfm.mean.v1");
        assert_eq!(options.clip_offset_seconds, Some(30.0));
    }

    #[test]
    fn test_audio_embedding_response_serializes_shape() {
        let response = AudioEmbeddingResponse {
            object: "audio_embedding".to_string(),
            model: "musicfm-msd".to_string(),
            namespace: "musicfm.mean.v1".to_string(),
            embedding: vec![0.1, 0.2],
            dim: 2,
            dtype: "float32".to_string(),
            metadata: serde_json::json!({}),
            model_info: serde_json::json!({ "layer": 7 }),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains(r#""object":"audio_embedding""#));
        assert!(json.contains(r#""modelInfo":{"layer":7}"#));
    }
}
