//! OCR request options, provider metadata, and response types.

use serde::{Deserialize, Serialize};

/// OCR output depth requested by a client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OcrMode {
    /// Plain text only.
    Text,
    /// Text blocks with bounding boxes and confidence.
    Layout,
    /// Full document structure when available, including tables.
    Document,
}

impl Default for OcrMode {
    fn default() -> Self {
        Self::Document
    }
}

/// Feature advertised by an OCR provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OcrFeature {
    Text,
    Layout,
    Tables,
    Pdf,
}

/// Client options for OCR requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrOptions {
    #[serde(default)]
    pub mode: OcrMode,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub max_pages: Option<u32>,
}

impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            mode: OcrMode::Document,
            languages: Vec::new(),
            max_pages: None,
        }
    }
}

/// OCR provider metadata advertised by runners.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrProviderInfo {
    pub provider: String,
    #[serde(default)]
    pub provider_version: Option<String>,
    pub modes: Vec<OcrMode>,
    pub features: Vec<OcrFeature>,
    #[serde(default)]
    pub languages: Vec<String>,
}

/// OCR response returned by runners and the backend gateway.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrResponse {
    pub text: String,
    pub pages: Vec<OcrPage>,
    #[serde(default)]
    pub blocks: Vec<OcrBlock>,
    #[serde(default)]
    pub tables: Vec<OcrTable>,
    pub metadata: OcrMetadata,
}

/// OCR content extracted from one page.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrPage {
    pub page: u32,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    pub text: String,
    #[serde(default)]
    pub blocks: Vec<OcrBlock>,
    #[serde(default)]
    pub tables: Vec<OcrTable>,
}

/// A detected layout block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrBlock {
    pub kind: String,
    pub text: String,
    #[serde(default)]
    pub confidence: Option<f32>,
    /// Bounding box as [x0, y0, x1, y1] in source image/page coordinates.
    #[serde(default)]
    pub bbox: Option<[f32; 4]>,
    #[serde(default)]
    pub language: Option<String>,
}

/// A detected table.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrTable {
    #[serde(default)]
    pub bbox: Option<[f32; 4]>,
    pub rows: Vec<Vec<String>>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

/// Metadata about OCR execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrMetadata {
    pub provider: String,
    #[serde(default)]
    pub provider_version: Option<String>,
    pub mode: OcrMode,
    pub page_count: u32,
    pub elapsed_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocr_mode_serializes_snake_case() {
        let json = serde_json::to_string(&OcrMode::Document).unwrap();
        assert_eq!(json, r#""document""#);
    }

    #[test]
    fn test_ocr_options_defaults_to_document() {
        let options: OcrOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(options.mode, OcrMode::Document);
        assert!(options.languages.is_empty());
    }
}
