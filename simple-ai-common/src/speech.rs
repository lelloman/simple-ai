//! Text-to-speech request options, provider metadata, and response helpers.

use serde::{Deserialize, Serialize};

/// Voice selector accepted by OpenAI-compatible speech requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SpeechVoice {
    /// Provider-specific voice id.
    Id(String),
    /// Object form used by newer OpenAI-compatible clients.
    Object { id: String },
}

impl SpeechVoice {
    pub fn id(&self) -> &str {
        match self {
            SpeechVoice::Id(id) => id,
            SpeechVoice::Object { id } => id,
        }
    }
}

/// Audio container/codec requested by the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeechResponseFormat {
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

impl Default for SpeechResponseFormat {
    fn default() -> Self {
        Self::Mp3
    }
}

impl SpeechResponseFormat {
    pub fn content_type(self) -> &'static str {
        match self {
            SpeechResponseFormat::Mp3 => "audio/mpeg",
            SpeechResponseFormat::Opus => "audio/opus",
            SpeechResponseFormat::Aac => "audio/aac",
            SpeechResponseFormat::Flac => "audio/flac",
            SpeechResponseFormat::Wav => "audio/wav",
            SpeechResponseFormat::Pcm => "audio/pcm",
        }
    }
}

/// Streaming envelope requested by the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeechStreamFormat {
    Audio,
    Sse,
}

impl Default for SpeechStreamFormat {
    fn default() -> Self {
        Self::Audio
    }
}

/// OpenAI-compatible speech generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: SpeechVoice,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Optional provider-specific language code. XTTS-style providers use this
    /// when the language cannot be inferred from the configured voice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<SpeechResponseFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    // Provider-specific expressiveness control. Chatterbox uses this as its
    // exaggeration/emotion strength knob.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exaggeration: Option<f32>,
    // Provider-specific classifier-free-guidance weight for expressive TTS
    // models such as Chatterbox.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_weight: Option<f32>,
    // Optional sampling temperature for providers that expose stochastic TTS.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    // Optional nucleus sampling threshold for providers that support it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    // Optional minimum probability threshold for providers that support it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    // Optional repetition penalty for providers that support it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    // Optional deterministic seed for providers that support seeded sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_format: Option<SpeechStreamFormat>,
}

impl SpeechRequest {
    pub fn response_format_or_default(&self) -> SpeechResponseFormat {
        self.response_format.unwrap_or_default()
    }

    pub fn stream_format_or_default(&self) -> SpeechStreamFormat {
        self.stream_format.unwrap_or_default()
    }
}

/// A loadable TTS model and the voices/formats it can produce.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechModelInfo {
    pub id: String,
    pub provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_version: Option<String>,
    #[serde(default)]
    pub voices: Vec<String>,
    #[serde(default)]
    pub response_formats: Vec<SpeechResponseFormat>,
    #[serde(default)]
    pub supports_sse: bool,
}

/// TTS provider metadata advertised by runners.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechProviderInfo {
    pub provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_version: Option<String>,
    pub models: Vec<SpeechModelInfo>,
    pub max_input_chars: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speech_request_defaults() {
        let request: SpeechRequest =
            serde_json::from_str(r#"{"model":"tts-local","input":"hello","voice":"alloy"}"#)
                .unwrap();

        assert_eq!(
            request.response_format_or_default(),
            SpeechResponseFormat::Mp3
        );
        assert_eq!(
            request.stream_format_or_default(),
            SpeechStreamFormat::Audio
        );
        assert_eq!(request.voice.id(), "alloy");
    }

    #[test]
    fn test_speech_voice_object() {
        let request: SpeechRequest = serde_json::from_str(
            r#"{"model":"tts-local","input":"hello","voice":{"id":"nova"},"language":"en","response_format":"wav","stream_format":"sse"}"#,
        )
        .unwrap();

        assert_eq!(request.voice.id(), "nova");
        assert_eq!(request.language.as_deref(), Some("en"));
        assert_eq!(
            request.response_format_or_default(),
            SpeechResponseFormat::Wav
        );
        assert_eq!(request.stream_format_or_default(), SpeechStreamFormat::Sse);
    }

    #[test]
    fn test_response_format_content_type() {
        assert_eq!(SpeechResponseFormat::Mp3.content_type(), "audio/mpeg");
        assert_eq!(SpeechResponseFormat::Wav.content_type(), "audio/wav");
    }
}
