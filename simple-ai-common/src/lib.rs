//! SimpleAI Common Types
//!
//! Shared types used by both the inference-runner and backend gateway.

pub mod capability;
pub mod chat;
pub mod ocr;
pub mod protocol;

pub use capability::{Capability, CapabilityInfo, CapabilityStatus};
pub use chat::{
    format_sse_chunk, format_sse_done, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, Choice, ChunkChoice, ToolCall, ToolFunction, Usage,
};
pub use ocr::{
    OcrBlock, OcrFeature, OcrMetadata, OcrMode, OcrOptions, OcrPage, OcrProviderInfo, OcrResponse,
    OcrTable,
};
pub use protocol::{
    CommandResponse, EngineStatus, GatewayMessage, ModelInfo, RunnerHealth, RunnerMessage,
    RunnerMetrics, RunnerRegistration, RunnerStatus, PROTOCOL_VERSION,
};
