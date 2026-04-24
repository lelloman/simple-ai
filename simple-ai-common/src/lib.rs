//! SimpleAI Common Types
//!
//! Shared types used by both the inference-runner and backend gateway.

pub mod capability;
pub mod chat;
pub mod ocr;
pub mod protocol;
pub mod responses;

pub use capability::{Capability, CapabilityInfo, CapabilityStatus};
pub use chat::{
    format_sse_chunk, format_sse_done, format_sse_metrics, ChatCompletionChunk,
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, ChunkChoice,
    InferenceMetrics, ToolCall, ToolFunction, Usage,
};
pub use ocr::{
    OcrBlock, OcrFeature, OcrMetadata, OcrMode, OcrOptions, OcrPage, OcrProviderInfo, OcrResponse,
    OcrTable,
};
pub use protocol::{
    CommandResponse, EngineStatus, GatewayMessage, ModelInfo, RunnerHealth, RunnerMessage,
    RunnerMetrics, RunnerRegistration, RunnerStatus, PROTOCOL_VERSION,
};
pub use responses::{
    ResponseContent, ResponseContentPart, ResponseCreateRequest, ResponseFunctionCall,
    ResponseInput, ResponseInputItem, ResponseInputMessage, ResponseObject, ResponseOutputContent,
    ResponseOutputItem, ResponseOutputMessage, ResponseTypedInputItem, ResponseUsage,
};
