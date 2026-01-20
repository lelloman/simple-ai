//! SimpleAI Common Types
//!
//! Shared types used by both the inference-runner and backend gateway.

pub mod capability;
pub mod chat;
pub mod protocol;

pub use capability::{Capability, CapabilityInfo, CapabilityStatus};
pub use chat::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, ToolCall, ToolFunction,
    Usage,
};
pub use protocol::{
    CommandResponse, EngineStatus, GatewayMessage, RunnerHealth, RunnerMessage, RunnerMetrics,
    RunnerRegistration, RunnerStatus, PROTOCOL_VERSION,
};
