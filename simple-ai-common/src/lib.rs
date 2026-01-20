//! SimpleAI Common Types
//!
//! Shared types used by both the inference-runner and backend gateway.

pub mod capability;
pub mod chat;

pub use capability::{Capability, CapabilityInfo, CapabilityStatus};
pub use chat::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, ToolCall, ToolFunction,
    Usage,
};
