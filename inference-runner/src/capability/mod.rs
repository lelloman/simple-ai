//! Capability management for the inference runner.
//!
//! This module handles tracking which models are loaded and which capabilities
//! are available.

// For now, we have a simple pass-through to engines.
// The full CapabilityManager with lifecycle management will be added
// when we implement the WebSocket protocol for gateway communication.

// Re-export for use in other modules (Phase 2+)
#[allow(unused_imports)]
pub use simple_ai_common::{Capability, CapabilityInfo, CapabilityStatus};
