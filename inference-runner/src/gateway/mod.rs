//! Gateway WebSocket client for runner-gateway communication.
//!
//! This module handles the WebSocket connection to the central gateway,
//! including registration, heartbeats, status reporting, and command handling.

mod client;
mod status;

pub use client::GatewayClient;
pub use status::StatusCollector;
