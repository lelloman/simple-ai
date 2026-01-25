//! Admin UI routes.
//!
//! Provides:
//! - Dashboard (`/admin`) - Overview with stats
//! - Users (`/admin/users`) - User management
//! - Requests (`/admin/requests`) - Request history
//! - SSE endpoint for real-time runner events (`/admin/runners/events`)
//! - WebSocket endpoint for real-time runner events with token refresh (`/admin/ws`)

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use askama::Template;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, Query, Request, State,
    },
    http::StatusCode,
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        Html, IntoResponse, Redirect, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::{stream::Stream, SinkExt, StreamExt as FuturesStreamExt};
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt as TokioStreamExt;

use crate::audit::{DashboardStats, RequestSummary, RequestWithResponse, UserWithStats};
use crate::gateway::RunnerEvent;
use crate::wol;
use crate::AppState;

/// Admin user info for templates.
#[derive(Clone)]
pub struct AdminUser {
    pub sub: String,
    pub email: Option<String>,
}

/// Middleware that requires an authenticated admin user.
async fn require_admin(
    State(state): State<Arc<AppState>>,
    mut request: Request,
    next: Next,
) -> Response {
    // Get Authorization header
    let auth_result = state.jwks_client.authenticate(request.headers()).await;

    match auth_result {
        Ok(user) => {
            if !user.is_admin() {
                return (
                    StatusCode::FORBIDDEN,
                    Html("<h1>403 Forbidden</h1><p>Admin access required. You need the 'admin' role for this application.</p>"),
                ).into_response();
            }

            // Store admin user for handlers
            request.extensions_mut().insert(AdminUser {
                sub: user.sub,
                email: user.email,
            });

            next.run(request).await
        }
        Err(_) => {
            // Return 401 with instructions
            (
                StatusCode::UNAUTHORIZED,
                Html("<h1>401 Unauthorized</h1><p>Please provide a valid Bearer token with admin privileges.</p>"),
            ).into_response()
        }
    }
}

// ========== Templates ==========

#[derive(Template)]
#[template(path = "admin/dashboard.html")]
struct DashboardTemplate {
    nav_active: &'static str,
    admin_email: String,
    stats: DashboardStats,
    recent_requests: Vec<RequestSummary>,
}

#[derive(Template)]
#[template(path = "admin/users.html")]
struct UsersTemplate {
    nav_active: &'static str,
    admin_email: String,
    users: Vec<UserWithStats>,
}

#[derive(Template)]
#[template(path = "admin/requests.html")]
struct RequestsTemplate {
    nav_active: &'static str,
    admin_email: String,
    requests: Vec<RequestWithResponse>,
    filter_user_id: String,
    filter_model: String,
    page: u32,
    total_pages: u32,
}

// ========== Handlers ==========

/// GET /admin - Dashboard
async fn dashboard(
    State(state): State<Arc<AppState>>,
    axum::Extension(admin): axum::Extension<AdminUser>,
) -> impl IntoResponse {
    let stats = state.audit_logger.get_stats().unwrap_or(DashboardStats {
        total_users: 0,
        total_requests: 0,
        requests_24h: 0,
        total_tokens: 0,
    });

    let recent_requests = state.audit_logger.get_recent_requests(10).unwrap_or_default();

    let template = DashboardTemplate {
        nav_active: "dashboard",
        admin_email: admin.email.unwrap_or_else(|| admin.sub.clone()),
        stats,
        recent_requests,
    };

    Html(template.render().unwrap_or_else(|e| format!("Template error: {}", e)))
}

/// GET /admin/users - User list
async fn users_list(
    State(state): State<Arc<AppState>>,
    axum::Extension(admin): axum::Extension<AdminUser>,
) -> impl IntoResponse {
    let users = state.audit_logger.get_users_with_stats().unwrap_or_default();

    let template = UsersTemplate {
        nav_active: "users",
        admin_email: admin.email.unwrap_or_else(|| admin.sub.clone()),
        users,
    };

    Html(template.render().unwrap_or_else(|e| format!("Template error: {}", e)))
}

#[derive(serde::Deserialize)]
struct RequestsQuery {
    user_id: Option<String>,
    model: Option<String>,
    page: Option<u32>,
}

/// GET /admin/requests - Request history
async fn requests_list(
    State(state): State<Arc<AppState>>,
    axum::Extension(admin): axum::Extension<AdminUser>,
    Query(query): Query<RequestsQuery>,
) -> impl IntoResponse {
    let page = query.page.unwrap_or(1).max(1);
    let per_page = 50;

    let (requests, total_pages) = state
        .audit_logger
        .get_requests_paginated(
            query.user_id.as_deref(),
            query.model.as_deref(),
            page,
            per_page,
        )
        .unwrap_or_default();

    let template = RequestsTemplate {
        nav_active: "requests",
        admin_email: admin.email.unwrap_or_else(|| admin.sub.clone()),
        requests,
        filter_user_id: query.user_id.unwrap_or_default(),
        filter_model: query.model.unwrap_or_default(),
        page,
        total_pages,
    };

    Html(template.render().unwrap_or_else(|e| format!("Template error: {}", e)))
}

/// POST /admin/users/:id/disable - Disable a user
async fn disable_user(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
) -> impl IntoResponse {
    match state.audit_logger.disable_user(&user_id) {
        Ok(_) => Redirect::to("/admin/users"),
        Err(e) => {
            tracing::error!("Failed to disable user {}: {}", user_id, e);
            Redirect::to("/admin/users")
        }
    }
}

/// POST /admin/users/:id/enable - Enable a user
async fn enable_user(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
) -> impl IntoResponse {
    match state.audit_logger.enable_user(&user_id) {
        Ok(_) => Redirect::to("/admin/users"),
        Err(e) => {
            tracing::error!("Failed to enable user {}: {}", user_id, e);
            Redirect::to("/admin/users")
        }
    }
}

/// Runner info for API response.
#[derive(Debug, Clone, Serialize)]
pub struct RunnerInfo {
    pub id: String,
    pub name: String,
    pub machine_type: Option<String>,
    pub health: String,
    pub loaded_models: Vec<String>,
    pub connected_at: String,
    pub last_heartbeat: String,
    pub http_base_url: Option<String>,
    pub mac_address: Option<String>,
    pub is_online: bool,
}

/// Response for /admin/runners endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct RunnersResponse {
    pub runners: Vec<RunnerInfo>,
    pub total: usize,
}

/// GET /admin/runners - List all runners (connected + offline from DB)
async fn list_runners(State(state): State<Arc<AppState>>) -> Json<RunnersResponse> {
    use std::collections::HashSet;

    let connected_runners = state.runner_registry.all().await;
    let connected_ids: HashSet<String> = connected_runners.iter().map(|r| r.id.clone()).collect();

    // Convert connected runners to RunnerInfo
    let mut runners: Vec<RunnerInfo> = connected_runners
        .into_iter()
        .map(|r| {
            let loaded_models = r.loaded_models();
            RunnerInfo {
                id: r.id,
                name: r.name,
                machine_type: r.machine_type,
                health: format!("{:?}", r.status.health),
                loaded_models,
                connected_at: r.connected_at.to_rfc3339(),
                last_heartbeat: r.last_heartbeat.to_rfc3339(),
                http_base_url: r.http_base_url,
                mac_address: r.mac_address,
                is_online: true,
            }
        })
        .collect();

    // Add offline runners from database
    let offline_count = match state.audit_logger.get_all_runners() {
        Ok(db_runners) => {
            let db_count = db_runners.len();
            let mut added = 0;
            for db_runner in db_runners {
                if !connected_ids.contains(&db_runner.id) {
                    tracing::debug!("Adding offline runner: {} (MAC: {:?})", db_runner.id, db_runner.mac_address);
                    runners.push(RunnerInfo {
                        id: db_runner.id,
                        name: db_runner.name,
                        machine_type: db_runner.machine_type,
                        health: "Offline".to_string(),
                        loaded_models: vec![],
                        connected_at: "".to_string(),
                        last_heartbeat: db_runner.last_seen_at,
                        http_base_url: None,
                        mac_address: db_runner.mac_address,
                        is_online: false,
                    });
                    added += 1;
                }
            }
            tracing::debug!("Runners: {} in DB, {} connected, {} offline added", db_count, connected_ids.len(), added);
            added
        }
        Err(e) => {
            tracing::error!("Failed to fetch runners from database: {}", e);
            0
        }
    };

    let total = runners.len();
    tracing::debug!("Returning {} runners ({} connected, {} offline)", total, connected_ids.len(), offline_count);
    Json(RunnersResponse { runners, total })
}

/// Model info for /admin/models API response.
#[derive(Debug, Clone, Serialize)]
pub struct AdminModelInfo {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<String>,
    pub loaded: bool,
    /// Runner IDs that have this model loaded (in GPU memory).
    pub runners: Vec<String>,
    /// Runner IDs that have this model available (on disk).
    pub available_on: Vec<String>,
}

/// Response for /admin/models endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct AdminModelsResponse {
    pub models: Vec<AdminModelInfo>,
    pub total: usize,
}

/// Dashboard stats for WebSocket state snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct DashboardStatsInfo {
    pub total_users: u64,
    pub total_requests: u64,
    pub requests_24h: u64,
    pub total_tokens: u64,
}

/// GET /admin/models - List all models with loaded status (JSON API)
async fn list_models(State(state): State<Arc<AppState>>) -> Json<AdminModelsResponse> {
    let models = state.inference_router.list_models_with_details().await.unwrap_or_default();
    let total = models.len();

    let models: Vec<AdminModelInfo> = models
        .into_iter()
        .map(|m| AdminModelInfo {
            id: m.id,
            name: m.name,
            size_bytes: m.size_bytes,
            parameter_count: m.parameter_count,
            context_length: m.context_length,
            quantization: m.quantization,
            modified_at: m.modified_at,
            loaded: m.loaded,
            runners: m.runners,
            available_on: m.available_on,
        })
        .collect();

    Json(AdminModelsResponse { models, total })
}

/// Response for wake endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct WakeResponse {
    pub success: bool,
    pub message: String,
}

/// POST /admin/runners/:id/wake - Send Wake-on-LAN packet to runner
async fn wake_runner(
    State(state): State<Arc<AppState>>,
    Path(runner_id): Path<String>,
) -> Result<Json<WakeResponse>, (StatusCode, Json<WakeResponse>)> {
    // First check if runner is already online
    if let Some(runner) = state.runner_registry.get(&runner_id).await {
        if runner.is_operational() {
            return Ok(Json(WakeResponse {
                success: true,
                message: "Runner is already online".to_string(),
            }));
        }
    }

    // Look up MAC address from connected runner or database
    let mac_address = if let Some(runner) = state.runner_registry.get(&runner_id).await {
        runner.mac_address
    } else {
        // Check database for offline runner
        state.audit_logger.get_runner(&runner_id)
            .ok()
            .flatten()
            .and_then(|r| r.mac_address)
    };

    let Some(mac) = mac_address else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(WakeResponse {
                success: false,
                message: "Runner has no MAC address configured".to_string(),
            }),
        ));
    };

    // Send WOL packet - via bouncer if configured, otherwise directly
    let result = if let Some(ref bouncer_url) = state.wol_config.bouncer_url {
        wol::send_wol_via_bouncer(bouncer_url, &mac, &state.wol_config.broadcast_address).await
    } else {
        wol::send_wol(&mac, &state.wol_config.broadcast_address, state.wol_config.port)
    };

    match result {
        Ok(()) => {
            tracing::info!("Sent WOL packet for runner {} (MAC: {})", runner_id, mac);
            Ok(Json(WakeResponse {
                success: true,
                message: format!("Wake-on-LAN packet sent to {}", mac),
            }))
        }
        Err(e) => {
            tracing::error!("Failed to send WOL packet for runner {}: {}", runner_id, e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(WakeResponse {
                    success: false,
                    message: format!("Failed to send WOL packet: {}", e),
                }),
            ))
        }
    }
}

/// Query parameters for SSE authentication.
#[derive(serde::Deserialize)]
pub struct SseAuthQuery {
    token: String,
}

// ========== WebSocket Admin Client Messages ==========

/// Messages sent from admin client to server.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AdminClientMessage {
    /// Authentication message with JWT token.
    Auth { token: String },
}

/// Messages sent from server to admin client.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AdminServerMessage {
    /// Authentication succeeded.
    AuthOk,
    /// Authentication failed.
    AuthError { message: String },
    /// Full state snapshot (sent after auth and on request).
    StateSnapshot {
        runners: Vec<RunnerInfo>,
        models: Vec<AdminModelInfo>,
        stats: DashboardStatsInfo,
    },
    /// A runner connected.
    RunnerConnected {
        runner_id: String,
        name: String,
        machine_type: Option<String>,
        health: String,
        loaded_models: Vec<String>,
    },
    /// A runner disconnected.
    RunnerDisconnected { runner_id: String },
    /// A runner's status changed.
    RunnerStatusChanged {
        runner_id: String,
        health: String,
        loaded_models: Vec<String>,
    },
    /// Models list updated (sent after runner connect/disconnect/status change).
    ModelsUpdated { models: Vec<AdminModelInfo> },
}

impl From<RunnerEvent> for AdminServerMessage {
    fn from(event: RunnerEvent) -> Self {
        match event {
            RunnerEvent::Connected {
                runner_id,
                name,
                machine_type,
                health,
                loaded_models,
            } => AdminServerMessage::RunnerConnected {
                runner_id,
                name,
                machine_type,
                health,
                loaded_models,
            },
            RunnerEvent::Disconnected { runner_id } => {
                AdminServerMessage::RunnerDisconnected { runner_id }
            }
            RunnerEvent::StatusChanged {
                runner_id,
                health,
                loaded_models,
            } => AdminServerMessage::RunnerStatusChanged {
                runner_id,
                health,
                loaded_models,
            },
        }
    }
}

/// GET /admin/ws - WebSocket endpoint for real-time runner events.
///
/// Protocol:
/// - Client sends `{ "type": "auth", "token": "<jwt>" }` to authenticate
/// - Server responds with `{ "type": "auth_ok" }` or `{ "type": "auth_error", "message": "..." }`
/// - After auth, server sends runner events as they occur
/// - Client can send auth message again to refresh token mid-connection
async fn admin_ws(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_admin_ws(socket, state))
}

/// Handle an admin WebSocket connection.
async fn handle_admin_ws(socket: WebSocket, state: Arc<AppState>) {
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Wait for initial auth message within 10 seconds
    let auth_timeout = Duration::from_secs(10);
    let initial_auth = match tokio::time::timeout(auth_timeout, FuturesStreamExt::next(&mut ws_rx)).await {
        Ok(Some(Ok(Message::Text(text)))) => {
            match serde_json::from_str::<AdminClientMessage>(&text) {
                Ok(AdminClientMessage::Auth { token }) => token,
                Err(e) => {
                    let _ = send_admin_message(
                        &mut ws_tx,
                        &AdminServerMessage::AuthError {
                            message: format!("Invalid message format: {}", e),
                        },
                    )
                    .await;
                    return;
                }
            }
        }
        Ok(Some(Ok(Message::Close(_)))) | Ok(None) => {
            tracing::debug!("Admin WS client disconnected before auth");
            return;
        }
        Ok(Some(Ok(_))) => {
            let _ = send_admin_message(
                &mut ws_tx,
                &AdminServerMessage::AuthError {
                    message: "Expected text message with auth".to_string(),
                },
            )
            .await;
            return;
        }
        Ok(Some(Err(e))) => {
            tracing::warn!("Admin WS error during auth: {}", e);
            return;
        }
        Err(_) => {
            let _ = send_admin_message(
                &mut ws_tx,
                &AdminServerMessage::AuthError {
                    message: "Authentication timeout".to_string(),
                },
            )
            .await;
            return;
        }
    };

    // Validate initial token
    if let Err(msg) = validate_admin_token(&state, &initial_auth).await {
        let _ = send_admin_message(
            &mut ws_tx,
            &AdminServerMessage::AuthError { message: msg },
        )
        .await;
        return;
    }

    // Auth successful
    if send_admin_message(&mut ws_tx, &AdminServerMessage::AuthOk)
        .await
        .is_err()
    {
        return;
    }

    tracing::debug!("Admin WS client authenticated");

    // Send initial state snapshot
    let runners = get_runners_snapshot(&state).await;
    let models = get_models_snapshot(&state).await;
    let stats = get_stats_snapshot(&state);
    if send_admin_message(&mut ws_tx, &AdminServerMessage::StateSnapshot { runners, models, stats })
        .await
        .is_err()
    {
        return;
    }

    // Subscribe to runner events
    let event_rx = state.runner_registry.subscribe_events();
    let mut event_stream = BroadcastStream::new(event_rx);

    // Ping interval for keep-alive (30 seconds)
    let mut ping_interval = tokio::time::interval(Duration::from_secs(30));
    ping_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            // Handle runner events from registry
            Some(event_result) = TokioStreamExt::next(&mut event_stream) => {
                match event_result {
                    Ok(event) => {
                        // Send the runner event
                        let msg = AdminServerMessage::from(event);
                        if send_admin_message(&mut ws_tx, &msg).await.is_err() {
                            break;
                        }
                        // Also send updated models (since model availability depends on runners)
                        let models = get_models_snapshot(&state).await;
                        if send_admin_message(&mut ws_tx, &AdminServerMessage::ModelsUpdated { models }).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(n)) => {
                        tracing::warn!("Admin WS client lagged, skipped {} events", n);
                    }
                }
            }

            // Handle incoming messages from client
            Some(msg_result) = FuturesStreamExt::next(&mut ws_rx) => {
                match msg_result {
                    Ok(Message::Text(text)) => {
                        // Handle re-auth messages
                        match serde_json::from_str::<AdminClientMessage>(&text) {
                            Ok(AdminClientMessage::Auth { token }) => {
                                match validate_admin_token(&state, &token).await {
                                    Ok(()) => {
                                        tracing::debug!("Admin WS client re-authenticated");
                                        if send_admin_message(&mut ws_tx, &AdminServerMessage::AuthOk)
                                            .await
                                            .is_err()
                                        {
                                            break;
                                        }
                                    }
                                    Err(msg) => {
                                        // Don't disconnect on re-auth failure, just report the error
                                        // The client can try again with a valid token
                                        let _ = send_admin_message(
                                            &mut ws_tx,
                                            &AdminServerMessage::AuthError { message: msg },
                                        )
                                        .await;
                                    }
                                }
                            }
                            Err(_) => {
                                // Ignore malformed messages
                                tracing::debug!("Admin WS received malformed message");
                            }
                        }
                    }
                    Ok(Message::Ping(data)) => {
                        if ws_tx.send(Message::Pong(data)).await.is_err() {
                            break;
                        }
                    }
                    Ok(Message::Pong(_)) => {
                        // Client responded to our ping, connection is alive
                    }
                    Ok(Message::Close(_)) => {
                        tracing::debug!("Admin WS client sent close frame");
                        break;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!("Admin WS error: {}", e);
                        break;
                    }
                }
            }

            // Send ping for keep-alive
            _ = ping_interval.tick() => {
                if ws_tx.send(Message::Ping(vec![])).await.is_err() {
                    break;
                }
            }

            else => break,
        }
    }

    tracing::debug!("Admin WS client disconnected");
}

/// Get current runners list (connected + offline from DB).
async fn get_runners_snapshot(state: &AppState) -> Vec<RunnerInfo> {
    use std::collections::HashSet;

    let connected_runners = state.runner_registry.all().await;
    let connected_ids: HashSet<String> = connected_runners.iter().map(|r| r.id.clone()).collect();

    let mut runners: Vec<RunnerInfo> = connected_runners
        .into_iter()
        .map(|r| {
            let loaded_models = r.loaded_models();
            RunnerInfo {
                id: r.id,
                name: r.name,
                machine_type: r.machine_type,
                health: format!("{:?}", r.status.health),
                loaded_models,
                connected_at: r.connected_at.to_rfc3339(),
                last_heartbeat: r.last_heartbeat.to_rfc3339(),
                http_base_url: r.http_base_url,
                mac_address: r.mac_address,
                is_online: true,
            }
        })
        .collect();

    // Add offline runners from database
    if let Ok(db_runners) = state.audit_logger.get_all_runners() {
        for db_runner in db_runners {
            if !connected_ids.contains(&db_runner.id) {
                runners.push(RunnerInfo {
                    id: db_runner.id,
                    name: db_runner.name,
                    machine_type: db_runner.machine_type,
                    health: "Offline".to_string(),
                    loaded_models: vec![],
                    connected_at: "".to_string(),
                    last_heartbeat: db_runner.last_seen_at,
                    http_base_url: None,
                    mac_address: db_runner.mac_address,
                    is_online: false,
                });
            }
        }
    }

    runners
}

/// Get current models list.
async fn get_models_snapshot(state: &AppState) -> Vec<AdminModelInfo> {
    state
        .inference_router
        .list_models_with_details()
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|m| AdminModelInfo {
            id: m.id,
            name: m.name,
            size_bytes: m.size_bytes,
            parameter_count: m.parameter_count,
            context_length: m.context_length,
            quantization: m.quantization,
            modified_at: m.modified_at,
            loaded: m.loaded,
            runners: m.runners,
            available_on: m.available_on,
        })
        .collect()
}

/// Get current dashboard stats.
fn get_stats_snapshot(state: &AppState) -> DashboardStatsInfo {
    let stats = state.audit_logger.get_stats().unwrap_or(DashboardStats {
        total_users: 0,
        total_requests: 0,
        requests_24h: 0,
        total_tokens: 0,
    });

    DashboardStatsInfo {
        total_users: stats.total_users,
        total_requests: stats.total_requests,
        requests_24h: stats.requests_24h,
        total_tokens: stats.total_tokens,
    }
}

/// Validate a JWT token for admin access.
async fn validate_admin_token(state: &AppState, token: &str) -> Result<(), String> {
    let user = state
        .jwks_client
        .validate_token(token)
        .await
        .map_err(|e| format!("Invalid token: {}", e))?;

    if !user.is_admin() {
        return Err("Admin access required".to_string());
    }

    Ok(())
}

/// Send an AdminServerMessage over WebSocket.
async fn send_admin_message<S>(
    sink: &mut S,
    msg: &AdminServerMessage,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    S: SinkExt<Message> + Unpin,
    S::Error: std::error::Error + Send + Sync + 'static,
{
    let json = serde_json::to_string(msg)?;
    sink.send(Message::Text(json)).await?;
    Ok(())
}

/// GET /admin/runners/events - SSE stream for real-time runner events.
///
/// Authentication is done via query parameter since EventSource doesn't support headers.
/// Note: Prefer WebSocket endpoint (/admin/ws) for new clients - it supports token refresh.
async fn runner_events(
    State(state): State<Arc<AppState>>,
    Query(auth): Query<SseAuthQuery>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, StatusCode> {
    // Validate JWT from query parameter
    let user = state
        .jwks_client
        .validate_token(&auth.token)
        .await
        .map_err(|_| StatusCode::UNAUTHORIZED)?;

    if !user.is_admin() {
        return Err(StatusCode::FORBIDDEN);
    }

    // Subscribe to registry events
    let rx = state.runner_registry.subscribe_events();

    // Convert broadcast receiver to stream
    let stream = TokioStreamExt::filter_map(BroadcastStream::new(rx), |result| {
        match result {
            Ok(event) => {
                // Determine event type name for SSE
                let event_type = match &event {
                    RunnerEvent::Connected { .. } => "runner_connected",
                    RunnerEvent::Disconnected { .. } => "runner_disconnected",
                    RunnerEvent::StatusChanged { .. } => "runner_status_changed",
                };

                // Serialize event data
                match serde_json::to_string(&event) {
                    Ok(data) => Some(Ok(Event::default().event(event_type).data(data))),
                    Err(_) => None,
                }
            }
            Err(_) => None, // Skip lagged errors
        }
    });

    Ok(Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15))))
}

/// Build the admin router.
pub fn router(state: Arc<AppState>) -> Router {
    // SSE endpoint with query-based auth (separate from middleware-protected routes)
    // Note: SSE is kept for backward compatibility, prefer WebSocket for new clients
    let sse_routes = Router::new()
        .route("/runners/events", get(runner_events))
        .with_state(state.clone());

    // WebSocket endpoint with message-based auth (supports token refresh mid-connection)
    let ws_routes = Router::new()
        .route("/ws", get(admin_ws))
        .with_state(state.clone());

    // Regular admin routes with middleware authentication
    let admin_routes = Router::new()
        .route("/", get(dashboard))
        .route("/users", get(users_list))
        .route("/users/:id/disable", post(disable_user))
        .route("/users/:id/enable", post(enable_user))
        .route("/requests", get(requests_list))
        .route("/runners", get(list_runners))
        .route("/runners/:id/wake", post(wake_runner))
        .route("/models", get(list_models))
        .layer(middleware::from_fn_with_state(state.clone(), require_admin))
        .with_state(state);

    // Merge routes: SSE and WS first (more specific paths), then admin routes
    sse_routes.merge(ws_routes).merge(admin_routes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_requests_query_defaults() {
        let query = RequestsQuery {
            user_id: None,
            model: None,
            page: None,
        };
        assert!(query.user_id.is_none());
        assert!(query.model.is_none());
        assert!(query.page.is_none());
    }

    #[tokio::test]
    async fn test_requests_query_with_values() {
        let query = RequestsQuery {
            user_id: Some("user123".to_string()),
            model: Some("llama2".to_string()),
            page: Some(2),
        };
        assert_eq!(query.user_id, Some("user123".to_string()));
        assert_eq!(query.model, Some("llama2".to_string()));
        assert_eq!(query.page, Some(2));
    }

    #[tokio::test]
    async fn test_admin_user_struct() {
        let admin = AdminUser {
            sub: "auth0|123".to_string(),
            email: Some("admin@example.com".to_string()),
        };
        assert_eq!(admin.sub, "auth0|123");
        assert_eq!(admin.email, Some("admin@example.com".to_string()));
    }

    #[tokio::test]
    async fn test_admin_user_without_email() {
        let admin = AdminUser {
            sub: "auth0|456".to_string(),
            email: None,
        };
        assert!(admin.email.is_none());
    }

    #[tokio::test]
    async fn test_dashboard_stats_default() {
        let stats = DashboardStats {
            total_users: 0,
            total_requests: 0,
            requests_24h: 0,
            total_tokens: 0,
        };
        assert_eq!(stats.total_users, 0);
        assert_eq!(stats.total_requests, 0);
    }

    #[tokio::test]
    async fn test_dashboard_stats_with_values() {
        let stats = DashboardStats {
            total_users: 100,
            total_requests: 1000,
            requests_24h: 50,
            total_tokens: 50000,
        };
        assert_eq!(stats.total_users, 100);
        assert_eq!(stats.total_requests, 1000);
        assert_eq!(stats.requests_24h, 50);
        assert_eq!(stats.total_tokens, 50000);
    }

    #[tokio::test]
    async fn test_users_template() {
        let template = UsersTemplate {
            nav_active: "users",
            admin_email: "admin@test.com".to_string(),
            users: vec![],
        };
        assert_eq!(template.nav_active, "users");
        assert!(template.users.is_empty());
    }

    #[tokio::test]
    async fn test_requests_template() {
        let template = RequestsTemplate {
            nav_active: "requests",
            admin_email: "admin@test.com".to_string(),
            requests: vec![],
            filter_user_id: "".to_string(),
            filter_model: "".to_string(),
            page: 1,
            total_pages: 1,
        };
        assert_eq!(template.page, 1);
        assert_eq!(template.total_pages, 1);
    }

    #[tokio::test]
    async fn test_dashboard_template() {
        let template = DashboardTemplate {
            nav_active: "dashboard",
            admin_email: "admin@test.com".to_string(),
            stats: DashboardStats {
                total_users: 10,
                total_requests: 100,
                requests_24h: 5,
                total_tokens: 1000,
            },
            recent_requests: vec![],
        };
        assert_eq!(template.nav_active, "dashboard");
        assert_eq!(template.stats.total_users, 10);
    }

    #[tokio::test]
    async fn test_disable_user_route_path_extraction() {
        let user_id = "test-user-123".to_string();
        assert_eq!(user_id, "test-user-123");
    }
}
