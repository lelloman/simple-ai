//! Authenticated schema-driven information extraction endpoint.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use simple_server::web::{
    extract::{ConnectInfo, State},
    http::{HeaderMap, StatusCode},
    routing::post,
    Json, Router,
};
use simple_ai_common::{ExtractionRequest, ExtractionResponse};

use super::auth_helpers::{authenticate_inference_request, extract_client_ip};
use crate::gateway::{can_request_model, ModelClass, ModelRequest, SchedulerError};
use crate::models::request::{Request, Response};
use crate::{AppState, RequestEvent};

async fn create_extractions(
    State(state): State<Arc<AppState>>,
    connect_info: Result<
        ConnectInfo<SocketAddr>,
        simple_server::extract::RejectionResponse,
    >,
    headers: HeaderMap,
    Json(request): Json<ExtractionRequest>,
) -> Result<Json<ExtractionResponse>, (StatusCode, String)> {
    let start = Instant::now();
    request
        .validate()
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let (auth_user, user) =
        authenticate_inference_request(&state, &headers, connect_info.as_ref().ok().map(|c| c.0)).await?;
    let model_request = ModelRequest::parse(&request.model);
    if matches!(&model_request, ModelRequest::Class(class) if *class != ModelClass::InformationExtraction)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "extractions require class:information_extraction or a specific extraction model"
                .into(),
        ));
    }

    if !can_request_model(&auth_user.roles, &model_request) {
        return Err((
            StatusCode::BAD_REQUEST,
            "Permission denied: request class:information_extraction or use a model:specific role"
                .to_string(),
        ));
    }
    if !state.config.gateway.enabled {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "text extraction requires gateway mode and an extraction-capable runner".to_string(),
        ));
    }

    let model = match &model_request {
        ModelRequest::Specific(model) => model.clone(),
        ModelRequest::Class(class) => format!("class:{class}"),
    };
    let mut req_log = Request::new(user.id.clone(), "/v1/extractions".to_string());
    req_log.model = Some(model.clone());
    req_log.client_ip = extract_client_ip(&headers, connect_info.as_ref().ok().map(|info| info.0));
    let request_id = state
        .audit_logger
        .log_request(&req_log)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;

    let scheduled = state
        .request_scheduler
        .extraction(&req_log.id, &model, &model_request, &request)
        .await;
    let (response, runner_id, wol_sent) = match scheduled {
        Ok(mut result) => {
            result.response.model = result.resolved_model;
            (result.response, Some(result.runner_id), result.wol_sent)
        }
        Err(error) => {
            let status = match &error {
                SchedulerError::Router(router_error) => router_error.client_status(),
                SchedulerError::Wake(_) => StatusCode::INTERNAL_SERVER_ERROR,
            };
            let mut response_log = Response::new(request_id, status.as_u16());
            response_log.latency_ms = start.elapsed().as_millis() as u64;
            response_log.response_body = error.to_string();
            let _ = state.audit_logger.log_response(&response_log);
            return Err((status, error.to_string()));
        }
    };

    let mut response_log = Response::new(request_id, 200);
    response_log.latency_ms = start.elapsed().as_millis() as u64;

    response_log.runner_id = runner_id.clone();
    response_log.wol_sent = wol_sent;
    response_log.model_class = model_request
        .effective_class(&state.config.models)
        .map(|class| class.as_str().to_string());
    let _ = state.audit_logger.log_response(&response_log);
    let _ = state.request_events.send(RequestEvent {
        id: req_log.id,
        timestamp: req_log.timestamp.to_rfc3339(),
        user_id: req_log.user_id,
        user_email: auth_user.email,
        request_path: req_log.request_path,
        model: req_log.model,
        client_ip: req_log.client_ip,
        status: Some(200),
        latency_ms: Some(response_log.latency_ms as i64),
        tokens_prompt: None,
        tokens_completion: None,
        runner_id,
        wol_sent,
    });
    Ok(Json(response))
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/extractions", post(create_extractions))
        .with_state(state)
}
