//! Generic zero-shot NLI classification endpoint.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{ConnectInfo, State},
    http::{HeaderMap, StatusCode},
    routing::post,
    Json, Router,
};
use simple_ai_common::{ClassificationRequest, ClassificationResponse};

use super::auth_helpers::{authenticate_request, extract_client_ip};
use crate::gateway::{can_request_model, ModelRequest, SchedulerError};
use crate::models::request::{Request, Response};
use crate::{AppState, RequestEvent};

const MAX_INPUTS: usize = 256;
const MAX_LABELS: usize = 32;

fn validate(request: &ClassificationRequest) -> Result<(), (StatusCode, String)> {
    let input_count = request.input.len();
    if request.input.is_empty() || input_count > MAX_INPUTS {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("input must contain between 1 and {MAX_INPUTS} non-empty texts"),
        ));
    }
    if request
        .input
        .clone()
        .into_vec()
        .iter()
        .any(|text| text.trim().is_empty())
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "input texts must not be empty".to_string(),
        ));
    }
    if request.labels.is_empty() || request.labels.len() > MAX_LABELS {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("labels must contain between 1 and {MAX_LABELS} hypotheses"),
        ));
    }
    if request.labels.iter().any(|label| {
        label.label.trim().is_empty()
            || label.hypothesis.trim().is_empty()
            || label.hypothesis.len() > 4096
    }) {
        return Err((
            StatusCode::BAD_REQUEST,
            "each label and hypothesis must be non-empty; hypotheses are limited to 4096 bytes"
                .to_string(),
        ));
    }
    Ok(())
}

async fn create_classifications(
    State(state): State<Arc<AppState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(request): Json<ClassificationRequest>,
) -> Result<Json<ClassificationResponse>, (StatusCode, String)> {
    let start = Instant::now();
    validate(&request)?;
    let (auth_user, user) = authenticate_request(&state, &headers).await?;
    let model_request = ModelRequest::parse(&request.model);
    if !can_request_model(&auth_user.roles, &model_request) {
        return Err((
            StatusCode::BAD_REQUEST,
            "Permission denied: request class:text_classification or use a model:specific role"
                .to_string(),
        ));
    }
    if !state.config.gateway.enabled {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "text classification requires gateway mode and a classification-capable runner"
                .to_string(),
        ));
    }

    let model = match &model_request {
        ModelRequest::Specific(model) => model.clone(),
        ModelRequest::Class(class) => format!("class:{class}"),
    };
    let prompt_tokens = request
        .input
        .clone()
        .into_vec()
        .iter()
        .map(|text| text.split_whitespace().count())
        .sum::<usize>()
        + request
            .labels
            .iter()
            .map(|label| label.hypothesis.split_whitespace().count() * request.input.len())
            .sum::<usize>();

    let mut req_log = Request::new(user.id.clone(), "/v1/classifications".to_string());
    req_log.model = Some(model.clone());
    req_log.client_ip = extract_client_ip(&headers, connect_info.map(|info| info.0));
    let request_id = state
        .audit_logger
        .log_request(&req_log)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;

    let scheduled = state
        .request_scheduler
        .classification(&req_log.id, &model, &model_request, &request)
        .await;
    let (response, runner_id, wol_sent) = match scheduled {
        Ok(result) => (result.response, Some(result.runner_id), result.wol_sent),
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
    response_log.tokens_prompt = Some(prompt_tokens as u32);
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
        tokens_prompt: Some(prompt_tokens as i64),
        tokens_completion: None,
        runner_id,
        wol_sent,
    });
    Ok(Json(response))
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/classifications", post(create_classifications))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use simple_ai_common::{ClassificationInput, ClassificationLabel};

    fn request(
        input: ClassificationInput,
        labels: Vec<ClassificationLabel>,
    ) -> ClassificationRequest {
        ClassificationRequest {
            model: "class:text_classification".to_string(),
            input,
            labels,
        }
    }

    #[test]
    fn validates_a_well_formed_request() {
        assert!(validate(&request(
            ClassificationInput::Single("torrent metadata".to_string()),
            vec![ClassificationLabel {
                label: "leak".to_string(),
                hypothesis: "This torrent contains leaked records.".to_string(),
            }],
        ))
        .is_ok());
    }

    #[test]
    fn rejects_empty_text_and_hypothesis() {
        assert!(validate(&request(
            ClassificationInput::Multiple(vec![" ".to_string()]),
            vec![ClassificationLabel {
                label: "leak".to_string(),
                hypothesis: "This is a leak.".to_string(),
            }],
        ))
        .is_err());
        assert!(validate(&request(
            ClassificationInput::Single("metadata".to_string()),
            vec![ClassificationLabel {
                label: "leak".to_string(),
                hypothesis: "".to_string(),
            }],
        ))
        .is_err());
    }
}
