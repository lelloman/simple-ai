use crate::{
    error::{Error, Result},
    state::AppState,
};
use simple_ai_common::{DecisionRequest, DecisionResponse, DECISION_BODY_LIMIT, DECISION_MODEL};
use simple_server::web::{extract::State, routing::post, Json, Router};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/decisions", post(decide)).layer(
        simple_server::body_limit::BodyLimit::max(DECISION_BODY_LIMIT),
    )
}
async fn decide(
    State(state): State<Arc<AppState>>,
    Json(mut request): Json<DecisionRequest>,
) -> Result<Json<DecisionResponse>> {
    request.validate().map_err(Error::InvalidRequest)?;
    let start = Instant::now();
    let requested = request.model.clone();
    let resolved = state
        .config
        .aliases
        .mappings
        .get(&requested)
        .cloned()
        .unwrap_or(requested.clone());
    let (engine, local) = state
        .engine_registry
        .resolve_engine_for_model(&resolved)
        .await
        .ok_or_else(|| Error::ModelNotFound(resolved.clone()))?;
    if engine.engine_type() != "decisions" || local != DECISION_MODEL {
        return Err(Error::InvalidRequest(
            "model does not support semantic decisions".into(),
        ));
    }
    request.model = local;
    let queue = state
        .decision_queue
        .clone()
        .try_acquire_owned()
        .map_err(|_| Error::UpstreamResponse {
            status: 429,
            body: "decision queue full".into(),
        })?;
    let slot = tokio::time::timeout(
        Duration::from_secs(120),
        state.decision_slots.clone().acquire_owned(),
    )
    .await
    .map_err(|_| Error::UpstreamResponse {
        status: 429,
        body: "decision admission timed out".into(),
    })?
    .map_err(|e| Error::Internal(e.to_string()))?;
    let queue_ms = start.elapsed().as_secs_f64() * 1000.;
    // The owned task keeps the GPU lease until inference/cleanup completes, even after disconnect.
    let work = state.engine_registry.drain.track_existing();
    tokio::spawn(async move {
        let (_queue, _slot, _work) = (queue, slot, work);
        let lease = state.engine_registry.acquire_model(&resolved).await?;
        let mut response = lease.engine.decide(&lease.engine_model, &request).await?;
        response.model = requested;
        response.timing.queue_ms = queue_ms;
        response.timing.total_ms = start.elapsed().as_secs_f64() * 1000.;
        Ok(Json(response))
    })
    .await
    .map_err(|e| Error::Internal(e.to_string()))?
}
