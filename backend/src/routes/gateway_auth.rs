use simple_server::web::{extract::State, routing::get, Json, Router};
use serde_json::{json, Value};
use crate::AppState;

/// Public login metadata only. No secrets or calling-app credentials.
async fn configuration(State(state): State<std::sync::Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "issuer": state.config.oidc.issuer,
        "client_id": state.config.oidc.android_client_id,
    }))
}

pub fn router(state: std::sync::Arc<AppState>) -> Router {
    Router::new().route("/.well-known/simple-ai", get(configuration)).with_state(state)
}
