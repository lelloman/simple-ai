//! Advisory pressure for clients: `GET /v1/pressure` and the response headers
//! `X-SimpleAI-Pressure` / `X-SimpleAI-Pressure-Group` on every `/v1` response.
//!
//! Pressure never rejects work; polite clients may defer deferrable calls while
//! their feature group is orange or red.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::Arc;

use serde::Serialize;
use simple_server::web::{
    extract::{Request, State},
    http::HeaderValue,
    middleware::Next,
    response::Response,
    routing::get,
    Json, Router,
};

use crate::gateway::{ModelClass, PressureLevel, PressureTracker};

pub const PRESSURE_HEADER: &str = "x-simpleai-pressure";
pub const PRESSURE_GROUP_HEADER: &str = "x-simpleai-pressure-group";

tokio::task_local! {
    /// Model class of the current request, noted by handlers that parse a model field.
    static REQUEST_CLASS: Cell<Option<ModelClass>>;
}

/// Record the class a chat/responses request resolved to, so its response
/// carries that feature group's level. No-op outside the pressure middleware.
pub fn note_class(class: Option<ModelClass>) {
    let _ = REQUEST_CLASS.try_with(|cell| cell.set(class));
}

/// Class implied by an endpoint whose feature does not depend on the model field.
fn class_for_path(path: &str) -> Option<ModelClass> {
    let path = path.trim_end_matches('/');
    [
        ("/audio/embeddings", ModelClass::AudioEmbeddings),
        ("/audio/speech", ModelClass::Tts),
        ("/embeddings", ModelClass::EmbedSmall),
        ("/classifications", ModelClass::TextClassification),
        ("/decisions", ModelClass::SemanticDecisions),
        ("/extractions", ModelClass::InformationExtraction),
    ]
    .into_iter()
    .find(|(suffix, _)| path.ends_with(suffix))
    .map(|(_, class)| class)
}

/// Group and level for a response: the request's feature group when known,
/// otherwise the server-wide level.
fn header_values(
    tracker: &PressureTracker,
    class: Option<ModelClass>,
) -> (PressureLevel, Option<String>) {
    let group = class.and_then(|class| tracker.group_for_class(class.as_str()));
    match group.and_then(|group| tracker.group_level(group).map(|level| (group, level))) {
        Some((group, level)) => (level, Some(group.to_string())),
        None => (tracker.snapshot().level, None),
    }
}

pub async fn pressure_header_middleware(
    State(tracker): State<Arc<PressureTracker>>,
    request: Request,
    next: Next,
) -> Response {
    let path_class = class_for_path(request.uri().path());
    let (noted, mut response) = REQUEST_CLASS
        .scope(Cell::new(None), async move {
            let response = next.run(request).await;
            (REQUEST_CLASS.with(Cell::get), response)
        })
        .await;
    let (level, group) = header_values(&tracker, noted.or(path_class));
    let headers = response.headers_mut();
    headers.insert(PRESSURE_HEADER, HeaderValue::from_static(level.as_str()));
    if let Some(value) = group.and_then(|group| HeaderValue::from_str(&group).ok()) {
        headers.insert(PRESSURE_GROUP_HEADER, value);
    }
    response
}

#[derive(Debug, Serialize)]
pub struct ClientGroupPressure {
    pub level: PressureLevel,
    pub available: bool,
    pub cold: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_secs: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ClientPressure {
    pub level: PressureLevel,
    pub updated_at: String,
    pub groups: BTreeMap<String, ClientGroupPressure>,
}

/// GET /v1/pressure - levels per feature group, without host details.
async fn get_pressure(State(tracker): State<Arc<PressureTracker>>) -> Json<ClientPressure> {
    let snapshot = tracker.snapshot();
    Json(ClientPressure {
        level: snapshot.level,
        updated_at: snapshot.updated_at,
        groups: snapshot
            .groups
            .into_iter()
            .map(|(name, group)| {
                (
                    name,
                    ClientGroupPressure {
                        level: group.level,
                        available: group.available,
                        cold: group.cold,
                        retry_after_secs: group.retry_after_secs,
                    },
                )
            })
            .collect(),
    })
}

pub fn router(tracker: Arc<PressureTracker>) -> Router {
    Router::new()
        .route("/pressure", get(get_pressure))
        .with_state(tracker)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::pressure::{HostInput, LaneInput};
    use simple_server::web::{body::Body, http::StatusCode, middleware, routing::post};
    use std::time::{Duration, Instant};
    use tower::ServiceExt;

    fn tracker() -> Arc<PressureTracker> {
        Arc::new(PressureTracker::new(Default::default(), Default::default()))
    }

    fn busy_big(tracker: &PressureTracker) {
        let lane = |classes: &[&str], waiting| LaneInput {
            lane: "gpu:0".into(),
            classes: classes.iter().map(|c| c.to_string()).collect(),
            waiting,
            ..Default::default()
        };
        let hosts = || {
            vec![
                HostInput {
                    runner_id: "halo".into(),
                    name: "Halo".into(),
                    online: true,
                    lanes: vec![lane(&["big"], 2)],
                },
                HostInput {
                    runner_id: "rtx".into(),
                    name: "RTX".into(),
                    online: true,
                    lanes: vec![lane(&["fast", "information_extraction"], 0)],
                },
            ]
        };
        let t0 = Instant::now();
        tracker.evaluate(hosts(), t0);
        tracker.evaluate(hosts(), t0 + Duration::from_secs(2));
    }

    async fn call(app: Router, path: &str) -> Response {
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn responses_carry_their_feature_group_level() {
        let tracker = tracker();
        busy_big(&tracker);
        let app = Router::new()
            .route(
                "/chat/completions",
                post(|| async {
                    note_class(Some(ModelClass::Big));
                    StatusCode::OK
                }),
            )
            .route("/extractions", post(|| async { StatusCode::OK }))
            .route("/detect-language", post(|| async { StatusCode::OK }))
            .layer(middleware::from_fn_with_state(
                tracker.clone(),
                pressure_header_middleware,
            ));

        let chat = call(app.clone(), "/chat/completions").await;
        assert_eq!(chat.headers()[PRESSURE_HEADER], "orange");
        assert_eq!(chat.headers()[PRESSURE_GROUP_HEADER], "big");

        let extraction = call(app.clone(), "/extractions").await;
        assert_eq!(extraction.headers()[PRESSURE_HEADER], "green");
        assert_eq!(extraction.headers()[PRESSURE_GROUP_HEADER], "extraction");

        // No feature group: the server-wide (worst host) level.
        let other = call(app, "/detect-language").await;
        assert_eq!(other.headers()[PRESSURE_HEADER], "orange");
        assert!(other.headers().get(PRESSURE_GROUP_HEADER).is_none());
    }

    #[tokio::test]
    async fn pressure_endpoint_lists_groups_without_hosts() {
        let tracker = tracker();
        busy_big(&tracker);
        let response = call_get(router(tracker), "/pressure").await;
        let body: serde_json::Value = serde_json::from_slice(
            &simple_server::web::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["level"], "orange");
        assert_eq!(body["groups"]["big"]["level"], "orange");
        assert_eq!(body["groups"]["big"]["retry_after_secs"], 15);
        assert_eq!(body["groups"]["fast"]["level"], "green");
        assert_eq!(body["groups"]["tts"]["available"], false);
        assert!(body.get("hosts").is_none());
        assert!(body["groups"]["big"].get("hosts").is_none());
    }

    async fn call_get(app: Router, path: &str) -> Response {
        app.oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap()
    }
}
