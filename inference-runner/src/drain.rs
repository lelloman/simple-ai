//! Runner-wide admission and ownership of accepted work.
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use simple_ai_common::{DrainAction, RunnerHealth};
use simple_server::web::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use tokio::sync::Notify;

#[derive(Default)]
struct Inner {
    action: Option<DrainAction>,
    active: usize,
    armed: bool,
    awaiting_ack: Option<String>,
    completed: bool,
    stop: bool,
}

#[derive(Default)]
pub struct Drain {
    inner: Mutex<Inner>,
    changed: Notify,
    stopped: Notify,
}

/// Clones retain ownership of the same accepted request, including detached work.
#[derive(Clone)]
pub struct Work {
    _owner: Arc<WorkOwner>,
}
struct WorkOwner(Arc<Drain>);
impl Drop for WorkOwner {
    fn drop(&mut self) {
        self.0.inner.lock().unwrap().active -= 1;
        self.0.changed.notify_one();
    }
}

impl Drain {
    pub fn admit(self: &Arc<Self>) -> Option<Work> {
        let mut inner = self.inner.lock().unwrap();
        if inner.action.is_some() {
            return None;
        }
        inner.active += 1;
        Some(Work {
            _owner: Arc::new(WorkOwner(self.clone())),
        })
    }

    /// Only for children of already admitted work, before the parent's guard drops.
    pub fn track_existing(self: &Arc<Self>) -> Work {
        self.inner.lock().unwrap().active += 1;
        Work {
            _owner: Arc::new(WorkOwner(self.clone())),
        }
    }

    pub fn begin(&self, action: DrainAction, request_id: &str) -> Result<(), &'static str> {
        let mut inner = self.inner.lock().unwrap();
        match inner.action {
            Some(previous) if previous == action => return Ok(()),
            // A drain-only request may later be upgraded to a terminal action.
            Some(DrainAction::Drain) => {}
            Some(_) => return Err("a different drain action is already pending"),
            None => {}
        }
        inner.action = Some(action);
        inner.completed = false;
        inner.armed = false;
        inner.awaiting_ack = Some(request_id.to_owned());
        self.changed.notify_one();
        Ok(())
    }

    /// Called after the command acknowledgment (or reconnect registration) is sent.
    pub fn arm(&self, request_id: Option<&str>) {
        let mut inner = self.inner.lock().unwrap();
        if request_id.is_none() || inner.awaiting_ack.as_deref() == request_id {
            inner.armed = true;
            self.changed.notify_one();
        }
    }

    pub fn health(&self) -> Option<RunnerHealth> {
        let inner = self.inner.lock().unwrap();
        inner.action.map(|_| {
            if inner.active == 0 {
                RunnerHealth::Drained
            } else {
                RunnerHealth::Draining
            }
        })
    }

    async fn next_action(&self) -> DrainAction {
        loop {
            let notified = self.changed.notified();
            {
                let mut inner = self.inner.lock().unwrap();
                if inner.armed && inner.active == 0 && !inner.completed {
                    if let Some(action) = inner.action {
                        inner.completed = true;
                        return action;
                    }
                }
            }
            notified.await;
        }
    }

    pub async fn stopped(&self) {
        loop {
            let notified = self.stopped.notified();
            if self.inner.lock().unwrap().stop {
                return;
            }
            notified.await;
        }
    }

    pub async fn run(&self, stop: simple_server::lifecycle::Shutdown) {
        loop {
            let action = tokio::select! {
                _ = stop.requested() => return,
                action = self.next_action() => action,
            };
            tracing::info!(?action, "Runner drain complete");
            match action {
                DrainAction::Drain => {}
                DrainAction::Stop => {
                    self.inner.lock().unwrap().stop = true;
                    self.stopped.notify_one();
                }
                DrainAction::Shutdown | DrainAction::Reboot => {
                    let verb = if action == DrainAction::Shutdown {
                        "poweroff"
                    } else {
                        "reboot"
                    };
                    // Fixed argv: the gateway cannot supply arbitrary shell commands.
                    let operation = tokio::time::timeout(
                        std::time::Duration::from_secs(30),
                        tokio::process::Command::new("systemctl")
                            .args(["--no-ask-password", verb])
                            .kill_on_drop(true)
                            .output(),
                    );
                    let result = tokio::select! {
                        _ = stop.requested() => return,
                        result = operation => result,
                    };
                    match result {
                        Ok(Ok(output)) if output.status.success() => {}
                        other => tracing::error!(
                            ?other,
                            ?action,
                            "Host action failed; runner remains drained; no automatic retry"
                        ),
                    }
                }
            }
        }
    }
}

pub async fn admission(
    State(state): State<Arc<crate::state::AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let Some(work) = state.engine_registry.drain.admit() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            simple_server::web::Json(serde_json::json!({"error": {
                "message": "Runner is draining", "type": "runner_draining"
            }})),
        )
            .into_response();
    };
    let response = next.run(request).await;
    let (parts, body) = response.into_parts();
    // Keep ownership until the last streaming chunk or response cancellation.
    let stream = futures_util::stream::unfold(
        (body.into_data_stream(), work),
        |(mut body, work)| async move { body.next().await.map(|chunk| (chunk, (body, work))) },
    );
    Response::from_parts(parts, Body::from_stream(stream))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn response_body_keeps_drain_open_until_stream_finishes_or_is_dropped() {
        use simple_server::web::{routing::get, Router};
        use tower::ServiceExt;
        for consume in [true, false] {
            let config = serde_json::from_value(serde_json::json!({
                "runner": {"id":"test", "name":"test"}, "api": {}
            }))
            .unwrap();
            let registry = Arc::new(crate::engine::EngineRegistry::new());
            let drain = registry.drain.clone();
            let state = Arc::new(crate::state::AppState::new(config, registry, None));
            let app = Router::new()
                .route(
                    "/stream",
                    get(|| async {
                        Body::from_stream(futures_util::stream::iter([
                            Ok::<_, std::io::Error>("first"),
                            Ok("last"),
                        ]))
                    }),
                )
                .layer(simple_server::web::middleware::from_fn_with_state(
                    state, admission,
                ));
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/stream")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            drain.begin(DrainAction::Drain, "test").unwrap();
            drain.arm(Some("test"));
            assert_eq!(drain.health(), Some(RunnerHealth::Draining));
            assert_eq!(
                app.oneshot(
                    Request::builder()
                        .uri("/stream")
                        .body(Body::empty())
                        .unwrap()
                )
                .await
                .unwrap()
                .status(),
                StatusCode::SERVICE_UNAVAILABLE
            );
            if consume {
                assert_eq!(
                    simple_server::web::body::to_bytes(response.into_body(), 100)
                        .await
                        .unwrap(),
                    "firstlast"
                );
            } else {
                drop(response);
            }
            assert_eq!(drain.next_action().await, DrainAction::Drain);
        }
    }

    #[tokio::test]
    async fn drain_waits_for_all_owners_and_rejects_new_work() {
        let drain = Arc::new(Drain::default());
        let work = drain.admit().unwrap();
        let detached = work.clone();
        drain.begin(DrainAction::Stop, "test").unwrap();
        drain.arm(Some("test"));
        assert!(drain.admit().is_none());
        assert_eq!(drain.health(), Some(RunnerHealth::Draining));
        drop(work);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), drain.next_action())
                .await
                .is_err()
        );
        drop(detached);
        assert_eq!(drain.next_action().await, DrainAction::Stop);
        assert_eq!(drain.health(), Some(RunnerHealth::Drained));
        drain.begin(DrainAction::Stop, "test").unwrap();
        assert!(drain.begin(DrainAction::Reboot, "test").is_err());
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), drain.next_action())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn drain_can_be_upgraded_but_waits_for_acknowledgment() {
        let drain = Arc::new(Drain::default());
        drain.begin(DrainAction::Drain, "test").unwrap();
        drain.arm(Some("test"));
        assert_eq!(drain.next_action().await, DrainAction::Drain);
        drain.begin(DrainAction::Reboot, "test").unwrap();
        drain.arm(Some("earlier-command"));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), drain.next_action())
                .await
                .is_err()
        );
        drain.arm(Some("test"));
        assert_eq!(drain.next_action().await, DrainAction::Reboot);
    }
}
