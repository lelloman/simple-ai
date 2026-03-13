use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use chrono::Utc;
use serde::Serialize;
use tokio::sync::{broadcast, RwLock};

use crate::audit::AuditLogger;
use crate::config::ModelsConfig;

use super::{BatchQueue, ConnectedRunner, ModelClass, RunnerRegistry};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct RouterEventRecord {
    pub timestamp: String,
    pub kind: String,
    pub message: String,
    pub request_id: Option<String>,
    pub runner_id: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouterRunnerState {
    pub runner_id: String,
    pub name: String,
    pub machine_type: Option<String>,
    pub is_online: bool,
    pub health: String,
    pub scheduler_state: String,
    pub target_model: Option<String>,
    pub active_requests: usize,
    pub loaded_models: Vec<String>,
    pub available_models: Vec<String>,
    pub protected_classes: Vec<String>,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouterStateSnapshot {
    pub runners: Vec<RouterRunnerState>,
    pub queues: HashMap<String, usize>,
    pub recent_events: Vec<RouterEventRecord>,
}

#[derive(Debug, Clone)]
struct RunnerTransientState {
    scheduler_state: String,
    target_model: Option<String>,
    updated_at: String,
}

pub struct RouterTelemetry {
    event_tx: broadcast::Sender<RouterEventRecord>,
    recent_events: RwLock<VecDeque<RouterEventRecord>>,
    runner_states: RwLock<HashMap<String, RunnerTransientState>>,
}

impl RouterTelemetry {
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self {
            event_tx,
            recent_events: RwLock::new(VecDeque::with_capacity(256)),
            runner_states: RwLock::new(HashMap::new()),
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<RouterEventRecord> {
        self.event_tx.subscribe()
    }

    pub async fn emit(
        &self,
        kind: impl Into<String>,
        message: impl Into<String>,
        request_id: Option<String>,
        runner_id: Option<String>,
        model: Option<String>,
    ) {
        let event = RouterEventRecord {
            timestamp: Utc::now().to_rfc3339(),
            kind: kind.into(),
            message: message.into(),
            request_id,
            runner_id,
            model,
        };

        {
            let mut recent = self.recent_events.write().await;
            recent.push_front(event.clone());
            while recent.len() > 200 {
                recent.pop_back();
            }
        }

        let _ = self.event_tx.send(event);
    }

    pub async fn set_runner_state(
        &self,
        runner_id: &str,
        scheduler_state: &str,
        target_model: Option<String>,
    ) {
        self.runner_states.write().await.insert(
            runner_id.to_string(),
            RunnerTransientState {
                scheduler_state: scheduler_state.to_string(),
                target_model,
                updated_at: Utc::now().to_rfc3339(),
            },
        );
    }

    pub async fn clear_runner_state(&self, runner_id: &str) {
        self.runner_states.write().await.remove(runner_id);
    }

    pub async fn snapshot(
        &self,
        registry: &Arc<RunnerRegistry>,
        audit_logger: &Arc<AuditLogger>,
        batch_queue: Option<&Arc<BatchQueue>>,
        models_config: &ModelsConfig,
    ) -> RouterStateSnapshot {
        let connected = registry.all().await;
        let connected_map: HashMap<String, ConnectedRunner> = connected
            .into_iter()
            .map(|runner| (runner.id.clone(), runner))
            .collect();
        let db_runners = audit_logger.get_all_runners().unwrap_or_default();
        let transient_states = self.runner_states.read().await.clone();
        let recent_events = self.recent_events.read().await.iter().cloned().collect();

        let mut runners = Vec::new();
        let mut all_known_ids = std::collections::BTreeSet::new();
        for id in connected_map.keys() {
            all_known_ids.insert(id.clone());
        }
        for db_runner in &db_runners {
            all_known_ids.insert(db_runner.id.clone());
        }
        for id in transient_states.keys() {
            all_known_ids.insert(id.clone());
        }

        let all_available_by_runner: HashMap<String, Vec<String>> = all_known_ids
            .iter()
            .map(|id| {
                if let Some(runner) = connected_map.get(id) {
                    (id.clone(), runner.available_models())
                } else if let Some(db_runner) = db_runners.iter().find(|r| &r.id == id) {
                    (id.clone(), db_runner.available_models.clone())
                } else {
                    (id.clone(), Vec::new())
                }
            })
            .collect();

        for runner_id in all_known_ids {
            let connected_runner = connected_map.get(&runner_id);
            let db_runner = db_runners.iter().find(|r| r.id == runner_id);
            let transient = transient_states.get(&runner_id);

            let is_online = connected_runner.is_some();
            let available_models = connected_runner
                .map(|r| r.available_models())
                .or_else(|| db_runner.map(|r| r.available_models.clone()))
                .unwrap_or_default();
            let protected_classes =
                compute_protected_classes(&runner_id, &all_available_by_runner, models_config);

            runners.push(RouterRunnerState {
                runner_id: runner_id.clone(),
                name: connected_runner
                    .map(|r| r.name.clone())
                    .or_else(|| db_runner.map(|r| r.name.clone()))
                    .unwrap_or_else(|| runner_id.clone()),
                machine_type: connected_runner
                    .and_then(|r| r.machine_type.clone())
                    .or_else(|| db_runner.and_then(|r| r.machine_type.clone())),
                is_online,
                health: connected_runner
                    .map(|r| format!("{:?}", r.status.health))
                    .unwrap_or_else(|| "Offline".to_string()),
                scheduler_state: transient.map(|t| t.scheduler_state.clone()).unwrap_or_else(
                    || {
                        if is_online {
                            "ready".to_string()
                        } else {
                            "offline".to_string()
                        }
                    },
                ),
                target_model: transient.and_then(|t| t.target_model.clone()),
                active_requests: connected_runner
                    .map(|r| r.active_requests.load(std::sync::atomic::Ordering::SeqCst))
                    .unwrap_or(0),
                loaded_models: connected_runner
                    .map(|r| r.loaded_models())
                    .unwrap_or_default(),
                available_models,
                protected_classes,
                updated_at: transient
                    .map(|t| t.updated_at.clone())
                    .or_else(|| connected_runner.map(|r| r.last_heartbeat.to_rfc3339()))
                    .or_else(|| db_runner.map(|r| r.last_seen_at.clone()))
                    .unwrap_or_else(|| Utc::now().to_rfc3339()),
            });
        }

        runners.sort_by(|a, b| a.runner_id.cmp(&b.runner_id));

        let queues = if let Some(queue) = batch_queue {
            queue
                .get_stats()
                .await
                .into_iter()
                .map(|(model, stats)| (model, stats.pending))
                .collect()
        } else {
            HashMap::new()
        };

        RouterStateSnapshot {
            runners,
            queues,
            recent_events,
        }
    }
}

impl Default for RouterTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

fn compute_protected_classes(
    runner_id: &str,
    all_available_by_runner: &HashMap<String, Vec<String>>,
    models_config: &ModelsConfig,
) -> Vec<String> {
    let mut protected = Vec::new();
    let Some(models) = all_available_by_runner.get(runner_id) else {
        return protected;
    };

    for class in [
        ModelClass::Fast,
        ModelClass::Big,
        ModelClass::EmbedSmall,
        ModelClass::EmbedLarge,
    ] {
        let has_class = models
            .iter()
            .any(|model| super::classify_model(model, models_config) == Some(class));
        if !has_class {
            continue;
        }

        let available_elsewhere = all_available_by_runner
            .iter()
            .any(|(other_id, other_models)| {
                other_id != runner_id
                    && other_models
                        .iter()
                        .any(|model| super::classify_model(model, models_config) == Some(class))
            });
        if !available_elsewhere {
            protected.push(class.as_str().to_string());
        }
    }

    protected
}
