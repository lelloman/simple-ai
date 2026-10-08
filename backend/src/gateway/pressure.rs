//! Advisory load levels ("pressure") for hosts and client-facing feature groups.
//!
//! Each host is split into lanes: engines that share a resource group (e.g. one
//! GPU) form one lane, and every engine without a resource group is its own lane.
//! A lane is green when nothing waits, orange when requests wait longer than the
//! grace period (beyond the loaded slots, or for a model load or a wake), and red
//! when waiting lasts past `red_after_secs` or the lane recently failed requests.
//! Levels rise immediately and fall one step per `cooldown_secs` of calm.
//!
//! A host's level is its worst lane; a feature group's level is the best lane
//! that can serve it, since the router will use whichever host has room.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::Utc;
use serde::Serialize;
use tokio::sync::watch;

use crate::audit::AuditLogger;
use crate::config::{ModelsConfig, PressureConfig};
use crate::models::request::Response;

use super::model_class::ModelRequest;
use super::registry::RunnerRegistry;
use super::telemetry::RouterTelemetry;

/// Class names in `ModelsConfig`, in display order.
const CLASSES: [&str; 9] = [
    "big",
    "fast",
    "embed_small",
    "embed_large",
    "audio_embeddings",
    "tts",
    "text_classification",
    "information_extraction",
    "semantic_decisions",
];

/// Capability engines serve their class whatever model ids they advertise.
fn class_for_engine_type(engine_type: &str) -> Option<&'static str> {
    match engine_type {
        "tts" => Some("tts"),
        "audio_embeddings" => Some("audio_embeddings"),
        "classification" => Some("text_classification"),
        "extraction" => Some("information_extraction"),
        "decisions" => Some("semantic_decisions"),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PressureLevel {
    #[default]
    Green,
    Orange,
    Red,
}

impl PressureLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Green => "green",
            Self::Orange => "orange",
            Self::Red => "red",
        }
    }

    fn step_down(self) -> Self {
        match self {
            Self::Red => Self::Orange,
            Self::Orange | Self::Green => Self::Green,
        }
    }

    /// Advisory delay before a polite client retries deferrable work.
    fn retry_after_secs(self) -> Option<u64> {
        match self {
            Self::Green => None,
            Self::Orange => Some(15),
            Self::Red => Some(60),
        }
    }
}

/// One lane as observed at sampling time.
#[derive(Debug, Clone, Default)]
pub struct LaneInput {
    pub lane: String,
    pub classes: BTreeSet<String>,
    /// Requests that cannot start now: beyond loaded slots, or waiting for a load or wake.
    pub waiting: usize,
    /// Start of a known wait (model load or wake in progress).
    pub waiting_since: Option<Instant>,
    /// What the lane is waiting on, when known ("loading model", "waking").
    pub waiting_on: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct HostInput {
    pub runner_id: String,
    pub name: String,
    pub online: bool,
    pub lanes: Vec<LaneInput>,
}

#[derive(Debug, Clone)]
struct Failure {
    runner_id: Option<String>,
    class: Option<String>,
    at: Instant,
}

#[derive(Debug, Default)]
struct LaneState {
    level: PressureLevel,
    waiting_since: Option<Instant>,
    calm_since: Option<Instant>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LanePressure {
    pub lane: String,
    pub level: PressureLevel,
    pub waiting: usize,
    pub classes: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct HostPressure {
    pub runner_id: String,
    pub name: String,
    pub online: bool,
    pub level: PressureLevel,
    pub reason: String,
    pub lanes: Vec<LanePressure>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GroupPressure {
    pub level: PressureLevel,
    /// False when no known host can serve the group.
    pub available: bool,
    /// Every host that could serve the group is offline (a request would wake one).
    pub cold: bool,
    pub hosts: Vec<String>,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PressureSnapshot {
    /// Worst host level.
    pub level: PressureLevel,
    pub updated_at: String,
    pub groups: BTreeMap<String, GroupPressure>,
    pub hosts: Vec<HostPressure>,
}

impl Default for PressureSnapshot {
    fn default() -> Self {
        Self {
            level: PressureLevel::Green,
            updated_at: Utc::now().to_rfc3339(),
            groups: BTreeMap::new(),
            hosts: Vec::new(),
        }
    }
}

impl PressureSnapshot {
    fn same_levels(&self, other: &Self) -> bool {
        let levels = |snapshot: &Self| {
            (
                snapshot.level,
                snapshot
                    .groups
                    .iter()
                    .map(|(name, group)| (name.clone(), group.level, group.cold, group.available))
                    .collect::<Vec<_>>(),
                snapshot
                    .hosts
                    .iter()
                    .map(|host| (host.runner_id.clone(), host.level, host.online))
                    .collect::<Vec<_>>(),
            )
        };
        levels(self) == levels(other)
    }
}

#[derive(Default)]
struct TrackerState {
    lanes: HashMap<(String, String), LaneState>,
    failures: Vec<Failure>,
}

pub struct PressureTracker {
    config: PressureConfig,
    models: ModelsConfig,
    state: Mutex<TrackerState>,
    snapshot_tx: watch::Sender<PressureSnapshot>,
}

impl PressureTracker {
    pub fn new(config: PressureConfig, models: ModelsConfig) -> Self {
        let (snapshot_tx, _) = watch::channel(PressureSnapshot::default());
        Self {
            config,
            models,
            state: Mutex::new(TrackerState::default()),
            snapshot_tx,
        }
    }

    /// Latest evaluated levels.
    pub fn snapshot(&self) -> PressureSnapshot {
        self.snapshot_tx.borrow().clone()
    }

    /// Receiver notified whenever a host or group level changes.
    pub fn subscribe(&self) -> watch::Receiver<PressureSnapshot> {
        self.snapshot_tx.subscribe()
    }

    /// Feature group for a model class name.
    pub fn group_for_class(&self, class: &str) -> Option<&str> {
        self.config
            .groups
            .iter()
            .find(|(_, classes)| classes.iter().any(|c| c == class))
            .map(|(group, _)| group.as_str())
    }

    /// Feature group for a requested model (`class:fast`, an alias, or a model id).
    pub fn group_for_model(&self, model: &str) -> Option<&str> {
        let class = ModelRequest::parse(model).effective_class(&self.models)?;
        self.group_for_class(class.as_str())
    }

    /// Level of a feature group, if the group is known.
    pub fn group_level(&self, group: &str) -> Option<PressureLevel> {
        self.snapshot_tx.borrow().groups.get(group).map(|g| g.level)
    }

    /// Count a server-side failure (status 500 and above) against the responsible lanes.
    pub fn record_response(&self, response: &Response) {
        if response.status < 500 {
            return;
        }
        self.record_failure(response.runner_id.clone(), response.model_class.clone());
    }

    pub fn record_failure(&self, runner_id: Option<String>, class: Option<String>) {
        if runner_id.is_none() && class.is_none() {
            return;
        }
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .failures
            .push(Failure {
                runner_id,
                class,
                at: Instant::now(),
            });
    }

    /// Evaluate levels from one observation and publish the result.
    pub fn evaluate(&self, hosts: Vec<HostInput>, now: Instant) -> PressureSnapshot {
        let grace = Duration::from_millis(self.config.grace_ms);
        let red_after = Duration::from_secs(self.config.red_after_secs);
        let failure_window = Duration::from_secs(self.config.failure_window_secs);
        let cooldown = Duration::from_secs(self.config.cooldown_secs);

        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state
            .failures
            .retain(|failure| now.saturating_duration_since(failure.at) < failure_window);
        let failures = state.failures.clone();
        let mut seen = BTreeSet::new();

        let mut host_results = Vec::new();
        for host in &hosts {
            let mut lanes = Vec::new();
            for lane in &host.lanes {
                let key = (host.runner_id.clone(), lane.lane.clone());
                seen.insert(key.clone());
                let lane_state = state.lanes.entry(key).or_default();

                lane_state.waiting_since = if lane.waiting > 0 {
                    let started = lane_state.waiting_since.unwrap_or(now);
                    Some(
                        lane.waiting_since
                            .map_or(started, |known| known.min(started)),
                    )
                } else {
                    None
                };
                let waited = lane_state
                    .waiting_since
                    .map(|since| now.saturating_duration_since(since))
                    .unwrap_or_default();
                let failed = failures
                    .iter()
                    .filter(|failure| {
                        failure
                            .runner_id
                            .as_ref()
                            .is_none_or(|id| *id == host.runner_id)
                            && failure
                                .class
                                .as_ref()
                                .is_none_or(|class| lane.classes.contains(class))
                    })
                    .count();

                let raw = if failed > 0 || (lane.waiting > 0 && waited >= red_after) {
                    PressureLevel::Red
                } else if lane.waiting > 0 && waited >= grace {
                    PressureLevel::Orange
                } else {
                    PressureLevel::Green
                };
                if raw > lane_state.level {
                    lane_state.level = raw;
                    lane_state.calm_since = None;
                } else if raw < lane_state.level {
                    let calm_since = *lane_state.calm_since.get_or_insert(now);
                    if now.saturating_duration_since(calm_since) >= cooldown {
                        lane_state.level = lane_state.level.step_down();
                        lane_state.calm_since = Some(now);
                    }
                } else {
                    lane_state.calm_since = None;
                }

                let reason = if failed > 0 {
                    format!("{failed} failed request(s) recently")
                } else if lane.waiting > 0 && waited >= grace {
                    let what = lane.waiting_on.as_deref().unwrap_or("waiting");
                    format!("{} {what} for {}s", lane.waiting, waited.as_secs())
                } else if lane_state.level > PressureLevel::Green {
                    "recovering".to_string()
                } else {
                    String::new()
                };
                lanes.push(LanePressure {
                    lane: lane.lane.clone(),
                    level: lane_state.level,
                    waiting: lane.waiting,
                    classes: lane.classes.iter().cloned().collect(),
                    reason,
                });
            }
            let worst = lanes.iter().max_by_key(|lane| lane.level);
            host_results.push(HostPressure {
                runner_id: host.runner_id.clone(),
                name: host.name.clone(),
                online: host.online,
                level: worst.map(|lane| lane.level).unwrap_or_default(),
                reason: worst.map(|lane| lane.reason.clone()).unwrap_or_default(),
                lanes,
            });
        }
        state.lanes.retain(|key, _| seen.contains(key));
        drop(state);

        let mut groups = BTreeMap::new();
        for (group, classes) in &self.config.groups {
            let serving: Vec<(&HostPressure, &LanePressure)> = host_results
                .iter()
                .flat_map(|host| host.lanes.iter().map(move |lane| (host, lane)))
                .filter(|(_, lane)| lane.classes.iter().any(|c| classes.contains(c)))
                .collect();
            let best = serving.iter().min_by_key(|(_, lane)| lane.level);
            let mut hosts: Vec<String> = serving
                .iter()
                .map(|(host, _)| host.runner_id.clone())
                .collect();
            hosts.dedup();
            let level = best
                .map(|(_, lane)| lane.level)
                .unwrap_or(PressureLevel::Red);
            groups.insert(
                group.clone(),
                GroupPressure {
                    level,
                    available: best.is_some(),
                    cold: !serving.is_empty() && serving.iter().all(|(host, _)| !host.online),
                    hosts,
                    reason: match best {
                        None => "no runner can serve this group".to_string(),
                        Some((host, lane)) if !lane.reason.is_empty() => {
                            format!("{}: {}", host.name, lane.reason)
                        }
                        Some(_) => String::new(),
                    },
                    retry_after_secs: level.retry_after_secs(),
                },
            );
        }

        let snapshot = PressureSnapshot {
            level: host_results
                .iter()
                .map(|host| host.level)
                .max()
                .unwrap_or_default(),
            updated_at: Utc::now().to_rfc3339(),
            groups,
            hosts: host_results,
        };
        self.snapshot_tx.send_if_modified(|current| {
            let changed = !current.same_levels(&snapshot);
            *current = snapshot.clone();
            changed
        });
        snapshot
    }

    /// Model classes an engine serves, from configured model lists and its engine type.
    fn engine_classes(
        &self,
        engine_type: &str,
        models: &BTreeSet<String>,
        key: impl Fn(&str) -> String,
    ) -> BTreeSet<String> {
        let mut classes: BTreeSet<String> = CLASSES
            .iter()
            .filter(|class| {
                self.models
                    .models_for_class(class)
                    .iter()
                    .any(|id| models.contains(&key(id)))
            })
            .map(|class| class.to_string())
            .collect();
        classes.extend(class_for_engine_type(engine_type).map(String::from));
        classes
    }

    /// Observe the fleet and evaluate levels.
    pub async fn sample(
        &self,
        registry: &RunnerRegistry,
        telemetry: &RouterTelemetry,
        audit_logger: &AuditLogger,
    ) -> PressureSnapshot {
        let now = Instant::now();
        let since_instant = |since: chrono::DateTime<Utc>| {
            let age = (Utc::now() - since).to_std().unwrap_or_default();
            now.checked_sub(age).unwrap_or(now)
        };
        let transient = telemetry.transient_runner_states().await;
        let connected = registry.all().await;
        let mut hosts = Vec::new();

        for runner in &connected {
            let key = |model: &str| runner.model_request_key(model);
            let mut lanes: BTreeMap<String, LaneInput> = BTreeMap::new();
            // (lane, models, loaded models, slots) per engine
            let mut engines = Vec::new();
            for engine in &runner.status.engines {
                let lane_id = engine
                    .resource_group
                    .clone()
                    .unwrap_or_else(|| engine.engine_type.clone());
                let loaded: BTreeSet<String> =
                    engine.loaded_models.iter().map(|m| key(m)).collect();
                let mut models: BTreeSet<String> =
                    engine.available_models.iter().map(|m| key(&m.id)).collect();
                models.extend(loaded.iter().cloned());
                let lane = lanes.entry(lane_id.clone()).or_insert_with(|| LaneInput {
                    lane: lane_id.clone(),
                    ..Default::default()
                });
                lane.classes
                    .extend(self.engine_classes(&engine.engine_type, &models, key));
                engines.push((lane_id, models, loaded, engine.batch_size.max(1) as usize));
            }

            for (model, active) in runner.active_requests_by_model() {
                let engine = engines
                    .iter()
                    .find(|(_, _, loaded, _)| loaded.contains(&model))
                    .or_else(|| {
                        engines
                            .iter()
                            .find(|(_, models, _, _)| models.contains(&model))
                    });
                if let Some((lane_id, _, loaded, slots)) = engine {
                    let capacity = if loaded.contains(&model) { *slots } else { 0 };
                    if let Some(lane) = lanes.get_mut(lane_id) {
                        lane.waiting += active.saturating_sub(capacity);
                    }
                }
            }

            if let Some((state, target, since)) = transient.get(&runner.id) {
                if state == "loading" {
                    let target_lane = target.as_ref().and_then(|model| {
                        let model = key(model);
                        engines
                            .iter()
                            .find(|(_, models, _, _)| models.contains(&model))
                            .map(|(lane_id, ..)| lane_id.clone())
                    });
                    for lane in lanes.values_mut() {
                        if target_lane.as_ref().is_none_or(|id| *id == lane.lane) {
                            lane.waiting = lane.waiting.max(1);
                            lane.waiting_since = Some(since_instant(*since));
                            lane.waiting_on = Some("loading model".into());
                        }
                    }
                }
            }

            hosts.push(HostInput {
                runner_id: runner.id.clone(),
                name: runner.name.clone(),
                online: true,
                lanes: lanes.into_values().collect(),
            });
        }

        // Known but disconnected runners are one lane each: idle unless being woken.
        let known = audit_logger.get_all_runners().unwrap_or_default();
        for record in known {
            if connected.iter().any(|runner| runner.id == record.id) {
                continue;
            }
            let models: BTreeSet<String> = record
                .available_models
                .iter()
                .map(|m| m.to_lowercase())
                .collect();
            let mut lane = LaneInput {
                lane: "host".into(),
                classes: self.engine_classes("", &models, |m| m.to_lowercase()),
                ..Default::default()
            };
            if let Some((state, _, since)) = transient.get(&record.id) {
                if state == "waking" {
                    lane.waiting = 1;
                    lane.waiting_since = Some(since_instant(*since));
                    lane.waiting_on = Some("waking".into());
                }
            }
            hosts.push(HostInput {
                runner_id: record.id.clone(),
                name: record.name.clone(),
                online: false,
                lanes: vec![lane],
            });
        }
        hosts.sort_by(|a, b| a.runner_id.cmp(&b.runner_id));
        self.evaluate(hosts, now)
    }

    /// Sample once per second until the process stops.
    pub fn spawn_sampler(
        self: &Arc<Self>,
        registry: Arc<RunnerRegistry>,
        telemetry: Arc<RouterTelemetry>,
        audit_logger: Arc<AuditLogger>,
    ) -> tokio::task::JoinHandle<()> {
        let tracker = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                tracker.sample(&registry, &telemetry, &audit_logger).await;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker() -> PressureTracker {
        let models = ModelsConfig {
            big: vec!["code:smart".into()],
            fast: vec!["qwen".into()],
            ..Default::default()
        };
        PressureTracker::new(PressureConfig::default(), models)
    }

    fn lane(name: &str, classes: &[&str], waiting: usize) -> LaneInput {
        LaneInput {
            lane: name.into(),
            classes: classes.iter().map(|c| c.to_string()).collect(),
            waiting,
            ..Default::default()
        }
    }

    fn host(id: &str, lanes: Vec<LaneInput>) -> HostInput {
        HostInput {
            runner_id: id.into(),
            name: id.into(),
            online: true,
            lanes,
        }
    }

    fn rtx(waiting: usize) -> HostInput {
        host(
            "rtx",
            vec![
                lane("cuda:0", &["fast", "tts", "text_classification"], waiting),
                lane("extraction", &["information_extraction"], 0),
            ],
        )
    }

    fn halo(id: &str, waiting: usize) -> HostInput {
        host(
            id,
            vec![
                lane("gpu:0", &["big", "semantic_decisions"], waiting),
                lane("extraction", &["information_extraction"], 0),
            ],
        )
    }

    #[test]
    fn waiting_turns_orange_after_grace_and_red_after_threshold() {
        let tracker = tracker();
        let t0 = Instant::now();
        let at = |secs: u64| t0 + Duration::from_secs(secs);
        assert_eq!(
            tracker.evaluate(vec![rtx(2)], t0).level,
            PressureLevel::Green
        );
        let snapshot = tracker.evaluate(vec![rtx(2)], at(1));
        assert_eq!(snapshot.level, PressureLevel::Orange);
        assert_eq!(snapshot.groups["tts"].level, PressureLevel::Orange);
        assert_eq!(snapshot.groups["tts"].retry_after_secs, Some(15));
        assert_eq!(snapshot.groups["extraction"].level, PressureLevel::Green);
        assert_eq!(
            tracker.evaluate(vec![rtx(2)], at(10)).level,
            PressureLevel::Red
        );
    }

    #[test]
    fn levels_fall_one_step_per_cooldown() {
        let tracker = tracker();
        let t0 = Instant::now();
        let at = |secs: u64| t0 + Duration::from_secs(secs);
        tracker.evaluate(vec![rtx(1)], t0);
        assert_eq!(
            tracker.evaluate(vec![rtx(1)], at(10)).level,
            PressureLevel::Red
        );
        assert_eq!(
            tracker.evaluate(vec![rtx(0)], at(11)).level,
            PressureLevel::Red
        );
        assert_eq!(
            tracker.evaluate(vec![rtx(0)], at(40)).level,
            PressureLevel::Red
        );
        assert_eq!(
            tracker.evaluate(vec![rtx(0)], at(41)).level,
            PressureLevel::Orange
        );
        assert_eq!(
            tracker.evaluate(vec![rtx(0)], at(70)).level,
            PressureLevel::Orange
        );
        assert_eq!(
            tracker.evaluate(vec![rtx(0)], at(71)).level,
            PressureLevel::Green
        );
        // A new wait resets the waiting clock and rises again.
        tracker.evaluate(vec![rtx(1)], at(72));
        assert_eq!(
            tracker.evaluate(vec![rtx(1)], at(73)).level,
            PressureLevel::Orange
        );
    }

    #[test]
    fn group_takes_best_serving_lane_and_host_takes_worst() {
        let tracker = tracker();
        let t0 = Instant::now();
        let late = t0 + Duration::from_secs(20);
        let hosts = |halo1_waiting| vec![halo("halo1", halo1_waiting), halo("halo2", 0), rtx(0)];
        tracker.evaluate(hosts(3), t0);
        let snapshot = tracker.evaluate(hosts(3), late);
        let halo1 = snapshot
            .hosts
            .iter()
            .find(|h| h.runner_id == "halo1")
            .unwrap();
        assert_eq!(halo1.level, PressureLevel::Red);
        assert_eq!(snapshot.level, PressureLevel::Red);
        // halo2 can still serve big, and CPU extraction lanes are unaffected.
        assert_eq!(snapshot.groups["big"].level, PressureLevel::Green);
        assert_eq!(snapshot.groups["extraction"].level, PressureLevel::Green);
        assert_eq!(snapshot.groups["big"].hosts, ["halo1", "halo2"]);

        let all_busy = || vec![halo("halo1", 3), halo("halo2", 3), rtx(0)];
        tracker.evaluate(all_busy(), late);
        let busy = tracker.evaluate(all_busy(), late + Duration::from_secs(2));
        assert_eq!(busy.groups["big"].level, PressureLevel::Orange);
        // decisions includes text_classification, which the idle RTX serves.
        assert_eq!(busy.groups["decisions"].level, PressureLevel::Green);
        assert_eq!(busy.groups["fast"].level, PressureLevel::Green);
    }

    #[test]
    fn server_failures_redden_matching_lanes_only() {
        let tracker = tracker();
        let t0 = Instant::now();
        let mut response = Response::new("r".into(), 404);
        response.runner_id = Some("rtx".into());
        tracker.record_response(&response);
        response.status = 499;
        tracker.record_response(&response);
        assert_eq!(
            tracker.evaluate(vec![rtx(0)], t0).level,
            PressureLevel::Green
        );

        response.status = 503;
        response.model_class = Some("fast".into());
        tracker.record_response(&response);
        let snapshot = tracker.evaluate(vec![rtx(0)], t0);
        assert_eq!(snapshot.groups["fast"].level, PressureLevel::Red);
        assert_eq!(snapshot.groups["extraction"].level, PressureLevel::Green);
        assert!(snapshot.groups["fast"].reason.contains("failed"));

        // Failures expire after the window; the level then cools down.
        let later = Instant::now() + Duration::from_secs(121);
        assert_eq!(
            tracker.evaluate(vec![rtx(0)], later).groups["fast"].level,
            PressureLevel::Red
        );
    }

    #[test]
    fn unserved_and_offline_groups_are_reported() {
        let tracker = tracker();
        let mut asleep = halo("halo1", 0);
        asleep.online = false;
        let snapshot = tracker.evaluate(vec![asleep], Instant::now());
        assert!(snapshot.groups["big"].cold);
        assert_eq!(snapshot.groups["big"].level, PressureLevel::Green);
        assert!(!snapshot.groups["tts"].available);
        assert_eq!(snapshot.groups["tts"].level, PressureLevel::Red);
    }

    #[tokio::test]
    async fn sampling_counts_requests_beyond_loaded_slots_per_lane() {
        use simple_ai_common::{EngineStatus, ModelInfo, RunnerHealth, RunnerStatus};
        let engine =
            |engine_type: &str, group: Option<&str>, models: &[&str], loaded: &[&str], slots| {
                EngineStatus {
                    engine_type: engine_type.into(),
                    resource_group: group.map(String::from),
                    is_healthy: true,
                    version: None,
                    loaded_models: loaded.iter().map(|m| m.to_string()).collect(),
                    available_models: models
                        .iter()
                        .map(|id| ModelInfo {
                            id: id.to_string(),
                            name: id.to_string(),
                            size_bytes: None,
                            parameter_count: None,
                            context_length: None,
                            quantization: None,
                            modified_at: None,
                            reasoning: None,
                        })
                        .collect(),
                    error: None,
                    batch_size: slots,
                    prompt_cache: None,
                }
            };
        let registry = RunnerRegistry::new();
        let (tx, _rx) = tokio::sync::mpsc::channel(4);
        registry
            .register(
                "rtx".into(),
                "RTX".into(),
                Some("gpu-server".into()),
                RunnerStatus {
                    health: RunnerHealth::Healthy,
                    capabilities: vec![],
                    engines: vec![
                        engine("llama_cpp", Some("cuda:0"), &["qwen"], &["qwen"], 2),
                        engine("tts", Some("cuda:0"), &["xtts"], &[], 1),
                        engine("extraction", None, &["gliner"], &["gliner"], 4),
                    ],
                    metrics: None,
                    model_aliases: Default::default(),
                },
                None,
                tx,
                None,
            )
            .await;
        let runner = registry.get("rtx").await.unwrap();
        // Two chat requests fill the loaded slots; TTS waits for the shared GPU.
        let _chat = [
            registry.reserve(&runner, "qwen"),
            registry.reserve(&runner, "QWEN"),
        ];
        let tts = registry.reserve(&runner, "xtts");
        let _extraction = registry.reserve(&runner, "gliner");

        let tracker = tracker();
        let telemetry = RouterTelemetry::new();
        let audit = AuditLogger::new(":memory:").unwrap();
        let snapshot = tracker.sample(&registry, &telemetry, &audit).await;
        let lanes = &snapshot.hosts[0].lanes;
        let lane = |name: &str| lanes.iter().find(|lane| lane.lane == name).unwrap();
        assert_eq!(lane("cuda:0").waiting, 1);
        assert_eq!(lane("cuda:0").classes, ["fast", "tts"]);
        assert_eq!(lane("extraction").waiting, 0);
        assert_eq!(lane("extraction").classes, ["information_extraction"]);

        drop(tts);
        assert_eq!(runner.active_requests_by_model().get("xtts"), None);
        telemetry
            .set_runner_state("rtx", "loading", Some("xtts".into()))
            .await;
        let snapshot = tracker.sample(&registry, &telemetry, &audit).await;
        let loading = snapshot.hosts[0]
            .lanes
            .iter()
            .find(|l| l.lane == "cuda:0")
            .unwrap();
        assert_eq!(loading.waiting, 1);
    }

    #[test]
    fn requests_map_to_groups() {
        let tracker = tracker();
        assert_eq!(tracker.group_for_model("class:fast"), Some("fast"));
        assert_eq!(tracker.group_for_model("code:smart"), Some("big"));
        assert_eq!(tracker.group_for_model("QWEN"), Some("fast"));
        assert_eq!(tracker.group_for_model("unknown-model"), None);
        assert_eq!(
            tracker.group_for_class("text_classification"),
            Some("decisions")
        );
        assert_eq!(
            tracker.group_for_class("audio_embeddings"),
            Some("embeddings")
        );
    }

    #[test]
    fn subscribers_are_notified_only_on_level_changes() {
        let tracker = tracker();
        let mut rx = tracker.subscribe();
        let t0 = Instant::now();
        tracker.evaluate(vec![rtx(0)], t0);
        assert!(rx.has_changed().unwrap());
        rx.borrow_and_update();
        tracker.evaluate(vec![rtx(0)], t0 + Duration::from_secs(1));
        assert!(!rx.has_changed().unwrap());
        tracker.evaluate(vec![rtx(1)], t0 + Duration::from_secs(2));
        tracker.evaluate(vec![rtx(1)], t0 + Duration::from_secs(4));
        assert!(rx.has_changed().unwrap());
    }
}
