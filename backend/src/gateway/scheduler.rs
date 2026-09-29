use std::sync::Arc;
use std::time::Duration;

use tokio::time::{sleep, timeout};

use simple_ai_common::{ChatCompletionRequest, ChatCompletionResponse, GatewayMessage};

use crate::config::RoutingConfig;
use crate::wol::{WakeError, WakeService};

use super::{
    AffinityContext, AffinityDecision, BatchQueue, InferenceRouter, ModelRequest, RoutePlan,
    RoutedStream, RouterError, RouterTelemetry, RunnerEvent, RunnerRegistry,
};
use crate::routes::embeddings::{EmbeddingRequest, EmbeddingResponse};
use simple_ai_common::{AudioEmbeddingResponse, SpeechRequest};
use simple_ai_common::{
    ClassificationRequest, ClassificationResponse, ExtractionRequest, ExtractionResponse,
};

#[derive(Debug)]
pub struct ScheduledResponse<T> {
    pub response: T,
    pub runner_id: String,
    pub resolved_model: String,
    pub wol_sent: bool,
}

#[derive(Debug)]
struct PreparedRequest {
    plan: RoutePlan,
    wol_sent: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("{0}")]
    Router(#[from] RouterError),
    #[error("{0}")]
    Wake(#[from] WakeError),
}

#[derive(Clone)]
pub struct RequestScheduler {
    decision_races: Arc<
        tokio::sync::Mutex<
            std::collections::HashMap<String, tokio::sync::watch::Receiver<Option<String>>>,
        >,
    >,
    inference_router: Arc<InferenceRouter>,
    runner_registry: Arc<RunnerRegistry>,
    wake_service: Arc<WakeService>,
    router_telemetry: Arc<RouterTelemetry>,
    batch_queue: Option<Arc<BatchQueue>>,
    routing_config: RoutingConfig,
}

impl RequestScheduler {
    pub fn new(
        inference_router: Arc<InferenceRouter>,
        runner_registry: Arc<RunnerRegistry>,
        wake_service: Arc<WakeService>,
        router_telemetry: Arc<RouterTelemetry>,
        batch_queue: Option<Arc<BatchQueue>>,
        routing_config: RoutingConfig,
    ) -> Self {
        Self {
            decision_races: Default::default(),
            inference_router,
            runner_registry,
            wake_service,
            router_telemetry,
            batch_queue,
            routing_config,
        }
    }

    pub async fn chat_completion(
        &self,
        request_id: &str,
        model: &str,
        model_request: &ModelRequest,
        affinity: Option<AffinityContext>,
        request: &ChatCompletionRequest,
        use_batching: bool,
    ) -> Result<ScheduledResponse<ChatCompletionResponse>, SchedulerError> {
        let prepared = self
            .prepare_for_request(request_id, model, model_request, affinity.clone())
            .await?;
        let _activity = self
            .wake_service
            .keep_runner_awake(prepared.plan.runner.id.clone());
        let routed = if use_batching {
            let batch_queue = self
                .batch_queue
                .as_ref()
                .ok_or(RouterError::ConnectionFailed(
                    "Batch queue unavailable".to_string(),
                ))?;
            let queue_plan = prepared.plan.clone();
            let rx = batch_queue
                .enqueue_with_context(
                    queue_plan.resolved_model,
                    request_id.to_string(),
                    model.to_string(),
                    queue_plan.class_hint,
                    affinity.clone(),
                    request.clone(),
                )
                .await;
            let batched = rx.await.map_err(|_| {
                RouterError::ConnectionFailed("Batch queue response channel closed".to_string())
            })??;
            super::RoutedResponse {
                response: batched.response,
                runner_id: batched.runner_id,
                resolved_model: batched.resolved_model,
            }
        } else {
            let plan = prepared.plan;
            let reserved = match self.inference_router.reserve_plan(plan).await {
                Ok(reserved) => reserved,
                Err(RouterError::StalePlan) => {
                    let retry = self
                        .inference_router
                        .plan_chat_request(model, affinity.clone())
                        .await?;
                    if !retry.is_loaded {
                        self.prepare_runner_model(request_id, &retry).await?;
                    }
                    self.inference_router.reserve_plan(retry).await?
                }
                Err(error) => return Err(error.into()),
            };
            let _selected_activity = self
                .wake_service
                .keep_runner_awake(reserved.plan.runner.id.clone());
            self.emit_affinity_decision(request_id, &reserved.plan)
                .await;
            self.inference_router
                .execute_chat_plan(reserved, request)
                .await?
        };
        Ok(ScheduledResponse {
            response: routed.response,
            runner_id: routed.runner_id,
            resolved_model: routed.resolved_model,
            wol_sent: prepared.wol_sent,
        })
    }

    pub async fn chat_completion_stream(
        &self,
        request_id: &str,
        model: &str,
        model_request: &ModelRequest,
        affinity: Option<AffinityContext>,
        request: &ChatCompletionRequest,
    ) -> Result<ScheduledResponse<RoutedStream>, SchedulerError> {
        let prepared = self
            .prepare_for_request(request_id, model, model_request, affinity.clone())
            .await?;
        let _activity = self
            .wake_service
            .keep_runner_awake(prepared.plan.runner.id.clone());
        let plan = prepared.plan;
        let reserved = match self.inference_router.reserve_plan(plan).await {
            Ok(reserved) => reserved,
            Err(RouterError::StalePlan) => {
                let retry = self
                    .inference_router
                    .plan_chat_request(model, affinity.clone())
                    .await?;
                if !retry.is_loaded {
                    self.prepare_runner_model(request_id, &retry).await?;
                }
                self.inference_router.reserve_plan(retry).await?
            }
            Err(error) => return Err(error.into()),
        };
        self.emit_affinity_decision(request_id, &reserved.plan)
            .await;
        let routed = self
            .inference_router
            .execute_chat_stream_plan(reserved, request)
            .await?;
        Ok(ScheduledResponse {
            runner_id: routed.runner_id.clone(),
            resolved_model: routed.resolved_model.clone(),
            response: routed,
            wol_sent: prepared.wol_sent,
        })
    }

    pub async fn embeddings(
        &self,
        request_id: &str,
        model: &str,
        model_request: &ModelRequest,
        request: &EmbeddingRequest,
    ) -> Result<ScheduledResponse<EmbeddingResponse>, SchedulerError> {
        let prepared = self
            .prepare_for_request(request_id, model, model_request, None)
            .await?;
        let _activity = self
            .wake_service
            .keep_runner_awake(prepared.plan.runner.id.clone());
        let routed = self
            .inference_router
            .embed::<EmbeddingRequest, EmbeddingResponse>(model, request)
            .await?;
        Ok(ScheduledResponse {
            response: routed.response,
            runner_id: routed.runner_id,
            resolved_model: routed.resolved_model,
            wol_sent: prepared.wol_sent,
        })
    }

    pub async fn classification(
        &self,
        request_id: &str,
        model: &str,
        model_request: &ModelRequest,
        request: &ClassificationRequest,
    ) -> Result<ScheduledResponse<ClassificationResponse>, SchedulerError> {
        let prepared = self
            .prepare_for_request(request_id, model, model_request, None)
            .await?;
        let _activity = self
            .wake_service
            .keep_runner_awake(prepared.plan.runner.id.clone());
        let routed = self
            .inference_router
            .classification::<ClassificationRequest, ClassificationResponse>(model, request)
            .await?;
        Ok(ScheduledResponse {
            response: routed.response,
            runner_id: routed.runner_id,
            resolved_model: routed.resolved_model,
            wol_sent: prepared.wol_sent,
        })
    }

    pub async fn extraction(
        &self,
        request_id: &str,
        model: &str,
        model_request: &ModelRequest,
        request: &ExtractionRequest,
    ) -> Result<ScheduledResponse<ExtractionResponse>, SchedulerError> {
        let prepared = self
            .prepare_for_request(request_id, model, model_request, None)
            .await?;
        let _activity = self
            .wake_service
            .keep_runner_awake(prepared.plan.runner.id.clone());
        let routed = self
            .inference_router
            .extraction::<ExtractionRequest, ExtractionResponse>(model, request)
            .await?;
        Ok(ScheduledResponse {
            response: routed.response,
            runner_id: routed.runner_id,
            resolved_model: routed.resolved_model,
            wol_sent: prepared.wol_sent,
        })
    }

    pub async fn decision(
        &self,
        request_id: &str,
        model: &str,
        model_request: &ModelRequest,
        request: &simple_ai_common::DecisionRequest,
    ) -> Result<ScheduledResponse<simple_ai_common::DecisionResponse>, SchedulerError> {
        let prepared = if self.routing_config.decision_ready_race {
            self.prepare_decision_race(request_id, model, model_request)
                .await?
        } else {
            self.prepare_for_request(request_id, model, model_request, None)
                .await?
        };
        let _activity = self
            .wake_service
            .keep_runner_awake(prepared.plan.runner.id.clone());
        let routed = self
            .inference_router
            .decision(&prepared.plan, request)
            .await?;
        Ok(ScheduledResponse {
            response: routed.response,
            runner_id: routed.runner_id,
            resolved_model: routed.resolved_model,
            wol_sent: prepared.wol_sent,
        })
    }

    pub async fn audio_embedding(
        &self,
        request_id: &str,
        model: &str,
        model_request: &ModelRequest,
        file_name: String,
        file_bytes: Vec<u8>,
        options_json: String,
    ) -> Result<ScheduledResponse<AudioEmbeddingResponse>, SchedulerError> {
        let prepared = self
            .prepare_for_request(request_id, model, model_request, None)
            .await?;
        let _activity = self
            .wake_service
            .keep_runner_awake(prepared.plan.runner.id.clone());
        let routed = self
            .inference_router
            .audio_embedding_multipart(model, file_name, file_bytes, options_json)
            .await?;
        Ok(ScheduledResponse {
            response: routed.response,
            runner_id: routed.runner_id,
            resolved_model: routed.resolved_model,
            wol_sent: prepared.wol_sent,
        })
    }

    pub async fn speech(
        &self,
        request_id: &str,
        model: &str,
        model_request: &ModelRequest,
        request: &SpeechRequest,
    ) -> Result<ScheduledResponse<reqwest::Response>, SchedulerError> {
        let prepared = self
            .prepare_for_request(request_id, model, model_request, None)
            .await?;
        let _activity = self
            .wake_service
            .keep_runner_awake(prepared.plan.runner.id.clone());
        let routed = self.inference_router.speech_raw(model, request).await?;
        Ok(ScheduledResponse {
            response: routed.response,
            runner_id: routed.runner_id,
            resolved_model: routed.resolved_model,
            wol_sent: prepared.wol_sent,
        })
    }

    async fn prepare_decision_race(
        &self,
        request_id: &str,
        model: &str,
        request: &ModelRequest,
    ) -> Result<PreparedRequest, SchedulerError> {
        let ready = self.inference_router.ready_decision_plan(model).await;
        let preferred = self
            .routing_config
            .class_preferences
            .get("semantic_decisions")
            .and_then(|types| types.first());
        if ready
            .as_ref()
            .is_some_and(|p| preferred.map(String::as_str) == p.runner.machine_type.as_deref())
        {
            return Ok(PreparedRequest {
                plan: ready.unwrap(),
                wol_sent: false,
            });
        }
        let targets = self.wake_service.decision_race_targets(request).await?;
        if targets.is_empty() {
            if let Some(plan) = ready {
                return Ok(PreparedRequest {
                    plan,
                    wol_sent: false,
                });
            }
            return Err(RouterError::NoRunners.into());
        }
        // Class and concrete selectors share a preparation flight for the same
        // model. It outlives the winning request so RTX still finishes loading.
        let key = match request {
            ModelRequest::Specific(id) => id.clone(),
            _ => simple_ai_common::DECISION_MODEL.to_string(),
        };
        let mut flights = self.decision_races.lock().await;
        let mut wol_sent = false;
        let completion = if let Some(receiver) = flights.get(&key) {
            receiver.clone()
        } else {
            for target in &targets {
                wol_sent |= self.runner_registry.get(&target.id).await.is_none();
            }
            let (tx, rx) = tokio::sync::watch::channel(None);
            flights.insert(key.clone(), rx.clone());
            let scheduler = self.clone();
            let selector = model.to_owned();
            let request_id = request_id.to_owned();
            tokio::spawn(async move {
                let outcomes = futures_util::future::join_all(targets.into_iter().map(|target| {
                    let scheduler = scheduler.clone();
                    let selector = selector.clone();
                    let request_id = request_id.clone();
                    async move {
                        let result = scheduler
                            .prepare_decision_target(&request_id, &selector, &target)
                            .await;
                        if let Err(ref e) = result {
                            tracing::warn!(runner=%target.id, error=%e, "JEV preparation failed");
                        }
                        result.err().map(|e| format!("{}: {}", target.id, e))
                    }
                }))
                .await;
                let errors = outcomes
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join("; ");
                let _ = tx.send(Some(errors));
                scheduler.decision_races.lock().await.remove(&key);
            });
            rx
        };
        drop(flights);
        // Only model preparation is raced; inference is dispatched once.
        let deadline = self.wake_service.wake_timeout()
            + Duration::from_secs(self.routing_config.model_prepare_timeout_secs);
        timeout(deadline, async {
            loop {
                if let Some(plan) = self.inference_router.ready_decision_plan(model).await {
                    self.router_telemetry
                        .emit(
                            "decision_ready_selected",
                            format!("Selected ready JEV runner {}", plan.runner.id),
                            Some(request_id.into()),
                            Some(plan.runner.id.clone()),
                            Some(plan.resolved_model.clone()),
                        )
                        .await;
                    return Ok(PreparedRequest { plan, wol_sent });
                }
                if let Some(errors) = completion.borrow().clone() {
                    return Err(SchedulerError::Router(RouterError::ConnectionFailed(
                        format!("No JEV runner became ready: {errors}"),
                    )));
                }
                sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .map_err(|_| RouterError::ConnectionFailed("Timed out preparing JEV runners".into()))?
    }

    async fn prepare_decision_target(
        &self,
        request_id: &str,
        selector: &str,
        target: &crate::audit::RunnerRecord,
    ) -> Result<(), SchedulerError> {
        let _activity = self.wake_service.keep_runner_awake(target.id.clone());
        if self.runner_registry.get(&target.id).await.is_none() {
            self.router_telemetry
                .set_runner_state(&target.id, "waking", None)
                .await;
            if let Err(error) = self.wake_service.wake_runner(target).await {
                self.router_telemetry.clear_runner_state(&target.id).await;
                return Err(error.into());
            }
        }
        let plan = timeout(self.wake_service.wake_timeout(), async {
            loop {
                if let Ok(plan) = self
                    .inference_router
                    .decision_plan_on(selector, &target.id, false)
                    .await
                {
                    return plan;
                }
                sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .map_err(|_| {
            RouterError::ConnectionFailed(format!("JEV runner {} did not connect", target.id))
        });
        let result = match plan {
            Ok(plan) => self.prepare_runner_model(request_id, &plan).await,
            Err(e) => Err(e.into()),
        };
        self.router_telemetry.clear_runner_state(&target.id).await;
        result
    }

    async fn emit_affinity_decision(&self, request_id: &str, plan: &RoutePlan) {
        if matches!(
            plan.affinity_decision,
            AffinityDecision::Unkeyed | AffinityDecision::Disabled
        ) {
            return;
        }
        self.router_telemetry
            .emit(
                plan.affinity_decision.as_str(),
                format!(
                    "Cache affinity decision: {}",
                    plan.affinity_decision.as_str()
                ),
                Some(request_id.to_string()),
                Some(plan.runner.id.clone()),
                Some(plan.resolved_model.clone()),
            )
            .await;
    }

    async fn prepare_for_request(
        &self,
        request_id: &str,
        model: &str,
        model_request: &ModelRequest,
        affinity: Option<AffinityContext>,
    ) -> Result<PreparedRequest, SchedulerError> {
        self.router_telemetry
            .emit(
                "request_received",
                format!("Scheduler received request for {}", model),
                Some(request_id.to_string()),
                None,
                Some(model.to_string()),
            )
            .await;

        match self.plan_chat_request(model, affinity.clone()).await {
            Ok(plan) => {
                self.router_telemetry
                    .emit(
                        "scheduler_planned",
                        format!(
                            "Selected runner {} for {} ({})",
                            plan.runner.id,
                            plan.resolved_model,
                            if plan.is_loaded { "ready" } else { "load" }
                        ),
                        Some(request_id.to_string()),
                        Some(plan.runner.id.clone()),
                        Some(plan.resolved_model.clone()),
                    )
                    .await;
                if !plan.is_loaded {
                    self.prepare_runner_model(request_id, &plan).await?;
                }
                Ok(PreparedRequest {
                    plan,
                    wol_sent: false,
                })
            }
            Err(RouterError::NoRunners) | Err(RouterError::NoModelsOfClass(_)) => {
                if self.wake_service.is_enabled()
                    && !self
                        .wake_service
                        .find_wakeable_runners(Some(model_request))
                        .await?
                        .is_empty()
                {
                    let wake_targets = self
                        .wake_service
                        .planned_wake_targets(model_request)
                        .await?;
                    for target in &wake_targets {
                        self.router_telemetry
                            .set_runner_state(&target.id, "waking", None)
                            .await;
                        self.router_telemetry
                            .emit(
                                "runner_marked_waking",
                                format!("Marked runner {} as waking", target.id),
                                Some(request_id.to_string()),
                                Some(target.id.clone()),
                                None,
                            )
                            .await;
                    }
                    self.router_telemetry
                        .emit(
                            "wake_started",
                            format!("Waking capacity for {}", model),
                            Some(request_id.to_string()),
                            None,
                            Some(model.to_string()),
                        )
                        .await;
                    if let Err(err) = self
                        .wake_service
                        .speculative_wake_and_wait(model_request)
                        .await
                    {
                        for target in &wake_targets {
                            self.router_telemetry.clear_runner_state(&target.id).await;
                        }
                        self.router_telemetry
                            .emit(
                                "wake_failed",
                                format!("Wake failed for {}: {}", model, err),
                                Some(request_id.to_string()),
                                None,
                                Some(model.to_string()),
                            )
                            .await;
                        return Err(err.into());
                    }
                    for target in &wake_targets {
                        self.router_telemetry.clear_runner_state(&target.id).await;
                    }
                    self.router_telemetry
                        .emit(
                            "wake_succeeded",
                            format!("Capacity became available for {}", model),
                            Some(request_id.to_string()),
                            None,
                            Some(model.to_string()),
                        )
                        .await;
                    let plan = self.plan_chat_request(model, affinity).await?;
                    self.router_telemetry
                        .emit(
                            "scheduler_planned",
                            format!(
                                "Selected runner {} for {} after wake",
                                plan.runner.id, plan.resolved_model
                            ),
                            Some(request_id.to_string()),
                            Some(plan.runner.id.clone()),
                            Some(plan.resolved_model.clone()),
                        )
                        .await;
                    if !plan.is_loaded {
                        self.prepare_runner_model(request_id, &plan).await?;
                    }
                    Ok(PreparedRequest {
                        plan,
                        wol_sent: true,
                    })
                } else {
                    Err(RouterError::NoRunners.into())
                }
            }
            Err(e) => Err(e.into()),
        }
    }

    async fn plan_chat_request(
        &self,
        model: &str,
        affinity: Option<AffinityContext>,
    ) -> Result<RoutePlan, RouterError> {
        match affinity {
            Some(context) => {
                self.inference_router
                    .plan_chat_request(model, Some(context))
                    .await
            }
            None => self.inference_router.plan_request(model).await,
        }
    }

    async fn prepare_runner_model(
        &self,
        request_id: &str,
        plan: &RoutePlan,
    ) -> Result<(), SchedulerError> {
        if plan.is_loaded {
            return Ok(());
        }

        let _activity = self.wake_service.keep_runner_awake(plan.runner.id.clone());
        self.router_telemetry
            .set_runner_state(
                &plan.runner.id,
                "loading",
                Some(plan.resolved_model.clone()),
            )
            .await;
        self.router_telemetry
            .emit(
                "model_load_started",
                format!(
                    "Loading {} on runner {}",
                    plan.resolved_model, plan.runner.id
                ),
                Some(request_id.to_string()),
                Some(plan.runner.id.clone()),
                Some(plan.resolved_model.clone()),
            )
            .await;

        let request_id = format!(
            "scheduler-load-{}-{}",
            plan.runner.id,
            uuid::Uuid::new_v4().simple()
        );
        // Subscribe before dispatch so immediate command failures are not lost.
        let events = self.runner_registry.subscribe_events();
        plan.runner
            .tx
            .send(GatewayMessage::LoadModel {
                model_id: plan.resolved_model.clone(),
                request_id: request_id.clone(),
            })
            .await
            .map_err(|e| RouterError::ConnectionFailed(e.to_string()))?;

        self.wait_for_model_ready(&plan.runner.id, &plan.resolved_model, &request_id, events)
            .await
    }

    async fn wait_for_model_ready(
        &self,
        runner_id: &str,
        model_id: &str,
        request_id: &str,
        mut events: tokio::sync::broadcast::Receiver<RunnerEvent>,
    ) -> Result<(), SchedulerError> {
        let timeout_duration = Duration::from_secs(self.routing_config.model_prepare_timeout_secs);

        timeout(timeout_duration, async {
            loop {
                if let Some(runner) = self.runner_registry.get(runner_id).await {
                    if runner.has_model_or_alias(model_id) {
                        self.router_telemetry.clear_runner_state(runner_id).await;
                        self.router_telemetry
                            .emit(
                                "model_load_ready",
                                format!("Model {} is ready on runner {}", model_id, runner_id),
                                None,
                                Some(runner_id.to_string()),
                                Some(model_id.to_string()),
                            )
                            .await;
                        return Ok(());
                    }
                }

                match events.recv().await {
                    Ok(RunnerEvent::StatusChanged {
                        runner_id: changed_runner,
                        ..
                    })
                    | Ok(RunnerEvent::Connected {
                        runner_id: changed_runner,
                        ..
                    }) => {
                        if changed_runner == runner_id {
                            continue;
                        }
                    }
                    Ok(RunnerEvent::CommandCompleted {
                        runner_id: changed_runner,
                        request_id: completed_request_id,
                        success,
                        error,
                    }) => {
                        if changed_runner == runner_id && completed_request_id == request_id {
                            if success {
                                continue;
                            }
                            self.router_telemetry.clear_runner_state(runner_id).await;
                            return Err(RouterError::ConnectionFailed(format!(
                                "Runner {} failed to prepare model {}: {}",
                                runner_id,
                                model_id,
                                error.unwrap_or_else(|| "unknown error".to_string())
                            )));
                        }
                    }
                    Ok(RunnerEvent::Disconnected {
                        runner_id: changed_runner,
                    }) => {
                        if changed_runner == runner_id {
                            self.router_telemetry.clear_runner_state(runner_id).await;
                            return Err(RouterError::ConnectionFailed(format!(
                                "Runner {} disconnected while preparing model {}",
                                runner_id, model_id
                            )));
                        }
                    }
                    Err(_) => sleep(Duration::from_millis(50)).await,
                }
            }
        })
        .await
        .map_err(|_| {
            let telemetry = self.router_telemetry.clone();
            let runner_id = runner_id.to_string();
            let cleanup_runner_id = runner_id.clone();
            tokio::spawn(async move {
                telemetry.clear_runner_state(&cleanup_runner_id).await;
            });
            RouterError::ConnectionFailed(format!(
                "Timed out waiting for runner {} to prepare model {}",
                runner_id, model_id
            ))
        })??;

        Ok(())
    }
}

#[cfg(test)]
mod decision_race_tests {
    use super::*;
    use crate::audit::AuditLogger;
    use crate::config::{GatewayConfig, ModelsConfig, WolConfig};
    use simple_ai_common::{EngineStatus, ModelInfo, RunnerHealth, RunnerStatus, DECISION_MODEL};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn status(loaded: bool) -> RunnerStatus {
        RunnerStatus {
            health: RunnerHealth::Healthy,
            capabilities: vec![],
            metrics: None,
            model_aliases: Default::default(),
            engines: vec![EngineStatus {
                engine_type: "decisions".into(),
                resource_group: Some("gpu:0".into()),
                is_healthy: true,
                version: None,
                error: None,
                batch_size: 1,
                prompt_cache: None,
                loaded_models: if loaded {
                    vec![DECISION_MODEL.into()]
                } else {
                    vec![]
                },
                available_models: vec![ModelInfo {
                    id: DECISION_MODEL.into(),
                    name: "JEV".into(),
                    size_bytes: None,
                    parameter_count: None,
                    context_length: Some(18432),
                    quantization: None,
                    modified_at: None,
                    reasoning: None,
                }],
            }],
        }
    }

    fn setup() -> (RequestScheduler, Arc<AuditLogger>) {
        let registry = Arc::new(RunnerRegistry::new());
        let audit = Arc::new(AuditLogger::new(":memory:").unwrap());
        let models = ModelsConfig {
            semantic_decisions: vec![DECISION_MODEL.into()],
            ..Default::default()
        };
        let routing = RoutingConfig {
            decision_ready_race: true,
            model_prepare_timeout_secs: 2,
            class_preferences: std::collections::HashMap::from([(
                "semantic_decisions".into(),
                vec!["gpu-server".into(), "halo".into()],
            )]),
            speculative_wake_targets: std::collections::HashMap::from([(
                "semantic_decisions".into(),
                vec!["gpu-server".into(), "halo".into()],
            )]),
            ..Default::default()
        };
        let wake = Arc::new(WakeService::new(
            registry.clone(),
            audit.clone(),
            GatewayConfig {
                wake_timeout_secs: 1,
                ..Default::default()
            },
            WolConfig::default(),
            models.clone(),
            routing.clone(),
        ));
        let router = Arc::new(InferenceRouter::new(
            registry.clone(),
            models,
            routing.clone(),
            audit.clone(),
        ));
        (
            RequestScheduler::new(
                router,
                registry,
                wake,
                Arc::new(RouterTelemetry::new()),
                None,
                routing,
            ),
            audit,
        )
    }

    async fn runner(
        s: &RequestScheduler,
        audit: &AuditLogger,
        id: &str,
        machine: &str,
        loaded: bool,
        delay: Duration,
        fail: bool,
    ) -> Arc<AtomicUsize> {
        audit
            .upsert_runner(id, id, None, Some(machine), Some(&[DECISION_MODEL.into()]))
            .unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        s.runner_registry
            .register(
                id.into(),
                id.into(),
                Some(machine.into()),
                status(loaded),
                None,
                tx,
                None,
            )
            .await;
        let registry = s.runner_registry.clone();
        let id = id.to_string();
        let count = Arc::new(AtomicUsize::new(0));
        let seen = count.clone();
        tokio::spawn(async move {
            while let Some(command) = rx.recv().await {
                if let GatewayMessage::LoadModel { request_id, .. } = command {
                    seen.fetch_add(1, Ordering::SeqCst);
                    sleep(delay).await;
                    if fail {
                        registry.emit_command_response(
                            &id,
                            &simple_ai_common::CommandResponse {
                                request_id,
                                success: false,
                                error: Some("test load failed".into()),
                                status: None,
                            },
                        );
                    } else {
                        registry.update_status(&id, status(true)).await;
                    }
                }
            }
        });
        count
    }

    #[tokio::test]
    async fn decision_race_first_ready_then_rtx_and_only_one_halo() {
        let (s, audit) = setup();
        let rtx = runner(
            &s,
            &audit,
            "rtx",
            "gpu-server",
            false,
            Duration::from_millis(600),
            false,
        )
        .await;
        let h1 = runner(
            &s,
            &audit,
            "halo1",
            "halo",
            false,
            Duration::from_millis(20),
            false,
        )
        .await;
        let h2 = runner(&s, &audit, "halo2", "halo", false, Duration::ZERO, false).await;
        let selector = "class:semantic_decisions";
        let request = ModelRequest::parse(selector);
        let (a, b) = tokio::join!(
            s.prepare_decision_race("a", selector, &request),
            s.prepare_decision_race("b", selector, &request)
        );
        assert_eq!(a.unwrap().plan.runner.id, "halo1");
        assert_eq!(b.unwrap().plan.runner.id, "halo1");
        assert_eq!(h1.load(Ordering::SeqCst), 1);
        assert_eq!(h2.load(Ordering::SeqCst), 0);
        timeout(Duration::from_secs(2), async {
            while !s
                .runner_registry
                .get("rtx")
                .await
                .unwrap()
                .has_model(DECISION_MODEL)
            {
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        let next = s
            .prepare_decision_race("c", DECISION_MODEL, &ModelRequest::parse(DECISION_MODEL))
            .await
            .unwrap();
        assert_eq!(next.plan.runner.id, "rtx");
        assert_eq!(rtx.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn decision_race_reuses_ready_halo2_and_survives_rtx_failure() {
        let (s, audit) = setup();
        let h1 = runner(&s, &audit, "halo1", "halo", false, Duration::ZERO, false).await;
        runner(&s, &audit, "halo2", "halo", true, Duration::ZERO, false).await;
        runner(&s, &audit, "rtx", "gpu-server", false, Duration::ZERO, true).await;
        let request = ModelRequest::parse("class:semantic_decisions");
        let selected = s
            .prepare_decision_race("test", "class:semantic_decisions", &request)
            .await
            .unwrap();
        assert_eq!(selected.plan.runner.id, "halo2");
        sleep(Duration::from_millis(100)).await;
        assert_eq!(h1.load(Ordering::SeqCst), 0);
        assert!(s
            .inference_router
            .ready_decision_plan("class:semantic_decisions")
            .await
            .is_some());
    }

    #[tokio::test]
    async fn decision_race_reports_all_load_failures() {
        let (s, audit) = setup();
        runner(&s, &audit, "rtx", "gpu-server", false, Duration::ZERO, true).await;
        runner(&s, &audit, "halo1", "halo", false, Duration::ZERO, true).await;
        let request = ModelRequest::parse("class:semantic_decisions");
        let error = s
            .prepare_decision_race("test", "class:semantic_decisions", &request)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("test load failed"));
    }
    #[tokio::test]
    async fn decision_race_offline_targets_are_one_per_type() {
        let (s, audit) = setup();
        for (id, machine) in [("rtx", "gpu-server"), ("halo1", "halo"), ("halo2", "halo")] {
            audit
                .upsert_runner(
                    id,
                    id,
                    Some("00:11:22:33:44:55"),
                    Some(machine),
                    Some(&[DECISION_MODEL.into()]),
                )
                .unwrap();
        }
        let wake = WakeService::new(
            s.runner_registry.clone(),
            audit.clone(),
            GatewayConfig {
                auto_wake_enabled: true,
                ..Default::default()
            },
            WolConfig::default(),
            ModelsConfig {
                semantic_decisions: vec![DECISION_MODEL.into()],
                ..Default::default()
            },
            s.routing_config.clone(),
        );
        let request = ModelRequest::parse("class:semantic_decisions");
        let targets = wake.decision_race_targets(&request).await.unwrap();
        assert_eq!(
            targets.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            vec!["rtx", "halo1"]
        );
        runner(&s, &audit, "halo2", "halo", true, Duration::ZERO, false).await;
        let targets = wake.decision_race_targets(&request).await.unwrap();
        assert_eq!(
            targets.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            vec!["rtx", "halo2"]
        );
    }

    #[tokio::test]
    async fn decision_race_never_dispatches_to_unhealthy_loaded_model() {
        let (s, audit) = setup();
        runner(&s, &audit, "rtx", "gpu-server", true, Duration::ZERO, false).await;
        runner(&s, &audit, "halo1", "halo", true, Duration::ZERO, false).await;
        let mut unhealthy = status(true);
        unhealthy.engines[0].is_healthy = false;
        s.runner_registry.update_status("rtx", unhealthy).await;
        let plan = s
            .inference_router
            .ready_decision_plan("class:semantic_decisions")
            .await
            .unwrap();
        assert_eq!(plan.runner.id, "halo1");
    }
}
