use std::sync::Arc;
use std::time::Duration;

use tokio::time::{sleep, timeout};

use simple_ai_common::{ChatCompletionRequest, ChatCompletionResponse, GatewayMessage};

use crate::config::RoutingConfig;
use crate::wol::{WakeError, WakeService};

use super::{
    BatchQueue, InferenceRouter, ModelRequest, RoutePlan, RouterError, RouterTelemetry,
    RunnerEvent, RunnerRegistry,
};
use crate::routes::embeddings::{EmbeddingRequest, EmbeddingResponse};

#[derive(Debug)]
pub struct ScheduledResponse<T> {
    pub response: T,
    pub runner_id: String,
    pub resolved_model: String,
    pub wol_sent: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("{0}")]
    Router(#[from] RouterError),
    #[error("{0}")]
    Wake(#[from] WakeError),
}

pub struct RequestScheduler {
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
        request: &ChatCompletionRequest,
        use_batching: bool,
    ) -> Result<ScheduledResponse<ChatCompletionResponse>, SchedulerError> {
        let wol_sent = self
            .prepare_for_request(request_id, model, model_request)
            .await?;
        let routed = if use_batching {
            let batch_queue = self
                .batch_queue
                .as_ref()
                .ok_or(RouterError::ConnectionFailed(
                    "Batch queue unavailable".to_string(),
                ))?;
            self.inference_router
                .chat_completion_batched(model, request, batch_queue)
                .await?
        } else {
            self.inference_router
                .chat_completion(model, request)
                .await?
        };
        Ok(ScheduledResponse {
            response: routed.response,
            runner_id: routed.runner_id,
            resolved_model: routed.resolved_model,
            wol_sent,
        })
    }

    pub async fn embeddings(
        &self,
        request_id: &str,
        model: &str,
        model_request: &ModelRequest,
        request: &EmbeddingRequest,
    ) -> Result<ScheduledResponse<EmbeddingResponse>, SchedulerError> {
        let wol_sent = self
            .prepare_for_request(request_id, model, model_request)
            .await?;
        let routed = self
            .inference_router
            .embed::<EmbeddingRequest, EmbeddingResponse>(model, request)
            .await?;
        Ok(ScheduledResponse {
            response: routed.response,
            runner_id: routed.runner_id,
            resolved_model: routed.resolved_model,
            wol_sent,
        })
    }

    async fn prepare_for_request(
        &self,
        request_id: &str,
        model: &str,
        model_request: &ModelRequest,
    ) -> Result<bool, SchedulerError> {
        self.router_telemetry
            .emit(
                "request_received",
                format!("Scheduler received request for {}", model),
                Some(request_id.to_string()),
                None,
                Some(model.to_string()),
            )
            .await;

        match self.inference_router.plan_request(model).await {
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
                Ok(false)
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
                    let plan = self.inference_router.plan_request(model).await?;
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
                    Ok(true)
                } else {
                    Err(RouterError::NoRunners.into())
                }
            }
            Err(e) => Err(e.into()),
        }
    }

    async fn prepare_runner_model(
        &self,
        request_id: &str,
        plan: &RoutePlan,
    ) -> Result<(), SchedulerError> {
        if plan.runner.has_model_or_alias(&plan.resolved_model) {
            return Ok(());
        }

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
        plan.runner
            .tx
            .send(GatewayMessage::LoadModel {
                model_id: plan.resolved_model.clone(),
                request_id: request_id.clone(),
            })
            .await
            .map_err(|e| RouterError::ConnectionFailed(e.to_string()))?;

        self.wait_for_model_ready(&plan.runner.id, &plan.resolved_model, &request_id)
            .await
    }

    async fn wait_for_model_ready(
        &self,
        runner_id: &str,
        model_id: &str,
        request_id: &str,
    ) -> Result<(), SchedulerError> {
        let mut events = self.runner_registry.subscribe_events();
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
