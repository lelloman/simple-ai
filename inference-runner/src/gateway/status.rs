//! Status collection from engines and capabilities.

use std::sync::Arc;

use simple_ai_common::{
    Capability, CapabilityInfo, CapabilityStatus, EngineStatus, ModelInfo, RunnerHealth, RunnerStatus,
};

use crate::config::Config;
use crate::engine::EngineRegistry;

/// Collects status from all engines and builds a RunnerStatus.
pub struct StatusCollector {
    config: Config,
    engine_registry: Arc<EngineRegistry>,
}

impl StatusCollector {
    pub fn new(config: Config, engine_registry: Arc<EngineRegistry>) -> Self {
        Self {
            config,
            engine_registry,
        }
    }

    /// Resolve a local engine model name to its canonical alias name.
    /// If an alias maps to this local name, returns the canonical name.
    /// Otherwise returns the original local name.
    fn resolve_to_canonical(&self, local_name: &str) -> String {
        // Reverse lookup: find alias key where value == local_name
        for (canonical, local) in &self.config.aliases.mappings {
            if local == local_name {
                return canonical.clone();
            }
        }
        local_name.to_string()
    }

    /// Collect current status from all engines.
    pub async fn collect(&self) -> RunnerStatus {
        let engines = self.collect_engine_status().await;
        let capabilities = self.collect_capabilities(&engines).await;
        let health = Self::compute_health(&engines);

        RunnerStatus {
            health,
            capabilities,
            engines,
            // TODO: Implement actual metrics collection (request counts, latency, GPU/CPU usage)
            metrics: None,
            model_aliases: self.config.aliases.mappings.clone(),
        }
    }

    /// Collect status from all registered engines.
    async fn collect_engine_status(&self) -> Vec<EngineStatus> {
        let mut statuses = Vec::new();

        for engine in self.engine_registry.all().await {
            let status = match engine.health_check().await {
                Ok(health) => {
                    // Get all available models from this engine
                    // Map local engine names to canonical alias names where configured
                    let available_models = match engine.list_models().await {
                        Ok(models) => models
                            .into_iter()
                            .map(|m| {
                                let canonical_id = self.resolve_to_canonical(&m.id);
                                let canonical_name = self.resolve_to_canonical(&m.name);
                                ModelInfo {
                                    id: canonical_id,
                                    name: canonical_name,
                                    size_bytes: m.size_bytes,
                                    parameter_count: m.parameter_count,
                                    context_length: m.context_length,
                                    quantization: m.quantization,
                                    modified_at: m.modified_at,
                                }
                            })
                            .collect(),
                        Err(_) => vec![],
                    };

                    // Map loaded model names to canonical aliases too
                    let loaded_models: Vec<String> = health
                        .models_loaded
                        .iter()
                        .map(|m| self.resolve_to_canonical(m))
                        .collect();

                    EngineStatus {
                        engine_type: engine.engine_type().to_string(),
                        is_healthy: health.is_healthy,
                        version: health.version,
                        loaded_models,
                        available_models,
                        error: None,
                        batch_size: engine.batch_size(),
                    }
                }
                Err(e) => EngineStatus {
                    engine_type: engine.engine_type().to_string(),
                    is_healthy: false,
                    version: None,
                    loaded_models: vec![],
                    available_models: vec![],
                    error: Some(e.to_string()),
                    batch_size: engine.batch_size(),
                },
            };
            statuses.push(status);
        }

        statuses
    }

    /// Collect capability status based on engine state and config mappings.
    async fn collect_capabilities(&self, engines: &[EngineStatus]) -> Vec<CapabilityInfo> {
        let mut capabilities = Vec::new();
        let loaded_models: Vec<&str> = engines
            .iter()
            .flat_map(|e| e.loaded_models.iter().map(|s| s.as_str()))
            .collect();

        // Check each configured capability mapping
        for (model_id, caps) in &self.config.capabilities.mappings {
            for cap in caps {
                let status = if loaded_models.contains(&model_id.as_str()) {
                    CapabilityStatus::Loaded
                } else {
                    CapabilityStatus::Unloaded
                };

                capabilities.push(CapabilityInfo {
                    capability: *cap,
                    status,
                    model_id: model_id.clone(),
                    active_requests: 0,
                    avg_latency_ms: None,
                });
            }
        }

        // Add default capabilities if no mappings configured
        if capabilities.is_empty() {
            // Report all loaded models as capable of large_chat by default
            for model in loaded_models {
                capabilities.push(CapabilityInfo {
                    capability: Capability::LargeChat,
                    status: CapabilityStatus::Loaded,
                    model_id: model.to_string(),
                    active_requests: 0,
                    avg_latency_ms: None,
                });
            }
        }

        capabilities
    }

    /// Determine overall health based on engine states.
    fn compute_health(engines: &[EngineStatus]) -> RunnerHealth {
        if engines.is_empty() {
            return RunnerHealth::Starting;
        }

        let healthy_count = engines.iter().filter(|e| e.is_healthy).count();

        if healthy_count == engines.len() {
            RunnerHealth::Healthy
        } else if healthy_count > 0 {
            RunnerHealth::Degraded
        } else {
            RunnerHealth::Unhealthy
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AliasesConfig, ApiConfig, CapabilitiesConfig, Config, EnginesConfig, RunnerConfig};
    use std::collections::HashMap;

    fn test_config() -> Config {
        Config {
            runner: RunnerConfig {
                id: "test".to_string(),
                name: "Test".to_string(),
                machine_type: None,
                mac_address: None,
            },
            api: ApiConfig::default(),
            gateway: None,
            engines: EnginesConfig::default(),
            capabilities: CapabilitiesConfig::default(),
            aliases: AliasesConfig::default(),
        }
    }

    fn test_config_with_mappings() -> Config {
        let mut config = test_config();
        let mut mappings = HashMap::new();
        mappings.insert("llama3.2:3b".to_string(), vec![Capability::FastChat]);
        mappings.insert("qwen2.5:72b".to_string(), vec![Capability::LargeChat]);
        config.capabilities.mappings = mappings;
        config
    }

    #[tokio::test]
    async fn test_collect_capabilities_with_mappings() {
        let config = test_config_with_mappings();
        let registry = std::sync::Arc::new(crate::engine::EngineRegistry::new());
        let collector = StatusCollector::new(config, registry);

        let engines = vec![EngineStatus {
            engine_type: "ollama".to_string(),
            is_healthy: true,
            version: None,
            loaded_models: vec!["llama3.2:3b".to_string()],
            available_models: vec![],
            error: None,
            batch_size: 1,
        }];

        let capabilities = collector.collect_capabilities(&engines).await;

        // Should have 2 capabilities (one loaded, one unloaded)
        assert_eq!(capabilities.len(), 2);

        let fast_chat = capabilities.iter().find(|c| c.capability == Capability::FastChat);
        assert!(fast_chat.is_some());
        assert_eq!(fast_chat.unwrap().status, CapabilityStatus::Loaded);

        let large_chat = capabilities.iter().find(|c| c.capability == Capability::LargeChat);
        assert!(large_chat.is_some());
        assert_eq!(large_chat.unwrap().status, CapabilityStatus::Unloaded);
    }

    #[tokio::test]
    async fn test_collect_capabilities_default_behavior() {
        let config = test_config(); // No mappings
        let registry = std::sync::Arc::new(crate::engine::EngineRegistry::new());
        let collector = StatusCollector::new(config, registry);

        let engines = vec![EngineStatus {
            engine_type: "ollama".to_string(),
            is_healthy: true,
            version: None,
            loaded_models: vec!["some-model".to_string()],
            available_models: vec![],
            error: None,
            batch_size: 1,
        }];

        let capabilities = collector.collect_capabilities(&engines).await;

        // Without mappings, loaded models default to LargeChat
        assert_eq!(capabilities.len(), 1);
        assert_eq!(capabilities[0].capability, Capability::LargeChat);
        assert_eq!(capabilities[0].model_id, "some-model");
    }

    #[tokio::test]
    async fn test_collect_capabilities_no_loaded_models() {
        let config = test_config();
        let registry = std::sync::Arc::new(crate::engine::EngineRegistry::new());
        let collector = StatusCollector::new(config, registry);

        let engines = vec![EngineStatus {
            engine_type: "ollama".to_string(),
            is_healthy: true,
            version: None,
            loaded_models: vec![],
            available_models: vec![],
            error: None,
            batch_size: 1,
        }];

        let capabilities = collector.collect_capabilities(&engines).await;

        // No mappings + no loaded models = empty capabilities
        assert!(capabilities.is_empty());
    }

    #[test]
    fn test_compute_health_all_healthy() {
        let engines = vec![
            EngineStatus {
                engine_type: "ollama".to_string(),
                is_healthy: true,
                version: None,
                loaded_models: vec![],
                available_models: vec![],
                error: None,
                batch_size: 1,
            },
            EngineStatus {
                engine_type: "llama_cpp".to_string(),
                is_healthy: true,
                version: None,
                loaded_models: vec![],
                available_models: vec![],
                error: None,
                batch_size: 1,
            },
        ];
        assert_eq!(StatusCollector::compute_health(&engines), RunnerHealth::Healthy);
    }

    #[test]
    fn test_compute_health_some_unhealthy() {
        let engines = vec![
            EngineStatus {
                engine_type: "ollama".to_string(),
                is_healthy: true,
                version: None,
                loaded_models: vec![],
                available_models: vec![],
                error: None,
                batch_size: 1,
            },
            EngineStatus {
                engine_type: "llama_cpp".to_string(),
                is_healthy: false,
                version: None,
                loaded_models: vec![],
                available_models: vec![],
                error: Some("Connection failed".to_string()),
                batch_size: 1,
            },
        ];
        assert_eq!(StatusCollector::compute_health(&engines), RunnerHealth::Degraded);
    }

    #[test]
    fn test_compute_health_all_unhealthy() {
        let engines = vec![EngineStatus {
            engine_type: "ollama".to_string(),
            is_healthy: false,
            version: None,
            loaded_models: vec![],
            available_models: vec![],
            error: Some("Down".to_string()),
            batch_size: 1,
        }];
        assert_eq!(StatusCollector::compute_health(&engines), RunnerHealth::Unhealthy);
    }

    #[test]
    fn test_compute_health_no_engines() {
        let engines: Vec<EngineStatus> = vec![];
        assert_eq!(StatusCollector::compute_health(&engines), RunnerHealth::Starting);
    }
}
