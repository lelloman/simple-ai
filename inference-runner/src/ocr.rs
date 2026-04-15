//! OCR provider abstraction and CLI-backed provider implementation.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use simple_ai_common::{OcrOptions, OcrProviderInfo, OcrResponse};
use tempfile::TempDir;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::config::OcrConfig;
use crate::error::{Error, Result};

#[async_trait]
pub trait OcrProvider: Send + Sync {
    async fn health_check(&self) -> Result<OcrProviderInfo>;
    async fn recognize(
        &self,
        input_path: &Path,
        filename: &str,
        options: &OcrOptions,
    ) -> Result<OcrResponse>;
    fn max_file_bytes(&self) -> u64;
}

/// CLI-backed OCR provider. The configured command is executed with:
/// --input <path> --filename <name> --options <json path> --output <json path>
pub struct CliOcrProvider {
    config: OcrConfig,
}

impl CliOcrProvider {
    pub fn new(config: OcrConfig) -> Result<Self> {
        if config.command.is_empty() {
            return Err(Error::InvalidRequest(
                "ocr.command is required when OCR is enabled".to_string(),
            ));
        }
        Ok(Self { config })
    }

    fn build_command(&self) -> Command {
        let mut command = Command::new(&self.config.command[0]);
        for arg in self.config.command.iter().skip(1) {
            command.arg(arg);
        }
        command
    }

    async fn run_provider(
        &self,
        input_path: &Path,
        filename: &str,
        options_path: &Path,
        output_path: &Path,
    ) -> Result<()> {
        let mut command = self.build_command();
        command
            .arg("--input")
            .arg(input_path)
            .arg("--filename")
            .arg(filename)
            .arg("--options")
            .arg(options_path)
            .arg("--output")
            .arg(output_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let timeout_duration = Duration::from_secs(self.config.timeout_secs);
        let output = tokio::time::timeout(timeout_duration, command.output())
            .await
            .map_err(|_| Error::InferenceFailed("OCR provider timed out".to_string()))?
            .map_err(|e| Error::Communication(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let message = if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("provider exited with {}", output.status)
            };
            return Err(Error::InferenceFailed(message));
        }

        Ok(())
    }

    async fn run_health_check(&self, output_path: &Path) -> Result<()> {
        let mut command = self.build_command();
        command
            .arg("--health")
            .arg("--output")
            .arg(output_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = tokio::time::timeout(Duration::from_secs(15), command.output())
            .await
            .map_err(|_| Error::InferenceFailed("OCR provider health check timed out".to_string()))?
            .map_err(|e| Error::Communication(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let message = if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("provider health check exited with {}", output.status)
            };
            return Err(Error::EngineNotAvailable(message));
        }

        Ok(())
    }
}

#[async_trait]
impl OcrProvider for CliOcrProvider {
    async fn health_check(&self) -> Result<OcrProviderInfo> {
        let temp_dir = TempDir::new().map_err(|e| Error::Internal(e.to_string()))?;
        let output_path = temp_dir.path().join("health.json");
        self.run_health_check(&output_path).await?;
        let bytes = tokio::fs::read(&output_path)
            .await
            .map_err(|e| Error::EngineNotAvailable(format!("OCR health output missing: {}", e)))?;
        let mut info: OcrProviderInfo = serde_json::from_slice(&bytes)
            .map_err(|e| Error::EngineNotAvailable(format!("invalid OCR health output: {}", e)))?;

        if info.provider.is_empty() {
            info.provider = self.config.provider.clone();
        }
        if info.modes.is_empty() {
            info.modes = self.config.modes.clone();
        }
        if info.features.is_empty() {
            info.features = self.config.features.clone();
        }
        if info.languages.is_empty() {
            info.languages = self.config.languages.clone();
        }

        Ok(OcrProviderInfo {
            provider: self.config.provider.clone(),
            provider_version: info.provider_version,
            modes: info.modes,
            features: info.features,
            languages: info.languages,
        })
    }

    async fn recognize(
        &self,
        input_path: &Path,
        filename: &str,
        options: &OcrOptions,
    ) -> Result<OcrResponse> {
        if !self.config.modes.contains(&options.mode) {
            return Err(Error::InvalidRequest(format!(
                "OCR mode {:?} is not supported by {}",
                options.mode, self.config.provider
            )));
        }

        if let Some(max_pages) = options.max_pages {
            if max_pages > self.config.max_pages {
                return Err(Error::InvalidRequest(format!(
                    "maxPages {} exceeds runner limit {}",
                    max_pages, self.config.max_pages
                )));
            }
        }

        let started = Instant::now();
        let temp_dir = TempDir::new().map_err(|e| Error::Internal(e.to_string()))?;
        let options_path = temp_dir.path().join("options.json");
        let output_path = temp_dir.path().join("output.json");

        let mut options_file = tokio::fs::File::create(&options_path)
            .await
            .map_err(|e| Error::Internal(e.to_string()))?;
        let options_json =
            serde_json::to_vec(options).map_err(|e| Error::InvalidRequest(e.to_string()))?;
        options_file
            .write_all(&options_json)
            .await
            .map_err(|e| Error::Internal(e.to_string()))?;
        options_file
            .flush()
            .await
            .map_err(|e| Error::Internal(e.to_string()))?;

        self.run_provider(input_path, filename, &options_path, &output_path)
            .await?;

        let bytes = tokio::fs::read(&output_path)
            .await
            .map_err(|e| Error::InferenceFailed(format!("OCR output missing: {}", e)))?;
        let mut response: OcrResponse =
            serde_json::from_slice(&bytes).map_err(|e| Error::InferenceFailed(e.to_string()))?;
        response.metadata.provider = self.config.provider.clone();
        response.metadata.mode = options.mode;
        if response.metadata.elapsed_ms == 0 {
            response.metadata.elapsed_ms = started.elapsed().as_millis() as u64;
        }
        Ok(response)
    }

    fn max_file_bytes(&self) -> u64 {
        self.config.max_file_mb * 1024 * 1024
    }
}

pub async fn write_upload(bytes: &[u8], filename: &str) -> Result<(TempDir, PathBuf)> {
    let temp_dir = TempDir::new().map_err(|e| Error::Internal(e.to_string()))?;
    let safe_name = filename
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect::<String>();
    let path = temp_dir.path().join(if safe_name.is_empty() {
        "upload.bin"
    } else {
        &safe_name
    });
    tokio::fs::write(&path, bytes)
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;
    Ok((temp_dir, path))
}
