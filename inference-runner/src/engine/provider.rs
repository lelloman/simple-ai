//! Startup policy shared by the four local Hugging Face providers.
use crate::error::{Error, Result};
use std::{
    collections::VecDeque,
    process::Stdio,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{
    io::AsyncReadExt,
    process::{Child, Command},
};

const STDERR_BYTES: usize = 8192;

pub(super) async fn start(
    command: &mut Command,
    client: &reqwest::Client,
    engine: &str,
    model: &str,
    port: u16,
    timeout_secs: u64,
) -> Result<Child> {
    // Async vLLM chat has a separate engine-core child, unlike offline providers.
    #[cfg(unix)]
    if engine == "tag" {
        command.process_group(0);
    }
    tracing::info!(engine, model, port, "Spawning provider (HF_HUB_OFFLINE=1)");
    let mut child = command
        .env("HF_HUB_OFFLINE", "1")
        .kill_on_drop(true)
        .stdout(Stdio::inherit())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            tracing::warn!(engine, model, %error, "Provider spawn failed");
            Error::LoadFailed(format!("{engine} provider for {model}: {error}"))
        })?;
    let mut startup_group = ProcessGroup(if engine == "tag" { child.id() } else { None });
    let mut stderr = child.stderr.take().expect("piped stderr");
    let tail = Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_BYTES)));
    let captured = tail.clone();
    let engine_name = engine.to_string();
    let model_name = model.to_string();
    let mut reader = tokio::spawn(async move {
        let mut bytes = [0; 1024];
        loop {
            let n = match stderr.read(&mut bytes).await {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            tracing::info!(engine = %engine_name, model = %model_name,
                stderr = %String::from_utf8_lossy(&bytes[..n]), "Provider stderr");
            // Fixed byte bound, including output without newlines or invalid UTF-8.
            let mut tail = captured.lock().unwrap();
            for byte in &bytes[..n] {
                if tail.len() == STDERR_BYTES {
                    tail.pop_front();
                }
                tail.push_back(*byte);
            }
        }
    });
    let url = format!("http://127.0.0.1:{port}/health");
    let ready = async {
        loop {
            tokio::select! {
                biased;
                status = child.wait() => {
                    return Err(format!("exited during startup with {}", status
                        .map_err(|e| e.to_string())?));
                }
                response = client.get(&url).timeout(Duration::from_secs(2)).send() => {
                    if response.is_ok_and(|r| r.status().is_success()) { return Ok(()); }
                }
            }
            tokio::select! {
                biased;
                status = child.wait() => {
                    return Err(format!("exited during startup with {}", status
                        .map_err(|e| e.to_string())?));
                }
                _ = tokio::time::sleep(Duration::from_millis(100)) => {}
            }
        }
    };
    let result = tokio::time::timeout(Duration::from_secs(timeout_secs), ready).await;
    match result {
        Ok(Ok(())) => {
            // Keep draining stderr for the lifetime of the healthy process.
            tracing::info!(engine, model, port, "Provider ready");
            startup_group.0 = None;
            Ok(child)
        }
        result => {
            let _ = child.kill().await;
            // A descendant may hold stderr open: never delay startup failure indefinitely.
            if tokio::time::timeout(Duration::from_millis(100), &mut reader)
                .await
                .is_err()
            {
                reader.abort();
            }
            let reason = match result {
                Ok(Err(reason)) => reason,
                _ => "startup timed out".into(),
            };
            let bytes: Vec<u8> = tail.lock().unwrap().iter().copied().collect();
            let stderr = String::from_utf8_lossy(&bytes);
            let message = format!("{engine} provider for {model} {reason}; stderr tail:\n{stderr}");
            tracing::warn!(engine, model, error = %message, "Provider startup failed");
            Err(Error::LoadFailed(message))
        }
    }
}

/// Kill all descendants of the async vLLM server, including on startup cancellation.
pub(super) struct ProcessGroup(pub Option<u32>);
impl ProcessGroup {
    pub fn terminate(&self) -> std::io::Result<()> {
        #[cfg(unix)]
        if let Some(pid) = self.0 {
            use nix::{
                errno::Errno,
                sys::signal::{kill, Signal},
                unistd::Pid,
            };
            match kill(Pid::from_raw(-(pid as i32)), Signal::SIGKILL) {
                Ok(()) | Err(Errno::ESRCH) => {}
                Err(error) => return Err(std::io::Error::from_raw_os_error(error as i32)),
            }
        }
        Ok(())
    }
}
impl Drop for ProcessGroup {
    fn drop(&mut self) {
        if let Err(error) = self.terminate() {
            tracing::error!(%error, "Failed to terminate tag provider group");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn exit_interrupts_even_a_stalled_health_response_and_captures_stderr() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let stalled = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.unwrap();
            std::future::pending::<()>().await;
        });
        let mut command = Command::new("sh");
        command.args(["-c", "sleep 0.1; printf 'startup failure\\n' >&2; exit 23"]);
        let error = tokio::time::timeout(
            Duration::from_secs(2),
            start(
                &mut command,
                &reqwest::Client::new(),
                "test",
                "broken",
                port,
                600,
            ),
        )
        .await
        .expect("must not wait for health or startup timeout")
        .unwrap_err()
        .to_string();
        stalled.abort();
        assert!(error.contains("exit status: 23"), "{error}");
        assert!(error.contains("startup failure"), "{error}");
    }

    #[tokio::test]
    async fn providers_receive_offline_environment() {
        let mut command = Command::new("sh");
        command.args(["-c", "printf 'offline=%s' \"$HF_HUB_OFFLINE\" >&2; exit 7"]);
        let error = start(
            &mut command,
            &reqwest::Client::new(),
            "test",
            "offline",
            0,
            600,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(error.contains("offline=1"), "{error}");
    }

    #[tokio::test]
    async fn stderr_capture_is_bounded_without_newlines() {
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "head -c 100000 /dev/zero >&2; printf 'last line' >&2; exit 9",
        ]);
        let error = start(
            &mut command,
            &reqwest::Client::new(),
            "test",
            "large",
            0,
            600,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(error.ends_with("last line"), "{error}");
        assert!(error.len() < STDERR_BYTES + 200);
    }
}
