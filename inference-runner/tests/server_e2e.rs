//! Real runner HTTP API against an isolated mock inference engine.
use serde_json::{json, Value};
use std::{process::Stdio, time::Duration};
use tokio::{
    process::Command,
    time::{sleep, timeout},
};
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

#[cfg(unix)]
#[tokio::test]
async fn actual_runner_drains_chat_on_sigterm() {
    actual_runner_discovers_models_and_proxies_chat(nix::sys::signal::Signal::SIGTERM, false).await;
}

#[cfg(unix)]
#[tokio::test]
async fn actual_runner_drains_chat_on_sigint_with_gateway() {
    actual_runner_discovers_models_and_proxies_chat(nix::sys::signal::Signal::SIGINT, true).await;
}

#[cfg(unix)]
async fn actual_runner_discovers_models_and_proxies_chat(
    signal: nix::sys::signal::Signal,
    gateway: bool,
) {
    let engine = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"models":[{"name":"e2e-model"}]})),
        )
        .mount(&engine)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"model_info":{}})))
        .mount(&engine)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"done":true})))
        .mount(&engine)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model":"e2e-model", "message":{"role":"assistant","content":"E2E reply"},
            "done":true, "prompt_eval_count":2, "eval_count":3
        })))
        .expect(1)
        .mount(&engine)
        .await;

    let gateway_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let gateway_addr = gateway_listener.local_addr().unwrap();
    let (registered_tx, registered_rx) = tokio::sync::oneshot::channel();
    let gateway_task = tokio::spawn(async move {
        use futures_util::{SinkExt, StreamExt};
        let (socket, _) = gateway_listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
        let registration = ws.next().await.unwrap().unwrap();
        assert!(registration.to_text().unwrap().contains("e2e-runner"));
        ws.send(tokio_tungstenite::tungstenite::Message::Text(
            r#"{"type":"register_ack","runner_id":"e2e-runner"}"#.to_owned(),
        ))
        .await
        .unwrap();
        registered_tx.send(()).unwrap();
        while let Some(message) = ws.next().await {
            if message.is_err() {
                break;
            }
        }
    });
    let directory = tempfile::tempdir().unwrap();
    let reservation = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = reservation.local_addr().unwrap().port();
    drop(reservation);
    std::fs::write(directory.path().join("config.toml"), format!(
        "[runner]\nid=\"e2e-runner\"\nname=\"E2E runner\"\n[api]\nhost=\"127.0.0.1\"\nport={port}\n[engines.ollama]\nenabled=true\nbase_url=\"{}\"\n", engine.uri()
    )).unwrap();
    if gateway {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(directory.path().join("config.toml"))
            .unwrap();
        writeln!(file, "[gateway]\nws_url=\"ws://{gateway_addr}\"\nauth_token=\"test\"\nheartbeat_interval_secs=1\nreconnect_delay_secs=60").unwrap();
    }
    let mut command = Command::new(env!("CARGO_BIN_EXE_simple-ai-runner"));
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("RUNNER_") {
            command.env_remove(key);
        }
    }
    let mut child = command
        .current_dir(directory.path())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let base = format!("http://127.0.0.1:{port}");
    timeout(Duration::from_secs(15), async {
        loop {
            assert!(
                child.try_wait().unwrap().is_none(),
                "runner exited before readiness"
            );
            if let Ok(response) = http.get(format!("{base}/health")).send().await {
                if response.status().is_success() {
                    let health: Value = response.json().await.unwrap();
                    assert_eq!(health["status"], "ok");
                    assert_eq!(health["engines"][0]["healthy"], true);
                    break;
                }
            }
            sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("runner readiness timeout");
    let models: Value = http
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["data"][0]["id"], "e2e-model");
    if gateway {
        timeout(Duration::from_secs(10), registered_rx)
            .await
            .unwrap()
            .unwrap();
    }
    let chat: Value = http.post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model":"e2e-model","messages":[{"role":"user","content":"Hello"}],"stream":false}))
        .send().await.unwrap().error_for_status().unwrap().json().await.unwrap();
    assert_eq!(chat["choices"][0]["message"]["content"], "E2E reply");
    assert_eq!(chat["usage"]["total_tokens"], 5);
    assert_eq!(
        http.post(format!("{base}/v1/chat/completions"))
            .json(&json!({}))
            .send()
            .await
            .unwrap()
            .status(),
        422
    );
    // A second request blocks upstream while the OS signal starts HTTP draining.
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(500))
                .set_body_json(json!({
                    "model":"e2e-model", "message":{"role":"assistant","content":"drained reply"},
                    "done":true, "prompt_eval_count":2, "eval_count":3
                })),
        )
        .with_priority(1)
        .expect(1)
        .mount(&engine)
        .await;
    let draining_http = http.clone();
    let draining_base = base.clone();
    let request = tokio::spawn(async move {
        draining_http.post(format!("{draining_base}/v1/chat/completions"))
            .json(&json!({"model":"e2e-model","messages":[{"role":"user","content":"Finish this"}],"stream":false}))
            .send().await.unwrap().error_for_status().unwrap().json::<Value>().await.unwrap()
    });
    timeout(Duration::from_secs(5), async {
        loop {
            if engine
                .received_requests()
                .await
                .unwrap()
                .iter()
                .filter(|r| r.url.path() == "/api/chat")
                .count()
                >= 2
            {
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(child.id().unwrap() as i32),
        signal,
    )
    .unwrap();
    let response = timeout(Duration::from_secs(5), request)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        response["choices"][0]["message"]["content"],
        "drained reply"
    );
    assert!(timeout(Duration::from_secs(5), child.wait())
        .await
        .unwrap()
        .unwrap()
        .success());
    assert!(http.get(format!("{base}/health")).send().await.is_err());
    if gateway {
        timeout(Duration::from_secs(5), gateway_task)
            .await
            .unwrap()
            .unwrap();
    } else {
        gateway_task.abort();
        let _ = gateway_task.await;
    }
}
