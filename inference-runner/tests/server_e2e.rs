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

#[tokio::test]
async fn actual_runner_discovers_models_and_proxies_chat() {
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

    let directory = tempfile::tempdir().unwrap();
    let reservation = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = reservation.local_addr().unwrap().port();
    drop(reservation);
    std::fs::write(directory.path().join("config.toml"), format!(
        "[runner]\nid=\"e2e-runner\"\nname=\"E2E runner\"\n[api]\nhost=\"127.0.0.1\"\nport={port}\n[engines.ollama]\nenabled=true\nbase_url=\"{}\"\n", engine.uri()
    )).unwrap();
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
    child.kill().await.unwrap();
    child.wait().await.unwrap();
}
