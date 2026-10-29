//! Provenance from the runner selected for this response, including streams.
use simple_server::web::http::{HeaderMap, HeaderValue};

pub fn headers(runner: &str, revision: Option<&str>) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (name, value) in [
        ("x-runner-id", Some(runner)),
        ("x-model-revision", revision),
    ] {
        if let Some(value) = value {
            match HeaderValue::from_str(value) {
                Ok(value) => {
                    headers.insert(name, value);
                }
                Err(error) => {
                    tracing::warn!(runner, header = name, %error, "Invalid model identity header")
                }
            }
        }
    }
    headers
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn emits_identity_and_tolerates_old_runners() {
        let h = headers("halo2", Some("sha256:abc"));
        assert_eq!(h["x-runner-id"], "halo2");
        assert_eq!(h["x-model-revision"], "sha256:abc");
        let old = headers("old", None);
        assert_eq!(old["x-runner-id"], "old");
        assert!(!old.contains_key("x-model-revision"));
        assert!(!headers("halo", Some("bad\nvalue")).contains_key("x-model-revision"));
    }
}
