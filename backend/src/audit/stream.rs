//! Capture client-visible SSE without delaying delivery or retaining unbounded output.
use std::sync::Arc;

use futures_util::StreamExt;
use simple_server::web::body::Body;

use super::AuditLogger;

const MAX_CAPTURE_BYTES: usize = 4 * 1024 * 1024;

struct Capture {
    logger: Arc<AuditLogger>,
    request_id: String,
    bytes: Vec<u8>,
    truncated: bool,
    complete: bool,
    failed: bool,
}

impl Drop for Capture {
    fn drop(&mut self) {
        let mut body = String::from_utf8_lossy(&self.bytes).into_owned();
        if self.truncated {
            body.push_str("\n[History capture truncated at 4 MiB]");
        }
        if !self.complete {
            body.push_str("\n[Stream interrupted; partial output]");
        }
        if let Err(error) = self.logger.save_stream_body(&self.request_id, &body) {
            tracing::error!(%error, "Cannot save streamed response history");
        }
    }
}

pub fn capture_response(body: Body, logger: Arc<AuditLogger>, request_id: String) -> Body {
    let capture = Capture {
        logger,
        request_id,
        bytes: Vec::new(),
        truncated: false,
        complete: false,
        failed: false,
    };
    let stream = futures_util::stream::unfold(
        (body.into_data_stream(), capture),
        |(mut stream, mut capture)| async move {
            match stream.next().await {
                Some(item) => {
                    capture.failed |= item.is_err();
                    if let Ok(bytes) = &item {
                        let remaining = MAX_CAPTURE_BYTES.saturating_sub(capture.bytes.len());
                        capture
                            .bytes
                            .extend_from_slice(&bytes[..bytes.len().min(remaining)]);
                        capture.truncated |= bytes.len() > remaining;
                    }
                    Some((item, (stream, capture)))
                }
                None => {
                    capture.complete = !capture.failed;
                    drop(capture);
                    None
                }
            }
        },
    );
    Body::from_stream(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::request::Request;
    use simple_server::web::body::Bytes;

    fn setup() -> (Arc<AuditLogger>, String) {
        let logger = Arc::new(AuditLogger::new(":memory:").unwrap());
        let user = logger.find_or_create_user("stream-user", None).unwrap();
        let request = Request::new(user.id, "/v1/chat/completions".into());
        logger.log_request(&request).unwrap();
        (logger, request.id)
    }

    #[tokio::test]
    async fn captures_complete_stream_without_changing_delivery() {
        let (logger, id) = setup();
        let body = capture_response(
            Body::from("data: hello\n\ndata: [DONE]\n\n"),
            logger.clone(),
            id.clone(),
        );
        let mut stream = body.into_data_stream();
        let mut delivered = Vec::new();
        while let Some(chunk) = stream.next().await {
            delivered.extend_from_slice(&chunk.unwrap());
        }
        assert_eq!(delivered, b"data: hello\n\ndata: [DONE]\n\n");
        assert_eq!(
            logger
                .get_request_bodies(&id)
                .unwrap()
                .unwrap()
                .response_body
                .unwrap()
                .as_bytes(),
            delivered
        );
    }

    #[tokio::test]
    async fn failed_stream_is_marked_incomplete() {
        let (logger, id) = setup();
        let chunks = futures_util::stream::iter(vec![
            Ok(Bytes::from_static(b"data: partial\n\n")),
            Err(std::io::Error::other("upstream failed")),
        ]);
        let mut stream = capture_response(Body::from_stream(chunks), logger.clone(), id.clone())
            .into_data_stream();
        assert!(stream.next().await.unwrap().is_ok());
        assert!(stream.next().await.unwrap().is_err());
        assert!(stream.next().await.is_none());
        let saved = logger
            .get_request_bodies(&id)
            .unwrap()
            .unwrap()
            .response_body
            .unwrap();
        assert!(saved.starts_with("data: partial"));
        assert!(saved.contains("partial output"));
    }

    #[tokio::test]
    async fn captures_partial_stream_and_bounds_storage() {
        let (logger, id) = setup();
        let chunks = futures_util::stream::iter(vec![Ok::<_, std::io::Error>(Bytes::from(vec![
                b'x';
                MAX_CAPTURE_BYTES
                    + 10
            ]))])
        .chain(futures_util::stream::pending());
        let mut stream = capture_response(Body::from_stream(chunks), logger.clone(), id.clone())
            .into_data_stream();
        assert_eq!(
            stream.next().await.unwrap().unwrap().len(),
            MAX_CAPTURE_BYTES + 10
        );
        drop(stream);
        let saved = logger
            .get_request_bodies(&id)
            .unwrap()
            .unwrap()
            .response_body
            .unwrap();
        assert!(saved.contains("truncated at 4 MiB"));
        assert!(saved.contains("partial output"));
        assert!(saved.len() < MAX_CAPTURE_BYTES + 150);
    }
}
