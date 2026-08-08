mod openai;

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::json;
use thiserror::Error;

use crate::{engine::EngineHandle, metrics::Metrics};

pub use openai::{ChatCompletionRequest, Message, Role};

#[derive(Clone)]
pub struct AppState {
    pub engine: EngineHandle,
    pub metrics: Arc<Metrics>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health/live", get(live))
        .route("/health/ready", get(ready))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(openai::models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .with_state(state)
}

async fn live() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "status": "live" })))
}

async fn ready(State(state): State<AppState>) -> Response {
    if state.metrics.is_ready() {
        (StatusCode::OK, Json(json!({ "status": "ready" }))).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "status": "not_ready" })),
        )
            .into_response()
    }
}

async fn metrics(State(state): State<AppState>) -> Response {
    let mut response = state.metrics.prometheus().into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    response
}

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("the inference server is overloaded")]
    Overloaded,
    #[error("the inference engine is unavailable")]
    Unavailable,
    #[error("generation failed: {0}")]
    Generation(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, kind) = match self {
            Self::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            Self::Overloaded => (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
            Self::Unavailable => (StatusCode::SERVICE_UNAVAILABLE, "server_error"),
            Self::Generation(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
        };
        let message = self.to_string();
        (
            status,
            Json(json!({
                "error": {
                    "message": message,
                    "type": kind,
                    "param": null,
                    "code": null
                }
            })),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use super::*;
    use crate::{
        config::EngineConfig,
        engine::{EngineHandle, testing::DeterministicBackend},
    };

    fn test_app() -> Router {
        let metrics = Arc::new(Metrics::default());
        let engine = EngineHandle::spawn(
            DeterministicBackend::new("test-model"),
            EngineConfig {
                max_pending: 8,
                max_running: 2,
                max_batch_tokens: 16,
                prefill_chunk_tokens: 4,
                max_sequence_length: 128,
                kv_capacity_tokens: 128,
                output_buffer: 8,
            },
            8,
            Arc::clone(&metrics),
        )
        .unwrap();
        router(AppState { engine, metrics })
    }

    fn chat_body(stream: bool) -> Value {
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 2,
            "temperature": 0,
            "stream": stream,
            "stream_options": {"include_usage": true}
        })
    }

    #[tokio::test]
    async fn non_streaming_chat_is_openai_shaped() {
        let response = test_app()
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(chat_body(false).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["choices"][0]["message"]["content"], " token0 token1");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn streaming_chat_is_sse_and_terminates() {
        let response = test_app()
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(chat_body(true).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("\"role\":\"assistant\""));
        assert!(body.contains(" token0"));
        assert!(body.contains("\"completion_tokens\":2"));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn rejects_unknown_model_with_typed_openai_error() {
        let mut body = chat_body(false);
        body["model"] = Value::String("not-loaded".into());
        let response = test_app()
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn health_and_metrics_are_exposed() {
        let app = test_app();
        let health = app
            .clone()
            .oneshot(Request::get("/health/live").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);

        let metrics = app
            .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(metrics.status(), StatusCode::OK);
        let body = to_bytes(metrics.into_body(), 64 * 1024).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("tesseract_running_requests"));
        assert!(body.contains("tesseract_kv_tokens_used"));
    }
}
