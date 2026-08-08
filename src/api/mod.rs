mod openai;

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, State},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
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
        .route("/v1/requests/{request_id}", delete(cancel_request))
        .with_state(state)
}

async fn cancel_request(
    State(state): State<AppState>,
    Path(request_id): Path<String>,
) -> Result<(StatusCode, Json<serde_json::Value>), ApiError> {
    let request_id = request_id.strip_prefix("chatcmpl-").unwrap_or(&request_id);
    let request_id = uuid::Uuid::parse_str(request_id)
        .map_err(|_| ApiError::InvalidRequest("request_id is not a valid UUID".into()))?;
    state
        .engine
        .try_cancel(request_id)
        .map_err(|error| match error {
            crate::engine::SubmitError::Overloaded => ApiError::Overloaded,
            crate::engine::SubmitError::Unavailable => ApiError::Unavailable,
        })?;
    Ok((
        StatusCode::ACCEPTED,
        Json(json!({"id": request_id, "cancelled": true})),
    ))
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
        engine::{EngineHandle, testing::DeterministicExecutor},
    };

    fn test_app_with(
        executor: DeterministicExecutor,
        config: EngineConfig,
    ) -> (Router, Arc<Metrics>) {
        let metrics = Arc::new(Metrics::default());
        let engine = EngineHandle::spawn(executor, config, 8, Arc::clone(&metrics)).unwrap();
        (
            router(AppState {
                engine,
                metrics: Arc::clone(&metrics),
            }),
            metrics,
        )
    }

    fn test_app() -> Router {
        test_app_with(
            DeterministicExecutor::new("test-model"),
            EngineConfig {
                max_pending: 8,
                max_running: 2,
                max_batch_tokens: 16,
                prefill_chunk_tokens: 4,
                max_sequence_length: 128,
                kv_capacity_tokens: 128,
                output_buffer: 8,
            },
        )
        .0
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
    async fn rejects_empty_messages_with_typed_openai_error() {
        let mut body = chat_body(false);
        body["messages"] = json!([]);
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
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("messages must not be empty")
        );
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

    #[tokio::test]
    async fn overload_returns_openai_rate_limit_error() {
        let (app, metrics) = test_app_with(
            DeterministicExecutor::new("test-model")
                .with_submission_delay(std::time::Duration::from_millis(50)),
            EngineConfig {
                max_pending: 0,
                max_running: 1,
                max_batch_tokens: 4,
                prefill_chunk_tokens: 1,
                max_sequence_length: 128,
                kv_capacity_tokens: 128,
                output_buffer: 8,
            },
        );
        let mut first_body = chat_body(false);
        first_body["max_tokens"] = Value::from(4);
        let first = tokio::spawn(
            app.clone().oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(first_body.to_string()))
                    .unwrap(),
            ),
        );
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while !metrics
                .prometheus()
                .contains("tesseract_running_requests 1")
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(chat_body(false).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "rate_limit_error");
        assert_eq!(first.await.unwrap().unwrap().status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn explicit_cancel_route_is_typed_and_validated() {
        let app = test_app();
        let request_id = uuid::Uuid::now_v7();
        let accepted = app
            .clone()
            .oneshot(
                Request::delete(format!("/v1/requests/chatcmpl-{}", request_id.simple()))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::ACCEPTED);
        let body = to_bytes(accepted.into_body(), 64 * 1024).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["id"], request_id.to_string());
        assert_eq!(body["cancelled"], true);

        let invalid = app
            .oneshot(
                Request::delete("/v1/requests/not-a-uuid")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
    }
}
