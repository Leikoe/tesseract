use std::{convert::Infallible, time::SystemTime};

use async_stream::stream;
use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response, Sse, sse::Event},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::Instrument;
use uuid::Uuid;

use crate::{
    api::{ApiError, AppState},
    engine::{
        FinishReason, GenerateRequest, GenerationEvent, GenerationParams, SubmitError, Usage,
    },
};

const DEFAULT_MAX_TOKENS: usize = 128;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Stop {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(stop) => vec![stop],
            Self::Many(stop) => stop,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: StreamOptions,
    #[serde(default, alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<Stop>,
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

#[derive(Debug, Serialize)]
struct CompletionMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct Choice<'a> {
    index: usize,
    message: CompletionMessage<'a>,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize)]
struct ChatCompletion<'a> {
    id: String,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: Vec<Choice<'a>>,
    usage: Usage,
}

pub async fn models(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.engine.model_id(),
            "object": "model",
            "owned_by": "tesseract"
        }]
    }))
}

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    if request.model != state.engine.model_id() {
        return Err(ApiError::InvalidRequest(format!(
            "model `{}` is not loaded",
            request.model
        )));
    }
    let prompt = render_llama3_chat(&request.messages)?;
    let id = Uuid::now_v7();
    let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let params = GenerationParams {
        max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        seed: request.seed.unwrap_or_else(|| id.as_u128() as u64),
        stop: request.stop.map(Stop::into_vec).unwrap_or_default(),
    };
    params.validate().map_err(ApiError::InvalidRequest)?;

    tracing::info!(request_id = %id, max_tokens, stream = request.stream, "request admitted");
    let output = state
        .engine
        .try_generate(GenerateRequest { id, prompt, params })
        .map_err(map_submit_error)?;
    let span = tracing::info_span!("generation", request_id = %id);

    if request.stream {
        Ok(streaming_response(
            output,
            state.engine.model_id().to_owned(),
            request.stream_options.include_usage,
        )
        .instrument(span)
        .await)
    } else {
        non_streaming_response(output, state.engine.model_id().to_owned())
            .instrument(span)
            .await
    }
}

fn map_submit_error(error: SubmitError) -> ApiError {
    match error {
        SubmitError::Overloaded => ApiError::Overloaded,
        SubmitError::Unavailable => ApiError::Unavailable,
    }
}

async fn non_streaming_response(
    mut output: crate::engine::RequestStream,
    model: String,
) -> Result<Response, ApiError> {
    let id = format!("chatcmpl-{}", output.request_id().simple());
    let mut text = String::new();
    let (reason, usage) = loop {
        match output.recv().await {
            Some(GenerationEvent::Delta { text: delta, .. }) => text.push_str(&delta),
            Some(GenerationEvent::Finished { reason, usage }) => break (reason, usage),
            Some(GenerationEvent::Failed { message }) => return Err(ApiError::Generation(message)),
            None => return Err(ApiError::Unavailable),
        }
    };
    let response = ChatCompletion {
        id,
        object: "chat.completion",
        created: unix_timestamp(),
        model: &model,
        choices: vec![Choice {
            index: 0,
            message: CompletionMessage {
                role: "assistant",
                content: &text,
            },
            finish_reason: reason,
        }],
        usage,
    };
    Ok(Json(response).into_response())
}

async fn streaming_response(
    mut output: crate::engine::RequestStream,
    model: String,
    include_usage: bool,
) -> Response {
    let response_id = format!("chatcmpl-{}", output.request_id().simple());
    let created = unix_timestamp();
    let events = stream! {
        let first = json!({
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        });
        yield Ok::<_, Infallible>(Event::default().data(first.to_string()));

        while let Some(event) = output.recv().await {
            match event {
                GenerationEvent::Delta { text, .. } => {
                    let chunk = json!({
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
                    });
                    yield Ok(Event::default().data(chunk.to_string()));
                }
                GenerationEvent::Finished { reason, usage } => {
                    let final_chunk = json!({
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]
                    });
                    yield Ok(Event::default().data(final_chunk.to_string()));
                    if include_usage {
                        let usage_chunk = json!({
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": usage
                        });
                        yield Ok(Event::default().data(usage_chunk.to_string()));
                    }
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                GenerationEvent::Failed { message } => {
                    let error = json!({"error": {"message": message, "type": "server_error"}});
                    yield Ok(Event::default().data(error.to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
            }
        }
    };
    Sse::new(events).into_response()
}

fn render_llama3_chat(messages: &[Message]) -> Result<String, ApiError> {
    if messages.is_empty() {
        return Err(ApiError::InvalidRequest(
            "messages must contain at least one item".into(),
        ));
    }
    if messages.iter().any(|message| message.content.is_empty()) {
        return Err(ApiError::InvalidRequest(
            "message content must not be empty".into(),
        ));
    }

    let mut prompt = String::from("<|begin_of_text|>");
    for message in messages {
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(message.role.as_str());
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(&message.content);
        prompt.push_str("<|eot_id|>");
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    Ok(prompt)
}

fn unix_timestamp() -> u64 {
    SystemTime::UNIX_EPOCH
        .elapsed()
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_llama3_chat_template() {
        let prompt = render_llama3_chat(&[
            Message {
                role: Role::System,
                content: "Be terse.".into(),
            },
            Message {
                role: Role::User,
                content: "Hello".into(),
            },
        ])
        .unwrap();
        assert_eq!(
            prompt,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe terse.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn rejects_empty_messages() {
        assert!(render_llama3_chat(&[]).is_err());
    }
}
