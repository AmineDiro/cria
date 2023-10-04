use async_stream::stream;
use axum::extract::State;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{response::sse::Event, Json};
use futures::Stream;
use serde::Deserialize;
use serde_json;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::inferer::{InferenceEvent, RequestQueue, StreamingResponse};
use crate::*;

/// Basically switches between completions and completions_stream methods
/// depending on request `stream` value
pub(crate) async fn compat_completions(
    queue: State<RequestQueue>,
    request: Json<CompletionRequest>,
) -> Response {
    tracing::debug!(
        "Received request with streaming set to: {}",
        &request.stream
    );
    if !request.stream {
        completions(queue, request).await.into_response()
    } else {
        completions_stream(queue, request).await.into_response()
    }
}

pub(crate) async fn completions_stream(
    State(mut queue): State<RequestQueue>,
    Json(request): Json<CompletionRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = flume::unbounded();
    let event = InferenceEvent::CompletionEvent(request, tx);
    let _ = queue.push(event).await;

    let stream = stream! {
            while let Ok(streaming_response) = rx.recv_async().await{
                // TODO : handle error
                if let Ok(StreamingResponse{token}) = streaming_response {
                let response= CompletionResponse{
                    id: format!("cmpl-{}", Uuid::new_v4()),
                    object: "text.completion.chunk".to_string(),
                    created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() ,
                    model:"llama-2".to_string(),
                    choices: vec![ CompletionResponseChoices {
                                text:token,
                                index:0,
                                logprobs: None,
                                finish_reason: None,
                            }
                        ],
                    usage: None,
                };
                // Note: this is entirely for openai python client to work
                // for some reason the python clients parses a SSE msg starting with b"data: " NOT b"data:"
                // This different from the http spec : https://developer.mozilla.org/en-US/docs/Web/API/MessageEvent/data
                let data = format!(" {}",serde_json::to_string(&response).unwrap());
                yield Ok(Event::default().data(&data));
                }
            }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

pub(crate) async fn completions(
    State(mut queue): State<RequestQueue>,
    Json(request): Json<CompletionRequest>,
) -> Json<CompletionResponse> {
    let (tx, rx) = flume::unbounded();
    let event = InferenceEvent::CompletionEvent(request, tx);
    let _ = queue.push(event).await;

    let mut response_tokens: Vec<String> = Vec::new();
    while let Ok(streaming_response) = rx.recv_async().await {
        // TODO : handle error
        if let Ok(StreamingResponse { token }) = streaming_response {
            let _ = &response_tokens.push(token);
        }
    }

    Json(CompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: "Llama-2".to_string(),
        choices: vec![CompletionResponseChoices {
            text: response_tokens.into_iter().collect::<String>(),
            index: 0,
            logprobs: None,
            finish_reason: Some(FinishReason::Length), // TODO: stop or length
        }],
        // TODO: figure out where to get this from : either the inferer or completions
        usage: Some(Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        }),
    })
}
#[derive(Deserialize, Debug)]
pub enum LogitBias {
    TokenIds,
    Tokens,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub struct CompletionRequest {
    #[serde(default, deserialize_with = "string_or_seq_string")]
    pub prompt: Vec<String>,
    suffix: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_microstat_mode")]
    pub mirostat_mode: usize,
    #[serde(default = "default_microstat_tau")]
    pub mirostat_tau: f32,
    #[serde(default = "default_microstat_eta")]
    pub mirostat_eta: f32,
    #[serde(default = "default_echo")]
    pub echo: bool,
    /// Whether to use SSE streaming with the stream terminated by a data: [DONE]
    #[serde(default = "default_stream")]
    pub stream: bool,
    pub stop: Option<Vec<String>>,
    pub logprobs: Option<usize>,
    #[serde(default = "default_presence_penalty")]
    pub presence_penalty: f32,
    #[serde(default = "default_frequence_penalty")]
    pub frequency_penalty: f32,
    pub logit_bias: Option<HashMap<String, f32>>,
    // llama.cpp specific parameters
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    pub logit_bias_type: Option<LogitBias>,
    // ignored or currently unsupported
    // Model name to use.
    pub model: Option<String>,
    // How many completions to generate for each prompt.
    pub n: Option<usize>,
    // Generates best_of completions server-side and returns the “best”.
    pub best_of: Option<usize>, // 1
    pub user: Option<String>,
}

#[derive(Serialize, Debug)]
pub enum FinishReason {
    Stop,
    Length,
}
#[derive(Serialize, Debug, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}
#[derive(Serialize, Debug)]
struct CompletionResponseChoices {
    text: String,
    index: usize,
    // TODO : Figure out what to return here
    logprobs: Option<()>,
    finish_reason: Option<FinishReason>,
}

#[derive(Serialize, Debug)]
pub struct CompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompletionResponseChoices>,
    usage: Option<Usage>,
}
