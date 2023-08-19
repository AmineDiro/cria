use async_stream::stream;
use axum::extract::State;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{response::sse::Event, Json};
use futures::Stream;
use llm::samplers::build_sampler;
use llm::TokenUtf8Buffer;
use llm::{feed_prompt_callback, InferenceError, InferenceFeedback};
use serde::Deserialize;
use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

use crate::*;

/// Basically switches between completions and completions_stream methods
/// depending on request `stream` value
pub(crate) async fn compat_completions(
    model: State<Arc<dyn Model>>,
    request: Json<CompletionRequest>,
) -> Response {
    tracing::debug!(
        "Received request with streaming set to: {}",
        &request.stream
    );
    if !request.stream {
        completions(model.clone(), request).await.into_response()
    } else {
        completions_stream(model.clone(), request)
            .await
            .into_response()
    }
}

pub(crate) async fn completions_stream(
    State(model): State<Arc<dyn Model>>,
    Json(request): Json<CompletionRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut session: llm::InferenceSession = model.start_session(Default::default());
    let mut response_tokens: Vec<String> = Vec::new();
    let maximum_token_count = request.max_tokens.min(usize::MAX);

    let repetition_penalty_last_n = 512;
    let sampler_args = vec![
        format!("topk:k={}", request.top_k),
        format!("topp:p={}", request.top_p),
        format!(
            "repetition:penalty={}:last_n={}",
            request.repeat_penalty, repetition_penalty_last_n
        ),
        format!("temperature:temperature={}", request.temperature),
    ];
    let sampler = build_sampler(0, &[], &sampler_args).unwrap();
    let stream = stream! {
    let prompt = request.prompt.into_iter().collect::<String>();
    if !prompt.is_empty() {
        session
            .feed_prompt(
                &*model,
                llm::Prompt::Text(&prompt),
                &mut Default::default(),
                feed_prompt_callback::<_>(|r| match r {
                    llm::InferenceResponse::PromptToken(_) => {
                        Ok::<InferenceFeedback, InferenceError>(llm::InferenceFeedback::Continue)
                    }
                    llm::InferenceResponse::InferredToken(t) => {
                        let _ = &response_tokens.push(t);
                        Ok(llm::InferenceFeedback::Continue)
                    }
                    _ => Ok(llm::InferenceFeedback::Continue),
                }),
            )
            .unwrap();
    }
    // After the prompt is consumed, sample tokens by repeatedly calling
    // `infer_next_token`. We generate tokens until the model returns an
    // EndOfText token, or we run out of space in the context window,
    // or we reach the specified limit.
    let mut tokens_processed = 0;
    let mut token_utf8_buf = TokenUtf8Buffer::new();

    while tokens_processed < maximum_token_count {
        let token = match session.infer_next_token(
            &*model,
            &llm::InferenceParameters {
                    sampler: sampler.clone(),
                        },
            &mut Default::default(),
            &mut rand::thread_rng(),
        ) {
            Ok(token) => token,
            Err(InferenceError::EndOfText) => break,
            // TODO: Handle actual errors
            Err(_) => break,
        };
        tokens_processed += 1;
        // Buffer the token until it's valid UTF-8, then call the callback.
        if let Some(tokens) = token_utf8_buf.push(&token) {
            yield Ok(Event::default().json_data(CompletionResponse{
                id: format!("cmpl-{}", Uuid::new_v4().to_string()),
                object: "text.completion.chunk".to_string(),
                model:"llama-2".to_string(),
                choices: vec![ CompletionResponseChoices {
                            text: tokens,
                            index:0,
                            // TODO : Figure out what to return here
                            logprobs: None,
                            finish_reason: None,
                        }
                    ],
                usage: None,
            }).unwrap());
        }
    }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

pub(crate) async fn completions(
    State(model): State<Arc<dyn Model>>,
    Json(request): Json<CompletionRequest>,
) -> Json<CompletionResponse> {
    let mut session: llm::InferenceSession = model.start_session(Default::default());
    let mut response_tokens: Vec<String> = Vec::new();
    let prompt = request.prompt.into_iter().collect::<String>();

    // TODO : where is this used in LLAMA ?
    let repetition_penalty_last_n = 512;
    let sampler_args = vec![
        format!("topk:k={}", request.top_k),
        format!("topp:p={}", request.top_p),
        format!(
            "repetition:penalty={}:last_n={}",
            request.repeat_penalty, repetition_penalty_last_n
        ),
        format!("temperature:temperature={}", request.temperature),
    ];
    let sampler = build_sampler(0, &[], &sampler_args).unwrap();
    let stats = session
        .infer::<Infallible>(
            &*model,
            &mut rand::thread_rng(),
            &llm::InferenceRequest {
                prompt: llm::Prompt::Text(&prompt),
                parameters: &llm::InferenceParameters { sampler: sampler },
                play_back_previous_tokens: false,
                maximum_token_count: Some(request.max_tokens),
            },
            &mut Default::default(),
            |r| match r {
                llm::InferenceResponse::PromptToken(_) => Ok(llm::InferenceFeedback::Continue),
                llm::InferenceResponse::InferredToken(t) => {
                    let _ = &response_tokens.push(t);
                    Ok(llm::InferenceFeedback::Continue)
                }
                _ => Ok(llm::InferenceFeedback::Continue),
            },
        )
        .unwrap();
    Json(CompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4().to_string()),
        object: "text_completion".to_string(),
        model: "Llama-2".to_string(),
        choices: vec![CompletionResponseChoices {
            text: response_tokens.into_iter().collect::<String>(),
            index: 0,
            logprobs: None,
            finish_reason: Some(FinishReason::Length), // TODO: stop or length
        }],
        usage: Some(Usage {
            prompt_tokens: stats.prompt_tokens,
            completion_tokens: stats.predict_tokens,
            total_tokens: stats.prompt_tokens + stats.predict_tokens,
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
    prompt: Vec<String>,
    suffix: Option<String>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default = "default_microstat_mode")]
    mirostat_mode: usize,
    #[serde(default = "default_microstat_tau")]
    mirostat_tau: f32,
    #[serde(default = "default_microstat_eta")]
    mirostat_eta: f32,
    #[serde(default = "default_echo")]
    echo: bool,
    /// Whether to use SSE streaming with the stream terminated by a data: [DONE]
    #[serde(default = "default_stream")]
    stream: bool,
    stop: Option<Vec<String>>,
    logprobs: Option<usize>,
    #[serde(default = "default_presence_penalty")]
    presence_penalty: f32,
    #[serde(default = "default_frequence_penalty")]
    frequency_penalty: f32,
    logit_bias: Option<HashMap<String, f32>>,
    // llama.cpp specific parameters
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_repeat_penalty")]
    repeat_penalty: f32,
    logit_bias_type: Option<LogitBias>,
    // ignored or currently unsupported
    // Model name to use.
    model: Option<String>,
    // How many completions to generate for each prompt.
    n: Option<usize>,
    // Generates best_of completions server-side and returns the “best”.
    best_of: Option<usize>, // 1
    user: Option<String>,
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
    model: String,
    choices: Vec<CompletionResponseChoices>,
    usage: Option<Usage>,
}
