use async_stream::stream;
use axum::extract::State;
use axum::response::sse::{KeepAlive, Sse};
use axum::{response::sse::Event, Json};
use futures::Stream;
use llm::TokenUtf8Buffer;
use llm::{feed_prompt_callback, samplers::TopPTopK, InferenceError, InferenceFeedback, TokenBias};
use serde::Deserialize;
use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

use crate::*;

pub async fn completions_stream(
    State(model): State<Arc<dyn Model>>,
    Json(request): Json<CompletionRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut session: llm::InferenceSession = model.start_session(Default::default());
    let mut response_tokens: Vec<String> = Vec::new();
    let maximum_token_count = request.max_tokens.min(usize::MAX);

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
                sampler: Arc::new(TopPTopK {
                    top_k: request.top_k,
                    top_p: request.top_p,
                    repeat_penalty: request.repeat_penalty,
                    temperature: request.temperature,
                    bias_tokens: TokenBias::empty(),
                    repetition_penalty_last_n: 512, // TODO : where is this used in LLAMA ?
                }),
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

pub async fn completions(
    State(model): State<Arc<dyn Model>>,
    Json(request): Json<CompletionRequest>,
) -> Json<CompletionResponse> {
    let mut session: llm::InferenceSession = model.start_session(Default::default());
    let mut response_tokens: Vec<String> = Vec::new();
    let prompt = request.prompt.into_iter().collect::<String>();
    let stats = session
        .infer::<Infallible>(
            &*model,
            &mut rand::thread_rng(),
            &llm::InferenceRequest {
                prompt: llm::Prompt::Text(&prompt),
                parameters: &llm::InferenceParameters {
                    sampler: Arc::new(TopPTopK {
                        top_k: request.top_k,
                        top_p: request.top_p,
                        repeat_penalty: request.repeat_penalty,
                        temperature: request.temperature,
                        bias_tokens: TokenBias::empty(),
                        repetition_penalty_last_n: 512, // TODO : where is this used in LLAMA ?
                    }),
                },
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
        usage: Some(CompletionResponseUsage {
            prompt_tokens: stats.prompt_tokens,
            completion_tokens: stats.predict_tokens,
            total_tokens: stats.prompt_tokens + stats.predict_tokens,
        }),
    })
}
#[derive(Deserialize, Debug)]
enum LogitBias {
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
fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    0.8
}
fn default_top_p() -> f32 {
    0.95
}
fn default_stream() -> bool {
    false
}
fn default_echo() -> bool {
    false
}
fn default_top_k() -> usize {
    40
}
fn default_repeat_penalty() -> f32 {
    1.1
}
fn default_presence_penalty() -> f32 {
    0.0
}
fn default_frequence_penalty() -> f32 {
    0.0
}
fn default_microstat_mode() -> usize {
    0
}
fn default_microstat_tau() -> f32 {
    5.0
}
fn default_microstat_eta() -> f32 {
    0.1
}

#[derive(Serialize, Debug)]
enum FinishReason {
    Stop,
    Length,
}
#[derive(Serialize, Debug, Default)]
struct CompletionResponseUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
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
    usage: Option<CompletionResponseUsage>,
}
// {
//   "choices": [
//     {
//       "delta": {
//         "role": "assistant"
//       },
//       "finish_reason": null,
//       "index": 0
//     }
//   ],
//   "created": 1677825464,
//   "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
//   "model": "gpt-3.5-turbo-0301",
//   "object": "chat.completion.chunk"
// }
