use axum::{Extension, Json};
use llm::{samplers::TopPTopK, TokenBias};
use serde::Deserialize;
use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

use crate::*;

pub async fn completions(
    Extension(model): Extension<Arc<dyn Model>>,
    Json(request): Json<CompletionRequest>,
) -> Json<CompletionResponse> {
    let mut session: llm::InferenceSession = model.start_session(Default::default());
    let mut response_tokens: Vec<String> = Vec::new();
    // TODO : batch this
    dbg!(&request);
    for prompt in request.prompt {
        dbg!(&prompt);
        let _ = session.infer::<Infallible>(
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
                llm::InferenceResponse::PromptToken(t)
                | llm::InferenceResponse::InferredToken(t) => {
                    let _ = &response_tokens.push(t);
                    Ok(llm::InferenceFeedback::Continue)
                }
                _ => Ok(llm::InferenceFeedback::Continue),
            },
        );
    }
    println!("{:?}", response_tokens);
    Json(CompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4().to_string()),
        object: "text_completion".to_string(),
        model: "Llama-2".to_string(),
        choices: vec![CompletionResponseChoices {
            text: response_tokens.into_iter().collect::<String>(),
            index: 0,
            logprobs: None,
            finish_reason: FinishReason::Length,
        }],
    })
}
#[derive(Deserialize, Debug)]
enum LogitBias {
    TokenIds,
    Tokens,
}

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

// {
//                         "id": completion_id,
//                         "object": "text_completion",
//                         "created": created,
//                         "model": model_name,
//                         "choices": [
//                             {
//                                 "text": self.detokenize([token]).decode(
//                                     "utf-8", errors="ignore"
//                                 ),
//                                 "index": 0,
//                                 "logprobs": logprobs_or_none,
//                                 "finish_reason": "length",
//                             }
//                         ],
//                     }

#[derive(Serialize, Debug)]
enum FinishReason {
    Stop,
    Length,
}
#[derive(Serialize, Debug)]
struct CompletionResponseChoices {
    text: String,
    index: usize,
    // TODO : Figure out what to return here
    logprobs: Option<()>,
    finish_reason: FinishReason,
}

#[derive(Serialize, Debug)]
pub struct CompletionResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<CompletionResponseChoices>,
}
