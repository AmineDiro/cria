use axum::extract::State;
use axum::Json;
use flume::Sender;
use llm::samplers::build_sampler;
use llm::{InferenceError, InferenceFeedback, InferenceResponse, Model};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::vec;
use std::{collections::HashMap, convert::Infallible};
use uuid::Uuid;

use super::completions::{FinishReason, LogitBias, Usage};
use crate::defaults::*;
use crate::inferer::{InferenceEvent, RequestQueue};

fn get_chat_history(messages: Vec<Message>) -> String {
    let mut history = String::new();
    for msg in messages {
        match msg.role {
            Role::System => {
                let ct = format!("SYSTEM: {}\n", msg.content);
                history.push_str(&ct)
            }
            Role::User => {
                let ct = format!("USER: {}\n", msg.content);
                history.push_str(&ct)
            }
            Role::Assistant => {
                let ct = format!("ASSISTANT: {}\n", msg.content);
                history.push_str(&ct)
            }
        }
    }
    history.push_str("ASSISTANT:");
    history
}

/// An [InferenceResponse] callback that will halt inference when a `stop_sequence` is generated.
pub fn chat_inference_callback<'a, E: std::error::Error + Send + Sync + 'static>(
    stop_sequence: &'a str,
    mut callback: impl FnMut(String) + 'a,
) -> impl FnMut(InferenceResponse) -> Result<InferenceFeedback, E> + 'a {
    let mut stop_sequence_buf = String::new();
    move |resp| match resp {
        InferenceResponse::InferredToken(token) => {
            // We've generated a token, so we need to check if it's contained in the stop sequence.
            let mut buf = stop_sequence_buf.clone();
            buf.push_str(&token);

            if buf.starts_with(stop_sequence) {
                // We've generated the stop sequence, so we're done.
                // Note that this will contain the extra tokens that were generated after the stop sequence,
                // which may affect generation. This is non-ideal, but it's the best we can do without
                // modifying the model.
                stop_sequence_buf.clear();
                return Ok(InferenceFeedback::Halt);
            } else if stop_sequence.starts_with(&buf) {
                // We've generated a prefix of the stop sequence, so we need to keep buffering.
                stop_sequence_buf = buf;
                return Ok(InferenceFeedback::Continue);
            }

            // We've generated a token that isn't part of the stop sequence, so we can
            // pass it to the callback.
            stop_sequence_buf.clear();
            callback(buf);
            Ok(InferenceFeedback::Continue)
        }
        InferenceResponse::EotToken => Ok(InferenceFeedback::Halt),
        _ => Ok(InferenceFeedback::Continue),
    }
}

pub fn chat_completion(
    model: &Box<dyn Model>,
    request: ChatCompletionRequest,
    request_tx: Sender<Result<ChatCompletionResponse, InferenceError>>,
) {
    let mut session = model.start_session(Default::default());

    // TODO: deal with result  error
    // Sampler are built using the following
    //`sampler_name:key1=value1:key2=value2`.
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

    let chat_history = get_chat_history(request.messages);
    tracing::debug!("Chat history : {}", &chat_history);
    let mut response_tokens: Vec<String> = Vec::new();
    // TODO : better error handling
    let stats = session
        .infer::<Infallible>(
            model.as_ref(),
            &mut rand::thread_rng(),
            &llm::InferenceRequest {
                prompt: llm::Prompt::Text(&chat_history),
                parameters: &llm::InferenceParameters { sampler: sampler },
                play_back_previous_tokens: false,
                maximum_token_count: Some(request.max_tokens),
            },
            &mut Default::default(),
            chat_inference_callback("USER", |r| response_tokens.push(r)),
        )
        .unwrap();

    let response = ChatCompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4().to_string()),
        object: "text_completion".to_string(),
        model: "Llama-2".to_string(),
        choices: vec![ChatCompletionResponseChoices {
            index: 0,
            finish_reason: Some(FinishReason::Length), // TODO: stop or length
            message: Message {
                role: Role::Assistant,
                content: response_tokens.into_iter().collect::<String>(),
            },
        }],
        usage: Some(Usage {
            prompt_tokens: stats.prompt_tokens,
            completion_tokens: stats.predict_tokens,
            total_tokens: stats.prompt_tokens + stats.predict_tokens,
        }),
    };

    // TODO: deal with sending error
    match request_tx.send_timeout(Ok(response), Duration::from_millis(100)) {
        Ok(_) => {}
        Err(_) => {
            tracing::info!("client closed channel")
        }
    }
}

pub(crate) async fn chat_completion_route(
    State(mut queue): State<RequestQueue>,
    Json(request): Json<ChatCompletionRequest>,
) -> Json<ChatCompletionResponse> {
    let (tx, rx) = flume::unbounded();
    let event = InferenceEvent::ChatEvent(request, tx);
    let _ = queue.push(event).await;
    // TODO : deal with errors
    let chat_response = rx.recv_async().await.unwrap();
    Json(chat_response.unwrap())
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}
#[derive(Serialize, Deserialize, Debug)]
struct Message {
    role: Role,
    content: String,
}

#[derive(Serialize, Debug)]
struct ChatCompletionResponseChoices {
    index: usize,
    message: Message,
    finish_reason: Option<FinishReason>,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<ChatCompletionResponseChoices>,
    usage: Option<Usage>,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub struct ChatCompletionRequest {
    messages: Vec<Message>,
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
    /// Whether to use SSE streaming with the stream terminated by a data: [DONE]
    #[serde(default = "default_stream")]
    stream: bool,
    stop: Option<Vec<String>>,
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

    /// ignored or currently unsupported
    model: Option<String>,
    // How many completions to generate for each prompt.
    n: Option<usize>,
    // Generates best_of completions server-side and returns the “best”.
    user: Option<String>,
}
