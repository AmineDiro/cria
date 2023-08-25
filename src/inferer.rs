use std::time::Duration;

use flume::{Receiver, Sender};
use llm::samplers::build_sampler;
use llm::Model;
use llm::TokenUtf8Buffer;
use llm::{feed_prompt_callback, InferenceError, InferenceFeedback};

use crate::routes::completions::CompletionRequest;

pub fn run_inference(model: Box<dyn Model>, rx_queue: Receiver<InferenceEvent>) {
    // runs loop to get event and do inference
    tracing::info!("Running model inference !");
    while let Ok(inference_request) = rx_queue.recv() {
        match inference_request {
            InferenceEvent::CompletionEvent(request, request_tx) => {
                let mut session: llm::InferenceSession = model.start_session(Default::default());
                let prompt = request.prompt.into_iter().collect::<String>();

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
                // First push the prompt to the model
                if !prompt.is_empty() {
                    session
                        .feed_prompt(
                            model.as_ref(),
                            llm::Prompt::Text(&prompt),
                            &mut Default::default(),
                            feed_prompt_callback::<_>(|r| match r {
                                llm::InferenceResponse::PromptToken(_) => {
                                    Ok::<InferenceFeedback, InferenceError>(
                                        llm::InferenceFeedback::Continue,
                                    )
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
                        model.as_ref(),
                        &llm::InferenceParameters {
                            sampler: sampler.clone(),
                        },
                        &mut Default::default(),
                        &mut rand::thread_rng(),
                    ) {
                        Ok(token) => token,
                        // TODO: Send an end of stream or end
                        Err(InferenceError::EndOfText) => break,
                        // TODO: Handle actual errors
                        Err(_) => break,
                    };
                    tokens_processed += 1;

                    // Buffer the token until it's valid UTF-8
                    if let Some(token) = token_utf8_buf.push(&token) {
                        let _res = request_tx
                            .send_timeout(
                                Ok(StreamingResponse { token }),
                                Duration::from_millis(10),
                            )
                            .unwrap();
                    }
                }
            }
            _ => {}
        }
    }
}

pub struct StreamingResponse {
    pub token: String,
}
pub enum InferenceEvent {
    CompletionEvent(
        CompletionRequest,
        Sender<Result<StreamingResponse, InferenceError>>,
    ),
    _EmbeddingEvent,
    _ChatEvent,
}

/// Requester holding
#[derive(Clone)]
pub struct RequestQueue {
    queue: Sender<InferenceEvent>,
}

impl RequestQueue {
    pub(crate) fn new(queue: Sender<InferenceEvent>) -> Self {
        Self { queue }
    }
    pub(crate) async fn append(&mut self, event: InferenceEvent) {
        // TODO: result
        self.queue.send_async(event).await.unwrap_or_else(|err| {
            panic!(
                "Can't append the event on inference queue. Err {:?}",
                err.to_string()
            )
        })
    }
    // fn generate_completion_stream() {}
    // fn generate_chat_stream() {}
    // fn generate_completion(&mut self) {}
    // fn generate_chat() {}
}
