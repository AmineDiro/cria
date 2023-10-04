use std::time::Duration;

use flume::{Receiver, Sender};
use llm::samplers::build_sampler;
use llm::InferenceSession;
use llm::Model;
use llm::TokenUtf8Buffer;
use llm::Tokenizer;
use llm::{feed_prompt_callback, InferenceError, InferenceFeedback};

use crate::routes::chat::chat_completion;
use crate::routes::chat::ChatCompletionRequest;
use crate::routes::chat::ChatCompletionResponse;
use crate::routes::completions::CompletionRequest;
use crate::routes::embeddings::Embedding;
use crate::routes::embeddings::EmbeddingRequest;

fn stream_completion(
    model: &dyn Model,
    request: CompletionRequest,
    request_tx: Sender<Result<StreamingResponse, InferenceError>>,
) {
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
                model,
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
            model,
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

        if let Some(token) = token_utf8_buf.push(&token) {
            match request_tx
                .send_timeout(Ok(StreamingResponse { token }), Duration::from_millis(100))
            {
                Ok(_) => {}
                Err(_) => {
                    // Stop generating this response as soon as error occurs
                    tracing::info!("client closed channel");
                    return;
                }
            };
        }
    }
}

fn embed_string(
    model: &dyn Model,
    session: &mut InferenceSession,
    vocab: &Tokenizer,
    query: &str,
) -> Embedding {
    let mut output_request = llm::OutputRequest {
        all_logits: None,
        embeddings: Some(Vec::new()),
    };
    let beginning_of_sentence = true;
    let query_token_ids = vocab
        .tokenize(query, beginning_of_sentence)
        .unwrap()
        .iter()
        .map(|(_, tok)| *tok)
        .collect::<Vec<_>>();
    model.evaluate(&mut *session, &query_token_ids, &mut output_request);

    Embedding {
        ntokens: query_token_ids.len(),
        embedding: output_request.embeddings,
    }
}

fn stream_embedding(
    model: &dyn Model,
    request: EmbeddingRequest,
    request_tx: Sender<Result<Embedding, InferenceError>>,
) {
    let mut session = model.start_session(Default::default());
    let vocab = model.tokenizer();

    for input in request.input {
        let embd = embed_string(model, &mut session, vocab, &input);
        request_tx
            .send_timeout(Ok(embd), Duration::from_millis(10))
            .unwrap();
    }
}

pub fn inference_loop(model: Box<dyn Model>, rx_queue: Receiver<InferenceEvent>) {
    // Runs inference loop :  match on event and stream inference
    tracing::info!("Running model inference thread !");
    while let Ok(inference_request) = rx_queue.recv() {
        match inference_request {
            InferenceEvent::CompletionEvent(request, request_tx) => {
                stream_completion(model.as_ref(), request, request_tx)
            }
            InferenceEvent::EmbeddingEvent(request, request_tx) => {
                stream_embedding(model.as_ref(), request, request_tx)
            }
            InferenceEvent::ChatEvent(request, request_tx) => {
                chat_completion(model.as_ref(), request, request_tx)
            }
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
    EmbeddingEvent(EmbeddingRequest, Sender<Result<Embedding, InferenceError>>),
    ChatEvent(
        ChatCompletionRequest,
        Sender<Result<ChatCompletionResponse, InferenceError>>,
    ),
}

/// Multi producer sender queue to put requests
#[derive(Clone)]
pub struct RequestQueue {
    queue: Sender<InferenceEvent>,
}

impl RequestQueue {
    pub(crate) fn new(queue: Sender<InferenceEvent>) -> Self {
        Self { queue }
    }
    pub(crate) async fn push(&mut self, event: InferenceEvent) {
        // TODO: deal with result
        self.queue.send_async(event).await.unwrap_or_else(|err| {
            panic!(
                "Can't append the event on inference queue. Err {:?}",
                err.to_string()
            )
        })
    }
}
