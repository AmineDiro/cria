use axum::{extract::State, Json};

use crate::*;
use llm;

fn get_embeddings(
    model: &dyn llm::Model,
    inference_parameters: &llm::InferenceParameters,
    query: &str,
) -> (usize, Vec<f32>) {
    let mut session = model.start_session(Default::default());
    let mut output_request = llm::OutputRequest {
        all_logits: None,
        embeddings: Some(Vec::new()),
    };
    let vocab = model.tokenizer();
    let beginning_of_sentence = true;
    let query_token_ids = vocab
        .tokenize(query, beginning_of_sentence)
        .unwrap()
        .iter()
        .map(|(_, tok)| *tok)
        .collect::<Vec<_>>();
    model.evaluate(&mut session, &query_token_ids, &mut output_request);
    // TODO: return a result
    (query_token_ids.len(), output_request.embeddings.unwrap())
}

pub(crate) async fn embeddings<T: Serialize>(
    State(model): State<Arc<dyn Model>>,
    Json(request): Json<EmbeddingRequest>,
) -> Json<EmbeddingResponse<T>> {
    let inference_parameters = llm::InferenceParameters::default();

    let input = request.input.into_iter().collect::<String>();
    let (ntokens, embd) = get_embeddings(&*model, &inference_parameters, &input);
    Json(EmbeddingResponse {
        object: "list".to_string(),
        model: "llama-2".to_string(),
        data: EmbeddingData {
            object: "embedding".to_string(),
            index: 0,
            embedding: embd.into_iter().map(|e| T::from(e)).collect(),
        },
        usage: Usage {
            prompt_tokens: ntokens,
            total_tokens: ntokens,
        },
    })
}

#[derive(Deserialize, Debug)]
pub struct EmbeddingRequest {
    model: Option<String>,
    #[serde(default, deserialize_with = "string_or_seq_string")]
    input: Vec<String>,
    user: Option<String>,
}

#[derive(Serialize, Debug)]
struct EmbeddingData<T: Serialize> {
    object: String,
    index: usize,
    embedding: Vec<T>,
}
#[derive(Serialize, Debug)]
pub struct EmbeddingResponse<T: Serialize> {
    object: String,
    model: String,
    data: EmbeddingData<T>,
    usage: Usage,
}

#[derive(Serialize, Debug, Default)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}
