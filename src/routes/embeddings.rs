use axum::{extract::State, Json};

use crate::*;
use llm;

fn get_embeddings(model: &dyn llm::Model, query: &str) -> (usize, Vec<f32>) {
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

pub(crate) async fn embeddings(
    State(model): State<Arc<dyn Model>>,
    Json(request): Json<EmbeddingRequest>,
) -> Json<EmbeddingResponse> {
    // let input = request.input.into_iter().collect::<String>();

    let mut data = Vec::new();
    let mut ntokens = 0;
    for input in request.input {
        let (ntokens_input, embd) = get_embeddings(&*model, &input);
        ntokens += ntokens_input;
        data.push(EmbeddingData {
            object: "embedding".to_string(),
            index: 0,
            embedding: embd,
        })
    }
    Json(EmbeddingResponse {
        object: "list".to_string(),
        model: "llama-2".to_string(),
        data,
        usage: Usage {
            prompt_tokens: ntokens,
            total_tokens: ntokens,
        },
    })
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub struct EmbeddingRequest {
    model: Option<String>,
    #[serde(default, deserialize_with = "string_or_seq_string")]
    input: Vec<String>,
    user: Option<String>,
}

#[derive(Serialize, Debug)]
struct EmbeddingData {
    object: String,
    index: usize,
    embedding: Vec<f32>,
}
#[derive(Serialize, Debug)]
pub struct EmbeddingResponse {
    object: String,
    model: String,
    data: Vec<EmbeddingData>,
    usage: Usage,
}

#[derive(Serialize, Debug, Default)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}
