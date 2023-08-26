use axum::{extract::State, Json};

use crate::{inferer::InferenceEvent, *};
use flume;

pub(crate) async fn embeddings(
    State(mut queue): State<RequestQueue>,
    Json(request): Json<EmbeddingRequest>,
) -> Json<EmbeddingResponse> {
    let (tx, rx) = flume::unbounded();

    let event = InferenceEvent::EmbeddingEvent(request, tx);
    let _ = queue.append(event).await;

    let mut data = Vec::new();
    let mut ntokens = 0;

    while let Ok(response) = rx.recv_async().await {
        match response {
            Ok((response_ntokens, emb)) => {
                ntokens += response_ntokens;

                data.push(EmbeddingData {
                    object: "embedding".to_string(),
                    index: 0,
                    embedding: emb.unwrap_or(Vec::new()),
                })
            }
            Err(_e) => {
                //TODO!
                tracing::error!("Error generating embedding ")
            }
        }
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
    pub model: Option<String>,
    #[serde(default, deserialize_with = "string_or_seq_string")]
    pub input: Vec<String>,
    pub user: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct EmbeddingData {
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
