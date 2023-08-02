use axum::{
    routing::{get, post},
    Router,
};
use llm::{InferenceError, InferenceStats, Model, ModelParameters, TokenizerSource};
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, io::Write, path::PathBuf};
mod routes;
use routes::get_models;

use crate::routes::completion;

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelList {
    pub models: Vec<String>,
}

fn run_inference(model: &dyn Model) -> Result<InferenceStats, InferenceError> {
    let mut session = model.start_session(Default::default());
    let prompt = "Rust is a cool programming language because";

    session.infer::<Infallible>(
        model,
        &mut rand::thread_rng(),
        &llm::InferenceRequest {
            prompt: prompt.into(),
            parameters: &llm::InferenceParameters::default(),
            play_back_previous_tokens: false,
            maximum_token_count: None,
        },
        // OutputRequest
        &mut Default::default(),
        |r| match r {
            llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
                print!("{t}");
                std::io::stdout().flush().unwrap();

                Ok(llm::InferenceFeedback::Continue)
            }
            _ => Ok(llm::InferenceFeedback::Continue),
        },
    )
}

pub async fn run_webserver(
    model_architecture: llm::ModelArchitecture,
    model_path: PathBuf,
    tokenizer_source: TokenizerSource,
) {
    let now = std::time::Instant::now();

    let _model = llm::load_dynamic(
        Some(model_architecture),
        &model_path,
        tokenizer_source,
        // TODO : move to Args
        ModelParameters {
            prefer_mmap: true,
            context_size: 2048,
            lora_adapters: None,
            use_gpu: true,
            gpu_layers: Some(32),
            rope_overrides: None,
        },
        |_l| {},
    )
    .unwrap_or_else(|err| {
        panic!("Failed to load {model_architecture} model from {model_path:?}: {err}")
    });

    println!(
        "Model fully loaded! Elapsed: {}ms",
        now.elapsed().as_millis()
    );

    let model_list = ModelList {
        models: vec!["llama-2".into()],
    };

    let app = Router::new()
        .route("/v1/models", get(get_models))
        .with_state(model_list)
        .route("/v1/completions", post(completion));
    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
