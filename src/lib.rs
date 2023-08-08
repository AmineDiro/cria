use axum::{
    routing::{get, post},
    Router,
};
use llm::{Model, ModelParameters, TokenizerSource};
use serde::de;
use serde::Serialize;
use serde::{Deserialize, Deserializer};
use std::marker::PhantomData;
use std::{convert::Infallible, path::PathBuf};
use std::{fmt, sync::Arc};
use tower_http::trace::{self, TraceLayer};

pub mod defaults;
use defaults::*;

use crate::routes::{
    chat::chat_completion,
    completions::{compat_completions, completions, completions_stream},
    embeddings::embeddings,
    models::get_models,
};
pub mod routes;

pub const N_SUPPORTED_MODELS: usize = 1;

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelList {
    pub models: [String; N_SUPPORTED_MODELS],
}

pub async fn run_webserver(
    model_architecture: llm::ModelArchitecture,
    model_path: PathBuf,
    tokenizer_source: TokenizerSource,
    model_params: ModelParameters,
) {
    tracing_subscriber::fmt().init();
    let now = std::time::Instant::now();

    let model: Arc<dyn Model> = Arc::from(
        llm::load_dynamic(
            Some(model_architecture),
            &model_path,
            tokenizer_source,
            model_params,
            |_l| {},
        )
        .unwrap_or_else(|err| {
            panic!("Failed to load {model_architecture} model from {model_path:?}: {err}")
        }),
    );

    tracing::info!(
        "Llama2 - fully loaded in: {}ms !",
        now.elapsed().as_millis()
    );

    let model_list = ModelList {
        models: ["llama-2".into()],
    };

    let app = Router::new()
        .route("/v1/models", get(get_models))
        .with_state(model_list)
        .route("/v1/chat/completions", post(chat_completion))
        .route("/v1/completions", post(compat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/completions_full", post(completions))
        .route("/v1/completions_stream", post(completions_stream))
        .with_state(model)
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(trace::DefaultMakeSpan::new().level(tracing::Level::INFO))
                .on_response(trace::DefaultOnResponse::new().level(tracing::Level::INFO))
                .on_request(trace::DefaultOnRequest::new().level(tracing::Level::INFO)),
        );

    // TODO : add port to clap
    tracing::info!("listening on :3000");
    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}

fn string_or_seq_string<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringOrVec(PhantomData<Vec<String>>);

    impl<'de> de::Visitor<'de> for StringOrVec {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("string or list of strings")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(vec![value.to_owned()])
        }

        fn visit_seq<S>(self, visitor: S) -> Result<Self::Value, S::Error>
        where
            S: de::SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(visitor))
        }
    }

    deserializer.deserialize_any(StringOrVec(PhantomData))
}
