use axum::{
    routing::{get, post},
    Extension, Router,
};
use llm::{InferenceError, InferenceStats, Model, ModelParameters, TokenizerSource};
use serde::de;
use serde::Serialize;
use serde::{Deserialize, Deserializer};
use std::marker::PhantomData;
use std::{convert::Infallible, io::Write, path::PathBuf};
use std::{fmt, sync::Arc};

use crate::routes::{completions::completions, models::get_models};
pub mod infer;
pub mod routes;

pub const N_SUPPORTED_MODELS: usize = 1;

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelList {
    pub models: [String; N_SUPPORTED_MODELS],
}

fn run_inference(model: Arc<dyn Model>) -> Result<InferenceStats, InferenceError> {
    let mut session = model.start_session(Default::default());
    let prompt = "Rust is a cool programming language because";

    session.infer::<Infallible>(
        &*model,
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
    model_params: ModelParameters,
) {
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

    println!(
        "Llama2 - fully loaded in: {}ms !",
        now.elapsed().as_millis()
    );
    // dbg!(run_inference(model.clone()));

    let model_list = ModelList {
        models: ["llama-2".into()],
    };

    let app = Router::new()
        .route("/v1/models", get(get_models))
        .with_state(model_list)
        .route("/v1/completions", post(completions))
        .layer(Extension(model));

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
