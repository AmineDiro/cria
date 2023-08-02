use crate::ModelList;
use axum::{extract::State, Json};
use serde::de;
use serde::{Deserialize, Deserializer};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

#[derive(Deserialize, Debug)]
enum LogitBias {
    InputIds,
    Tokens,
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
    model: Option<String>,
    n: Option<usize>,       // 1
    best_of: Option<usize>, // 1
    user: Option<String>,
}
fn default_max_tokens() -> usize {
    16
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

pub async fn get_models(State(state): State<ModelList>) -> Json<ModelList> {
    Json(state)
}

pub async fn completion(Json(request): Json<CompletionRequest>) {
    println!("Got request: {:?}", request)
}
