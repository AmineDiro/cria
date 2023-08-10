use llm;
use serde::{Deserialize, Deserializer, Serialize};
use std::path::PathBuf;

/// The global configuration for cria components.
#[derive(Serialize, Deserialize)]
pub struct Config {
    #[serde(
        deserialize_with = "model_architecture_deserialize",
        default = "default_model_architecture"
    )]
    pub model_architecture: llm::ModelArchitecture,
    pub model_path: PathBuf,
    #[serde(default)]
    pub tokenizer_path: Option<PathBuf>,
    #[serde(default)]
    pub tokenizer_repository: Option<String>,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: usize,
    #[serde(default = "default_prefer_mmap")]
    pub prefer_mmap: bool,
    #[serde(default = "default_context_size")]
    pub context_size: usize,
    #[serde(default)]
    pub lora_adapters: Option<Vec<PathBuf>>,
    #[serde(default = "default_use_gpu")]
    pub use_gpu: bool,
    #[serde(default)]
    pub gpu_layers: Option<usize>,
    #[serde(default = "default_service_name")]
    pub service_name: String,
    #[serde(default)]
    pub zipkin_endpoint: Option<String>,
}

fn model_architecture_deserialize<'de, D>(
    deserializer: D,
) -> Result<llm::ModelArchitecture, D::Error>
where
    D: Deserializer<'de>,
{
    let maybe_model_architecture: Option<String> = Option::deserialize(deserializer)?;
    if maybe_model_architecture.is_none() {
        return Ok(llm::ModelArchitecture::Llama);
    }
    Ok(match maybe_model_architecture.unwrap() {
        case if case.eq_ignore_ascii_case("llama") => llm::ModelArchitecture::Llama,
        case if case.eq_ignore_ascii_case("gpt2") => llm::ModelArchitecture::Gpt2,
        case if case.eq_ignore_ascii_case("gptj") => llm::ModelArchitecture::GptJ,
        case if case.eq_ignore_ascii_case("gpt-neo-x") => llm::ModelArchitecture::GptNeoX,
        case if case.eq_ignore_ascii_case("mpt") => llm::ModelArchitecture::Mpt,
        case => panic!("Unknown model architecture: {}", case),
    })
}

impl Config {
    pub fn to_tokenizer_source(&self) -> llm::TokenizerSource {
        match (&self.tokenizer_path, &self.tokenizer_repository) {
            (Some(_), Some(_)) => {
                panic!("Cannot specify both --tokenizer-path and --tokenizer-repository");
            }
            (Some(path), None) => llm::TokenizerSource::HuggingFaceTokenizerFile(path.to_owned()),
            (None, Some(repo)) => llm::TokenizerSource::HuggingFaceRemote(repo.to_owned()),
            (None, None) => llm::TokenizerSource::Embedded,
        }
    }
    pub fn extract_model_params(&self) -> llm::ModelParameters {
        llm::ModelParameters {
            prefer_mmap: self.prefer_mmap,
            context_size: self.context_size,
            lora_adapters: self.lora_adapters.clone(),
            use_gpu: self.use_gpu,
            gpu_layers: self.gpu_layers,
            rope_overrides: None,
        }
    }
}

// config defaults

fn default_host() -> String {
    String::from("0.0.0.0")
}

fn default_port() -> usize {
    3000
}
fn default_use_gpu() -> bool {
    true
}
fn default_context_size() -> usize {
    2048
}
fn default_prefer_mmap() -> bool {
    true
}

fn default_model_architecture() -> llm::ModelArchitecture {
    llm::ModelArchitecture::Llama
}

fn default_service_name() -> String {
    String::from("cria")
}
