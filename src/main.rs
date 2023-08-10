use clap::Parser;
use cria::run_webserver;
use llm::ModelParameters;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    model_architecture: llm::ModelArchitecture,
    model_path: PathBuf,
    #[arg(long, short = 'v')]
    pub tokenizer_path: Option<PathBuf>,
    #[arg(long, short = 'r')]
    pub tokenizer_repository: Option<String>,

    #[arg(long, short, default_value_t = String::from("0.0.0.0"))]
    pub host: String,
    #[arg(long, short, default_value_t = 3000)]
    pub port: usize,
    #[arg(long, short, default_value_t = true)]
    pub prefer_mmap: bool,
    #[arg(long, short, default_value_t = 2048)]
    pub context_size: usize,
    #[arg(long, short)]
    pub lora_adapters: Option<Vec<PathBuf>>,
    #[arg(long, short, default_value_t = true)]
    pub use_gpu: bool,
    #[arg(long, short)]
    pub gpu_layers: Option<usize>,
}
impl Args {
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
    fn extract_model_params(&self) -> llm::ModelParameters {
        ModelParameters {
            prefer_mmap: self.prefer_mmap,
            context_size: self.context_size,
            lora_adapters: self.lora_adapters.clone(),
            use_gpu: self.use_gpu,
            gpu_layers: self.gpu_layers,
            rope_overrides: None,
        }
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let tokenizer_source = args.to_tokenizer_source();
    let model_params = args.extract_model_params();
    let Args {
        model_architecture,
        model_path,
        host,
        port,
        ..
    } = args;

    run_webserver(
        model_architecture,
        model_path,
        tokenizer_source,
        model_params,
        host,
        port
    )
    .await;
}
