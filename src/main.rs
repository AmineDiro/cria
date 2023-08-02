use clap::Parser;
use cria::run_webserver;
use std::error::Error;
use std::{convert::Infallible, io::Write, path::PathBuf, sync::TryLockError};

#[derive(Parser)]
struct Args {
    model_architecture: llm::ModelArchitecture,
    model_path: PathBuf,
    #[arg(long, short = 'v')]
    pub tokenizer_path: Option<PathBuf>,
    #[arg(long, short = 'r')]
    pub tokenizer_repository: Option<String>,
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
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let tokenizer_source = args.to_tokenizer_source();
    let Args {
        model_architecture,
        model_path,
        tokenizer_path,
        tokenizer_repository,
    } = args;
    run_webserver(model_architecture, model_path, tokenizer_source).await;
}
