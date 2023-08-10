use clap::Parser;
use cria::run_webserver;
use figment::{
    providers::{
        Env,
        Serialized,
    },
    Figment,
};
pub mod cli;
use cli::Args;
pub mod config;
use cria::config::Config;

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // hierarchical config. cli args override Env vars 
    let config: Config = Figment::new()
        .merge(Env::prefixed("APP_"))
        .merge(Serialized::defaults(args))
        .extract().unwrap();

    run_webserver(
        config
    )
    .await;
}
