use clap::Parser;
use cria::run_webserver;
use figment::{
    providers::{Env, Serialized},
    Figment,
};
use opentelemetry::global;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
pub mod cli;
use cli::Args;
pub mod config;
use cria::config::Config;

#[tokio::main]
async fn main() {
    // let args = Args::parse();

    // hierarchical config. cli args override Env vars
    let config: Config = Figment::new()
        .merge(Env::prefixed("APP_"))
        .merge(Serialized::defaults(Args::parse()))
        .extract()
        .unwrap();

    // setup opentelemetry
    opentelemetry::global::set_text_map_propagator(opentelemetry_zipkin::Propagator::new());

    let subscriber = tracing_subscriber::fmt::layer().json();

    let level = EnvFilter::new("debug".to_owned());

    let registry = tracing_subscriber::registry().with(subscriber).with(level);

    if config.zipkin_endpoint.is_some() {
        let host = config.host.clone();
        let port = config.port;
        let zipkin_endpoint = config.zipkin_endpoint.clone().unwrap();
        let tracer = opentelemetry_zipkin::new_pipeline()
            .with_service_name(config.service_name.clone())
            .with_service_address(format!("{host}:{port}").parse().unwrap())
            .with_collector_endpoint(zipkin_endpoint)
            .install_batch(opentelemetry::runtime::Tokio)
            .expect("unable to install zipkin tracer");
        let tracer = tracing_opentelemetry::layer().with_tracer(tracer.clone());

        registry.with(tracer).init();
    } else {
        registry.init();
    }

    run_webserver(config).await;
    global::shutdown_tracer_provider();
}
