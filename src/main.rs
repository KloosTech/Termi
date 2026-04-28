mod cli;
mod error;
mod explore;
mod ollama;
mod workflow;

use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use tracing_subscriber::{fmt, EnvFilter};

use cli::{Cli, Command};
use explore::{ExploreConfig, ExplorePipeline};
use ollama::{EmbeddingsRequest, MockOllamaClient, OllamaClient, RealOllamaClient};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(true)
        .compact()
        .init();

    let cli = Cli::parse();

    let client: Arc<dyn OllamaClient> = if cli.mock {
        tracing::info!("using mock Ollama client");
        Arc::new(MockOllamaClient::new(&cli.model))
    } else {
        Arc::new(RealOllamaClient::new(&cli.ollama_url))
    };

    match cli.command {
        Command::Explore { path } => {
            let config = ExploreConfig { model: cli.model.clone(), ..Default::default() };
            let pipeline = ExplorePipeline::new(Arc::clone(&client), config);
            let summary = pipeline.run(&path).await.context("explore pipeline failed")?;

            println!("\n=== Project Summary ===\n");
            println!("{}", summary);
        }

        Command::ListModels => {
            let tags = client.list_models().await.context("failed to list models")?;
            if tags.models.is_empty() {
                println!("No models found.");
            } else {
                for m in &tags.models {
                    println!("{:40} {}", m.name, m.details.parameter_size);
                }
            }
        }

        Command::Embed { text } => {
            let req = EmbeddingsRequest { model: cli.model.clone(), prompt: text, options: None };
            let resp = client.embeddings(req).await.context("failed to get embeddings")?;
            println!("{:?}", resp.embedding);
        }
    }

    Ok(())
}
