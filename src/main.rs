mod cli;
mod error;
mod explore;
mod searchtor;
mod ollama;
mod tui;
mod wizard;
mod workflow;

use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use tracing_subscriber::{fmt, EnvFilter};

use cli::{Cli, Command};
use explore::{ExploreConfig, ExplorePipeline};
use searchtor::SearchtorPipeline;
use ollama::{EmbeddingsRequest, MockOllamaClient, OllamaClient, RealOllamaClient};
use workflow::StepEvent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // The TUI takes over the terminal for the real explore path, so we only
    // initialise the tracing subscriber when it won't collide with ratatui.
    // `New` never uses the TUI; other commands do unless --mock is set.
    let will_run_tui = matches!(cli.command, Command::Explore { .. } | Command::Searchtor { .. }) && !cli.mock;
    if !will_run_tui {
        fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .with_target(true)
            .compact()
            .init();
    }

    let client: Arc<dyn OllamaClient> = if cli.mock {
        tracing::info!("using mock Ollama client");
        Arc::new(MockOllamaClient::new(&cli.model))
    } else {
        Arc::new(RealOllamaClient::new(&cli.ollama_url))
    };

    match cli.command {
        Command::Explore { path } => {
            let config = ExploreConfig {
                model: cli.model.clone(),
                ..Default::default()
            };

            if cli.mock {
                // Mock / test path: plain stdout, no TUI.
                let pipeline = ExplorePipeline::new(Arc::clone(&client), config);
                let summary = pipeline
                    .run(&path)
                    .await
                    .context("explore pipeline failed")?;
                println!("\n=== Project Summary ===\n");
                println!("{}", summary);
            } else {
                // Real path: stream tokens into a live ratatui TUI.
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = ExplorePipeline::new(Arc::clone(&client), config).with_events(tx);

                // Run the pipeline in a background task so the TUI stays responsive.
                let path_clone = path.clone();
                let handle = tokio::spawn(async move { pipeline.run(&path_clone).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "explore".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                // Retrieve the final summary once the TUI has exited.
                let summary = handle.await.context("pipeline task panicked")??;

                println!("\n=== Project Summary ===\n");
                println!("{}", summary);
            }
        }

        Command::Searchtor { query, depth } => {
            let query = query.join(" ");
            if cli.mock {
                let pipeline = SearchtorPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_depth(depth);
                let result = pipeline.run(query).await.context("searchtor pipeline failed")?;
                println!("\n=== Searchtor ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = SearchtorPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_depth(depth)
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(query).await });

                tui::run(rx, cli.model.clone(), "searchtor".to_string(), Arc::clone(&client), cli.debug)
                    .await.context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Searchtor ===\n");
                println!("{}", result);
            }
        }
        Command::New { name } => {
            wizard::run(name)?;
        }

        Command::ListModels => {
            let tags = client
                .list_models()
                .await
                .context("failed to list models")?;
            if tags.models.is_empty() {
                println!("No models found.");
            } else {
                for m in &tags.models {
                    println!("{:40} {}", m.name, m.details.parameter_size);
                }
            }
        }

        Command::Embed { text } => {
            let req = EmbeddingsRequest {
                model: cli.model.clone(),
                prompt: text,
                options: None,
            };
            let resp = client
                .embeddings(req)
                .await
                .context("failed to get embeddings")?;
            println!("{:?}", resp.embedding);
        }
    }

    Ok(())
}
