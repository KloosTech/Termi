mod cli;
mod error;
mod explore;
mod lidarr;
mod media;
mod ollama;
mod radarr;
mod searchtor;
mod sonarr;
mod tui;
mod wizard;
mod workflow;

use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use tracing_subscriber::{fmt, EnvFilter};

use dialoguer::{theme::ColorfulTheme, Select};

use cli::{Cli, Command};
use explore::{ExploreConfig, ExplorePipeline};
use lidarr::LidarrPipeline;
use ollama::{EmbeddingsRequest, MockOllamaClient, OllamaClient, RealOllamaClient};
use radarr::RadarrPipeline;
use searchtor::SearchtorPipeline;
use sonarr::SonarrPipeline;
use workflow::StepEvent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // The TUI takes over the terminal for the real explore path, so we only
    // initialise the tracing subscriber when it won't collide with ratatui.
    // `New` never uses the TUI; other commands do unless --mock is set.
    let will_run_tui = matches!(
        cli.command,
        Command::Explore { .. } | Command::Searchtor { .. }
    ) && !cli.mock;
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
            let vault_path = "/Users/jack/Library/Mobile Documents/iCloud~md~obsidian/Documents/MainVault/Personal/Knowlege";
            if cli.mock {
                let pipeline = SearchtorPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_depth(depth)
                    .with_vault(vault_path);
                let result = pipeline
                    .run(query)
                    .await
                    .context("searchtor pipeline failed")?;
                println!("\n=== Searchtor ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = SearchtorPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_depth(depth)
                    .with_vault(vault_path)
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(query).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "searchtor".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Searchtor ===\n");
                println!("{}", result);
            }
        }
        Command::Sonarr {
            query,
            url,
            api_key,
        } => {
            let query_str = query.join(" ");
            let pipeline = SonarrPipeline::new(Arc::clone(&client), cli.model.clone());
            let output = pipeline
                .run(&query_str, &url, &api_key)
                .await
                .context("sonarr pipeline failed")?;

            if output.results.is_empty() {
                println!("No results found for \"{}\".", output.corrected_query);
                return Ok(());
            }

            let labels: Vec<&str> = output.results.iter().map(|r| r.display.as_str()).collect();
            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt(format!(
                    "Results for \"{}\" — select to add",
                    output.corrected_query
                ))
                .items(&labels)
                .interact_opt()
                .context("selection failed")?;

            let idx = match selection {
                None => return Ok(()),
                Some(i) => i,
            };
            let item = &output.results[idx];
            if item.already_added {
                println!("\"{}\" is already in Sonarr.", item.display);
                return Ok(());
            }
            media::post_add_media(&url, &api_key, "/api/v3/series", &item.raw)
                .await
                .context("failed to add to Sonarr")?;
            println!("✓  Added \"{}\" to Sonarr.", item.display);
        }

        Command::Radarr {
            query,
            url,
            api_key,
        } => {
            let query_str = query.join(" ");
            let pipeline = RadarrPipeline::new(Arc::clone(&client), cli.model.clone());
            let output = pipeline
                .run(&query_str, &url, &api_key)
                .await
                .context("radarr pipeline failed")?;

            if output.results.is_empty() {
                println!("No results found for \"{}\".", output.corrected_query);
                return Ok(());
            }

            let labels: Vec<&str> = output.results.iter().map(|r| r.display.as_str()).collect();
            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt(format!(
                    "Results for \"{}\" — select to add",
                    output.corrected_query
                ))
                .items(&labels)
                .interact_opt()
                .context("selection failed")?;

            let idx = match selection {
                None => return Ok(()),
                Some(i) => i,
            };
            let item = &output.results[idx];
            if item.already_added {
                println!("\"{}\" is already in Radarr.", item.display);
                return Ok(());
            }
            media::post_add_media(&url, &api_key, "/api/v3/movie", &item.raw)
                .await
                .context("failed to add to Radarr")?;
            println!("✓  Added \"{}\" to Radarr.", item.display);
        }

        Command::Lidarr {
            query,
            url,
            api_key,
        } => {
            let query_str = query.join(" ");
            let pipeline = LidarrPipeline::new(Arc::clone(&client), cli.model.clone());
            let output = pipeline
                .run(&query_str, &url, &api_key)
                .await
                .context("lidarr pipeline failed")?;

            if output.results.is_empty() {
                println!("No results found for \"{}\".", output.corrected_query);
                return Ok(());
            }

            let labels: Vec<&str> = output.results.iter().map(|r| r.display.as_str()).collect();
            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt(format!(
                    "Results for \"{}\" — select to add",
                    output.corrected_query
                ))
                .items(&labels)
                .interact_opt()
                .context("selection failed")?;

            let idx = match selection {
                None => return Ok(()),
                Some(i) => i,
            };
            let item = &output.results[idx];
            if item.already_added {
                println!("\"{}\" is already in Lidarr.", item.display);
                return Ok(());
            }
            media::post_add_media(&url, &api_key, "/api/v1/artist", &item.raw)
                .await
                .context("failed to add to Lidarr")?;
            println!("✓  Added \"{}\" to Lidarr.", item.display);
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
