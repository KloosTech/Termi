mod auto_docs;
mod cli;
mod competitive;
mod changelog;
mod dead_code;
mod dep_audit;
mod deploy_check;
mod error;
mod error_detective;
mod explore;
mod gen_tests;
mod log_anomaly;
mod migration;
mod onboard;
mod refactor;
mod review;
mod searchtor;
mod tech_radar;
mod ollama;
mod tui;
mod wizard;
mod workflow;

use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use tracing_subscriber::{fmt, EnvFilter};

use auto_docs::AutoDocsPipeline;
use cli::{Cli, Command};
use competitive::CompetitivePipeline;
use changelog::ChangelogPipeline;
use dead_code::DeadCodePipeline;
use dep_audit::DepAuditPipeline;
use deploy_check::DeployCheckPipeline;
use error_detective::ErrorDetectivePipeline;
use explore::{ExploreConfig, ExplorePipeline};
use gen_tests::GenTestsPipeline;
use log_anomaly::LogAnomalyPipeline;
use migration::MigrationPipeline;
use onboard::OnboardPipeline;
use refactor::RefactorPipeline;
use review::ReviewPipeline;
use searchtor::SearchtorPipeline;
use tech_radar::TechRadarPipeline;
use ollama::{EmbeddingsRequest, MockOllamaClient, OllamaClient, RealOllamaClient};
use workflow::StepEvent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let will_run_tui = matches!(
        cli.command,
        Command::Explore { .. }
            | Command::Searchtor { .. }
            | Command::Review { .. }
            | Command::DeadCode { .. }
            | Command::Refactor { .. }
            | Command::AutoDocs { .. }
            | Command::Onboard { .. }
            | Command::Changelog { .. }
            | Command::TechRadar { .. }
            | Command::DepAudit { .. }
            | Command::Competitive { .. }
            | Command::GenTests { .. }
            | Command::Migration { .. }
            | Command::ErrorDetective { .. }
            | Command::LogAnomaly { .. }
            | Command::DeployCheck { .. }
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
                let pipeline = ExplorePipeline::new(Arc::clone(&client), config);
                let summary = pipeline
                    .run(&path)
                    .await
                    .context("explore pipeline failed")?;
                println!("\n=== Project Summary ===\n");
                println!("{}", summary);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = ExplorePipeline::new(Arc::clone(&client), config).with_events(tx);

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

        Command::Review { base, head } => {
            if cli.mock {
                let pipeline = ReviewPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline.run(&base, &head).await.context("review pipeline failed")?;
                println!("\n=== Code Review ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = ReviewPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&base, &head).await });

                tui::run(rx, cli.model.clone(), "review".to_string(), Arc::clone(&client), cli.debug)
                    .await.context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Code Review ===\n");
                println!("{}", result);
            }
        }

        Command::DeadCode { path } => {
            if cli.mock {
                let pipeline = DeadCodePipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline.run(&path).await.context("dead-code pipeline failed")?;
                println!("\n=== Dead Code Report ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = DeadCodePipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path).await });

                tui::run(rx, cli.model.clone(), "dead-code".to_string(), Arc::clone(&client), cli.debug)
                    .await.context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Dead Code Report ===\n");
                println!("{}", result);
            }
        }

        Command::Refactor { path } => {
            if cli.mock {
                let pipeline = RefactorPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline.run(&path).await.context("refactor pipeline failed")?;
                println!("\n=== Refactoring Plan ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = RefactorPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path).await });

                tui::run(rx, cli.model.clone(), "refactor".to_string(), Arc::clone(&client), cli.debug)
                    .await.context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Refactoring Plan ===\n");
                println!("{}", result);
            }
        }

        Command::AutoDocs { path } => {
            if cli.mock {
                let pipeline = AutoDocsPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline.run(&path).await.context("auto-docs pipeline failed")?;
                println!("\n=== API Documentation ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = AutoDocsPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path).await });

                tui::run(rx, cli.model.clone(), "auto-docs".to_string(), Arc::clone(&client), cli.debug)
                    .await.context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== API Documentation ===\n");
                println!("{}", result);
            }
        }

        Command::Onboard { path } => {
            if cli.mock {
                let pipeline = OnboardPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline.run(&path).await.context("onboard pipeline failed")?;
                println!("\n=== Onboarding Guide ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = OnboardPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path).await });

                tui::run(rx, cli.model.clone(), "onboard".to_string(), Arc::clone(&client), cli.debug)
                    .await.context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Onboarding Guide ===\n");
                println!("{}", result);
            }
        }

        Command::Changelog { from, to } => {
            if cli.mock {
                let pipeline = ChangelogPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline.run(from.as_deref(), &to).await.context("changelog pipeline failed")?;
                println!("\n=== Changelog ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = ChangelogPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(from.as_deref(), &to).await });

                tui::run(rx, cli.model.clone(), "changelog".to_string(), Arc::clone(&client), cli.debug)
                    .await.context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Changelog ===\n");
                println!("{}", result);
            }
        }

        Command::TechRadar { topic } => {
            if cli.mock {
                let pipeline = TechRadarPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline.run(&topic).await.context("tech-radar pipeline failed")?;
                println!("\n=== Tech Radar ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = TechRadarPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&topic).await });

                tui::run(rx, cli.model.clone(), "tech-radar".to_string(), Arc::clone(&client), cli.debug)
                    .await.context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Tech Radar ===\n");
                println!("{}", result);
            }
        }

        Command::DepAudit { path } => {
            if cli.mock {
                let pipeline = DepAuditPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline.run(&path).await.context("dep-audit pipeline failed")?;
                println!("\n=== Dependency Audit ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = DepAuditPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path).await });

                tui::run(rx, cli.model.clone(), "dep-audit".to_string(), Arc::clone(&client), cli.debug)
                    .await.context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Dependency Audit ===\n");
                println!("{}", result);
            }
        }

        Command::Competitive { crate_name, vs } => {
            if cli.mock {
                let pipeline = CompetitivePipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline.run(&crate_name, &vs).await.context("competitive pipeline failed")?;
                println!("\n=== Competitive Analysis ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = CompetitivePipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&crate_name, &vs).await });

                tui::run(rx, cli.model.clone(), "competitive".to_string(), Arc::clone(&client), cli.debug)
                    .await.context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Competitive Analysis ===\n");
                println!("{}", result);
            }
        }

        Command::GenTests { path } => {
            if cli.mock {
                let pipeline = GenTestsPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline.run(&path).await.context("gen-tests pipeline failed")?;
                println!("\n=== Generated Tests ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = GenTestsPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path).await });

                tui::run(rx, cli.model.clone(), "gen-tests".to_string(), Arc::clone(&client), cli.debug)
                    .await.context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Generated Tests ===\n");
                println!("{}", result);
            }
        }

        Command::Migration { path, to } => {
            if cli.mock {
                let pipeline = MigrationPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline.run(&path, &to).await.context("migration pipeline failed")?;
                println!("\n=== Migration Guide ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = MigrationPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path, &to).await });

                tui::run(rx, cli.model.clone(), "migration".to_string(), Arc::clone(&client), cli.debug)
                    .await.context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Migration Guide ===\n");
                println!("{}", result);
            }
        }

        Command::ErrorDetective { path, log } => {
            if cli.mock {
                let pipeline = ErrorDetectivePipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline.run(&path, log.as_deref()).await.context("error-detective pipeline failed")?;
                println!("\n=== Error Diagnosis ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = ErrorDetectivePipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path, log.as_deref()).await });

                tui::run(rx, cli.model.clone(), "error-detective".to_string(), Arc::clone(&client), cli.debug)
                    .await.context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Error Diagnosis ===\n");
                println!("{}", result);
            }
        }

        Command::LogAnomaly { log, lines } => {
            if cli.mock {
                let pipeline = LogAnomalyPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline.run(&log, lines).await.context("log-anomaly pipeline failed")?;
                println!("\n=== Log Anomaly Report ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = LogAnomalyPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&log, lines).await });

                tui::run(rx, cli.model.clone(), "log-anomaly".to_string(), Arc::clone(&client), cli.debug)
                    .await.context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Log Anomaly Report ===\n");
                println!("{}", result);
            }
        }

        Command::DeployCheck { path } => {
            if cli.mock {
                let pipeline = DeployCheckPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline.run(&path).await.context("deploy-check pipeline failed")?;
                println!("\n=== Deployment Readiness ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = DeployCheckPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path).await });

                tui::run(rx, cli.model.clone(), "deploy-check".to_string(), Arc::clone(&client), cli.debug)
                    .await.context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Deployment Readiness ===\n");
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
