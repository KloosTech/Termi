mod auto_docs;
mod changelog;
mod cli;
mod commit_gen;
mod competitive;
mod dead_code;
mod dep_audit;
mod deploy_check;
mod error;
mod error_detective;
mod explore;
mod gen_tests;
mod lidarr;
mod log_anomaly;
mod mail;
mod media;
mod migration;
mod ollama;
mod onboard;
mod radarr;
mod refactor;
mod review;
mod searchtor;
mod sonarr;
mod tech_radar;
mod tts;
mod tui;
mod vault;
mod wizard;
mod workflow;
mod workflows;

use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use dialoguer::{theme::ColorfulTheme, Select};
use tracing_subscriber::{fmt, EnvFilter};

use auto_docs::AutoDocsPipeline;
use changelog::ChangelogPipeline;
use cli::{Cli, Command};
use commit_gen::CommitGenPipeline;
use competitive::CompetitivePipeline;
use dead_code::DeadCodePipeline;
use dep_audit::DepAuditPipeline;
use deploy_check::DeployCheckPipeline;
use error_detective::ErrorDetectivePipeline;
use explore::{ExploreConfig, ExplorePipeline};
use gen_tests::GenTestsPipeline;
use lidarr::LidarrPipeline;
use log_anomaly::LogAnomalyPipeline;
use mail::MailPipeline;

use migration::MigrationPipeline;
use ollama::{EmbeddingsRequest, MockOllamaClient, OllamaClient, RealOllamaClient};
use onboard::OnboardPipeline;
use radarr::RadarrPipeline;
use refactor::RefactorPipeline;
use review::ReviewPipeline;
use searchtor::SearchtorPipeline;
use sonarr::SonarrPipeline;
use tech_radar::TechRadarPipeline;
use workflow::StepEvent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // The TUI takes over the terminal for the real pipeline paths, so we only
    // initialise the tracing subscriber when it won't collide with ratatui.
    // `New` never uses the TUI; other commands do unless --mock is set.
    let will_run_tui = matches!(
        cli.command,
        Command::Explore { .. }
            | Command::Searchtor { .. }
            | Command::Review { .. }
            | Command::CommitGen
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
            | Command::Sonarr { .. }
            | Command::Radarr { .. }
            | Command::Lidarr { .. }
            | Command::Mail { .. }
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
        // High-availability Ollama: Try the remote server first, fallback to local.
        // Only attempt discovery if the user hasn't overridden the default local URL.
        let remote_url = "http://192.168.1.8:11434";
        let local_url = &cli.ollama_url;

        let selected_url = if local_url == "http://localhost:11434" {
            let discovery_client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_millis(1500))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new());

            if let Ok(resp) = discovery_client
                .get(format!("{}/api/tags", remote_url))
                .send()
                .await
            {
                if resp.status().is_success() {
                    tracing::info!("Ollama: Using remote instance at {}", remote_url);
                    remote_url.to_string()
                } else {
                    tracing::info!(
                        "Ollama: Remote at {} returned {}, falling back to {}",
                        remote_url,
                        resp.status(),
                        local_url
                    );
                    local_url.clone()
                }
            } else {
                tracing::info!(
                    "Ollama: Remote at {} unreachable, falling back to {}",
                    remote_url,
                    local_url
                );
                local_url.clone()
            }
        } else {
            local_url.clone()
        };

        Arc::new(RealOllamaClient::new(selected_url))
    };

    match cli.command {
        Command::Explore { path } => {
            let config = ExploreConfig {
                model: cli.model.clone(),
                ..Default::default()
            };

            let summary = if cli.mock {
                let pipeline = ExplorePipeline::new(Arc::clone(&client), config);
                pipeline
                    .run(&path)
                    .await
                    .context("explore pipeline failed")?
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

                handle.await.context("pipeline task panicked")??
            };

            println!("\n=== Project Summary ===\n");
            println!("{}", summary);

            if let Some(v) = &cli.vault {
                let title = format!("Explore - {}", path.display());
                vault::save(v, &title, &summary, &["project-analysis", "explore"], &None).await;
            }

            if cli.audio {
                let title = format!("Explore - {}", path.display());
                let output = tts::output_filename(&title);
                tts::generate(summary, output)
                    .await
                    .context("audio generation failed")?;
            }
        }

        Command::Searchtor { query, depth } => {
            let query_str = query.join(" ");

            let result = if cli.mock {
                SearchtorPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_depth(depth)
                    .run(query_str.clone())
                    .await
                    .context("searchtor pipeline failed")?
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = SearchtorPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_depth(depth)
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(query_str.clone()).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "searchtor".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                handle.await.context("pipeline task panicked")??
            };

            println!("\n=== Searchtor ===\n");
            println!("{}", result);

            if let Some(v) = &cli.vault {
                vault::save(
                    v,
                    &query.join(" "),
                    &result,
                    &["research", "searchtor"],
                    &None,
                )
                .await;
            }

            if cli.audio {
                let output = tts::output_filename(&query.join(" "));
                tts::generate(result, output)
                    .await
                    .context("audio generation failed")?;
            }
        }

        Command::Sonarr {
            query,
            url,
            api_key,
        } => {
            let query_str = query.join(" ");
            let output = if cli.mock {
                SonarrPipeline::new(Arc::clone(&client), cli.model.clone())
                    .run(&query_str, &url, &api_key)
                    .await
                    .context("sonarr pipeline failed")?
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    SonarrPipeline::new(Arc::clone(&client), cli.model.clone()).with_events(tx);
                let (q, u, k) = (query_str.clone(), url.clone(), api_key.clone());
                let handle = tokio::spawn(async move { pipeline.run(&q, &u, &k).await });
                tui::run(
                    rx,
                    cli.model.clone(),
                    "sonarr".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;
                handle.await.context("sonarr pipeline panicked")??
            };

            if output.results.is_empty() {
                println!("No results found for \"{}\".", output.corrected_query);
                return Ok(());
            }
            let labels: Vec<&str> = output.results.iter().map(|r| r.display.as_str()).collect();
            let Some(idx) = Select::with_theme(&ColorfulTheme::default())
                .with_prompt(format!(
                    "Results for \"{}\" — select to add",
                    output.corrected_query
                ))
                .items(&labels)
                .interact_opt()
                .context("selection failed")?
            else {
                return Ok(());
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
            let output = if cli.mock {
                RadarrPipeline::new(Arc::clone(&client), cli.model.clone())
                    .run(&query_str, &url, &api_key)
                    .await
                    .context("radarr pipeline failed")?
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    RadarrPipeline::new(Arc::clone(&client), cli.model.clone()).with_events(tx);
                let (q, u, k) = (query_str.clone(), url.clone(), api_key.clone());
                let handle = tokio::spawn(async move { pipeline.run(&q, &u, &k).await });
                tui::run(
                    rx,
                    cli.model.clone(),
                    "radarr".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;
                handle.await.context("radarr pipeline panicked")??
            };

            if output.results.is_empty() {
                println!("No results found for \"{}\".", output.corrected_query);
                return Ok(());
            }
            let labels: Vec<&str> = output.results.iter().map(|r| r.display.as_str()).collect();
            let Some(idx) = Select::with_theme(&ColorfulTheme::default())
                .with_prompt(format!(
                    "Results for \"{}\" — select to add",
                    output.corrected_query
                ))
                .items(&labels)
                .interact_opt()
                .context("selection failed")?
            else {
                return Ok(());
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
            let output = if cli.mock {
                LidarrPipeline::new(Arc::clone(&client), cli.model.clone())
                    .run(&query_str, &url, &api_key)
                    .await
                    .context("lidarr pipeline failed")?
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    LidarrPipeline::new(Arc::clone(&client), cli.model.clone()).with_events(tx);
                let (q, u, k) = (query_str.clone(), url.clone(), api_key.clone());
                let handle = tokio::spawn(async move { pipeline.run(&q, &u, &k).await });
                tui::run(
                    rx,
                    cli.model.clone(),
                    "lidarr".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;
                handle.await.context("lidarr pipeline panicked")??
            };

            if output.results.is_empty() {
                println!("No results found for \"{}\".", output.corrected_query);
                return Ok(());
            }
            let labels: Vec<&str> = output.results.iter().map(|r| r.display.as_str()).collect();
            let Some(idx) = Select::with_theme(&ColorfulTheme::default())
                .with_prompt(format!(
                    "Results for \"{}\" — select to add",
                    output.corrected_query
                ))
                .items(&labels)
                .interact_opt()
                .context("selection failed")?
            else {
                return Ok(());
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

        Command::Review { base, head } => {
            if cli.mock {
                let pipeline = ReviewPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline
                    .run(&base, &head)
                    .await
                    .context("review pipeline failed")?;
                println!("\n=== Code Review ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    ReviewPipeline::new(Arc::clone(&client), cli.model.clone()).with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&base, &head).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "review".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Code Review ===\n");
                println!("{}", result);
            }
        }

        Command::CommitGen => {
            if cli.mock {
                let pipeline = CommitGenPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline.run().await.context("commit-gen pipeline failed")?;
                println!("\n=== Commit Suggestion ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    CommitGenPipeline::new(Arc::clone(&client), cli.model.clone()).with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run().await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "commit-gen".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Commit Suggestion ===\n");
                println!("{}", result);
            }
        }

        Command::DeadCode { path } => {
            if cli.mock {
                let pipeline = DeadCodePipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline
                    .run(&path)
                    .await
                    .context("dead-code pipeline failed")?;
                println!("\n=== Dead Code Report ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    DeadCodePipeline::new(Arc::clone(&client), cli.model.clone()).with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "dead-code".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Dead Code Report ===\n");
                println!("{}", result);
            }
        }

        Command::Refactor { path } => {
            if cli.mock {
                let pipeline = RefactorPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline
                    .run(&path)
                    .await
                    .context("refactor pipeline failed")?;
                println!("\n=== Refactoring Plan ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    RefactorPipeline::new(Arc::clone(&client), cli.model.clone()).with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "refactor".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Refactoring Plan ===\n");
                println!("{}", result);
            }
        }

        Command::AutoDocs { path } => {
            if cli.mock {
                let pipeline = AutoDocsPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline
                    .run(&path)
                    .await
                    .context("auto-docs pipeline failed")?;
                println!("\n=== API Documentation ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    AutoDocsPipeline::new(Arc::clone(&client), cli.model.clone()).with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "auto-docs".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== API Documentation ===\n");
                println!("{}", result);
            }
        }

        Command::Onboard { path } => {
            if cli.mock {
                let pipeline = OnboardPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline
                    .run(&path)
                    .await
                    .context("onboard pipeline failed")?;
                println!("\n=== Onboarding Guide ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    OnboardPipeline::new(Arc::clone(&client), cli.model.clone()).with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "onboard".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Onboarding Guide ===\n");
                println!("{}", result);
            }
        }

        Command::Changelog { from, to } => {
            if cli.mock {
                let pipeline = ChangelogPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline
                    .run(from.as_deref(), &to)
                    .await
                    .context("changelog pipeline failed")?;
                println!("\n=== Changelog ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    ChangelogPipeline::new(Arc::clone(&client), cli.model.clone()).with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(from.as_deref(), &to).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "changelog".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Changelog ===\n");
                println!("{}", result);
            }
        }

        Command::TechRadar { topic } => {
            if cli.mock {
                let pipeline = TechRadarPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline
                    .run(&topic)
                    .await
                    .context("tech-radar pipeline failed")?;
                println!("\n=== Tech Radar ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    TechRadarPipeline::new(Arc::clone(&client), cli.model.clone()).with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&topic).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "tech-radar".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Tech Radar ===\n");
                println!("{}", result);
            }
        }

        Command::DepAudit { path } => {
            if cli.mock {
                let pipeline = DepAuditPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline
                    .run(&path)
                    .await
                    .context("dep-audit pipeline failed")?;
                println!("\n=== Dependency Audit ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    DepAuditPipeline::new(Arc::clone(&client), cli.model.clone()).with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "dep-audit".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Dependency Audit ===\n");
                println!("{}", result);
            }
        }

        Command::Competitive { crate_name, vs } => {
            if cli.mock {
                let pipeline = CompetitivePipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline
                    .run(&crate_name, &vs)
                    .await
                    .context("competitive pipeline failed")?;
                println!("\n=== Competitive Analysis ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = CompetitivePipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&crate_name, &vs).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "competitive".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Competitive Analysis ===\n");
                println!("{}", result);
            }
        }

        Command::GenTests { path } => {
            if cli.mock {
                let pipeline = GenTestsPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline
                    .run(&path)
                    .await
                    .context("gen-tests pipeline failed")?;
                println!("\n=== Generated Tests ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    GenTestsPipeline::new(Arc::clone(&client), cli.model.clone()).with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "gen-tests".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Generated Tests ===\n");
                println!("{}", result);
            }
        }

        Command::Migration { path, to } => {
            if cli.mock {
                let pipeline = MigrationPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline
                    .run(&path, &to)
                    .await
                    .context("migration pipeline failed")?;
                println!("\n=== Migration Guide ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    MigrationPipeline::new(Arc::clone(&client), cli.model.clone()).with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path, &to).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "migration".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Migration Guide ===\n");
                println!("{}", result);
            }
        }

        Command::ErrorDetective { path, log } => {
            if cli.mock {
                let pipeline = ErrorDetectivePipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline
                    .run(&path, log.as_deref())
                    .await
                    .context("error-detective pipeline failed")?;
                println!("\n=== Error Diagnosis ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = ErrorDetectivePipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path, log.as_deref()).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "error-detective".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Error Diagnosis ===\n");
                println!("{}", result);
            }
        }

        Command::LogAnomaly { log, lines } => {
            if cli.mock {
                let pipeline = LogAnomalyPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline
                    .run(&log, lines)
                    .await
                    .context("log-anomaly pipeline failed")?;
                println!("\n=== Log Anomaly Report ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    LogAnomalyPipeline::new(Arc::clone(&client), cli.model.clone()).with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&log, lines).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "log-anomaly".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Log Anomaly Report ===\n");
                println!("{}", result);
            }
        }

        Command::DeployCheck { path } => {
            if cli.mock {
                let pipeline = DeployCheckPipeline::new(Arc::clone(&client), cli.model.clone());
                let result = pipeline
                    .run(&path)
                    .await
                    .context("deploy-check pipeline failed")?;
                println!("\n=== Deployment Readiness ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline = DeployCheckPipeline::new(Arc::clone(&client), cli.model.clone())
                    .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run(&path).await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "deploy-check".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("pipeline task panicked")??;
                println!("\n=== Deployment Readiness ===\n");
                println!("{}", result);
            }
        }

        Command::Mail { limit } => {
            let username = std::env::var("MAIL_USERNAME").context("MAIL_USERNAME env not set")?;
            let password = std::env::var("MAIL_PASSWORD").context("MAIL_PASSWORD env not set")?;

            if cli.mock {
                let pipeline =
                    MailPipeline::new(Arc::clone(&client), cli.model.clone(), username, password)
                        .with_limit(limit);
                let result = pipeline.run().await.context("mail pipeline failed")?;
                println!("\n=== Email Briefing ===\n");
                println!("{}", result);
            } else {
                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);
                let pipeline =
                    MailPipeline::new(Arc::clone(&client), cli.model.clone(), username, password)
                        .with_limit(limit)
                        .with_events(tx);

                let handle = tokio::spawn(async move { pipeline.run().await });

                tui::run(
                    rx,
                    cli.model.clone(),
                    "mail".to_string(),
                    Arc::clone(&client),
                    cli.debug,
                )
                .await
                .context("TUI error")?;

                let result = handle.await.context("mail pipeline panicked")??;
                println!("\n=== Email Briefing ===\n");
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
