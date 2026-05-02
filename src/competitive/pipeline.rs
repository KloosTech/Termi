use std::sync::Arc;

use tokio::sync::mpsc;

use crate::error::TermiError;
use crate::ollama::client::OllamaClient;
use crate::workflow::context::WorkflowContext;
use crate::workflow::events::StepEvent;
use crate::workflow::runner::Workflow;
use crate::workflow::shell::ShellStepBuilder;
use crate::workflow::step::StepBuilder;

pub struct CompetitivePipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl CompetitivePipeline {
    pub fn new(client: Arc<dyn OllamaClient>, model: String) -> Self {
        Self { client, model, events: None }
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    pub async fn run(&self, crate_name: &str, vs: &[String]) -> Result<String, TermiError> {
        let mut b = Workflow::builder();
        if let Some(tx) = self.events.clone() {
            b = b.with_events(tx);
        }

        let crate_name = crate_name.to_string();
        let crate2 = crate_name.clone();
        let crate3 = crate_name.clone();
        let alts = vs.to_vec();
        let alts2 = alts.clone();

        let alts_list = if alts.is_empty() {
            "none specified".to_string()
        } else {
            alts.join(", ")
        };

        let ctx = WorkflowContext::new()
            .with("crate_name", &crate_name)
            .with("alternatives", &alts_list);

        let ctx = b
            .shell(
                ShellStepBuilder::new("fetch_main_crate")
                    .command(move |_ctx| {
                        format!(
                            "curl -s 'https://crates.io/api/v1/crates/{}' -H 'User-Agent: termi/0.1' 2>/dev/null \
                             | grep -o '\"name\":\"[^\"]*\"\\|\"downloads\":[0-9]*\\|\"description\":\"[^\"]*\"\\|\"repository\":\"[^\"]*\"\\|\"version\":\"[^\"]*\"' \
                             | head -30 || echo 'fetch failed'",
                            crate_name
                        )
                    })
                    .store_stdout_as("main_crate_data")
                    .timeout_secs(20),
            )
            .shell(
                ShellStepBuilder::new("fetch_alternatives")
                    .command(move |_ctx| {
                        if alts.is_empty() {
                            "echo 'No alternatives specified'".to_string()
                        } else {
                            alts.iter()
                                .map(|name| {
                                    format!(
                                        "echo '=== {} ==='; curl -s 'https://crates.io/api/v1/crates/{}' -H 'User-Agent: termi/0.1' 2>/dev/null | grep -o '\"downloads\":[0-9]*\\|\"description\":\"[^\"]*\"\\|\"version\":\"[^\"]*\"' | head -10",
                                        name, name
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join("; ")
                        }
                    })
                    .store_stdout_as("alts_data")
                    .timeout_secs(30),
            )
            .shell(
                ShellStepBuilder::new("fetch_main_readme")
                    .command(move |_ctx| {
                        format!(
                            "curl -s 'https://crates.io/api/v1/crates/{}/readme' -H 'User-Agent: termi/0.1' 2>/dev/null | head -200 || echo 'README not available'",
                            crate2
                        )
                    })
                    .store_stdout_as("main_readme")
                    .timeout_secs(20),
            )
            .shell(
                ShellStepBuilder::new("fetch_alt_readmes")
                    .command(move |_ctx| {
                        if alts2.is_empty() {
                            "echo 'No alternatives'".to_string()
                        } else {
                            alts2.iter()
                                .map(|name| {
                                    format!(
                                        "echo '=== {} README ==='; curl -s 'https://crates.io/api/v1/crates/{}/readme' -H 'User-Agent: termi/0.1' 2>/dev/null | head -80 || echo 'not available'",
                                        name, name
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join("; ")
                        }
                    })
                    .store_stdout_as("alt_readmes")
                    .timeout_secs(30),
            )
            .shell(
                ShellStepBuilder::new("fetch_github_stats")
                    .command(move |_ctx| {
                        format!(
                            "curl -s 'https://crates.io/api/v1/crates/{}/versions' -H 'User-Agent: termi/0.1' 2>/dev/null \
                             | grep -o '\"num\":\"[^\"]*\"\\|\"downloads\":[0-9]*\\|\"created_at\":\"[^\"]*\"' \
                             | head -30 || echo 'version history unavailable'",
                            crate3
                        )
                    })
                    .store_stdout_as("version_history")
                    .timeout_secs(20),
            )
            .step(
                StepBuilder::new("compare_crates")
                    .model(self.model.clone())
                    .system_prompt("You are a Rust ecosystem expert. Respond only with valid JSON.")
                    .prompt(|ctx| {
                        format!(
                            "Compare the crate '{}' against its alternatives: {}. \
                            Return a JSON object with:\n\
                            - 'matrix': array of objects with fields: crate (string), \
                              downloads (string), latest_version (string), \
                              strengths (array of strings), weaknesses (array of strings), \
                              best_for (string, use case), maintenance (active|maintained|slow|abandoned)\n\
                            - 'recommendation': string (which crate to use and why)\n\n\
                            Main crate data:\n{}\n\nMain crate README:\n{}\n\n\
                            Alternatives data:\n{}\n\nAlternatives READMEs:\n{}\n\nVersion history:\n{}",
                            ctx.get_str("crate_name"),
                            ctx.get_str("alternatives"),
                            ctx.get_str("main_crate_data"),
                            ctx.get_str("main_readme"),
                            ctx.get_str("alts_data"),
                            ctx.get_str("alt_readmes"),
                            ctx.get_str("version_history"),
                        )
                    })
                    .output_json()
                    .store_as("comparison"),
            )
            .step(
                StepBuilder::new("write_analysis")
                    .model(self.model.clone())
                    .system_prompt("You are a Rust ecosystem expert writing a crate evaluation report.")
                    .prompt(|ctx| {
                        format!(
                            "Write a competitive analysis report comparing '{}' against {}. \
                            Structure it as:\n\
                            ## Crate Comparison: {} vs Alternatives\n\
                            ### TL;DR Recommendation\n\
                            ### Comparison Matrix\n(table format: features, downloads, activity)\n\
                            ### {} — Deep Dive\n\
                            ### Alternatives Analysis\n\
                            ### Decision Guide\n(when to choose each option)\n\
                            ### Migration Notes\n(if switching between them)\n\n\
                            Comparison data:\n{}",
                            ctx.get_str("crate_name"),
                            ctx.get_str("alternatives"),
                            ctx.get_str("crate_name"),
                            ctx.get_str("crate_name"),
                            ctx.get_str("comparison"),
                        )
                    })
                    .output_text()
                    .store_as("analysis"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        Ok(ctx.get_str("analysis").to_string())
    }
}
