use std::path::Path;
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::error::TermiError;
use crate::ollama::client::OllamaClient;
use crate::workflow::context::WorkflowContext;
use crate::workflow::events::StepEvent;
use crate::workflow::runner::Workflow;
use crate::workflow::shell::ShellStepBuilder;
use crate::workflow::step::StepBuilder;

pub struct OnboardPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl OnboardPipeline {
    pub fn new(client: Arc<dyn OllamaClient>, model: String) -> Self {
        Self { client, model, events: None }
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    pub async fn run(&self, path: &Path) -> Result<String, TermiError> {
        let mut b = Workflow::builder();
        if let Some(tx) = self.events.clone() {
            b = b.with_events(tx);
        }

        let path_str = path.to_string_lossy().to_string();
        let path2 = path_str.clone();
        let path3 = path_str.clone();
        let path4 = path_str.clone();
        let path5 = path_str.clone();

        let ctx = WorkflowContext::new().with("path", &path_str);

        let ctx = b
            .shell(
                ShellStepBuilder::new("get_tree")
                    .command(move |_ctx| {
                        format!(
                            "find {} -maxdepth 4 -not -path '*/.git/*' -not -path '*/target/*' -not -path '*/node_modules/*' 2>/dev/null | sort | head -150",
                            path_str
                        )
                    })
                    .store_stdout_as("project_tree")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("read_cargo")
                    .command(move |_ctx| {
                        format!("cat {}/Cargo.toml 2>/dev/null || echo 'No Cargo.toml found'", path2)
                    })
                    .store_stdout_as("cargo_toml")
                    .timeout_secs(5),
            )
            .shell(
                ShellStepBuilder::new("find_entry_points")
                    .command(move |_ctx| {
                        format!(
                            "grep -rn 'fn main\\|#\\[tokio::main\\]\\|#\\[actix_web::main\\]' {}/src --include='*.rs' 2>/dev/null",
                            path3
                        )
                    })
                    .store_stdout_as("entry_points")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("read_readme")
                    .command(move |_ctx| {
                        format!(
                            "cat {p}/README.md 2>/dev/null || cat {p}/README.rst 2>/dev/null || echo 'No README found'",
                            p = path4
                        )
                    })
                    .store_stdout_as("readme_content")
                    .timeout_secs(5),
            )
            .shell(
                ShellStepBuilder::new("find_key_types")
                    .command(move |_ctx| {
                        format!(
                            "grep -rn '^pub struct \\|^pub enum \\|^pub trait ' {}/src --include='*.rs' 2>/dev/null | head -80",
                            path5
                        )
                    })
                    .store_stdout_as("key_types")
                    .timeout_secs(10),
            )
            .step(
                StepBuilder::new("write_guide")
                    .model(self.model.clone())
                    .system_prompt("You are a senior developer writing an onboarding guide for new contributors.")
                    .prompt(|ctx| {
                        format!(
                            "Write a comprehensive onboarding guide for this Rust project. \
                            Structure it as:\n\
                            # Developer Onboarding Guide\n\
                            ## What Is This Project?\n\
                            ## Quick Start\n(setup, build, run commands)\n\
                            ## Architecture Overview\n(modules, data flow, key abstractions)\n\
                            ## Key Types & Concepts\n(most important structs/enums/traits)\n\
                            ## Entry Points\n(where execution starts, how requests flow)\n\
                            ## Common Development Tasks\n(how to add a feature, run tests, etc.)\n\
                            ## Project Layout\n(what each directory/file does)\n\n\
                            README:\n{}\n\nCargo.toml:\n{}\n\nProject tree:\n{}\n\nEntry points:\n{}\n\nKey types:\n{}",
                            ctx.get_str("readme_content"),
                            ctx.get_str("cargo_toml"),
                            ctx.get_str("project_tree"),
                            ctx.get_str("entry_points"),
                            ctx.get_str("key_types"),
                        )
                    })
                    .output_text()
                    .store_as("guide"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete(None)).await;
        }

        Ok(ctx.get_str("guide").to_string())
    }
}
