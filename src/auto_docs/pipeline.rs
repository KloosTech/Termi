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

pub struct AutoDocsPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl AutoDocsPipeline {
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

        let ctx = WorkflowContext::new().with("path", &path_str);

        let ctx = b
            .shell(
                ShellStepBuilder::new("find_pub_api")
                    .command(move |_ctx| {
                        format!(
                            "grep -rn '^pub fn \\|^pub struct \\|^pub enum \\|^pub trait \\|^pub mod \\|^pub type \\|^pub const ' {}/src --include='*.rs' 2>/dev/null | head -200",
                            path_str
                        )
                    })
                    .store_stdout_as("public_api")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("read_source")
                    .command(move |_ctx| {
                        format!(
                            "find {}/src -name '*.rs' 2>/dev/null | head -15 | xargs cat 2>&1 | head -3000",
                            path2
                        )
                    })
                    .store_stdout_as("source_content")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("check_existing_docs")
                    .command(move |_ctx| {
                        format!(
                            "grep -rn '///' {}/src --include='*.rs' 2>/dev/null | head -80",
                            path3
                        )
                    })
                    .store_stdout_as("existing_docs")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("read_cargo_meta")
                    .command(move |_ctx| {
                        format!("cat {}/Cargo.toml 2>/dev/null | head -30", path4)
                    })
                    .store_stdout_as("cargo_meta")
                    .timeout_secs(5),
            )
            .step(
                StepBuilder::new("generate_docs")
                    .model(self.model.clone())
                    .system_prompt("You are a technical writer specialising in Rust API documentation.")
                    .prompt(|ctx| {
                        format!(
                            "Generate comprehensive API documentation in Markdown for this Rust codebase. \
                            Structure it as:\n\
                            # API Reference\n\
                            ## Overview\n\
                            ## Modules\n(list each public module with description)\n\
                            ## Structs\n(each public struct: fields, purpose, example)\n\
                            ## Enums\n(each public enum: variants, purpose)\n\
                            ## Traits\n(each public trait: methods, implementors)\n\
                            ## Functions\n(each public fn: signature, params, returns, example)\n\
                            ## Constants & Types\n\n\
                            Use existing doc comments where available and expand them.\n\n\
                            Crate metadata:\n{}\n\nPublic API:\n{}\n\nSource code:\n{}\n\nExisting doc comments:\n{}",
                            ctx.get_str("cargo_meta"),
                            ctx.get_str("public_api"),
                            ctx.get_str("source_content"),
                            ctx.get_str("existing_docs"),
                        )
                    })
                    .output_text()
                    .store_as("api_docs"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        Ok(ctx.get_str("api_docs").to_string())
    }
}
