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

pub struct MigrationPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl MigrationPipeline {
    pub fn new(client: Arc<dyn OllamaClient>, model: String) -> Self {
        Self { client, model, events: None }
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    pub async fn run(&self, path: &Path, to_edition: &str) -> Result<String, TermiError> {
        let mut b = Workflow::builder();
        if let Some(tx) = self.events.clone() {
            b = b.with_events(tx);
        }

        let path_str = path.to_string_lossy().to_string();
        let path2 = path_str.clone();
        let path3 = path_str.clone();
        let path4 = path_str.clone();
        let path5 = path_str.clone();
        let to_edition = to_edition.to_string();

        let ctx = WorkflowContext::new()
            .with("path", &path_str)
            .with("to_edition", &to_edition);

        let ctx = b
            .shell(
                ShellStepBuilder::new("read_cargo")
                    .command(move |_ctx| {
                        format!("cat {}/Cargo.toml 2>/dev/null || echo 'Cargo.toml not found'", path_str)
                    })
                    .store_stdout_as("cargo_toml")
                    .timeout_secs(5),
            )
            .shell(
                ShellStepBuilder::new("find_deprecated")
                    .command(move |_ctx| {
                        format!(
                            "cd {} && cargo check 2>&1 | grep -i 'deprecated\\|warning.*use.*instead' | head -80",
                            path2
                        )
                    })
                    .store_stdout_as("deprecations")
                    .timeout_secs(120),
            )
            .shell(
                ShellStepBuilder::new("find_unsafe")
                    .command(move |_ctx| {
                        format!(
                            "grep -rn 'unsafe' {}/src --include='*.rs' 2>/dev/null | head -50",
                            path3
                        )
                    })
                    .store_stdout_as("unsafe_usage")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("find_migration_hints")
                    .command(move |_ctx| {
                        format!(
                            "grep -rn 'TODO.*migrat\\|FIXME.*upgrad\\|deprecated\\|#\\!\\[feature' {}/src --include='*.rs' 2>/dev/null | head -40",
                            path4
                        )
                    })
                    .store_stdout_as("migration_hints")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("read_source_sample")
                    .command(move |_ctx| {
                        format!(
                            "find {}/src -name '*.rs' 2>/dev/null | head -8 | xargs cat 2>&1 | head -2000",
                            path5
                        )
                    })
                    .store_stdout_as("source_sample")
                    .timeout_secs(15),
            )
            .step(
                StepBuilder::new("plan_migration")
                    .model(self.model.clone())
                    .system_prompt("You are a Rust migration expert. Respond only with valid JSON.")
                    .prompt(|ctx| {
                        format!(
                            "Create a structured migration plan for upgrading this Rust project to edition {}. \
                            Return a JSON array of objects with fields: \
                            step (number), file (string or 'all'), change (string, what needs changing), \
                            priority (blocking|high|medium|low), effort (hours as number), \
                            risk (high|medium|low), notes (string, caveats or gotchas).\n\n\
                            Current Cargo.toml:\n{}\n\nDeprecation warnings:\n{}\n\n\
                            Unsafe usage:\n{}\n\nMigration hints in code:\n{}\n\nSource sample:\n{}",
                            ctx.get_str("to_edition"),
                            ctx.get_str("cargo_toml"),
                            ctx.get_str("deprecations"),
                            ctx.get_str("unsafe_usage"),
                            ctx.get_str("migration_hints"),
                            ctx.get_str("source_sample"),
                        )
                    })
                    .output_json()
                    .store_as("migration_plan"),
            )
            .step(
                StepBuilder::new("write_migration_guide")
                    .model(self.model.clone())
                    .system_prompt("You are a Rust migration expert writing a practical step-by-step guide.")
                    .prompt(|ctx| {
                        format!(
                            "Write a detailed migration guide for upgrading this project to Rust edition {}. \
                            For each step: explain the change, show before/after code examples where possible, \
                            provide the exact commands to run. Highlight any breaking changes.\n\n\
                            Migration plan:\n{}\n\nDeprecation warnings:\n{}\n\nSource sample:\n{}",
                            ctx.get_str("to_edition"),
                            ctx.get_str("migration_plan"),
                            ctx.get_str("deprecations"),
                            ctx.get_str("source_sample"),
                        )
                    })
                    .output_text()
                    .store_as("migration_guide"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete(None)).await;
        }

        Ok(ctx.get_str("migration_guide").to_string())
    }
}
