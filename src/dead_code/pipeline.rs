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

pub struct DeadCodePipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl DeadCodePipeline {
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
                ShellStepBuilder::new("compiler_check")
                    .command(move |_ctx| {
                        format!(
                            "cd {} && cargo check 2>&1 | grep -E 'warning:.*never used|warning:.*is never read|dead_code' | head -100",
                            path_str
                        )
                    })
                    .store_stdout_as("compiler_warnings")
                    .timeout_secs(120),
            )
            .shell(
                ShellStepBuilder::new("scan_pub_items")
                    .command(move |_ctx| {
                        format!(
                            "grep -rn '^pub fn \\|^pub struct \\|^pub enum \\|^pub trait \\|^pub const \\|^pub type ' {}/src 2>/dev/null | head -200",
                            path2
                        )
                    })
                    .store_stdout_as("pub_items")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("scan_usages")
                    .command(move |_ctx| {
                        format!(
                            "grep -rn 'use \\|::' {}/src --include='*.rs' 2>/dev/null | grep -v '^.*//.*use ' | head -300",
                            path3
                        )
                    })
                    .store_stdout_as("usage_lines")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("scan_allow_dead")
                    .command(move |_ctx| {
                        format!(
                            "grep -rn '#\\[allow(dead_code)\\]\\|#\\[allow(unused' {}/src --include='*.rs' 2>/dev/null | head -50",
                            path4
                        )
                    })
                    .store_stdout_as("suppressed_warnings")
                    .timeout_secs(15),
            )
            .step(
                StepBuilder::new("find_dead_items")
                    .model(self.model.clone())
                    .system_prompt("You are a Rust code quality expert. Respond only with valid JSON.")
                    .prompt(|ctx| {
                        format!(
                            "Analyse the following Rust codebase data and identify dead/unused code. \
                            Return a JSON array of objects with fields: \
                            item (string, the symbol name), file (string), kind (fn|struct|enum|trait|const|type), \
                            confidence (high|medium|low), reason (string explaining why it appears unused).\n\n\
                            Compiler warnings:\n{}\n\nPublic items:\n{}\n\nUsage patterns:\n{}\n\nSuppressed warnings:\n{}",
                            ctx.get_str("compiler_warnings"),
                            ctx.get_str("pub_items"),
                            ctx.get_str("usage_lines"),
                            ctx.get_str("suppressed_warnings"),
                        )
                    })
                    .output_json()
                    .store_as("dead_items"),
            )
            .step(
                StepBuilder::new("write_report")
                    .model(self.model.clone())
                    .system_prompt("You are a Rust code quality expert writing actionable reports.")
                    .prompt(|ctx| {
                        format!(
                            "Write a dead code report for this Rust project. \
                            For each dead item explain the safe removal steps. \
                            Group by: High Priority (safe to delete), Medium (verify before deleting), Low (keep with #[allow]).\n\n\
                            Dead items identified:\n{}\n\nCompiler warnings:\n{}",
                            ctx.get_str("dead_items"),
                            ctx.get_str("compiler_warnings"),
                        )
                    })
                    .output_text()
                    .store_as("report"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        Ok(ctx.get_str("report").to_string())
    }
}
