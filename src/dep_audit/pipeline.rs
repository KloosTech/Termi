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

pub struct DepAuditPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl DepAuditPipeline {
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
                ShellStepBuilder::new("read_cargo_toml")
                    .command(move |_ctx| {
                        format!("cat {}/Cargo.toml 2>/dev/null || echo 'Cargo.toml not found'", path_str)
                    })
                    .store_stdout_as("cargo_toml")
                    .timeout_secs(5),
            )
            .shell(
                ShellStepBuilder::new("read_cargo_lock")
                    .command(move |_ctx| {
                        format!("cat {}/Cargo.lock 2>/dev/null | head -500 || echo 'Cargo.lock not found'", path2)
                    })
                    .store_stdout_as("cargo_lock_sample")
                    .timeout_secs(5),
            )
            .shell(
                ShellStepBuilder::new("run_audit")
                    .command(move |_ctx| {
                        format!(
                            "cd {} && cargo audit 2>&1 || echo 'cargo-audit not installed. Run: cargo install cargo-audit'",
                            path3
                        )
                    })
                    .store_stdout_as("audit_raw")
                    .store_exit_code_as("audit_exit")
                    .timeout_secs(120),
            )
            .shell(
                ShellStepBuilder::new("check_outdated")
                    .command(move |_ctx| {
                        format!(
                            "cd {} && cargo outdated 2>&1 || echo 'cargo-outdated not installed. Run: cargo install cargo-outdated'",
                            path4
                        )
                    })
                    .store_stdout_as("outdated_raw")
                    .timeout_secs(120),
            )
            .step(
                StepBuilder::new("analyze_deps")
                    .model(self.model.clone())
                    .system_prompt("You are a Rust security expert specialising in supply chain security. Respond only with valid JSON.")
                    .prompt(|ctx| {
                        format!(
                            "Analyse these Rust dependencies for security issues, outdated packages, and license concerns. \
                            Return a JSON array of objects with fields: \
                            crate (string), version (string, current), \
                            category (security|outdated|license|quality), \
                            severity (critical|high|medium|low|info), \
                            issue (string, description of the concern), \
                            remediation (string, how to fix).\n\n\
                            Cargo.toml:\n{}\n\nCargo.lock sample:\n{}\n\nAudit output:\n{}\n\nOutdated check:\n{}",
                            ctx.get_str("cargo_toml"),
                            ctx.get_str("cargo_lock_sample"),
                            ctx.get_str("audit_raw"),
                            ctx.get_str("outdated_raw"),
                        )
                    })
                    .output_json()
                    .store_as("dep_issues"),
            )
            .step(
                StepBuilder::new("write_audit_report")
                    .model(self.model.clone())
                    .system_prompt("You are a security auditor writing a dependency health report.")
                    .prompt(|ctx| {
                        format!(
                            "Write a comprehensive dependency audit report. Structure it as:\n\
                            ## Dependency Audit Report\n\
                            ### Executive Summary\n\
                            ### Critical & High Severity Issues\n\
                            ### Outdated Dependencies\n\
                            ### License Concerns\n\
                            ### Recommendations\n\
                            ### Upgrade Commands\n\n\
                            Issues found:\n{}\n\nRaw audit output:\n{}",
                            ctx.get_str("dep_issues"),
                            ctx.get_str("audit_raw"),
                        )
                    })
                    .output_text()
                    .store_as("audit_report"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        Ok(ctx.get_str("audit_report").to_string())
    }
}
