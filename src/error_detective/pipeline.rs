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

pub struct ErrorDetectivePipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl ErrorDetectivePipeline {
    pub fn new(client: Arc<dyn OllamaClient>, model: String) -> Self {
        Self { client, model, events: None }
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    pub async fn run(&self, path: &Path, log_path: Option<&Path>) -> Result<String, TermiError> {
        let mut b = Workflow::builder();
        if let Some(tx) = self.events.clone() {
            b = b.with_events(tx);
        }

        let path_str = path.to_string_lossy().to_string();
        let path2 = path_str.clone();
        let path3 = path_str.clone();
        let log_str = log_path
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        let log_str2 = log_str.clone();

        let ctx = WorkflowContext::new()
            .with("path", &path_str)
            .with("log_path", &log_str);

        let ctx = b
            .shell(
                ShellStepBuilder::new("read_logs")
                    .command(move |_ctx| {
                        if log_str.is_empty() {
                            "echo 'No log file specified'".to_string()
                        } else {
                            format!("tail -n 500 {} 2>/dev/null || echo 'Log file not found'", log_str)
                        }
                    })
                    .store_stdout_as("log_content")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("extract_errors")
                    .command(move |_ctx| {
                        if log_str2.is_empty() {
                            "echo 'No log file to scan'".to_string()
                        } else {
                            format!(
                                "grep -E 'ERROR|WARN|panic|thread.*panicked|called.*unwrap.*on.*Err|FATAL' {} 2>/dev/null | head -100",
                                log_str2
                            )
                        }
                    })
                    .store_stdout_as("error_lines")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("check_build")
                    .command(move |_ctx| {
                        format!("cd {} && cargo build 2>&1 | head -100", path_str)
                    })
                    .store_stdout_as("build_output")
                    .store_exit_code_as("build_exit")
                    .timeout_secs(180),
            )
            .shell(
                ShellStepBuilder::new("scan_error_sites")
                    .command(move |_ctx| {
                        format!(
                            "grep -rn '\\.unwrap()\\|\\.expect(\\|panic!\\|todo!\\|unimplemented!' {}/src --include='*.rs' 2>/dev/null | head -80",
                            path2
                        )
                    })
                    .store_stdout_as("error_sites")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("scan_error_types")
                    .command(move |_ctx| {
                        format!(
                            "grep -rn '^pub enum.*Error\\|#\\[derive.*Error\\|impl.*Error for' {}/src --include='*.rs' 2>/dev/null | head -30",
                            path3
                        )
                    })
                    .store_stdout_as("error_types")
                    .timeout_secs(10),
            )
            .step(
                StepBuilder::new("cluster_errors")
                    .model(self.model.clone())
                    .system_prompt("You are a debugging expert. Respond only with valid JSON.")
                    .prompt(|ctx| {
                        format!(
                            "Analyse these error signals from a Rust project and group them by root cause. \
                            Return a JSON array of objects with fields: \
                            cluster_name (string), root_cause (string), \
                            severity (critical|high|medium|low), \
                            affected_files (array of strings), \
                            error_count (number), \
                            symptoms (array of strings, actual error messages).\n\n\
                            Log errors:\n{}\n\nBuild output:\n{}\n\nError-prone code sites:\n{}\n\nError types defined:\n{}",
                            ctx.get_str("error_lines"),
                            ctx.get_str("build_output"),
                            ctx.get_str("error_sites"),
                            ctx.get_str("error_types"),
                        )
                    })
                    .output_json()
                    .store_as("error_clusters"),
            )
            .step(
                StepBuilder::new("diagnose")
                    .model(self.model.clone())
                    .system_prompt("You are a senior Rust debugging expert.")
                    .prompt(|ctx| {
                        format!(
                            "Diagnose the errors in this Rust project and provide actionable fixes. \
                            For each error cluster:\n\
                            1. Explain the root cause clearly\n\
                            2. Show the exact code fix (before/after)\n\
                            3. Explain how to prevent it in the future\n\
                            4. Rate the fix complexity (quick-fix/refactor/redesign)\n\n\
                            Error clusters:\n{}\n\nBuild output:\n{}\n\nError sites in code:\n{}",
                            ctx.get_str("error_clusters"),
                            ctx.get_str("build_output"),
                            ctx.get_str("error_sites"),
                        )
                    })
                    .output_text()
                    .store_as("diagnosis"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        Ok(ctx.get_str("diagnosis").to_string())
    }
}
