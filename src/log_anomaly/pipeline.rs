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

pub struct LogAnomalyPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl LogAnomalyPipeline {
    pub fn new(client: Arc<dyn OllamaClient>, model: String) -> Self {
        Self { client, model, events: None }
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    pub async fn run(&self, log_path: &Path, lines: u64) -> Result<String, TermiError> {
        let mut b = Workflow::builder();
        if let Some(tx) = self.events.clone() {
            b = b.with_events(tx);
        }

        let log_str = log_path.to_string_lossy().to_string();
        let log2 = log_str.clone();
        let log3 = log_str.clone();
        let log4 = log_str.clone();

        let ctx = WorkflowContext::new()
            .with("log_path", &log_str)
            .with("lines", lines as i64);

        let ctx = b
            .shell(
                ShellStepBuilder::new("read_tail")
                    .command(move |ctx| {
                        let n = ctx.get_i64("lines").unwrap_or(1000);
                        format!("tail -n {} {} 2>/dev/null || echo 'Log file not found or empty'", n, log_str)
                    })
                    .store_stdout_as("log_tail")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("count_errors")
                    .command(move |_ctx| {
                        format!(
                            "echo 'ERROR:' $(grep -cE 'ERROR|error' {} 2>/dev/null || echo 0); \
                             echo 'WARN:' $(grep -cE 'WARN|warn' {} 2>/dev/null || echo 0); \
                             echo 'CRITICAL:' $(grep -cE 'CRITICAL|FATAL|critical|fatal' {} 2>/dev/null || echo 0)",
                            log2, log2, log2
                        )
                    })
                    .store_stdout_as("error_counts")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("top_error_patterns")
                    .command(move |_ctx| {
                        format!(
                            "grep -E 'ERROR|WARN|CRITICAL|FATAL' {} 2>/dev/null | sort | uniq -c | sort -nr | head -30",
                            log3
                        )
                    })
                    .store_stdout_as("error_patterns")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("hourly_distribution")
                    .command(move |_ctx| {
                        format!(
                            "grep -oE '[0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}}' {} 2>/dev/null | cut -d: -f1 | sort | uniq -c | sort -k2 -n || echo 'No timestamps found'",
                            log4
                        )
                    })
                    .store_stdout_as("hourly_dist")
                    .timeout_secs(10),
            )
            .step(
                StepBuilder::new("detect_anomalies")
                    .model(self.model.clone())
                    .system_prompt("You are a log analysis expert specialising in anomaly detection. Respond only with valid JSON.")
                    .prompt(|ctx| {
                        format!(
                            "Analyse these log statistics and identify anomalies. \
                            Return a JSON array of objects with fields: \
                            anomaly (string, short name), type (spike|pattern|absence|sequence|rate_change), \
                            severity (critical|high|medium|low), \
                            time_range (string, when it occurred or 'ongoing'), \
                            description (string, what is unusual), \
                            evidence (string, the log lines or counts that show it), \
                            recommended_action (string).\n\n\
                            Error counts:\n{}\n\nTop error patterns:\n{}\n\nHourly distribution:\n{}\n\nLog sample:\n{}",
                            ctx.get_str("error_counts"),
                            ctx.get_str("error_patterns"),
                            ctx.get_str("hourly_dist"),
                            ctx.get_str("log_tail"),
                        )
                    })
                    .output_json()
                    .store_as("anomalies"),
            )
            .step(
                StepBuilder::new("write_anomaly_report")
                    .model(self.model.clone())
                    .system_prompt("You are an SRE writing a clear, actionable log anomaly report.")
                    .prompt(|ctx| {
                        format!(
                            "Write an anomaly report for this log file. Structure it as:\n\
                            ## Log Anomaly Report\n\
                            ### Executive Summary\n\
                            ### Critical Anomalies\n\
                            ### High Severity\n\
                            ### Medium / Low Severity\n\
                            ### Recommended Actions\n(ordered by priority)\n\
                            ### Normal Baseline\n(what looks expected)\n\n\
                            Anomalies detected:\n{}\n\nError counts:\n{}\n\nTop patterns:\n{}",
                            ctx.get_str("anomalies"),
                            ctx.get_str("error_counts"),
                            ctx.get_str("error_patterns"),
                        )
                    })
                    .output_text()
                    .store_as("anomaly_report"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete(None)).await;
        }

        Ok(ctx.get_str("anomaly_report").to_string())
    }
}
