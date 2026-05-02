use std::sync::Arc;

use tokio::sync::mpsc;

use crate::error::TermiError;
use crate::ollama::client::OllamaClient;
use crate::workflow::context::WorkflowContext;
use crate::workflow::events::StepEvent;
use crate::workflow::runner::Workflow;
use crate::workflow::shell::ShellStepBuilder;
use crate::workflow::step::StepBuilder;

pub struct ReviewPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl ReviewPipeline {
    pub fn new(client: Arc<dyn OllamaClient>, model: String) -> Self {
        Self { client, model, events: None }
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    pub async fn run(&self, base: &str, head: &str) -> Result<String, TermiError> {
        let mut b = Workflow::builder();
        if let Some(tx) = self.events.clone() {
            b = b.with_events(tx);
        }

        let base = base.to_string();
        let head = head.to_string();
        let base2 = base.clone();
        let head2 = head.clone();
        let base3 = base.clone();
        let head3 = head.clone();

        let ctx = WorkflowContext::new()
            .with("base", &base)
            .with("head", &head);

        let ctx = b
            .shell(
                ShellStepBuilder::new("gather_commits")
                    .command(move |_ctx| {
                        format!("git log --oneline {}..{} 2>&1 | head -100", base, head)
                    })
                    .store_stdout_as("commit_list")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("gather_stat")
                    .command(move |_ctx| {
                        format!("git diff --stat {}..{} 2>&1", base2, head2)
                    })
                    .store_stdout_as("diff_stat")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("gather_diff")
                    .command(move |_ctx| {
                        format!("git diff {}..{} 2>&1 | head -3000", base3, head3)
                    })
                    .store_stdout_as("diff_content")
                    .timeout_secs(30),
            )
            .step(
                StepBuilder::new("analyze_issues")
                    .model(self.model.clone())
                    .system_prompt("You are a senior code reviewer. Respond only with valid JSON.")
                    .prompt(|ctx| {
                        format!(
                            "Analyse this git diff and identify issues. Return a JSON array of objects with fields: \
                            type (bug|security|style|performance|logic), severity (high|medium|low), \
                            file (string), line (string), description (string), suggestion (string).\n\n\
                            Commits:\n{}\n\nDiff stats:\n{}\n\nDiff:\n{}",
                            ctx.get_str("commit_list"),
                            ctx.get_str("diff_stat"),
                            ctx.get_str("diff_content"),
                        )
                    })
                    .output_json()
                    .store_as("issues"),
            )
            .step(
                StepBuilder::new("write_review")
                    .model(self.model.clone())
                    .system_prompt("You are a senior code reviewer writing a professional review.")
                    .prompt(|ctx| {
                        format!(
                            "Write a thorough code review based on the following analysis.\n\n\
                            Commits:\n{}\n\nDiff stats:\n{}\n\nIdentified issues (JSON):\n{}\n\n\
                            Structure the review with sections: Summary, Critical Issues, \
                            Warnings, Suggestions, Verdict (approve/request-changes/needs-discussion).",
                            ctx.get_str("commit_list"),
                            ctx.get_str("diff_stat"),
                            ctx.get_str("issues"),
                        )
                    })
                    .output_text()
                    .store_as("review"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        Ok(ctx.get_str("review").to_string())
    }
}
