use std::sync::Arc;

use tokio::sync::mpsc;

use crate::error::TermiError;
use crate::ollama::client::OllamaClient;
use crate::workflow::context::WorkflowContext;
use crate::workflow::events::StepEvent;
use crate::workflow::runner::Workflow;
use crate::workflow::shell::ShellStepBuilder;
use crate::workflow::step::StepBuilder;

pub struct CommitGenPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl CommitGenPipeline {
    pub fn new(client: Arc<dyn OllamaClient>, model: String) -> Self {
        Self {
            client,
            model,
            events: None,
        }
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    pub async fn run(&self) -> Result<String, TermiError> {
        let mut b = Workflow::builder();
        if let Some(tx) = self.events.clone() {
            b = b.with_events(tx);
        }

        let ctx = WorkflowContext::new();

        let ctx = b
            .shell(
                ShellStepBuilder::new("check_staged")
                    .command(|_| "git diff --cached --quiet".to_string())
                    .store_stdout_as("staged_check")
                    .store_exit_code_as("staged_status")
                    .timeout_secs(5),
            )
            .shell(
                ShellStepBuilder::new("get_diff")
                    .command(|ctx| {
                        let status = ctx.get_i64("staged_status").unwrap_or(0);
                        if status != 0 {
                            // Staged changes exist; prioritise them for the commit message.
                            "git diff --cached 2>&1 | head -n 2000".to_string()
                        } else {
                            // No staged changes; fall back to summarising unstaged changes.
                            "git diff 2>&1 | head -n 2000".to_string()
                        }
                    })
                    .store_stdout_as("git_diff")
                    .timeout_secs(10),
            )
            .step(
                StepBuilder::new("summarize_changes")
                    .model(self.model.clone())
                    .system_prompt("You are a helpful technical lead summarising code changes.")
                    .prompt(|ctx| {
                        format!(
                            "Briefly summarise what changed in these files. Focus on the 'why' and 'what', not the raw diff syntax.\n\nDiff:\n{}",
                            ctx.get_str("git_diff")
                        )
                    })
                    .output_text()
                    .store_as("summary"),
            )
            .step(
                StepBuilder::new("generate_commit_message")
                    .model(self.model.clone())
                    .system_prompt("You are an expert at writing Conventional Commits.")
                    .prompt(|ctx| {
                        format!(
                            "Write a git commit message following Conventional Commits format for the following changes.\n\
                             Include a short header (type: description) and a bulleted list for the body if appropriate.\n\n\
                             Summary:\n{}\n\nDiff snippet:\n{}",
                            ctx.get_str("summary"),
                            ctx.get_str("git_diff")
                        )
                    })
                    .output_text()
                    .store_as("commit_message"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        let result = format!(
            "### Summary of Changes\n\n{}\n\n### Suggested Commit Message\n\n```\n{}\n```",
            ctx.get_str("summary"),
            ctx.get_str("commit_message").trim()
        );

        Ok(result)
    }
}
