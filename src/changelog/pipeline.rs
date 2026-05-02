use std::sync::Arc;

use tokio::sync::mpsc;

use crate::error::TermiError;
use crate::ollama::client::OllamaClient;
use crate::workflow::context::WorkflowContext;
use crate::workflow::events::StepEvent;
use crate::workflow::runner::Workflow;
use crate::workflow::shell::ShellStepBuilder;
use crate::workflow::step::StepBuilder;

pub struct ChangelogPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl ChangelogPipeline {
    pub fn new(client: Arc<dyn OllamaClient>, model: String) -> Self {
        Self { client, model, events: None }
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    pub async fn run(&self, from: Option<&str>, to: &str) -> Result<String, TermiError> {
        let mut b = Workflow::builder();
        if let Some(tx) = self.events.clone() {
            b = b.with_events(tx);
        }

        let to = to.to_string();
        let to2 = to.clone();
        let to3 = to.clone();
        let explicit_from = from.map(|s| s.to_string());

        let ctx = WorkflowContext::new().with("to_ref", &to);

        let ctx = b
            .shell(
                ShellStepBuilder::new("get_base_ref")
                    .command(move |_ctx| {
                        if let Some(f) = &explicit_from {
                            format!("echo '{}'", f)
                        } else {
                            "git describe --tags --abbrev=0 HEAD^ 2>/dev/null || git rev-list --max-parents=0 HEAD 2>/dev/null".to_string()
                        }
                    })
                    .store_stdout_as("base_ref")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("get_commits")
                    .command(move |ctx| {
                        let from_ref = ctx.get_str("base_ref").trim().to_string();
                        format!(
                            "git log --pretty=format:\"%h|%s|%an|%ad\" --date=short {}..{} 2>&1 | head -200",
                            from_ref, to
                        )
                    })
                    .store_stdout_as("raw_commits")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("get_tags")
                    .command(move |_ctx| {
                        format!("git tag --sort=-version:refname | head -10; echo '---'; git log --oneline {}..{} 2>&1 | wc -l", to2, to3)
                    })
                    .store_stdout_as("tag_info")
                    .timeout_secs(10),
            )
            .step(
                StepBuilder::new("categorize")
                    .model(self.model.clone())
                    .system_prompt("You are a changelog writer. Respond only with valid JSON.")
                    .prompt(|ctx| {
                        format!(
                            "Categorise these git commits into groups. Return a JSON object with keys: \
                            feat (array), fix (array), refactor (array), docs (array), chore (array), perf (array), other (array). \
                            Each item is a string summarising the commit.\n\nCommits:\n{}\n\nTag info:\n{}",
                            ctx.get_str("raw_commits"),
                            ctx.get_str("tag_info"),
                        )
                    })
                    .output_json()
                    .store_as("categorized"),
            )
            .step(
                StepBuilder::new("write_changelog")
                    .model(self.model.clone())
                    .system_prompt("You write clean, developer-friendly changelogs in Markdown.")
                    .prompt(|ctx| {
                        format!(
                            "Write a CHANGELOG.md section for these changes. Use standard Keep a Changelog format \
                            (### Added / Fixed / Changed / Removed / Performance / Other). \
                            Be concise and user-facing. Include the date today.\n\n\
                            Base ref: {}\nTo ref: {}\n\nCategorised commits:\n{}",
                            ctx.get_str("base_ref"),
                            ctx.get_str("to_ref"),
                            ctx.get_str("categorized"),
                        )
                    })
                    .output_text()
                    .store_as("changelog"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        Ok(ctx.get_str("changelog").to_string())
    }
}
