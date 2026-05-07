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

    pub async fn run(&self, base: &str, head: &str) -> Result<String, TermiError> {
        let mut b = Workflow::builder();
        if let Some(tx) = self.events.clone() {
            b = b.with_events(tx);
        }

        let base = base.to_string();
        let head = head.to_string();

        let ctx = WorkflowContext::new()
            .with("base", &base)
            .with("head", &head);

        let ctx = b
            .shell(
                ShellStepBuilder::new("resolve_head")
                    .command(|ctx| {
                        let head = ctx.get_str("head");
                        if head == "HEAD" {
                            "git rev-parse --abbrev-ref HEAD 2>/dev/null || echo HEAD".to_string()
                        } else {
                            format!("echo {}", head)
                        }
                    })
                    .store_stdout_as("resolved_head")
                    .timeout_secs(5),
            )
            .shell(
                ShellStepBuilder::new("gather_commits")
                    .command(|ctx| {
                        format!(
                            "git log --oneline {}..{} 2>&1 | head -100",
                            ctx.get_str("base"),
                            ctx.get_str("head")
                        )
                    })
                    .store_stdout_as("commit_list")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("gather_files")
                    .command(|ctx| {
                        format!(
                            "git diff --name-only {}..{} 2>&1",
                            ctx.get_str("base"),
                            ctx.get_str("head")
                        )
                    })
                    .store_stdout_as("files_changed")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("gather_tree")
                    .command(|_ctx| {
                        "tree -L 2 -I 'target|.git' 2>/dev/null || find . -maxdepth 2 -not -path '*/.*' -not -path './target*'"
                            .to_string()
                    })
                    .store_stdout_as("project_tree")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("gather_stat")
                    .command(|ctx| {
                        format!(
                            "git diff --stat {}..{} 2>&1",
                            ctx.get_str("base"),
                            ctx.get_str("head")
                        )
                    })
                    .store_stdout_as("diff_stat")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("gather_whitespace")
                    .command(|ctx| {
                        format!(
                            "git diff --check {}..{} 2>&1",
                            ctx.get_str("base"),
                            ctx.get_str("head")
                        )
                    })
                    .store_stdout_as("whitespace_issues")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("gather_base_history")
                    .command(|ctx| format!("git log -n 20 --oneline {} 2>&1", ctx.get_str("base")))
                    .store_stdout_as("base_history")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("gather_diff")
                    .command(move |ctx| {
                        format!(
                            "git diff {}..{} 2>&1 | head -3000",
                            ctx.get_str("base"),
                            ctx.get_str("head")
                        )
                    })
                    .store_stdout_as("diff_content")
                    .timeout_secs(30),
            )
            .step(
                StepBuilder::new("analyze_issues")
                    .model(self.model.clone())
                    .system_prompt("You are a senior code reviewer helping a developer prepare their branch for a merge into main. Respond only with valid JSON.")
                    .prompt(|ctx| {
                        format!(
                            "Analyse the changes in branch '{}' for a merge request into '{}'. Identify bugs, architectural concerns, and potential improvements.\n\n\
                            PROJECT STRUCTURE:\n{}\n\n\
                            FILES CHANGED:\n{}\n\n\
                            WHITESPACE/SYNTAX ISSUES:\n{}\n\n\
                            COMMITS:\n{}\n\n\
                            DIFF STATS:\n{}\n\n\
                            RECENT HISTORY IN TARGET BRANCH ({}):\n{}\n\n\
                            FULL DIFF CONTENT:\n{}\n\n\
                            Return a JSON array of objects with fields: \
                            type (bug|security|style|performance|logic|documentation), \
                            severity (high|medium|low), \
                            file (string), line (string), description (string), suggestion (string).",
                            ctx.get_str("resolved_head").trim(),
                            ctx.get_str("base"),
                            ctx.get_str("project_tree"),
                            ctx.get_str("files_changed"),
                            ctx.get_str("whitespace_issues"),
                            ctx.get_str("commit_list"),
                            ctx.get_str("diff_stat"),
                            ctx.get_str("base"),
                            ctx.get_str("base_history"),
                            ctx.get_str("diff_content"),
                        )
                    })
                    .output_json()
                    .store_as("issues"),
            )
            .step(
                StepBuilder::new("write_review")
                    .model(self.model.clone())
                    .system_prompt("You are a senior code reviewer providing a high-quality review for a merge request.")
                    .prompt(|ctx| {
                        format!(
                            "Write a thorough code review for merging '{}' into '{}', focused on merge readiness. \
                            Evaluate the quality of changes, impact on the target branch, and technical debt. \
                            Use the FULL DIFF CONTENT to verify implementation specifics and ensure the code changes match the intent described in the commits.\n\n\
                            PROJECT STRUCTURE:\n{}\n\n\
                            FILES CHANGED:\n{}\n\n\
                            COMMITS:\n{}\n\n\
                            DIFF STATS:\n{}\n\n\
                            IDENTIFIED ISSUES (JSON):\n{}\n\n\
                            FULL DIFF CONTENT:\n{}\n\n\
                            Structure the review with sections:\n\
                            1. Summary: Overview of the changes and overall quality.\n\
                            2. Merge Readiness: Is this branch safe and appropriate to merge into {}?\n\
                            3. Critical Issues: Serious bugs or architectural flaws.\n\
                            4. Technical Debt & Suggestions: Style, documentation, and small improvements.\n\
                            5. Verdict: One of (approve / request-changes / needs-discussion).",
                            ctx.get_str("resolved_head").trim(),
                            ctx.get_str("base"),
                            ctx.get_str("project_tree"),
                            ctx.get_str("files_changed"),
                            ctx.get_str("commit_list"),
                            ctx.get_str("diff_stat"),
                            ctx.get_str("issues"),
                            ctx.get_str("diff_content"),
                            ctx.get_str("base"),
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
