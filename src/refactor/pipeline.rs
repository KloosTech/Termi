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

pub struct RefactorPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl RefactorPipeline {
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
                ShellStepBuilder::new("run_clippy")
                    .command(move |_ctx| {
                        format!("cd {} && cargo clippy 2>&1 | head -150", path_str)
                    })
                    .store_stdout_as("clippy_output")
                    .timeout_secs(120),
            )
            .shell(
                ShellStepBuilder::new("read_source")
                    .command(move |_ctx| {
                        format!(
                            "find {}/src -name '*.rs' | head -10 | xargs cat 2>&1 | head -3000",
                            path2
                        )
                    })
                    .store_stdout_as("source_content")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("measure_complexity")
                    .command(move |_ctx| {
                        format!(
                            "grep -rn '^fn \\|^pub fn \\|^async fn \\|^pub async fn ' {}/src --include='*.rs' 2>/dev/null | head -100",
                            path3
                        )
                    })
                    .store_stdout_as("fn_list")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("find_long_fns")
                    .command(move |_ctx| {
                        // Count lines per function by detecting fn boundaries
                        format!(
                            "awk '/^[[:space:]]*(pub )?(async )?fn /{{fn=$0; count=0}} {{count++}} /^}}$/{{if(count>30) print count, fn}}' {}/src/*.rs 2>/dev/null | sort -rn | head -20",
                            path4
                        )
                    })
                    .store_stdout_as("long_fns")
                    .timeout_secs(15),
            )
            .step(
                StepBuilder::new("identify_smells")
                    .model(self.model.clone())
                    .system_prompt("You are a Rust refactoring expert. Respond only with valid JSON.")
                    .prompt(|ctx| {
                        format!(
                            "Identify code smells in this Rust codebase. Return a JSON array of objects with fields: \
                            smell (string, e.g. 'long function', 'duplicated logic', 'magic number', 'deep nesting', 'god struct'), \
                            location (string, file:line or function name), severity (high|medium|low), \
                            description (string), refactoring (string, specific action to take).\n\n\
                            Clippy output:\n{}\n\nSource sample:\n{}\n\nFunction list:\n{}\n\nPotentially long functions:\n{}",
                            ctx.get_str("clippy_output"),
                            ctx.get_str("source_content"),
                            ctx.get_str("fn_list"),
                            ctx.get_str("long_fns"),
                        )
                    })
                    .output_json()
                    .store_as("smells"),
            )
            .step(
                StepBuilder::new("write_plan")
                    .model(self.model.clone())
                    .system_prompt("You are a senior Rust engineer writing a practical refactoring guide.")
                    .prompt(|ctx| {
                        format!(
                            "Create a prioritised refactoring roadmap for this Rust project. \
                            For each issue: explain the problem, show before/after pseudocode where helpful, \
                            estimate effort (small/medium/large), and note any risks.\n\n\
                            Identified smells:\n{}\n\nClipper hints:\n{}",
                            ctx.get_str("smells"),
                            ctx.get_str("clippy_output"),
                        )
                    })
                    .output_text()
                    .store_as("plan"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        Ok(ctx.get_str("plan").to_string())
    }
}
