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

pub struct GenTestsPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl GenTestsPipeline {
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
                ShellStepBuilder::new("find_source_files")
                    .command(move |_ctx| {
                        format!(
                            "find {}/src -name '*.rs' -not -name '*_test*' -not -path '*/tests/*' 2>/dev/null | head -20",
                            path_str
                        )
                    })
                    .store_stdout_as("source_files")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("check_existing_tests")
                    .command(move |_ctx| {
                        format!(
                            "grep -rn '#\\[test\\]\\|#\\[cfg(test)\\]' {}/src --include='*.rs' 2>/dev/null | head -60",
                            path2
                        )
                    })
                    .store_stdout_as("existing_tests")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("read_source")
                    .command(move |_ctx| {
                        format!(
                            "find {}/src -name '*.rs' -not -path '*/tests/*' 2>/dev/null | head -8 | xargs cat 2>&1 | head -3000",
                            path3
                        )
                    })
                    .store_stdout_as("source_content")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("run_tests")
                    .command(move |_ctx| {
                        format!("cd {} && cargo test 2>&1 | tail -30", path4)
                    })
                    .store_stdout_as("test_output")
                    .store_exit_code_as("test_exit")
                    .timeout_secs(120),
            )
            .step(
                StepBuilder::new("find_gaps")
                    .model(self.model.clone())
                    .system_prompt("You are a Rust testing expert. Respond only with valid JSON.")
                    .prompt(|ctx| {
                        format!(
                            "Analyse this Rust code and identify functions/modules that lack test coverage. \
                            Return a JSON array of objects with fields: \
                            function (string, the fn name), file (string), \
                            test_type (unit|integration|property), \
                            priority (high|medium|low), \
                            reason (string, why this needs a test), \
                            inputs_to_cover (array of strings, interesting test cases).\n\n\
                            Source files:\n{}\n\nExisting tests:\n{}\n\nSource code:\n{}\n\nTest run output:\n{}",
                            ctx.get_str("source_files"),
                            ctx.get_str("existing_tests"),
                            ctx.get_str("source_content"),
                            ctx.get_str("test_output"),
                        )
                    })
                    .output_json()
                    .store_as("test_gaps"),
            )
            .step(
                StepBuilder::new("generate_tests")
                    .model(self.model.clone())
                    .system_prompt("You are a Rust testing expert. Write complete, compilable test code.")
                    .prompt(|ctx| {
                        format!(
                            "Write Rust test code for all the identified gaps. \
                            For each function: write a #[cfg(test)] mod with multiple test cases covering \
                            happy path, edge cases, and error cases. Use assert!, assert_eq!, and if appropriate \
                            #[should_panic]. Include use statements needed. \
                            Format as ready-to-paste Rust code with clear comments.\n\n\
                            Source code:\n{}\n\nGaps to cover:\n{}",
                            ctx.get_str("source_content"),
                            ctx.get_str("test_gaps"),
                        )
                    })
                    .output_text()
                    .store_as("test_code"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        Ok(ctx.get_str("test_code").to_string())
    }
}
