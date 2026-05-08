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

pub struct DeployCheckPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl DeployCheckPipeline {
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
        let path5 = path_str.clone();
        let path6 = path_str.clone();

        let ctx = WorkflowContext::new().with("path", &path_str);

        let ctx = b
            .shell(
                ShellStepBuilder::new("check_git")
                    .command(move |_ctx| {
                        format!(
                            "git -C {p} status --porcelain 2>&1; echo '---'; git -C {p} log --oneline -5 2>&1",
                            p = path_str
                        )
                    })
                    .store_stdout_as("git_status")
                    .timeout_secs(10),
            )
            .shell(
                ShellStepBuilder::new("run_tests")
                    .command(move |_ctx| {
                        format!("cd {} && cargo test 2>&1 | tail -30", path2)
                    })
                    .store_stdout_as("test_results")
                    .store_exit_code_as("test_exit")
                    .timeout_secs(180),
            )
            .shell(
                ShellStepBuilder::new("run_clippy")
                    .command(move |_ctx| {
                        format!("cd {} && cargo clippy 2>&1 | head -60", path3)
                    })
                    .store_stdout_as("lint_results")
                    .store_exit_code_as("lint_exit")
                    .timeout_secs(120),
            )
            .shell(
                ShellStepBuilder::new("scan_todos")
                    .command(move |_ctx| {
                        format!(
                            "grep -rn 'TODO\\|FIXME\\|HACK\\|XXX\\|BROKEN' {}/src --include='*.rs' 2>/dev/null | head -30",
                            path4
                        )
                    })
                    .store_stdout_as("todo_scan")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("scan_secrets")
                    .command(move |_ctx| {
                        format!(
                            "grep -rni 'password[[:space:]]*=[[:space:]]*\"\\|secret[[:space:]]*=[[:space:]]*\"\\|api_key[[:space:]]*=[[:space:]]*\"\\|token[[:space:]]*=[[:space:]]*\"' {p} --include='*.rs' --include='*.toml' 2>/dev/null | grep -v '.git' | head -20",
                            p = path5
                        )
                    })
                    .store_stdout_as("secret_scan")
                    .timeout_secs(15),
            )
            .shell(
                ShellStepBuilder::new("check_version")
                    .command(move |_ctx| {
                        format!("grep '^version' {}/Cargo.toml 2>/dev/null", path6)
                    })
                    .store_stdout_as("version_info")
                    .timeout_secs(5),
            )
            .step(
                StepBuilder::new("assess_checklist")
                    .model(self.model.clone())
                    .system_prompt("You are a DevOps engineer doing a pre-deployment review. Respond only with valid JSON.")
                    .prompt(|ctx| {
                        format!(
                            "Evaluate each deployment check. Return a JSON array of objects with fields: \
                            check (string), status (PASS|WARN|FAIL), details (string, brief explanation).\n\n\
                            Checks to evaluate:\n\
                            1. Git working tree (are there uncommitted changes?)\n\
                            2. Test suite (did all tests pass?)\n\
                            3. Lint/Clippy (any warnings or errors?)\n\
                            4. TODO/FIXME items (critical ones blocking release?)\n\
                            5. Secrets scan (any hardcoded secrets?)\n\
                            6. Version bump (is there a version in Cargo.toml?)\n\n\
                            Git status:\n{}\n\nTest results:\n{}\n\nLint results:\n{}\n\nTODOs:\n{}\n\nSecrets scan:\n{}\n\nVersion:\n{}",
                            ctx.get_str("git_status"),
                            ctx.get_str("test_results"),
                            ctx.get_str("lint_results"),
                            ctx.get_str("todo_scan"),
                            ctx.get_str("secret_scan"),
                            ctx.get_str("version_info"),
                        )
                    })
                    .output_json()
                    .store_as("checklist"),
            )
            .step(
                StepBuilder::new("final_decision")
                    .model(self.model.clone())
                    .system_prompt("You are a release manager making deployment decisions.")
                    .prompt(|ctx| {
                        format!(
                            "Based on the pre-deployment checklist, give a final GO / NO-GO decision. \
                            Structure your response as:\n\
                            ## Deployment Readiness: [GO ✓ / NO-GO ✗]\n\
                            ### Summary\n(2-3 sentences)\n\
                            ### Blockers\n(list any FAIL items)\n\
                            ### Warnings\n(list WARN items to monitor)\n\
                            ### Passing Checks\n(brief list)\n\n\
                            Checklist:\n{}",
                            ctx.get_str("checklist"),
                        )
                    })
                    .output_text()
                    .store_as("decision"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete(None)).await;
        }

        Ok(ctx.get_str("decision").to_string())
    }
}
