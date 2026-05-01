use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::future::join_all;
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::error::TermiError;
use crate::ollama::client::{BoxStream, OllamaClient};
use crate::ollama::types::{ChatRequest, ChatStreamChunk, Message};
use crate::workflow::context::WorkflowContext;
use crate::workflow::events::StepEvent;
use crate::workflow::http::{HttpStep, HttpStepBuilder, JsRendering};
use crate::workflow::shell::{ShellStep, ShellStepBuilder};
use crate::workflow::step::{Step, StepBuilder};

// ── WorkflowNode ──────────────────────────────────────────────────────────────

pub enum WorkflowNode {
    /// A single LLM streaming call.
    Step(Step),
    /// Run a shell command and capture its output into the context.
    Shell(ShellStep),
    /// Fetch a URL and store the body (optionally as Markdown) in the context.
    Http(HttpStep),
    Parallel(Vec<Step>),
    Transform {
        name: &'static str,
        f: Box<dyn Fn(&mut WorkflowContext) + Send + Sync>,
    },
    Conditional {
        condition: Box<dyn Fn(&WorkflowContext) -> bool + Send + Sync>,
        if_true: Box<WorkflowNode>,
        if_false: Option<Box<WorkflowNode>>,
    },
    LoopWhile {
        condition: Box<dyn Fn(&WorkflowContext) -> bool + Send + Sync>,
        body: Box<WorkflowNode>,
        max_iterations: usize,
    },
}

// ── Workflow ──────────────────────────────────────────────────────────────────

pub struct Workflow {
    nodes: Vec<WorkflowNode>,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl Workflow {
    pub fn builder() -> WorkflowBuilder {
        WorkflowBuilder::new()
    }

    pub async fn run(
        &self,
        client: Arc<dyn OllamaClient>,
        ctx: WorkflowContext,
    ) -> Result<WorkflowContext, TermiError> {
        info!("workflow starting ({} nodes)", self.nodes.len());
        let mut ctx = ctx;
        for node in &self.nodes {
            ctx = run_node(node, Arc::clone(&client), ctx, self.events.clone()).await?;
        }
        info!("workflow complete");
        Ok(ctx)
    }
}

// ── Node execution ────────────────────────────────────────────────────────────

type NodeFuture<'a> =
    Pin<Box<dyn Future<Output = Result<WorkflowContext, TermiError>> + Send + 'a>>;

fn run_node<'n>(
    node: &'n WorkflowNode,
    client: Arc<dyn OllamaClient>,
    ctx: WorkflowContext,
    events: Option<mpsc::Sender<StepEvent>>,
) -> NodeFuture<'n> {
    Box::pin(async move {
        match node {
            WorkflowNode::Step(step) => run_step(step, client, ctx, events).await,
            WorkflowNode::Shell(shell) => run_shell(shell, ctx, &events).await,
            WorkflowNode::Http(http) => run_http(http, ctx, &events).await,

            WorkflowNode::Parallel(steps) => {
                info!("▶  parallel block ({} steps)", steps.len());
                let futs: Vec<_> = steps
                    .iter()
                    .map(|step| run_step(step, Arc::clone(&client), ctx.clone(), events.clone()))
                    .collect();
                let results = join_all(futs).await;
                let mut merged = ctx;
                for r in results {
                    merged.extend(&r?);
                }
                info!("✓  parallel block complete");
                Ok(merged)
            }

            WorkflowNode::Transform { name, f } => {
                debug!(name = name, "transform node");
                let mut ctx = ctx;
                f(&mut ctx);
                Ok(ctx)
            }

            WorkflowNode::Conditional { condition, if_true, if_false } => {
                if condition(&ctx) {
                    run_node(if_true, client, ctx, events).await
                } else if let Some(else_node) = if_false {
                    run_node(else_node, client, ctx, events).await
                } else {
                    Ok(ctx)
                }
            }

            WorkflowNode::LoopWhile { condition, body, max_iterations } => {
                let mut ctx = ctx;
                let mut i = 0usize;
                while condition(&ctx) {
                    if i >= *max_iterations {
                        return Err(TermiError::Pipeline(format!(
                            "loop_step exceeded max_iterations ({})",
                            max_iterations
                        )));
                    }
                    ctx = run_node(body, Arc::clone(&client), ctx, events.clone()).await?;
                    i += 1;
                }
                Ok(ctx)
            }
        }
    })
}

// ── Streaming accumulator ─────────────────────────────────────────────────────

async fn collect_stream(
    mut stream: BoxStream<ChatStreamChunk>,
    step_name: &'static str,
    events: &Option<mpsc::Sender<StepEvent>>,
) -> Result<(String, u32), TermiError> {
    let mut full_text = String::new();
    let mut token_count = 0u32;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if !chunk.message.content.is_empty() {
            if let Some(tx) = events {
                let _ = tx
                    .send(StepEvent::Token {
                        step: step_name,
                        text: chunk.message.content.clone(),
                    })
                    .await;
            }
            full_text.push_str(&chunk.message.content);
        }
        if chunk.done {
            // Use the server's authoritative token count from the final chunk.
            token_count = chunk.eval_count.unwrap_or(token_count);
        }
    }

    Ok((full_text, token_count))
}

// ── Step execution ────────────────────────────────────────────────────────────

async fn run_step(
    step: &Step,
    client: Arc<dyn OllamaClient>,
    mut ctx: WorkflowContext,
    events: Option<mpsc::Sender<StepEvent>>,
) -> Result<WorkflowContext, TermiError> {
    if let Some(skip_fn) = &step.skip_if {
        if skip_fn(&ctx) {
            info!("⏭  step \"{}\" skipped", step.name);
            if let Some(tx) = &events {
                let _ = tx.send(StepEvent::StepSkipped { name: step.name }).await;
            }
            return Ok(ctx);
        }
    }

    let prompt = (step.prompt_fn)(&ctx);
    debug!(step = step.name, model = %step.model, prompt_len = prompt.len(), "building prompt");
    info!("▶  step \"{}\"  (model: {})", step.name, step.model);

    if let Some(tx) = &events {
        let _ = tx
            .send(StepEvent::StepStarted {
                name: step.name,
                model: step.model.clone(),
            })
            .await;
    }

    let t = Instant::now();

    let mut messages = Vec::new();
    if let Some(sys) = &step.system_prompt {
        messages.push(Message::system(sys.clone()));
    }
    messages.push(Message::user(prompt));

    let req = ChatRequest {
        model: step.model.clone(),
        messages,
        stream: Some(true),
        format: step.output_format.ollama_format(),
        options: step.options.clone(),
        ..Default::default()
    };

    let mut attempts = 0u32;
    let (raw_text, token_count) = loop {
        match client.chat_stream(req.clone()).await {
            Ok(stream) => {
                match collect_stream(stream, step.name, &events).await {
                    Ok(result) => break result,
                    Err(e) if attempts < step.max_retries => {
                        attempts += 1;
                        warn!(step = step.name, attempt = attempts, error = %e, "step failed mid-stream, retrying");
                        // Reset the TUI buffer for this step on retry.
                        if let Some(tx) = &events {
                            let _ = tx
                                .send(StepEvent::StepStarted {
                                    name: step.name,
                                    model: step.model.clone(),
                                })
                                .await;
                        }
                    }
                    Err(e) => {
                        error!(step = step.name, error = %e, "step failed");
                        return Err(e);
                    }
                }
            }
            Err(e) if attempts < step.max_retries => {
                attempts += 1;
                warn!(step = step.name, attempt = attempts, error = %e, "step failed, retrying");
            }
            Err(e) => {
                error!(step = step.name, error = %e, "step failed");
                return Err(e);
            }
        }
    };

    let raw = raw_text.trim().to_string();
    let elapsed_ms = t.elapsed().as_millis();

    debug!(step = step.name, raw_len = raw.len(), "raw LLM response");

    let value = step.output_format.parse_and_validate(&raw).map_err(|e| {
        error!(step = step.name, error = %e, "output validation failed");
        e
    })?;

    let value = match &step.transform_output {
        Some(f) => f(value, &ctx),
        None => value,
    };

    ctx.set(step.output_key, &value);
    emit_snapshot(&ctx, &events).await;

    info!("✓  step \"{}\"  ({} tokens, {}ms)", step.name, token_count, elapsed_ms);

    if let Some(tx) = &events {
        let _ = tx
            .send(StepEvent::StepCompleted {
                name: step.name,
                total_tokens: token_count,
                elapsed_ms,
            })
            .await;
    }

    Ok(ctx)
}

// ── Context snapshot helper ───────────────────────────────────────────────────

async fn emit_snapshot(ctx: &WorkflowContext, events: &Option<mpsc::Sender<StepEvent>>) {
    if let Some(tx) = events {
        let _ = tx
            .send(StepEvent::ContextSnapshot { entries: ctx.snapshot() })
            .await;
    }
}

// ── Shell step execution ──────────────────────────────────────────────────────

async fn run_shell(
    shell: &ShellStep,
    mut ctx: WorkflowContext,
    events: &Option<mpsc::Sender<StepEvent>>,
) -> Result<WorkflowContext, TermiError> {
    if let Some(skip_fn) = &shell.skip_if {
        if skip_fn(&ctx) {
            info!("⏭  shell \"{}\" skipped", shell.name);
            if let Some(tx) = events {
                let _ = tx.send(StepEvent::StepSkipped { name: shell.name }).await;
            }
            return Ok(ctx);
        }
    }

    let cmd = (shell.command_fn)(&ctx);
    let working_dir = shell
        .working_dir_fn
        .as_ref()
        .map(|f| f(&ctx))
        .unwrap_or_else(|| ".".to_string());

    info!("▶  shell \"{}\"  $ {}", shell.name, &cmd);

    if let Some(tx) = events {
        let _ = tx.send(StepEvent::StepStarted {
            name: shell.name,
            model: "shell".to_string(),
        }).await;
        let _ = tx.send(StepEvent::StatusUpdate {
            message: format!("$ {}", &cmd),
        }).await;
    }

    let t = Instant::now();

    let output = tokio::time::timeout(
        Duration::from_secs(shell.timeout_secs),
        tokio::process::Command::new("sh")
            .arg("-c")
            .arg(&cmd)
            .current_dir(&working_dir)
            .output(),
    )
    .await
    .map_err(|_| TermiError::Pipeline(format!(
        "shell \"{}\" timed out after {}s", shell.name, shell.timeout_secs
    )))?
    .map_err(|e| TermiError::Pipeline(format!(
        "shell \"{}\" failed to launch: {}", shell.name, e
    )))?;

    let elapsed_ms = t.elapsed().as_millis();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code().unwrap_or(-1);

    ctx.set(shell.store_stdout_as, &stdout);
    if let Some(key) = shell.store_stderr_as {
        ctx.set(key, &stderr);
    }
    if let Some(key) = shell.store_exit_code_as {
        ctx.set(key, &(exit_code as i64));
    }
    emit_snapshot(&ctx, events).await;

    info!("✓  shell \"{}\"  exit={} ({}ms)", shell.name, exit_code, elapsed_ms);

    if let Some(tx) = events {
        let _ = tx.send(StepEvent::StepCompleted {
            name: shell.name,
            total_tokens: 0,
            elapsed_ms,
        }).await;
    }

    Ok(ctx)
}

// ── HTTP step execution ───────────────────────────────────────────────────────

async fn run_http(
    step: &HttpStep,
    mut ctx: WorkflowContext,
    events: &Option<mpsc::Sender<StepEvent>>,
) -> Result<WorkflowContext, TermiError> {
    if let Some(skip_fn) = &step.skip_if {
        if skip_fn(&ctx) {
            info!("⏭  http \"{}\" skipped", step.name);
            if let Some(tx) = events {
                let _ = tx.send(StepEvent::StepSkipped { name: step.name }).await;
            }
            return Ok(ctx);
        }
    }

    let url = (step.url_fn)(&ctx);
    info!("▶  http \"{}\"  GET {}", step.name, &url);

    if let Some(tx) = events {
        let _ = tx.send(StepEvent::StepStarted {
            name: step.name,
            model: "http".to_string(),
        }).await;
        let _ = tx.send(StepEvent::StatusUpdate {
            message: format!("GET {}", &url),
        }).await;
    }

    let t = Instant::now();

    let raw_html = match &step.js_rendering {
        JsRendering::None => fetch_static(&url, &step.headers, step.timeout_secs, step.store_status_as, &mut ctx).await?,
        JsRendering::Headless => fetch_js(&url, step.timeout_secs).await?,
    };

    let body = if step.strip_html {
        htmd::convert(&raw_html)
            .map_err(|e| TermiError::Pipeline(format!("HTML→Markdown failed: {e}")))?
    } else {
        raw_html
    };

    ctx.set(step.store_as, &body);
    emit_snapshot(&ctx, events).await;

    let elapsed_ms = t.elapsed().as_millis();
    info!("✓  http \"{}\"  ({} chars, {}ms)", step.name, body.len(), elapsed_ms);

    if let Some(tx) = events {
        let _ = tx.send(StepEvent::StepCompleted {
            name: step.name,
            total_tokens: 0,
            elapsed_ms,
        }).await;
    }

    Ok(ctx)
}

async fn fetch_static(
    url: &str,
    headers: &[(String, String)],
    timeout_secs: u64,
    store_status_as: Option<&'static str>,
    ctx: &mut WorkflowContext,
) -> Result<String, TermiError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .user_agent("termi/0.1")
        .build()
        .map_err(|e| TermiError::Pipeline(format!("HTTP client build failed: {e}")))?;

    let mut req = client.get(url);
    for (name, value) in headers {
        req = req.header(name.as_str(), value.as_str());
    }

    let response = req
        .send()
        .await
        .map_err(|e| TermiError::Pipeline(format!("HTTP request to {url} failed: {e}")))?;

    let status = response.status();
    let status_code = status.as_u16() as i64;

    if let Some(key) = store_status_as {
        ctx.set(key, &status_code);
    }

    if !status.is_success() && store_status_as.is_none() {
        return Err(TermiError::Pipeline(format!("HTTP {status_code} from {url}")));
    }

    response
        .text()
        .await
        .map_err(|e| TermiError::Pipeline(format!("Failed to read response body: {e}")))
}

// Two cfg variants of fetch_js: one that actually calls Playwright, one that
// returns a helpful runtime error when the feature is not compiled in.

#[cfg(feature = "js-render")]
async fn fetch_js(url: &str, _timeout_secs: u64) -> Result<String, TermiError> {
    use playwright_rs::Playwright;

    let playwright = Playwright::initialize().await.map_err(|e| {
        TermiError::Pipeline(format!(
            "Playwright init failed: {e}. \
             Ensure Node.js 18+ is on PATH and run: npx playwright@1.59.1 install chromium"
        ))
    })?;

    let browser = playwright
        .chromium()
        .launcher()
        .headless(true)
        .launch()
        .await
        .map_err(|e| TermiError::Pipeline(format!("Chromium launch failed: {e}")))?;

    let context = browser
        .context_builder()
        .build()
        .await
        .map_err(|e| TermiError::Pipeline(format!("Browser context failed: {e}")))?;

    let page = context
        .new_page()
        .await
        .map_err(|e| TermiError::Pipeline(format!("New page failed: {e}")))?;

    page.goto(url, None)
        .await
        .map_err(|e| TermiError::Pipeline(format!("Navigation to {url} failed: {e}")))?;

    // Give JS time to settle after initial load.
    tokio::time::sleep(Duration::from_secs(2)).await;

    let html = page
        .content()
        .await
        .map_err(|e| TermiError::Pipeline(format!("Page content extraction failed: {e}")))?;

    browser
        .close()
        .await
        .map_err(|e| TermiError::Pipeline(format!("Browser close failed: {e}")))?;

    Ok(html)
}

#[cfg(not(feature = "js-render"))]
async fn fetch_js(_url: &str, _timeout_secs: u64) -> Result<String, TermiError> {
    Err(TermiError::Pipeline(
        "JS rendering requires the `js-render` Cargo feature. \
         Rebuild with: cargo build --features js-render"
            .to_string(),
    ))
}

// ── Builder ───────────────────────────────────────────────────────────────────

pub struct WorkflowBuilder {
    nodes: Vec<WorkflowNode>,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl WorkflowBuilder {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), events: None }
    }

    pub fn step(mut self, step: StepBuilder) -> Self {
        self.nodes.push(WorkflowNode::Step(step.finish()));
        self
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    /// Add a shell-command step.
    pub fn shell(mut self, step: ShellStepBuilder) -> Self {
        self.nodes.push(WorkflowNode::Shell(step.finish()));
        self
    }

    /// Add an HTTP-fetch step.
    pub fn http(mut self, step: HttpStepBuilder) -> Self {
        self.nodes.push(WorkflowNode::Http(step.finish()));
        self
    }

    pub fn parallel(mut self, steps: Vec<StepBuilder>) -> Self {
        let steps = steps.into_iter().map(|s| s.finish()).collect();
        self.nodes.push(WorkflowNode::Parallel(steps));
        self
    }

    pub fn if_step<F>(mut self, condition: F, step: StepBuilder) -> Self
    where
        F: Fn(&WorkflowContext) -> bool + Send + Sync + 'static,
    {
        self.nodes.push(WorkflowNode::Conditional {
            condition: Box::new(condition),
            if_true: Box::new(WorkflowNode::Step(step.finish())),
            if_false: None,
        });
        self
    }

    pub fn if_else_step<F>(
        mut self,
        condition: F,
        if_step: StepBuilder,
        else_step: StepBuilder,
    ) -> Self
    where
        F: Fn(&WorkflowContext) -> bool + Send + Sync + 'static,
    {
        self.nodes.push(WorkflowNode::Conditional {
            condition: Box::new(condition),
            if_true: Box::new(WorkflowNode::Step(if_step.finish())),
            if_false: Some(Box::new(WorkflowNode::Step(else_step.finish()))),
        });
        self
    }

    pub fn transform<F>(mut self, name: &'static str, f: F) -> Self
    where
        F: Fn(&mut WorkflowContext) + Send + Sync + 'static,
    {
        self.nodes.push(WorkflowNode::Transform { name, f: Box::new(f) });
        self
    }

    pub fn loop_step<F>(mut self, condition: F, step: StepBuilder, max_iterations: usize) -> Self
    where
        F: Fn(&WorkflowContext) -> bool + Send + Sync + 'static,
    {
        self.nodes.push(WorkflowNode::LoopWhile {
            condition: Box::new(condition),
            body: Box::new(WorkflowNode::Step(step.finish())),
            max_iterations,
        });
        self
    }

    pub fn build(self) -> Workflow {
        Workflow { nodes: self.nodes, events: self.events }
    }
}

impl Default for WorkflowBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::{json, Value};

    use super::*;
    use crate::ollama::mock::{MockCall, MockOllamaClient};
    use crate::workflow::step::StepBuilder;

    fn make_client(response: &str) -> Arc<MockOllamaClient> {
        Arc::new(MockOllamaClient::new("llama3").with_chat_response(response))
    }

    #[tokio::test]
    async fn test_workflow_runs_steps_in_order() {
        let client = make_client("some text");

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("step_a")
                    .model("llama3")
                    .prompt(|_| "prompt A".to_string())
                    .output_text()
                    .store_as("result_a"),
            )
            .step(
                StepBuilder::new("step_b")
                    .model("llama3")
                    .prompt(|_| "prompt B".to_string())
                    .output_text()
                    .store_as("result_b"),
            )
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        assert!(ctx.contains("result_a"));
        assert!(ctx.contains("result_b"));

        let calls = client.recorded_calls().await;
        assert_eq!(calls.len(), 2);
        assert!(matches!(&calls[0], MockCall::ChatStream { model, .. } if model == "llama3"));
        assert!(matches!(&calls[1], MockCall::ChatStream { model, .. } if model == "llama3"));
    }

    #[tokio::test]
    async fn test_workflow_context_passes_between_steps() {
        let client = make_client("hello");

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("step_a")
                    .model("llama3")
                    .prompt(|_| "produce a greeting".to_string())
                    .output_text()
                    .store_as("msg"),
            )
            .step(
                StepBuilder::new("step_b")
                    .model("llama3")
                    .prompt(|ctx| format!("you said: {}", ctx.get_str("msg")))
                    .output_text()
                    .store_as("echo"),
            )
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        assert_eq!(ctx.get_str("msg"), "hello");
        assert!(ctx.contains("echo"));
    }

    #[tokio::test]
    async fn test_workflow_json_schema_validation_rejects_bad_output() {
        let client = make_client("not an array at all");

        let schema = json!({"type": "array", "items": {"type": "string"}});
        let wf = Workflow::builder()
            .step(
                StepBuilder::new("filter")
                    .model("llama3")
                    .prompt(|_| "list files".to_string())
                    .output_json_schema(schema)
                    .store_as("files"),
            )
            .build();

        let result = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TermiError::Pipeline(_)));
    }

    #[tokio::test]
    async fn test_workflow_different_models_per_step() {
        let client = Arc::new(MockOllamaClient::new("llama3").with_chat_response("response"));

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s1")
                    .model("llama3:8b")
                    .prompt(|_| "p1".to_string())
                    .output_text()
                    .store_as("r1"),
            )
            .step(
                StepBuilder::new("s2")
                    .model("mistral:latest")
                    .prompt(|_| "p2".to_string())
                    .output_text()
                    .store_as("r2"),
            )
            .build();

        wf.run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        let calls = client.recorded_calls().await;
        assert_eq!(calls.len(), 2);
        assert!(matches!(&calls[0], MockCall::ChatStream { model, .. } if model == "llama3:8b"));
        assert!(matches!(&calls[1], MockCall::ChatStream { model, .. } if model == "mistral:latest"));
    }

    #[tokio::test]
    async fn test_workflow_valid_json_schema_passes() {
        let client = make_client(r#"["src/main.rs","src/lib.rs"]"#);
        let schema = json!({"type": "array", "items": {"type": "string"}});

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("filter")
                    .model("llama3")
                    .prompt(|_| "list files".to_string())
                    .output_json_schema(schema)
                    .store_as("files"),
            )
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        let files = ctx.get_array("files");
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].as_str().unwrap(), "src/main.rs");
    }

    #[tokio::test]
    async fn test_system_prompt_sends_extra_message() {
        let client = make_client("ok");

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s")
                    .model("llama3")
                    .system_prompt("You are a helpful assistant.")
                    .prompt(|_| "hello".to_string())
                    .output_text()
                    .store_as("out"),
            )
            .build();

        wf.run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        let calls = client.recorded_calls().await;
        assert!(matches!(&calls[0], MockCall::ChatStream { has_system: true, .. }));
    }

    #[tokio::test]
    async fn test_no_system_prompt_sends_single_message() {
        let client = make_client("ok");

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s")
                    .model("llama3")
                    .prompt(|_| "hello".to_string())
                    .output_text()
                    .store_as("out"),
            )
            .build();

        wf.run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        let calls = client.recorded_calls().await;
        assert!(matches!(&calls[0], MockCall::ChatStream { has_system: false, .. }));
    }

    #[tokio::test]
    async fn test_skip_if_true_prevents_llm_call() {
        let client = make_client("should not be called");

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s")
                    .model("llama3")
                    .prompt(|_| "hello".to_string())
                    .output_text()
                    .store_as("out")
                    .skip_if(|_| true),
            )
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        assert!(client.recorded_calls().await.is_empty());
        assert!(!ctx.contains("out"));
    }

    #[tokio::test]
    async fn test_skip_if_false_allows_execution() {
        let client = make_client("done");

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s")
                    .model("llama3")
                    .prompt(|_| "hello".to_string())
                    .output_text()
                    .store_as("out")
                    .skip_if(|_| false),
            )
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        assert!(ctx.contains("out"));
    }

    #[tokio::test]
    async fn test_skip_if_reads_context() {
        let client = make_client("done");
        let ctx = WorkflowContext::new().with("skip", true);

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s")
                    .model("llama3")
                    .prompt(|_| "hello".to_string())
                    .output_text()
                    .store_as("out")
                    .skip_if(|ctx| ctx.get_bool("skip")),
            )
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();

        assert!(client.recorded_calls().await.is_empty());
        assert!(!ctx.contains("out"));
    }

    #[tokio::test]
    async fn test_retries_succeed_after_failures() {
        let client = Arc::new(
            MockOllamaClient::new("llama3")
                .with_chat_response("final")
                .with_fail_first_n(2),
        );

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s")
                    .model("llama3")
                    .prompt(|_| "hello".to_string())
                    .output_text()
                    .store_as("out")
                    .with_retries(3),
            )
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        assert_eq!(ctx.get_str("out"), "final");
        assert_eq!(client.recorded_calls().await.len(), 3); // 2 failures + 1 success
    }

    #[tokio::test]
    async fn test_retries_exhausted_returns_error() {
        let client = Arc::new(
            MockOllamaClient::new("llama3").with_fail_first_n(10),
        );

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s")
                    .model("llama3")
                    .prompt(|_| "hello".to_string())
                    .output_text()
                    .store_as("out")
                    .with_retries(2),
            )
            .build();

        let result = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await;

        assert!(result.is_err());
        assert_eq!(client.recorded_calls().await.len(), 3); // 1 initial + 2 retries
    }

    #[tokio::test]
    async fn test_transform_output_post_processes_value() {
        let client = make_client(r#"{"name":"alice","score":99}"#);

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s")
                    .model("llama3")
                    .prompt(|_| "get user".to_string())
                    .output_json()
                    .store_as("name")
                    .transform_output(|v, _| v.get("name").cloned().unwrap_or(Value::Null)),
            )
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        assert_eq!(ctx.get_str("name"), "alice");
    }

    #[tokio::test]
    async fn test_parallel_all_steps_execute_and_merge() {
        let client = make_client("response");

        let wf = Workflow::builder()
            .parallel(vec![
                StepBuilder::new("p1")
                    .model("llama3")
                    .prompt(|_| "first".to_string())
                    .output_text()
                    .store_as("r1"),
                StepBuilder::new("p2")
                    .model("llama3")
                    .prompt(|_| "second".to_string())
                    .output_text()
                    .store_as("r2"),
            ])
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        assert!(ctx.contains("r1"));
        assert!(ctx.contains("r2"));
        assert_eq!(client.recorded_calls().await.len(), 2);
    }

    #[tokio::test]
    async fn test_if_step_runs_when_condition_true() {
        let client = make_client("yes");
        let ctx = WorkflowContext::new().with("flag", true);

        let wf = Workflow::builder()
            .if_step(
                |ctx| ctx.get_bool("flag"),
                StepBuilder::new("s")
                    .model("llama3")
                    .prompt(|_| "do it".to_string())
                    .output_text()
                    .store_as("result"),
            )
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();

        assert!(ctx.contains("result"));
        assert_eq!(client.recorded_calls().await.len(), 1);
    }

    #[tokio::test]
    async fn test_if_step_skipped_when_condition_false() {
        let client = make_client("yes");
        let ctx = WorkflowContext::new().with("flag", false);

        let wf = Workflow::builder()
            .if_step(
                |ctx| ctx.get_bool("flag"),
                StepBuilder::new("s")
                    .model("llama3")
                    .prompt(|_| "do it".to_string())
                    .output_text()
                    .store_as("result"),
            )
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();

        assert!(!ctx.contains("result"));
        assert!(client.recorded_calls().await.is_empty());
    }

    #[tokio::test]
    async fn test_if_else_step_runs_else_branch() {
        let client = make_client("branch_response");
        let ctx = WorkflowContext::new().with("flag", false);

        let wf = Workflow::builder()
            .if_else_step(
                |ctx| ctx.get_bool("flag"),
                StepBuilder::new("if_branch")
                    .model("llama3")
                    .prompt(|_| "if".to_string())
                    .output_text()
                    .store_as("if_result"),
                StepBuilder::new("else_branch")
                    .model("llama3")
                    .prompt(|_| "else".to_string())
                    .output_text()
                    .store_as("else_result"),
            )
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();

        assert!(!ctx.contains("if_result"));
        assert!(ctx.contains("else_result"));
        assert_eq!(client.recorded_calls().await.len(), 1);
    }

    #[tokio::test]
    async fn test_transform_node_mutates_context_without_llm() {
        let client = make_client("unused");
        let ctx = WorkflowContext::new().with("count", 5i64);

        let wf = Workflow::builder()
            .transform("double", |ctx| {
                let n = ctx.get_i64("count").unwrap_or(0);
                ctx.set("count", n * 2);
            })
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();

        assert_eq!(ctx.get_i64("count"), Some(10));
        assert!(client.recorded_calls().await.is_empty());
    }

    #[tokio::test]
    async fn test_loop_step_iterates_until_condition_false() {
        let client = Arc::new(MockOllamaClient::new("llama3").with_chat_response("_"));

        let wf = Workflow::builder()
            .transform("init", |ctx| ctx.set("counter", 0i64))
            .loop_step(
                |ctx| ctx.get_i64("counter").unwrap_or(0) < 3,
                StepBuilder::new("inc")
                    .model("llama3")
                    .prompt(|ctx| format!("iter {}", ctx.get_i64("counter").unwrap_or(0)))
                    .output_text()
                    .store_as("counter")
                    .transform_output(|_, ctx| {
                        json!(ctx.get_i64("counter").unwrap_or(0) + 1)
                    }),
                10,
            )
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        assert_eq!(ctx.get_i64("counter"), Some(3));
        assert_eq!(client.recorded_calls().await.len(), 3);
    }

    #[tokio::test]
    async fn test_loop_step_max_iterations_guard() {
        let client = make_client("x");

        let wf = Workflow::builder()
            .loop_step(
                |_| true,
                StepBuilder::new("noop")
                    .model("llama3")
                    .prompt(|_| "x".to_string())
                    .output_text()
                    .store_as("out"),
                3,
            )
            .build();

        let result = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TermiError::Pipeline(_)));
    }
}
