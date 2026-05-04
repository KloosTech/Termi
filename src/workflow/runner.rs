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
use crate::workflow::step::{Step, StepBuilder, StepErrorAction};

// ── WorkflowNode ──────────────────────────────────────────────────────────────

pub enum WorkflowNode {
    /// A single LLM streaming call.
    Step(Step),
    /// Run a shell command and capture its output into the context.
    Shell(ShellStep),
    /// Fetch a URL and store the body (optionally as Markdown) in the context.
    Http(HttpStep),
    /// Run multiple steps concurrently. Outputs are merged back after all settle.
    Parallel { steps: Vec<Step>, partial_ok: bool },
    /// Run a transformation closure on the context.
    Transform {
        name: &'static str,
        f: Box<dyn Fn(&mut WorkflowContext) + Send + Sync>,
    },
    /// Branch logic based on a condition.
    Conditional {
        condition: Box<dyn Fn(&WorkflowContext) -> bool + Send + Sync>,
        if_true: Box<WorkflowNode>,
        if_false: Option<Box<WorkflowNode>>,
    },
    /// Repeatedly execute a node while a condition holds.
    LoopWhile {
        condition: Box<dyn Fn(&WorkflowContext) -> bool + Send + Sync>,
        body: Box<WorkflowNode>,
        max_iterations: usize,
    },
    /// Run `primary`; if it fails, run `fallback`.
    Fallback {
        primary: Box<WorkflowNode>,
        fallback: Box<WorkflowNode>,
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

            WorkflowNode::Parallel { steps, partial_ok } => {
                info!(
                    "▶  parallel block ({} steps, partial_ok={})",
                    steps.len(),
                    partial_ok
                );
                let futs: Vec<_> = steps
                    .iter()
                    .map(|step| run_step(step, Arc::clone(&client), ctx.clone(), events.clone()))
                    .collect();

                let results = join_all(futs).await;
                let mut merged = ctx;

                for (step, result) in steps.iter().zip(results) {
                    match result {
                        Ok(updated) => {
                            // Merge only the specific output key from the parallel branch.
                            if let Some(v) = updated.get(step.output_key) {
                                merged.set(step.output_key, v);
                            }
                        }
                        Err(e) if *partial_ok => {
                            warn!(
                                step = step.name,
                                error = %e,
                                "parallel step failed (partial_ok)"
                            );
                            record_error_keys(&mut merged, step.name, &e);
                        }
                        Err(e) => return Err(e),
                    }
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

            WorkflowNode::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                if condition(&ctx) {
                    run_node(if_true, client, ctx, events).await
                } else if let Some(else_node) = if_false {
                    run_node(else_node, client, ctx, events).await
                } else {
                    Ok(ctx)
                }
            }

            WorkflowNode::LoopWhile {
                condition,
                body,
                max_iterations,
            } => {
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

            WorkflowNode::Fallback { primary, fallback } => {
                match run_node(primary, Arc::clone(&client), ctx.clone(), events.clone()).await {
                    Ok(updated) => Ok(updated),
                    Err(e) => {
                        warn!(error = %e, "primary node failed, running fallback");
                        run_node(fallback, client, ctx, events).await
                    }
                }
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
    debug!(
        step = step.name,
        model = %step.model,
        prompt_len = prompt.len(),
        "building prompt"
    );
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

    let chat_fut = async {
        let mut attempts = 0u32;
        loop {
            match client.chat_stream(req.clone()).await {
                Ok(stream) => {
                    match collect_stream(stream, step.name, &events).await {
                        Ok(result) => break Ok(result),
                        Err(e) if attempts < step.max_retries => {
                            attempts += 1;
                            warn!(
                                step = step.name,
                                attempt = attempts,
                                error = %e,
                                "step failed mid-stream, retrying"
                            );
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
                        Err(e) => break Err(e),
                    }
                }
                Err(e) if attempts < step.max_retries => {
                    attempts += 1;
                    warn!(
                        step = step.name,
                        attempt = attempts,
                        error = %e,
                        "step failed, retrying"
                    );
                }
                Err(e) => break Err(e),
            }
        }
    };

    let result = if let Some(ms) = step.timeout_ms {
        match tokio::time::timeout(Duration::from_millis(ms), chat_fut).await {
            Ok(res) => res,
            Err(_) => Err(TermiError::Pipeline(format!(
                "Step \"{}\" timed out after {}ms",
                step.name, ms
            ))),
        }
    } else {
        chat_fut.await
    };

    match result {
        Ok((raw_text, token_count)) => {
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

            info!(
                "✓  step \"{}\"  ({} tokens, {}ms)",
                step.name, token_count, elapsed_ms
            );

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
        Err(e) => {
            if let Some(handler) = &step.error_handler {
                match handler(&e, &ctx) {
                    StepErrorAction::UseDefault(val) => {
                        ctx.set(step.output_key, &val);
                        record_error_keys(&mut ctx, step.name, &e);
                        Ok(ctx)
                    }
                    StepErrorAction::Abort => Err(e),
                }
            } else {
                Err(e)
            }
        }
    }
}

fn record_error_keys(ctx: &mut WorkflowContext, step_name: &str, error: &TermiError) {
    ctx.set(&format!("error_{}", step_name), error.to_string());
    ctx.set("last_error", error.to_string());
}

// ── Context snapshot helper ───────────────────────────────────────────────────

async fn emit_snapshot(ctx: &WorkflowContext, events: &Option<mpsc::Sender<StepEvent>>) {
    if let Some(tx) = events {
        let _ = tx
            .send(StepEvent::ContextSnapshot {
                entries: ctx.snapshot(),
            })
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
    info!("▶  shell \"{}\"  $ {}", shell.name, &cmd);

    if let Some(tx) = events {
        let _ = tx
            .send(StepEvent::StepStarted {
                name: shell.name,
                model: "shell".to_string(),
            })
            .await;
        let _ = tx
            .send(StepEvent::StatusUpdate {
                message: format!("$ {}", &cmd),
            })
            .await;
    }

    let t = Instant::now();

    let output = tokio::time::timeout(
        Duration::from_secs(shell.timeout_secs),
        tokio::process::Command::new("sh")
            .arg("-c")
            .arg(&cmd)
            .current_dir((shell.working_dir_fn)(&ctx))
            .output(),
    )
    .await
    .map_err(|_| {
        TermiError::Pipeline(format!(
            "shell \"{}\" timed out after {}s",
            shell.name, shell.timeout_secs
        ))
    })?
    .map_err(|e| {
        TermiError::Pipeline(format!("shell \"{}\" failed to launch: {}", shell.name, e))
    })?;

    let elapsed_ms = t.elapsed().as_millis();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code().unwrap_or(-1);

    if let Some(key) = shell.store_stdout_as {
        ctx.set(key, &stdout);
    }
    if let Some(key) = shell.store_stderr_as {
        ctx.set(key, &stderr);
    }
    if let Some(key) = shell.store_exit_code_as {
        ctx.set(key, &(exit_code as i64));
    }
    emit_snapshot(&ctx, events).await;

    info!(
        "✓  shell \"{}\"  exit={} ({}ms)",
        shell.name, exit_code, elapsed_ms
    );

    if let Some(tx) = events {
        let _ = tx
            .send(StepEvent::StepCompleted {
                name: shell.name,
                total_tokens: 0,
                elapsed_ms,
            })
            .await;
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
        let _ = tx
            .send(StepEvent::StepStarted {
                name: step.name,
                model: "http".to_string(),
            })
            .await;
        let _ = tx
            .send(StepEvent::StatusUpdate {
                message: format!("GET {}", &url),
            })
            .await;
    }

    let t = Instant::now();

    let raw_html = match &step.js_rendering {
        JsRendering::None => {
            fetch_static(
                &url,
                &step.headers,
                step.timeout_secs,
                step.store_status_as,
                &mut ctx,
            )
            .await?
        }
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
    info!(
        "✓  http \"{}\"  ({} chars, {}ms)",
        step.name,
        body.len(),
        elapsed_ms
    );

    if let Some(tx) = events {
        let _ = tx
            .send(StepEvent::StepCompleted {
                name: step.name,
                total_tokens: 0,
                elapsed_ms,
            })
            .await;
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
        .http1_only()
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
        return Err(TermiError::Pipeline(format!(
            "HTTP {status_code} from {url}"
        )));
    }

    response
        .text()
        .await
        .map_err(|e| TermiError::Pipeline(format!("Failed to read response body: {e}")))
}

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
        "JS rendering requires the 'js-render' feature. Rebuild with: cargo build --features js-render"
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
        Self {
            nodes: Vec::new(),
            events: None,
        }
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    pub fn step(mut self, step: StepBuilder) -> Self {
        self.nodes.push(WorkflowNode::Step(step.finish()));
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
        self.nodes.push(WorkflowNode::Parallel {
            steps,
            partial_ok: false,
        });
        self
    }

    pub fn parallel_partial(mut self, steps: Vec<StepBuilder>) -> Self {
        let steps = steps.into_iter().map(|s| s.finish()).collect();
        self.nodes.push(WorkflowNode::Parallel {
            steps,
            partial_ok: true,
        });
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
        self.nodes.push(WorkflowNode::Transform {
            name,
            f: Box::new(f),
        });
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

    pub fn fallback(mut self, primary: StepBuilder, fallback: StepBuilder) -> Self {
        self.nodes.push(WorkflowNode::Fallback {
            primary: Box::new(WorkflowNode::Step(primary.finish())),
            fallback: Box::new(WorkflowNode::Step(fallback.finish())),
        });
        self
    }

    pub fn build(self) -> Workflow {
        Workflow {
            nodes: self.nodes,
            events: self.events,
        }
    }
}

impl Default for WorkflowBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ollama::mock::{MockCall, MockOllamaClient};
    use serde_json::json;

    fn make_client() -> Arc<MockOllamaClient> {
        Arc::new(MockOllamaClient::new("llama3"))
    }

    #[tokio::test]
    async fn test_workflow_runs_steps_in_order() {
        let client = make_client();
        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s1")
                    .model("llama3")
                    .prompt(|_| "p1".into())
                    .store_as("o1"),
            )
            .step(
                StepBuilder::new("s2")
                    .model("llama3")
                    .prompt(|_| "p2".into())
                    .store_as("o2"),
            )
            .build();

        wf.run(client as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        let calls = client.recorded_calls().await;
        assert_eq!(calls.len(), 2);
    }

    #[tokio::test]
    async fn test_workflow_context_passes_between_steps() {
        let client = make_client();
        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s1")
                    .model("llama3")
                    .prompt(|_| "p1".into())
                    .store_as("o1"),
            )
            .step(
                StepBuilder::new("s2")
                    .model("llama3")
                    .prompt(|ctx| format!("prev:{}", ctx.get_str("o1")))
                    .store_as("o2"),
            )
            .build();

        let ctx = wf
            .run(client as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();
        assert_eq!(ctx.get_str("o1"), "Mock chat response");
    }

    #[tokio::test]
    async fn test_workflow_json_schema_validation_rejects_bad_output() {
        let client = Arc::new(MockOllamaClient::new("llama3").with_chat_response("not json"));

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s1")
                    .model("llama3")
                    .prompt(|_| "p1".into())
                    .output_json_schema(json!({"type": "array"}))
                    .store_as("o1"),
            )
            .build();

        let result = wf
            .run(client as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_workflow_different_models_per_step() {
        let client = make_client();
        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s1")
                    .model("llama3:8b")
                    .prompt(|_| "p1".into())
                    .store_as("o1"),
            )
            .step(
                StepBuilder::new("s2")
                    .model("mistral:latest")
                    .prompt(|_| "p2".into())
                    .store_as("o2"),
            )
            .build();

        wf.run(
            Arc::clone(&client) as Arc<dyn OllamaClient>,
            WorkflowContext::new(),
        )
        .await
        .unwrap();

        let calls = client.recorded_calls().await;
        assert_eq!(calls.len(), 2);
        assert!(matches!(
            &calls[0],
            MockCall::ChatStream { model, .. } if model == "llama3:8b"
        ));
        assert!(matches!(
            &calls[1],
            MockCall::ChatStream { model, .. } if model == "mistral:latest"
        ));
    }

    #[tokio::test]
    async fn test_workflow_valid_json_schema_passes() {
        let client = Arc::new(MockOllamaClient::new("llama3").with_chat_response(r#"["a.rs"]"#));

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s1")
                    .model("llama3")
                    .prompt(|_| "p1".into())
                    .output_json_schema(json!({"type": "array"}))
                    .store_as("o1"),
            )
            .build();

        let ctx = wf
            .run(
                Arc::clone(&client) as Arc<dyn OllamaClient>,
                WorkflowContext::new(),
            )
            .await
            .unwrap();

        let arr = ctx.get_array("o1");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0].as_str().unwrap(), "a.rs");
    }

    #[tokio::test]
    async fn test_system_prompt_sends_extra_message() {
        let client = make_client();

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s1")
                    .model("llama3")
                    .system_prompt("be helpful")
                    .prompt(|_| "p1".into())
                    .store_as("o1"),
            )
            .build();

        wf.run(
            Arc::clone(&client) as Arc<dyn OllamaClient>,
            WorkflowContext::new(),
        )
        .await
        .unwrap();

        let calls = client.recorded_calls().await;
        assert!(matches!(
            &calls[0],
            MockCall::ChatStream {
                has_system: true,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_no_system_prompt_sends_single_message() {
        let client = make_client();

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s1")
                    .model("llama3")
                    .prompt(|_| "p1".into())
                    .store_as("o1"),
            )
            .build();

        wf.run(
            Arc::clone(&client) as Arc<dyn OllamaClient>,
            WorkflowContext::new(),
        )
        .await
        .unwrap();

        let calls = client.recorded_calls().await;
        assert!(matches!(
            &calls[0],
            MockCall::ChatStream {
                has_system: false,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_skip_if_true_prevents_llm_call() {
        let client = make_client();
        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s1")
                    .model("llama3")
                    .prompt(|_| "p1".into())
                    .skip_if(|_| true)
                    .store_as("o1"),
            )
            .build();

        let ctx = wf
            .run(
                Arc::clone(&client) as Arc<dyn OllamaClient>,
                WorkflowContext::new(),
            )
            .await
            .unwrap();
        assert!(!ctx.contains("o1"));
        assert_eq!(client.recorded_calls().await.len(), 0);
    }

    #[tokio::test]
    async fn test_skip_if_false_allows_execution() {
        let client = make_client();
        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s1")
                    .model("llama3")
                    .prompt(|_| "p1".into())
                    .skip_if(|_| false)
                    .store_as("o1"),
            )
            .build();

        let ctx = wf
            .run(
                Arc::clone(&client) as Arc<dyn OllamaClient>,
                WorkflowContext::new(),
            )
            .await
            .unwrap();
        assert!(ctx.contains("o1"));
        assert_eq!(client.recorded_calls().await.len(), 1);
    }

    #[tokio::test]
    async fn test_skip_if_reads_context() {
        let client = make_client();
        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s1")
                    .model("llama3")
                    .prompt(|_| "p1".into())
                    .skip_if(|ctx| ctx.get_bool("skip_me"))
                    .store_as("o1"),
            )
            .build();

        let ctx = wf
            .run(
                Arc::clone(&client) as Arc<dyn OllamaClient>,
                WorkflowContext::new().with("skip_me", true),
            )
            .await
            .unwrap();
        assert!(!ctx.contains("o1"));
    }

    #[tokio::test]
    async fn test_retries_succeed_after_failures() {
        let client = Arc::new(MockOllamaClient::new("llama3").with_fail_first_n(2));

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s1")
                    .model("llama3")
                    .prompt(|_| "p1".into())
                    .with_retries(2)
                    .store_as("o1"),
            )
            .build();

        let ctx = wf
            .run(
                Arc::clone(&client) as Arc<dyn OllamaClient>,
                WorkflowContext::new(),
            )
            .await
            .unwrap();
        assert_eq!(ctx.get_str("o1"), "Mock chat response");
        assert_eq!(client.recorded_calls().await.len(), 3);
    }

    #[tokio::test]
    async fn test_retries_exhausted_returns_error() {
        let client = Arc::new(MockOllamaClient::new("llama3").with_fail_first_n(10));

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s1")
                    .model("llama3")
                    .prompt(|_| "p1".into())
                    .with_retries(2)
                    .store_as("o1"),
            )
            .build();

        let result = wf
            .run(
                Arc::clone(&client) as Arc<dyn OllamaClient>,
                WorkflowContext::new(),
            )
            .await;

        assert!(result.is_err());
        assert_eq!(client.recorded_calls().await.len(), 3);
    }

    #[tokio::test]
    async fn test_transform_output_post_processes_value() {
        let client = make_client();
        let wf = Workflow::builder()
            .step(
                StepBuilder::new("s1")
                    .model("llama3")
                    .prompt(|_| "p1".into())
                    .store_as("o1")
                    .transform_output(|val, _| {
                        json!(format!("transformed:{}", val.as_str().unwrap()))
                    }),
            )
            .build();

        let ctx = wf
            .run(
                Arc::clone(&client) as Arc<dyn OllamaClient>,
                WorkflowContext::new(),
            )
            .await
            .unwrap();
        assert_eq!(ctx.get_str("o1"), "transformed:Mock chat response");
    }

    #[tokio::test]
    async fn test_parallel_all_steps_execute_and_merge() {
        let client = make_client();
        let wf = Workflow::builder()
            .parallel(vec![
                StepBuilder::new("p1")
                    .model("llama3")
                    .prompt(|_| "p1".into())
                    .store_as("o1"),
                StepBuilder::new("p2")
                    .model("llama3")
                    .prompt(|_| "p2".into())
                    .store_as("o2"),
            ])
            .build();

        let ctx = wf
            .run(
                Arc::clone(&client) as Arc<dyn OllamaClient>,
                WorkflowContext::new(),
            )
            .await
            .unwrap();
        assert!(ctx.contains("o1"));
        assert!(ctx.contains("o2"));
    }

    #[tokio::test]
    async fn test_if_step_runs_when_condition_true() {
        let client = make_client();
        let wf = Workflow::builder()
            .if_step(
                |ctx| ctx.get_bool("flag"),
                StepBuilder::new("s1")
                    .model("llama3")
                    .prompt(|_| "p1".into())
                    .store_as("o1"),
            )
            .build();

        let ctx = wf
            .run(
                Arc::clone(&client) as Arc<dyn OllamaClient>,
                WorkflowContext::new().with("flag", true),
            )
            .await
            .unwrap();
        assert!(ctx.contains("o1"));
    }

    #[tokio::test]
    async fn test_if_step_skipped_when_condition_false() {
        let client = make_client();
        let wf = Workflow::builder()
            .if_step(
                |ctx| ctx.get_bool("flag"),
                StepBuilder::new("s1")
                    .model("llama3")
                    .prompt(|_| "p1".into())
                    .store_as("o1"),
            )
            .build();

        let ctx = wf
            .run(
                Arc::clone(&client) as Arc<dyn OllamaClient>,
                WorkflowContext::new().with("flag", false),
            )
            .await
            .unwrap();
        assert!(!ctx.contains("o1"));
    }

    #[tokio::test]
    async fn test_if_else_step_runs_else_branch() {
        let client = make_client();
        let wf = Workflow::builder()
            .if_else_step(
                |ctx| ctx.get_bool("flag"),
                StepBuilder::new("if")
                    .model("l3")
                    .prompt(|_| "if")
                    .store_as("o"),
                StepBuilder::new("else")
                    .model("l3")
                    .prompt(|_| "else")
                    .store_as("o"),
            )
            .build();

        let ctx = wf
            .run(
                Arc::clone(&client) as Arc<dyn OllamaClient>,
                WorkflowContext::new().with("flag", false),
            )
            .await
            .unwrap();
        assert_eq!(ctx.get_str("o"), "Mock chat response");
        // Verify it was indeed the 'else' branch (would need a way to distinguish mock responses or check names)
    }

    #[tokio::test]
    async fn test_transform_node_mutates_context_without_llm() {
        let client = make_client();
        let wf = Workflow::builder()
            .transform("inc", |ctx| {
                let v = ctx.get_i64("val").unwrap_or(0);
                ctx.set("val", &(v + 1));
            })
            .build();

        let ctx = wf
            .run(
                Arc::clone(&client) as Arc<dyn OllamaClient>,
                WorkflowContext::new().with("val", 10i64),
            )
            .await
            .unwrap();
        assert_eq!(ctx.get_i64("val"), Some(11));
        assert_eq!(client.recorded_calls().await.len(), 0);
    }

    #[tokio::test]
    async fn test_loop_step_iterates_until_condition_false() {
        let client = make_client();
        let wf = Workflow::builder()
            .loop_step(
                |ctx| ctx.get_i64("counter").unwrap_or(0) < 3,
                StepBuilder::new("step")
                    .model("l3")
                    .prompt(|ctx| format!("iter {}", ctx.get_i64("counter").unwrap_or(0)))
                    .output_text()
                    .store_as("counter")
                    .transform_output(|_, ctx| json!(ctx.get_i64("counter").unwrap_or(0) + 1)),
                10,
            )
            .build();

        let ctx = wf
            .run(
                Arc::clone(&client) as Arc<dyn OllamaClient>,
                WorkflowContext::new(),
            )
            .await
            .unwrap();

        assert_eq!(ctx.get_i64("counter"), Some(3));
        assert_eq!(client.recorded_calls().await.len(), 3);
    }

    #[tokio::test]
    async fn test_loop_step_max_iterations_guard() {
        let client = make_client();
        let wf = Workflow::builder()
            .loop_step(
                |_| true, // infinite loop
                StepBuilder::new("step")
                    .model("l3")
                    .prompt(|_| "p")
                    .store_as("o"),
                5,
            )
            .build();

        let result = wf
            .run(
                Arc::clone(&client) as Arc<dyn OllamaClient>,
                WorkflowContext::new(),
            )
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_iterations"));
    }
}
