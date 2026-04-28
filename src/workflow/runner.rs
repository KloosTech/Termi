use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use futures_util::future::join_all;
use tracing::{debug, error, info, warn};

use crate::error::TermiError;
use crate::ollama::client::OllamaClient;
use crate::ollama::types::{ChatRequest, Message};
use crate::workflow::context::WorkflowContext;
use crate::workflow::step::{Step, StepBuilder};

// ── WorkflowNode ──────────────────────────────────────────────────────────────

/// A node in a workflow graph. Nodes are composed via `WorkflowBuilder`.
pub enum WorkflowNode {
    /// A single LLM call.
    Step(Step),

    /// Multiple LLM steps executed concurrently; all results are merged into
    /// the context when every step completes.
    Parallel(Vec<Step>),

    /// A pure context transformation — no LLM call is made.
    Transform {
        name: &'static str,
        f: Box<dyn Fn(&mut WorkflowContext) + Send + Sync>,
    },

    /// Runs one of two branches depending on a runtime condition.
    Conditional {
        condition: Box<dyn Fn(&WorkflowContext) -> bool + Send + Sync>,
        if_true: Box<WorkflowNode>,
        if_false: Option<Box<WorkflowNode>>,
    },

    /// Repeats `body` while `condition` returns `true`, up to `max_iterations`.
    LoopWhile {
        condition: Box<dyn Fn(&WorkflowContext) -> bool + Send + Sync>,
        body: Box<WorkflowNode>,
        max_iterations: usize,
    },
}

// ── Workflow ──────────────────────────────────────────────────────────────────

pub struct Workflow {
    nodes: Vec<WorkflowNode>,
}

impl Workflow {
    pub fn builder() -> WorkflowBuilder {
        WorkflowBuilder::new()
    }

    /// Execute all nodes sequentially, passing the updated context forward.
    pub async fn run(
        &self,
        client: Arc<dyn OllamaClient>,
        ctx: WorkflowContext,
    ) -> Result<WorkflowContext, TermiError> {
        info!("workflow starting ({} nodes)", self.nodes.len());
        let mut ctx = ctx;
        for node in &self.nodes {
            ctx = run_node(node, Arc::clone(&client), ctx).await?;
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
) -> NodeFuture<'n> {
    Box::pin(async move {
        match node {
            WorkflowNode::Step(step) => run_step(step, client, ctx).await,

            WorkflowNode::Parallel(steps) => {
                info!("▶  parallel block ({} steps)", steps.len());
                let futs: Vec<_> = steps
                    .iter()
                    .map(|step| run_step(step, Arc::clone(&client), ctx.clone()))
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
                    run_node(if_true, client, ctx).await
                } else if let Some(else_node) = if_false {
                    run_node(else_node, client, ctx).await
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
                    ctx = run_node(body, Arc::clone(&client), ctx).await?;
                    i += 1;
                }
                Ok(ctx)
            }
        }
    })
}

async fn run_step(
    step: &Step,
    client: Arc<dyn OllamaClient>,
    mut ctx: WorkflowContext,
) -> Result<WorkflowContext, TermiError> {
    if let Some(skip_fn) = &step.skip_if {
        if skip_fn(&ctx) {
            info!("⏭  step \"{}\" skipped", step.name);
            return Ok(ctx);
        }
    }

    let prompt = (step.prompt_fn)(&ctx);
    debug!(step = step.name, model = %step.model, prompt_len = prompt.len(), "building prompt");
    info!("▶  step \"{}\"  (model: {})", step.name, step.model);
    let t = Instant::now();

    let mut messages = Vec::new();
    if let Some(sys) = &step.system_prompt {
        messages.push(Message::system(sys.clone()));
    }
    messages.push(Message::user(prompt));

    let req = ChatRequest {
        model: step.model.clone(),
        messages,
        stream: Some(false),
        format: step.output_format.ollama_format(),
        options: step.options.clone(),
        ..Default::default()
    };

    let mut attempts = 0u32;
    let resp = loop {
        match client.chat(req.clone()).await {
            Ok(r) => break r,
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

    let raw = resp.message.content.trim().to_string();
    let elapsed_ms = t.elapsed().as_millis();
    let tokens = resp.eval_count.unwrap_or(0);

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

    info!("✓  step \"{}\"  ({} tokens, {}ms)", step.name, tokens, elapsed_ms);

    Ok(ctx)
}

// ── Builder ───────────────────────────────────────────────────────────────────

pub struct WorkflowBuilder {
    nodes: Vec<WorkflowNode>,
}

impl WorkflowBuilder {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a sequential LLM step.
    pub fn step(mut self, step: StepBuilder) -> Self {
        self.nodes.push(WorkflowNode::Step(step.finish()));
        self
    }

    /// Run multiple LLM steps concurrently. All outputs are merged into the
    /// context once every step has completed.
    pub fn parallel(mut self, steps: Vec<StepBuilder>) -> Self {
        let steps = steps.into_iter().map(|s| s.finish()).collect();
        self.nodes.push(WorkflowNode::Parallel(steps));
        self
    }

    /// Run `step` only when `condition` returns `true` at execution time.
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

    /// Run `if_step` when `condition` is `true`, otherwise run `else_step`.
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

    /// Insert a pure context transformation (no LLM call). The closure
    /// receives `&mut WorkflowContext` and may read or write any key.
    pub fn transform<F>(mut self, name: &'static str, f: F) -> Self
    where
        F: Fn(&mut WorkflowContext) + Send + Sync + 'static,
    {
        self.nodes.push(WorkflowNode::Transform { name, f: Box::new(f) });
        self
    }

    /// Repeat `step` while `condition` returns `true`. Fails with an error if
    /// the loop runs more than `max_iterations` times.
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
        Workflow { nodes: self.nodes }
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

    // ── existing tests (preserved) ────────────────────────────────────────────

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
        assert!(matches!(&calls[0], MockCall::Chat { model, .. } if model == "llama3"));
        assert!(matches!(&calls[1], MockCall::Chat { model, .. } if model == "llama3"));
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
        assert!(matches!(&calls[0], MockCall::Chat { model, .. } if model == "llama3:8b"));
        assert!(matches!(&calls[1], MockCall::Chat { model, .. } if model == "mistral:latest"));
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

    // ── system_prompt ─────────────────────────────────────────────────────────

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
        assert!(matches!(&calls[0], MockCall::Chat { has_system: true, .. }));
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
        assert!(matches!(&calls[0], MockCall::Chat { has_system: false, .. }));
    }

    // ── skip_if ───────────────────────────────────────────────────────────────

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

    // ── with_retries ──────────────────────────────────────────────────────────

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

    // ── transform_output ──────────────────────────────────────────────────────

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

    // ── parallel ─────────────────────────────────────────────────────────────

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

    // ── if_step / if_else_step ────────────────────────────────────────────────

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

    // ── transform node ────────────────────────────────────────────────────────

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

    // ── loop_step ─────────────────────────────────────────────────────────────

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
