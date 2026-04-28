use std::sync::Arc;
use std::time::Instant;

use futures_util::future::join_all;
use tracing::{debug, error, info, warn};

use crate::error::TermiError;
use crate::ollama::client::OllamaClient;
use crate::ollama::types::{ChatRequest, Message};
use crate::workflow::context::WorkflowContext;
use crate::workflow::step::{Step, StepBuilder, StepErrorAction};

// ── StepNode ──────────────────────────────────────────────────────────────────

/// Internal representation of a workflow node. The public builder API
/// constructs these; callers never name this type directly.
enum StepNode {
    Single(Step),
    /// All steps run concurrently with the same input context.
    /// Outputs are merged back after all futures settle.
    /// `partial_ok`: when true, individual failures write error context keys
    /// and are skipped; when false, the first failure aborts the workflow.
    Parallel { steps: Vec<Step>, partial_ok: bool },
    /// Run `primary`; if it fails, run `fallback` with the same context.
    Fallback { primary: Step, fallback: Step },
}

// ── Workflow ──────────────────────────────────────────────────────────────────

pub struct Workflow {
    nodes: Vec<StepNode>,
}

impl Workflow {
    pub fn builder() -> WorkflowBuilder {
        WorkflowBuilder::new()
    }

    /// Execute all nodes sequentially (parallel nodes run their branches
    /// concurrently), passing the updated context forward.
    pub async fn run(
        &self,
        client: Arc<dyn OllamaClient>,
        mut ctx: WorkflowContext,
    ) -> Result<WorkflowContext, TermiError> {
        info!("workflow starting ({} nodes)", self.nodes.len());

        for node in &self.nodes {
            match node {
                StepNode::Single(step) => {
                    ctx = run_single(step, ctx, Arc::clone(&client)).await?;
                }

                StepNode::Parallel { steps, partial_ok } => {
                    // Clone context for every branch, run concurrently.
                    let futs: Vec<_> = steps
                        .iter()
                        .map(|s| run_single(s, ctx.clone(), Arc::clone(&client)))
                        .collect();

                    let results = join_all(futs).await;

                    for (step, result) in steps.iter().zip(results) {
                        match result {
                            Ok(updated) => {
                                // Merge only the output key from this branch.
                                if let Some(v) = updated.get(step.output_key) {
                                    ctx.set(step.output_key, v);
                                }
                            }
                            Err(e) if *partial_ok => {
                                warn!(step = step.name, error = %e, "parallel step failed (partial_ok)");
                                record_error_keys(&mut ctx, step.name, &e);
                            }
                            Err(e) => return Err(e),
                        }
                    }
                }

                StepNode::Fallback { primary, fallback } => {
                    match run_single(primary, ctx.clone(), Arc::clone(&client)).await {
                        Ok(updated) => ctx = updated,
                        Err(e) => {
                            warn!(
                                step = primary.name,
                                error = %e,
                                "primary step failed, running fallback"
                            );
                            ctx = run_single(fallback, ctx, Arc::clone(&client)).await?;
                        }
                    }
                }
            }
        }

        info!("workflow complete");
        Ok(ctx)
    }
}

// ── Per-step helper ───────────────────────────────────────────────────────────

/// Execute one step: build prompt → call LLM (with optional timeout) →
/// validate output → store in context → return updated context.
///
/// Handles `on_error` handlers and writes `__last_error_*` / `__error_count`
/// context keys when a default value is used.
async fn run_single(
    step: &Step,
    mut ctx: WorkflowContext,
    client: Arc<dyn OllamaClient>,
) -> Result<WorkflowContext, TermiError> {
    let prompt = (step.prompt_fn)(&ctx);
    debug!(step = step.name, model = %step.model, prompt_len = prompt.len(), "building prompt");

    info!("▶  step \"{}\"  (model: {})", step.name, step.model);
    let t = Instant::now();

    let req = ChatRequest {
        model: step.model.clone(),
        messages: vec![Message::user(prompt)],
        stream: Some(false),
        format: step.output_format.ollama_format(),
        ..Default::default()
    };

    // LLM call, optionally wrapped in a timeout.
    let chat_result = if let Some(ms) = step.timeout_ms {
        tokio::time::timeout(
            std::time::Duration::from_millis(ms),
            client.chat(req),
        )
        .await
        .map_err(|_| TermiError::Timeout { step: step.name.to_string(), ms })?
    } else {
        client.chat(req).await
    };

    let resp = match chat_result {
        Ok(r) => r,
        Err(e) => {
            error!(step = step.name, error = %e, "step failed");
            let wrapped = TermiError::StepFailed {
                step: step.name.to_string(),
                source: Box::new(e),
            };
            return handle_step_error(step, wrapped, ctx);
        }
    };

    let raw = resp.message.content.trim().to_string();
    let elapsed_ms = t.elapsed().as_millis();
    let tokens = resp.eval_count.unwrap_or(0);
    debug!(step = step.name, raw_len = raw.len(), "raw LLM response");

    let value = match step.output_format.parse_and_validate(&raw) {
        Ok(v) => v,
        Err(e) => {
            error!(step = step.name, error = %e, "output validation failed");
            let wrapped = TermiError::StepFailed {
                step: step.name.to_string(),
                source: Box::new(e),
            };
            return handle_step_error(step, wrapped, ctx);
        }
    };

    ctx.set(step.output_key, &value);

    info!(
        "✓  step \"{}\"  ({} tokens, {}ms)",
        step.name, tokens, elapsed_ms
    );

    Ok(ctx)
}

/// Invoke the step's error handler (if any); otherwise propagate.
fn handle_step_error(
    step: &Step,
    err: TermiError,
    mut ctx: WorkflowContext,
) -> Result<WorkflowContext, TermiError> {
    if let Some(handler) = &step.error_handler {
        match handler(&err, &ctx) {
            StepErrorAction::UseDefault(v) => {
                ctx.set(step.output_key, &v);
                record_error_keys(&mut ctx, step.name, &err);
                return Ok(ctx);
            }
            StepErrorAction::Abort => {}
        }
    }
    Err(err)
}

/// Write the three standard error context keys.
fn record_error_keys(ctx: &mut WorkflowContext, step_name: &str, err: &TermiError) {
    ctx.set("__last_error_step", step_name);
    ctx.set("__last_error_msg", err.to_string());
    let count = ctx.get("__error_count").and_then(|v| v.as_u64()).unwrap_or(0);
    ctx.set("__error_count", count + 1);
}

// ── Builder ───────────────────────────────────────────────────────────────────

pub struct WorkflowBuilder {
    nodes: Vec<StepNode>,
}

impl WorkflowBuilder {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a sequential step.
    pub fn step(mut self, step: StepBuilder) -> Self {
        self.nodes.push(StepNode::Single(step.finish()));
        self
    }

    /// Add a group of steps that run concurrently.  All must succeed (use
    /// `parallel_partial` to tolerate individual failures).
    pub fn parallel(mut self, steps: Vec<StepBuilder>) -> Self {
        self.nodes.push(StepNode::Parallel {
            steps: steps.into_iter().map(|s| s.finish()).collect(),
            partial_ok: false,
        });
        self
    }

    /// Like `parallel`, but individual step failures are recorded as error
    /// context keys and skipped rather than aborting the workflow.
    pub fn parallel_partial(mut self, steps: Vec<StepBuilder>) -> Self {
        self.nodes.push(StepNode::Parallel {
            steps: steps.into_iter().map(|s| s.finish()).collect(),
            partial_ok: true,
        });
        self
    }

    /// Run `primary`; if it fails, run `fallback` with the same context.
    pub fn step_with_fallback(mut self, primary: StepBuilder, fallback: StepBuilder) -> Self {
        self.nodes.push(StepNode::Fallback {
            primary: primary.finish(),
            fallback: fallback.finish(),
        });
        self
    }

    /// Absorb all nodes from `other` into this builder (composition primitive).
    pub fn extend(mut self, other: WorkflowBuilder) -> Self {
        self.nodes.extend(other.nodes);
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

    // ── existing tests (unchanged behaviour) ─────────────────────────────────

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
        assert!(matches!(result.unwrap_err(), TermiError::StepFailed { .. }));
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

    // ── new tests ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_extend_merges_steps() {
        let client = make_client("ok");

        let part_a = Workflow::builder().step(
            StepBuilder::new("a")
                .model("llama3")
                .prompt(|_| "pa".to_string())
                .output_text()
                .store_as("ra"),
        );
        let part_b = Workflow::builder().step(
            StepBuilder::new("b")
                .model("llama3")
                .prompt(|_| "pb".to_string())
                .output_text()
                .store_as("rb"),
        );

        let ctx = part_a
            .extend(part_b)
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        assert!(ctx.contains("ra"));
        assert!(ctx.contains("rb"));
        assert_eq!(client.recorded_calls().await.len(), 2);
    }

    #[tokio::test]
    async fn test_on_error_use_default_continues() {
        // Mock returns invalid JSON but step has on_error → UseDefault([])
        let client = make_client("not json");
        let schema = json!({"type": "array", "items": {"type": "string"}});

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("risky")
                    .model("llama3")
                    .prompt(|_| "list things".to_string())
                    .output_json_schema(schema)
                    .store_as("things")
                    .on_error(|_err, _ctx| {
                        StepErrorAction::UseDefault(Value::Array(vec![]))
                    }),
            )
            .step(
                StepBuilder::new("next")
                    .model("llama3")
                    .prompt(|_| "continue".to_string())
                    .output_text()
                    .store_as("result"),
            )
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        // Default value was stored
        assert_eq!(ctx.get_array("things").len(), 0);
        // Error context keys were written
        assert_eq!(ctx.get_str("__last_error_step"), "risky");
        assert!(!ctx.get_str("__last_error_msg").is_empty());
        assert_eq!(ctx.get("__error_count").and_then(|v| v.as_u64()), Some(1));
        // Subsequent step still ran
        assert!(ctx.contains("result"));
    }

    #[tokio::test]
    async fn test_on_error_abort_propagates() {
        let client = make_client("not json");
        let schema = json!({"type": "array", "items": {"type": "string"}});

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("risky")
                    .model("llama3")
                    .prompt(|_| "list things".to_string())
                    .output_json_schema(schema)
                    .store_as("things")
                    .on_error(|_err, _ctx| StepErrorAction::Abort),
            )
            .build();

        let result = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_step_with_fallback_uses_fallback_on_failure() {
        // First call → invalid JSON (primary fails), second call → "fallback ok" (fallback runs)
        let client = Arc::new(
            MockOllamaClient::new("llama3").with_responses(["not json", "fallback ok"]),
        );
        let schema = json!({"type": "array", "items": {"type": "string"}});

        let wf = Workflow::builder()
            .step_with_fallback(
                StepBuilder::new("primary")
                    .model("llama3")
                    .prompt(|_| "primary prompt".to_string())
                    .output_json_schema(schema)
                    .store_as("out"),
                StepBuilder::new("fallback")
                    .model("llama3")
                    .prompt(|_| "fallback prompt".to_string())
                    .output_text()
                    .store_as("out"),
            )
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        assert_eq!(ctx.get_str("out"), "fallback ok");
        assert_eq!(client.recorded_calls().await.len(), 2);
    }

    #[tokio::test]
    async fn test_step_with_fallback_skips_fallback_on_success() {
        let client = Arc::new(
            MockOllamaClient::new("llama3")
                .with_responses([r#"["a","b"]"#, "should not be called"]),
        );
        let schema = json!({"type": "array", "items": {"type": "string"}});

        let wf = Workflow::builder()
            .step_with_fallback(
                StepBuilder::new("primary")
                    .model("llama3")
                    .prompt(|_| "primary prompt".to_string())
                    .output_json_schema(schema)
                    .store_as("out"),
                StepBuilder::new("fallback")
                    .model("llama3")
                    .prompt(|_| "fallback prompt".to_string())
                    .output_text()
                    .store_as("out"),
            )
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        assert_eq!(ctx.get_array("out").len(), 2);
        assert_eq!(client.recorded_calls().await.len(), 1);
    }

    #[tokio::test]
    async fn test_parallel_all_succeed() {
        let client = Arc::new(
            MockOllamaClient::new("llama3").with_responses(["resp_x", "resp_y"]),
        );

        let wf = Workflow::builder()
            .parallel(vec![
                StepBuilder::new("px")
                    .model("llama3")
                    .prompt(|_| "prompt x".to_string())
                    .output_text()
                    .store_as("rx"),
                StepBuilder::new("py")
                    .model("llama3")
                    .prompt(|_| "prompt y".to_string())
                    .output_text()
                    .store_as("ry"),
            ])
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        assert!(ctx.contains("rx"));
        assert!(ctx.contains("ry"));
        assert_eq!(client.recorded_calls().await.len(), 2);
    }

    #[tokio::test]
    async fn test_parallel_partial_continues_past_failure() {
        // One branch returns invalid JSON, one returns valid text.
        let client = Arc::new(
            MockOllamaClient::new("llama3").with_responses(["not json", "good"]),
        );
        let schema = json!({"type": "array", "items": {"type": "string"}});

        let wf = Workflow::builder()
            .parallel_partial(vec![
                StepBuilder::new("bad")
                    .model("llama3")
                    .prompt(|_| "bad prompt".to_string())
                    .output_json_schema(schema)
                    .store_as("bad_out"),
                StepBuilder::new("good")
                    .model("llama3")
                    .prompt(|_| "good prompt".to_string())
                    .output_text()
                    .store_as("good_out"),
            ])
            .build();

        let ctx = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await
            .unwrap();

        // Good branch stored its output.
        assert_eq!(ctx.get_str("good_out"), "good");
        // Error context keys were written for the bad branch.
        assert_eq!(ctx.get_str("__last_error_step"), "bad");
        assert_eq!(ctx.get("__error_count").and_then(|v| v.as_u64()), Some(1));
    }

    #[tokio::test]
    async fn test_timeout_returns_timeout_error() {
        use std::sync::Arc;
        use tokio::time::Duration;

        // Use a client that sleeps longer than the step timeout.
        let client = Arc::new(crate::ollama::mock::SlowMockOllamaClient::new(200));

        let wf = Workflow::builder()
            .step(
                StepBuilder::new("slow")
                    .model("llama3")
                    .prompt(|_| "slow prompt".to_string())
                    .output_text()
                    .store_as("out")
                    .timeout_ms(50),
            )
            .build();

        let result = wf
            .run(client as Arc<dyn OllamaClient>, WorkflowContext::new())
            .await;

        assert!(
            matches!(result, Err(TermiError::Timeout { ms: 50, .. })),
            "expected Timeout, got: {:?}",
            result
        );

        // suppress unused import warning in non-test code
        let _ = Duration::from_millis(0);
    }
}
