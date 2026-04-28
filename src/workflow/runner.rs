use std::sync::Arc;
use std::time::Instant;

use tracing::{debug, error, info};

use crate::error::TermiError;
use crate::ollama::client::OllamaClient;
use crate::ollama::types::{ChatRequest, Message};
use crate::workflow::context::WorkflowContext;
use crate::workflow::step::{Step, StepBuilder};

pub struct Workflow {
    steps: Vec<Step>,
}

impl Workflow {
    pub fn builder() -> WorkflowBuilder {
        WorkflowBuilder::new()
    }

    /// Execute every step sequentially, passing the updated context forward.
    /// Logs progress at `INFO` level and request/response detail at `DEBUG`.
    pub async fn run(
        &self,
        client: Arc<dyn OllamaClient>,
        mut ctx: WorkflowContext,
    ) -> Result<WorkflowContext, TermiError> {
        info!("workflow starting ({} steps)", self.steps.len());

        for step in &self.steps {
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

            let resp = client.chat(req).await.map_err(|e| {
                error!(step = step.name, error = %e, "step failed");
                e
            })?;

            let raw = resp.message.content.trim().to_string();
            let elapsed_ms = t.elapsed().as_millis();
            let tokens = resp.eval_count.unwrap_or(0);

            debug!(step = step.name, raw_len = raw.len(), "raw LLM response");

            let value = step.output_format.parse_and_validate(&raw).map_err(|e| {
                error!(step = step.name, error = %e, "output validation failed");
                e
            })?;

            ctx.set(step.output_key, &value);

            info!(
                "✓  step \"{}\"  ({} tokens, {}ms)",
                step.name, tokens, elapsed_ms
            );
        }

        info!("workflow complete");
        Ok(ctx)
    }
}

// ── Builder ───────────────────────────────────────────────────────────────────

pub struct WorkflowBuilder {
    steps: Vec<Step>,
}

impl WorkflowBuilder {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a step defined via `StepBuilder::finish()`.
    pub fn step(mut self, step: StepBuilder) -> Self {
        self.steps.push(step.finish());
        self
    }

    pub fn build(self) -> Workflow {
        Workflow { steps: self.steps }
    }
}

impl Default for WorkflowBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

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
        assert!(matches!(&calls[0], MockCall::Chat { model, .. } if model == "llama3"));
        assert!(matches!(&calls[1], MockCall::Chat { model, .. } if model == "llama3"));
    }

    #[tokio::test]
    async fn test_workflow_context_passes_between_steps() {
        // step_a writes "hello" to ctx["msg"], step_b reads it back in its prompt
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
        // mock returns a plain string, but step expects a JSON array
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
        // Two steps using different model names — verify the mock sees both.
        let client = Arc::new(
            MockOllamaClient::new("llama3").with_chat_response("response"),
        );

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
}
