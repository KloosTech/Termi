use serde_json::Value;

use crate::workflow::context::WorkflowContext;
use crate::workflow::output::OutputFormat;

/// A single step in a workflow.
pub struct Step {
    /// Human-readable name used in log output.
    pub name: &'static str,
    /// Ollama model to call for this step.
    pub model: String,
    /// Builds the prompt from the current context.
    pub prompt_fn: Box<dyn Fn(&WorkflowContext) -> String + Send + Sync>,
    /// How the LLM output should be interpreted and validated.
    pub output_format: OutputFormat,
    /// Context key under which the parsed output is stored.
    pub output_key: &'static str,
}

// ── Fluent builder ─────────────────────────────────────────────────────────────

/// Builder for a `Step`. Obtain one via `Step::build("name")`.
pub struct StepBuilder {
    name: &'static str,
    model: Option<String>,
    prompt_fn: Option<Box<dyn Fn(&WorkflowContext) -> String + Send + Sync>>,
    output_format: OutputFormat,
    output_key: Option<&'static str>,
}

impl StepBuilder {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            model: None,
            prompt_fn: None,
            output_format: OutputFormat::Text,
            output_key: None,
        }
    }

    /// Set the model for this step.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the prompt builder closure. The closure receives the current
    /// `WorkflowContext` and must return the prompt string.
    pub fn prompt<F>(mut self, f: F) -> Self
    where
        F: Fn(&WorkflowContext) -> String + Send + Sync + 'static,
    {
        self.prompt_fn = Some(Box::new(f));
        self
    }

    /// Expect plain-text output (default).
    pub fn output_text(mut self) -> Self {
        self.output_format = OutputFormat::Text;
        self
    }

    /// Expect any valid JSON output.
    pub fn output_json(mut self) -> Self {
        self.output_format = OutputFormat::Json;
        self
    }

    /// Expect JSON output that conforms to `schema`.
    pub fn output_json_schema(mut self, schema: Value) -> Self {
        self.output_format = OutputFormat::JsonSchema(schema);
        self
    }

    /// Set the context key where the parsed output will be stored.
    pub fn store_as(mut self, key: &'static str) -> Self {
        self.output_key = Some(key);
        self
    }

    /// Finalise the builder and return a `Step`.
    ///
    /// # Panics
    /// Panics if `model`, `prompt`, or `store_as` were not called.
    pub fn finish(self) -> Step {
        Step {
            name: self.name,
            model: self.model.unwrap_or_else(|| {
                panic!("Step \"{}\": model() must be called before finish()", self.name)
            }),
            prompt_fn: self.prompt_fn.unwrap_or_else(|| {
                panic!("Step \"{}\": prompt() must be called before finish()", self.name)
            }),
            output_format: self.output_format,
            output_key: self.output_key.unwrap_or_else(|| {
                panic!("Step \"{}\": store_as() must be called before finish()", self.name)
            }),
        }
    }
}
