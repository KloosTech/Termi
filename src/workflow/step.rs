use serde_json::Value;

use crate::error::TermiError;
use crate::ollama::types::ModelOptions;
use crate::workflow::context::WorkflowContext;
use crate::workflow::output::OutputFormat;

/// What a step's error handler returns to the runner.
#[derive(Debug, Clone)]
pub enum StepErrorAction {
    /// Propagate the error and abort the workflow.
    Abort,
    /// Store this value in the context key and continue.
    UseDefault(Value),
}

/// A single step in a workflow.
pub struct Step {
    pub name: &'static str,
    pub model: String,
    pub prompt_fn: Box<dyn Fn(&WorkflowContext) -> String + Send + Sync>,
    pub output_format: OutputFormat,
    pub output_key: &'static str,
    /// Optional system message prepended before the user prompt.
    pub system_prompt: Option<String>,
    /// Inference options (temperature, max_tokens, top_p, seed).
    pub options: Option<ModelOptions>,
    /// Retry this step up to `max_retries` additional times on error.
    pub max_retries: u32,
    /// When `Some`, the step is skipped if this closure returns `true`.
    pub skip_if: Option<Box<dyn Fn(&WorkflowContext) -> bool + Send + Sync>>,
    /// Optional post-processing applied to the parsed output before storing.
    pub transform_output: Option<Box<dyn Fn(Value, &WorkflowContext) -> Value + Send + Sync>>,
    /// Optional per-step error recovery.
    pub error_handler:
        Option<Box<dyn Fn(&TermiError, &WorkflowContext) -> StepErrorAction + Send + Sync>>,
    /// If set, the LLM call is cancelled after this many milliseconds.
    pub timeout_ms: Option<u64>,
}

// ── Fluent builder ────────────────────────────────────────────────────────────

/// Builder for a `Step`. Obtain one via `StepBuilder::new("name")`.
pub struct StepBuilder {
    name: &'static str,
    model: Option<String>,
    prompt_fn: Option<Box<dyn Fn(&WorkflowContext) -> String + Send + Sync>>,
    output_format: OutputFormat,
    output_key: Option<&'static str>,
    system_prompt: Option<String>,
    options: ModelOptions,
    max_retries: u32,
    skip_if: Option<Box<dyn Fn(&WorkflowContext) -> bool + Send + Sync>>,
    transform_output: Option<Box<dyn Fn(Value, &WorkflowContext) -> Value + Send + Sync>>,
    error_handler:
        Option<Box<dyn Fn(&TermiError, &WorkflowContext) -> StepErrorAction + Send + Sync>>,
    timeout_ms: Option<u64>,
}

impl StepBuilder {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            model: None,
            prompt_fn: None,
            output_format: OutputFormat::Text,
            output_key: None,
            system_prompt: None,
            options: ModelOptions::default(),
            max_retries: 0,
            skip_if: None,
            transform_output: None,
            error_handler: None,
            timeout_ms: None,
        }
    }

    /// Set the Ollama model for this step.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the prompt builder closure. Receives the current `WorkflowContext`
    /// and must return the prompt string.
    pub fn prompt<F>(mut self, f: F) -> Self
    where
        F: Fn(&WorkflowContext) -> String + Send + Sync + 'static,
    {
        self.prompt_fn = Some(Box::new(f));
        self
    }

    /// Prepend a system message before the user prompt.
    pub fn system_prompt(mut self, text: impl Into<String>) -> Self {
        self.system_prompt = Some(text.into());
        self
    }

    /// Set the inference temperature (0.0 = deterministic, 1.0 = creative).
    pub fn temperature(mut self, t: f32) -> Self {
        self.options.temperature = Some(t);
        self
    }

    /// Limit the number of tokens the model may generate.
    pub fn max_tokens(mut self, n: i32) -> Self {
        self.options.num_predict = Some(n);
        self
    }

    /// Set the top-p nucleus sampling probability.
    pub fn top_p(mut self, p: f32) -> Self {
        self.options.top_p = Some(p);
        self
    }

    /// Set a fixed random seed for reproducible outputs.
    pub fn seed(mut self, s: u32) -> Self {
        self.options.seed = Some(s);
        self
    }

    /// Retry the step up to `n` additional times on any error before failing.
    pub fn with_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Skip this step entirely when the closure returns `true`.
    pub fn skip_if<F>(mut self, f: F) -> Self
    where
        F: Fn(&WorkflowContext) -> bool + Send + Sync + 'static,
    {
        self.skip_if = Some(Box::new(f));
        self
    }

    /// Apply a transformation to the parsed LLM output before it is stored in
    /// the context. Receives the parsed `Value` and the current context.
    pub fn transform_output<F>(mut self, f: F) -> Self
    where
        F: Fn(Value, &WorkflowContext) -> Value + Send + Sync + 'static,
    {
        self.transform_output = Some(Box::new(f));
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

    /// Set the context key under which the parsed output will be stored.
    pub fn store_as(mut self, key: &'static str) -> Self {
        self.output_key = Some(key);
        self
    }

    /// Attach a per-step error handler. Called when the LLM call or output
    /// validation fails. Return `StepErrorAction::UseDefault(v)` to store a
    /// fallback value and continue, or `StepErrorAction::Abort` (the default
    /// when no handler is set) to propagate the error.
    pub fn on_error<F>(mut self, f: F) -> Self
    where
        F: Fn(&TermiError, &WorkflowContext) -> StepErrorAction + Send + Sync + 'static,
    {
        self.error_handler = Some(Box::new(f));
        self
    }

    /// Cancel the LLM call if it takes longer than `ms` milliseconds.
    /// Returns `TermiError::Timeout` when triggered.
    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    /// Finalise the builder and return a `Step`.
    ///
    /// # Panics
    /// Panics if `model`, `prompt`, or `store_as` were not called.
    pub fn finish(self) -> Step {
        let options_empty = self.options.temperature.is_none()
            && self.options.top_p.is_none()
            && self.options.top_k.is_none()
            && self.options.num_predict.is_none()
            && self.options.stop.is_none()
            && self.options.seed.is_none();

        Step {
            name: self.name,
            model: self.model.unwrap_or_else(|| {
                panic!(
                    "Step \"{}\": model() must be called before finish()",
                    self.name
                )
            }),
            prompt_fn: self.prompt_fn.unwrap_or_else(|| {
                panic!(
                    "Step \"{}\": prompt() must be called before finish()",
                    self.name
                )
            }),
            output_format: self.output_format,
            output_key: self.output_key.unwrap_or_else(|| {
                panic!(
                    "Step \"{}\": store_as() must be called before finish()",
                    self.name
                )
            }),
            system_prompt: self.system_prompt,
            options: if options_empty {
                None
            } else {
                Some(self.options)
            },
            max_retries: self.max_retries,
            skip_if: self.skip_if,
            transform_output: self.transform_output,
            error_handler: self.error_handler,
            timeout_ms: self.timeout_ms,
        }
    }
}
