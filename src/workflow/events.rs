/// Events emitted by the workflow engine and consumed by the TUI (or ignored).
#[derive(Debug)]
pub enum StepEvent {
    /// A step is about to make its first LLM streaming call.
    StepStarted { name: &'static str, model: String },
    /// A single streamed token chunk arrived.
    Token { step: &'static str, text: String },
    /// A step finished successfully.
    StepCompleted {
        name: &'static str,
        total_tokens: u32,
        elapsed_ms: u128,
    },
    /// A step was skipped via `skip_if`.
    StepSkipped { name: &'static str },
    /// A non-LLM status message (e.g. "Reading N files…").
    StatusUpdate { message: String },
    /// Full snapshot of the WorkflowContext after a step wrote to it.
    /// Consumed by the debug panel in the TUI; ignored otherwise.
    ContextSnapshot {
        entries: Vec<(String, serde_json::Value)>,
    },
    /// All workflow nodes have completed.
    WorkflowComplete,
    /// The workflow encountered a fatal error.
    WorkflowFailed { message: String },
}
