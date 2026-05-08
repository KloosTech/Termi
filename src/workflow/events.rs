use tokio::sync::oneshot;

/// Events emitted by the workflow engine and consumed by the TUI (or ignored).
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
    /// All workflow nodes have completed. Optionally includes a summary
    /// string to display in the TUI's Reading phase.
    WorkflowComplete(Option<String>),
    /// The workflow encountered a fatal error.
    WorkflowFailed { message: String },
    /// Request user selection from a list of options.
    /// The TUI should display the options and send back the chosen index.
    SelectRequest {
        prompt: String,
        options: Vec<String>,
        reply: oneshot::Sender<Option<usize>>,
    },
}

impl std::fmt::Debug for StepEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StepStarted { name, model } => f
                .debug_struct("StepStarted")
                .field("name", name)
                .field("model", model)
                .finish(),
            Self::Token { step, text } => f
                .debug_struct("Token")
                .field("step", step)
                .field("text", text)
                .finish(),
            Self::StepCompleted {
                name,
                total_tokens,
                elapsed_ms,
            } => f
                .debug_struct("StepCompleted")
                .field("name", name)
                .field("total_tokens", total_tokens)
                .field("elapsed_ms", elapsed_ms)
                .finish(),
            Self::StepSkipped { name } => {
                f.debug_struct("StepSkipped").field("name", name).finish()
            }
            Self::StatusUpdate { message } => f
                .debug_struct("StatusUpdate")
                .field("message", message)
                .finish(),
            Self::ContextSnapshot { entries } => f
                .debug_struct("ContextSnapshot")
                .field("entries", entries)
                .finish(),
            Self::WorkflowComplete(summary) => {
                f.debug_tuple("WorkflowComplete").field(summary).finish()
            }
            Self::WorkflowFailed { message } => f
                .debug_struct("WorkflowFailed")
                .field("message", message)
                .finish(),
            Self::SelectRequest {
                prompt, options, ..
            } => f
                .debug_struct("SelectRequest")
                .field("prompt", prompt)
                .field("options", options)
                .field("reply", &"<oneshot::Sender>")
                .finish(),
        }
    }
}
