use thiserror::Error;

#[derive(Debug, Error)]
pub enum TermiError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Directory walk error: {0}")]
    Walk(#[from] walkdir::Error),

    #[error("Ollama API error: status={status}, body={body}")]
    OllamaApi { status: u16, body: String },

    #[error("Streaming error: {0}")]
    Stream(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("step {step:?} failed: {source}")]
    StepFailed {
        step: String,
        #[source]
        source: Box<TermiError>,
    },

    #[error("step {step:?} validation failed: {message}")]
    ValidationFailed { step: String, message: String },

    #[error("loop limit {limit} exceeded in step {step:?}")]
    LoopLimitExceeded { step: String, limit: usize },

    #[error("step {step:?} timed out after {ms}ms")]
    Timeout { step: String, ms: u64 },
}
