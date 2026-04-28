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
}
