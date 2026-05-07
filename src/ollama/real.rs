use std::time::Duration;

use tracing::warn;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt, TryStreamExt};
use tokio_util::codec::{FramedRead, LinesCodec};

use crate::error::TermiError;
use crate::ollama::client::{BoxStream, OllamaClient};
use crate::ollama::types::*;

pub struct RealOllamaClient {
    base_url: String,
    http: reqwest::Client,
}

impl RealOllamaClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap_or_else(|e| {
                warn!(
                    "Failed to build reqwest client with custom timeout: {}. Using default client.",
                    e
                );
                reqwest::Client::new()
            });
        Self {
            base_url: base_url.into(),
            http,
        }
    }

    async fn check_status(&self, resp: reqwest::Response) -> Result<reqwest::Response, TermiError> {
        if resp.status().is_success() {
            return Ok(resp);
        }
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        Err(TermiError::OllamaApi { status, body })
    }
}

/// Convert a reqwest byte stream to an async line-by-line stream.
fn byte_stream_to_lines(resp: reqwest::Response) -> impl Stream<Item = Result<String, TermiError>> {
    let byte_stream = resp
        .bytes_stream()
        .map_err(|e| std::io::Error::other(e.to_string()));
    let reader = tokio_util::io::StreamReader::new(byte_stream);
    FramedRead::new(reader, LinesCodec::new()).map_err(|e| TermiError::Stream(e.to_string()))
}

#[async_trait]
impl OllamaClient for RealOllamaClient {
    async fn chat(&self, mut req: ChatRequest) -> Result<ChatResponse, TermiError> {
        req.stream = Some(false);
        let resp = self
            .http
            .post(format!("{}/api/chat", self.base_url))
            .json(&req)
            .send()
            .await?;
        let resp = self.check_status(resp).await?;
        Ok(resp.json::<ChatResponse>().await?)
    }

    async fn chat_stream(
        &self,
        mut req: ChatRequest,
    ) -> Result<BoxStream<ChatStreamChunk>, TermiError> {
        req.stream = Some(true);
        let resp = self
            .http
            .post(format!("{}/api/chat", self.base_url))
            .json(&req)
            .send()
            .await?;
        let resp = self.check_status(resp).await?;
        let lines = byte_stream_to_lines(resp);
        let stream = lines.filter_map(|line_result| async move {
            match line_result {
                Err(e) => Some(Err(e)),
                Ok(line) if line.trim().is_empty() => None,
                Ok(line) => {
                    Some(serde_json::from_str::<ChatStreamChunk>(&line).map_err(TermiError::Json))
                }
            }
        });
        Ok(Box::pin(stream))
    }

    async fn generate(&self, mut req: GenerateRequest) -> Result<GenerateResponse, TermiError> {
        req.stream = Some(false);
        let resp = self
            .http
            .post(format!("{}/api/generate", self.base_url))
            .json(&req)
            .send()
            .await?;
        let resp = self.check_status(resp).await?;
        Ok(resp.json::<GenerateResponse>().await?)
    }

    async fn generate_stream(
        &self,
        mut req: GenerateRequest,
    ) -> Result<BoxStream<GenerateStreamChunk>, TermiError> {
        req.stream = Some(true);
        let resp = self
            .http
            .post(format!("{}/api/generate", self.base_url))
            .json(&req)
            .send()
            .await?;
        let resp = self.check_status(resp).await?;
        let lines = byte_stream_to_lines(resp);
        let stream = lines.filter_map(|line_result| async move {
            match line_result {
                Err(e) => Some(Err(e)),
                Ok(line) if line.trim().is_empty() => None,
                Ok(line) => Some(
                    serde_json::from_str::<GenerateStreamChunk>(&line).map_err(TermiError::Json),
                ),
            }
        });
        Ok(Box::pin(stream))
    }

    async fn list_models(&self) -> Result<TagsResponse, TermiError> {
        let resp = self
            .http
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await?;
        let resp = self.check_status(resp).await?;
        Ok(resp.json::<TagsResponse>().await?)
    }

    async fn embeddings(&self, req: EmbeddingsRequest) -> Result<EmbeddingsResponse, TermiError> {
        let resp = self
            .http
            .post(format!("{}/api/embeddings", self.base_url))
            .json(&req)
            .send()
            .await?;
        let resp = self.check_status(resp).await?;
        Ok(resp.json::<EmbeddingsResponse>().await?)
    }
}
