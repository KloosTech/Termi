use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

use crate::error::TermiError;
use crate::ollama::types::{
    ChatRequest, ChatResponse, ChatStreamChunk, EmbeddingsRequest, EmbeddingsResponse,
    GenerateRequest, GenerateResponse, GenerateStreamChunk, TagsResponse,
};

pub type BoxStream<T> = Pin<Box<dyn Stream<Item = Result<T, TermiError>> + Send>>;

#[async_trait]
pub trait OllamaClient: Send + Sync {
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, TermiError>;
    async fn chat_stream(&self, req: ChatRequest) -> Result<BoxStream<ChatStreamChunk>, TermiError>;

    async fn generate(&self, req: GenerateRequest) -> Result<GenerateResponse, TermiError>;
    async fn generate_stream(
        &self,
        req: GenerateRequest,
    ) -> Result<BoxStream<GenerateStreamChunk>, TermiError>;

    async fn list_models(&self) -> Result<TagsResponse, TermiError>;
    async fn embeddings(&self, req: EmbeddingsRequest) -> Result<EmbeddingsResponse, TermiError>;
}
