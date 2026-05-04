use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::stream;
use tokio::sync::Mutex;

use crate::error::TermiError;
use crate::ollama::client::{BoxStream, OllamaClient};
use crate::ollama::types::*;

/// Records which method was called and key parameters, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum MockCall {
    Chat {
        model: String,
        message_count: usize,
        has_system: bool,
    },
    ChatStream {
        model: String,
        message_count: usize,
        has_system: bool,
    },
    Generate {
        model: String,
        prompt_len: usize,
    },
    GenerateStream {
        model: String,
        prompt_len: usize,
    },
    ListModels,
    Embeddings {
        model: String,
        prompt: String,
    },
}

pub struct MockOllamaClient {
    pub calls: Arc<Mutex<Vec<MockCall>>>,
    pub model: String,
    pub chat_response_text: String,
    pub generate_response_text: String,
    pub model_list: Vec<String>,
    pub embedding: Vec<f32>,
    /// How many subsequent calls to `chat()` or `chat_stream()` should return an error.
    fail_remaining: Arc<Mutex<u32>>,
    /// Per-call response queue. When non-empty, `chat()` or `chat_stream()` pops from the front;
    /// falls back to `chat_response_text` when the queue is exhausted.
    response_queue: Arc<Mutex<VecDeque<String>>>,
}

impl MockOllamaClient {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            model: model.into(),
            chat_response_text: "Mock chat response".to_string(),
            generate_response_text: "Mock generate response".to_string(),
            model_list: vec!["llama3:latest".to_string()],
            embedding: vec![0.1, 0.2, 0.3],
            fail_remaining: Arc::new(Mutex::new(0)),
            response_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub fn with_chat_response(mut self, text: impl Into<String>) -> Self {
        self.chat_response_text = text.into();
        self
    }

    pub fn with_generate_response(mut self, text: impl Into<String>) -> Self {
        self.generate_response_text = text.into();
        self
    }

    /// Make the first `n` calls to `chat()` or `chat_stream()` return a `Pipeline` error.
    pub fn with_fail_first_n(mut self, n: u32) -> Self {
        self.fail_remaining = Arc::new(Mutex::new(n));
        self
    }

    /// Pre-load an ordered list of chat responses. Each call pops one
    /// from the front; once exhausted, `chat_response_text` is used instead.
    pub fn with_responses(
        mut self,
        responses: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let queue: VecDeque<String> = responses.into_iter().map(|s| s.into()).collect();
        self.response_queue = Arc::new(Mutex::new(queue));
        self
    }

    pub async fn recorded_calls(&self) -> Vec<MockCall> {
        self.calls.lock().await.clone()
    }

    async fn next_chat_text(&self) -> String {
        let mut q = self.response_queue.lock().await;
        q.pop_front()
            .unwrap_or_else(|| self.chat_response_text.clone())
    }

    fn make_chat_response(&self, model: &str, text: String) -> ChatResponse {
        ChatResponse {
            model: model.to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            message: Message::assistant(text),
            done: true,
            done_reason: Some("stop".to_string()),
            total_duration: Some(100_000_000),
            eval_count: Some(10),
        }
    }

    fn make_generate_response(&self, model: &str) -> GenerateResponse {
        GenerateResponse {
            model: model.to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            response: self.generate_response_text.clone(),
            done: true,
            context: None,
            total_duration: Some(100_000_000),
        }
    }
}

#[async_trait]
impl OllamaClient for MockOllamaClient {
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, TermiError> {
        let has_system = req.messages.iter().any(|m| m.role == "system");

        // Fail-first check
        {
            let mut remaining = self.fail_remaining.lock().await;
            if *remaining > 0 {
                *remaining -= 1;
                self.calls.lock().await.push(MockCall::Chat {
                    model: req.model.clone(),
                    message_count: req.messages.len(),
                    has_system,
                });
                return Err(TermiError::Pipeline("mock failure".to_string()));
            }
        }

        self.calls.lock().await.push(MockCall::Chat {
            model: req.model.clone(),
            message_count: req.messages.len(),
            has_system,
        });

        let text = self.next_chat_text().await;
        Ok(self.make_chat_response(&req.model, text))
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<ChatStreamChunk>, TermiError> {
        let has_system = req.messages.iter().any(|m| m.role == "system");

        // Fail-first check
        {
            let mut remaining = self.fail_remaining.lock().await;
            if *remaining > 0 {
                *remaining -= 1;
                self.calls.lock().await.push(MockCall::ChatStream {
                    model: req.model.clone(),
                    message_count: req.messages.len(),
                    has_system,
                });
                return Err(TermiError::Pipeline("mock failure".to_string()));
            }
        }

        self.calls.lock().await.push(MockCall::ChatStream {
            model: req.model.clone(),
            message_count: req.messages.len(),
            has_system,
        });

        let model = req.model.clone();
        let text = self.next_chat_text().await;
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        let word_count = words.len();

        let chunks: Vec<ChatStreamChunk> = words
            .into_iter()
            .enumerate()
            .map(|(i, word)| {
                let is_last = i == word_count - 1;
                ChatStreamChunk {
                    model: model.clone(),
                    created_at: "2024-01-01T00:00:00Z".to_string(),
                    message: Message::assistant(format!("{} ", word)),
                    done: is_last,
                    done_reason: if is_last {
                        Some("stop".to_string())
                    } else {
                        None
                    },
                    eval_count: if is_last {
                        Some(word_count as u32)
                    } else {
                        None
                    },
                    eval_duration: None,
                }
            })
            .collect();
        Ok(Box::pin(stream::iter(chunks.into_iter().map(Ok))))
    }

    async fn generate(&self, req: GenerateRequest) -> Result<GenerateResponse, TermiError> {
        self.calls.lock().await.push(MockCall::Generate {
            model: req.model.clone(),
            prompt_len: req.prompt.len(),
        });
        Ok(self.make_generate_response(&req.model))
    }

    async fn generate_stream(
        &self,
        req: GenerateRequest,
    ) -> Result<BoxStream<GenerateStreamChunk>, TermiError> {
        self.calls.lock().await.push(MockCall::GenerateStream {
            model: req.model.clone(),
            prompt_len: req.prompt.len(),
        });
        let model = req.model.clone();
        let word_count = self.generate_response_text.split_whitespace().count();
        let chunks: Vec<GenerateStreamChunk> = self
            .generate_response_text
            .split_whitespace()
            .enumerate()
            .map(|(i, word)| GenerateStreamChunk {
                model: model.clone(),
                response: format!("{} ", word),
                done: i == word_count - 1,
            })
            .collect();
        Ok(Box::pin(stream::iter(chunks.into_iter().map(Ok))))
    }

    async fn list_models(&self) -> Result<TagsResponse, TermiError> {
        self.calls.lock().await.push(MockCall::ListModels);
        let models = self
            .model_list
            .iter()
            .map(|name| ModelInfo {
                name: name.clone(),
                model: name.clone(),
                modified_at: "2024-01-01T00:00:00Z".to_string(),
                size: 4_000_000_000,
                digest: "sha256:mock".to_string(),
                details: ModelDetails {
                    format: "gguf".to_string(),
                    family: "llama".to_string(),
                    parameter_size: "8B".to_string(),
                    quantization_level: "Q4_0".to_string(),
                },
            })
            .collect();
        Ok(TagsResponse { models })
    }

    async fn embeddings(&self, req: EmbeddingsRequest) -> Result<EmbeddingsResponse, TermiError> {
        self.calls.lock().await.push(MockCall::Embeddings {
            model: req.model.clone(),
            prompt: req.prompt.clone(),
        });
        Ok(EmbeddingsResponse {
            embedding: self.embedding.clone(),
        })
    }
}

// ── SlowMockOllamaClient ──────────────────────────────────────────────────────

/// A mock that sleeps for `delay_ms` before returning, used to test timeouts.
pub struct SlowMockOllamaClient {
    delay_ms: u64,
}

impl SlowMockOllamaClient {
    pub fn new(delay_ms: u64) -> Self {
        Self { delay_ms }
    }
}

#[async_trait]
impl OllamaClient for SlowMockOllamaClient {
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, TermiError> {
        tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
        Ok(ChatResponse {
            model: req.model.clone(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            message: Message::assistant("slow response".to_string()),
            done: true,
            done_reason: Some("stop".to_string()),
            total_duration: None,
            eval_count: None,
        })
    }

    async fn chat_stream(
        &self,
        _req: ChatRequest,
    ) -> Result<BoxStream<ChatStreamChunk>, TermiError> {
        unimplemented!("SlowMockOllamaClient::chat_stream")
    }

    async fn generate(&self, _req: GenerateRequest) -> Result<GenerateResponse, TermiError> {
        unimplemented!("SlowMockOllamaClient::generate")
    }

    async fn generate_stream(
        &self,
        _req: GenerateRequest,
    ) -> Result<BoxStream<GenerateStreamChunk>, TermiError> {
        unimplemented!("SlowMockOllamaClient::generate_stream")
    }

    async fn list_models(&self) -> Result<TagsResponse, TermiError> {
        unimplemented!("SlowMockOllamaClient::list_models")
    }

    async fn embeddings(&self, _req: EmbeddingsRequest) -> Result<EmbeddingsResponse, TermiError> {
        unimplemented!("SlowMockOllamaClient::embeddings")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_chat_records_call() {
        let mock = MockOllamaClient::new("llama3");
        let req = ChatRequest {
            model: "llama3".into(),
            messages: vec![Message::user("hello")],
            ..Default::default()
        };
        let resp = mock.chat(req).await.unwrap();
        assert_eq!(resp.message.content, "Mock chat response");
        assert!(resp.done);
        let calls = mock.recorded_calls().await;
        assert_eq!(calls.len(), 1);
        assert!(matches!(
            &calls[0],
            MockCall::Chat {
                model,
                message_count: 1,
                ..
            } if model == "llama3"
        ));
    }

    #[tokio::test]
    async fn test_mock_chat_records_has_system() {
        let mock = MockOllamaClient::new("llama3");
        let req = ChatRequest {
            model: "llama3".into(),
            messages: vec![Message::system("be helpful"), Message::user("hi")],
            ..Default::default()
        };
        mock.chat(req).await.unwrap();
        let calls = mock.recorded_calls().await;
        assert!(matches!(
            &calls[0],
            MockCall::Chat {
                has_system: true,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_mock_fail_first_n_then_succeed() {
        let mock = MockOllamaClient::new("llama3")
            .with_chat_response("ok")
            .with_fail_first_n(2);
        let req = ChatRequest {
            model: "llama3".into(),
            messages: vec![Message::user("hi")],
            ..Default::default()
        };
        assert!(mock.chat(req.clone()).await.is_err());
        assert!(mock.chat(req.clone()).await.is_err());
        assert!(mock.chat(req).await.is_ok());
        assert_eq!(mock.recorded_calls().await.len(), 3);
    }

    #[tokio::test]
    async fn test_mock_with_responses_queue() {
        let mock = MockOllamaClient::new("llama3").with_responses(["first", "second", "third"]);
        let make_req = || ChatRequest {
            model: "llama3".into(),
            messages: vec![Message::user("hi")],
            ..Default::default()
        };

        let r1 = mock.chat(make_req()).await.unwrap();
        let r2 = mock.chat(make_req()).await.unwrap();
        let r3 = mock.chat(make_req()).await.unwrap();
        // Queue exhausted — falls back to default
        let r4 = mock.chat(make_req()).await.unwrap();

        assert_eq!(r1.message.content, "first");
        assert_eq!(r2.message.content, "second");
        assert_eq!(r3.message.content, "third");
        assert_eq!(r4.message.content, "Mock chat response");
    }

    #[tokio::test]
    async fn test_mock_chat_stream_records_call_and_yields_chunks() {
        use futures_util::StreamExt;

        let mock = MockOllamaClient::new("llama3").with_chat_response("hello world");
        let req = ChatRequest {
            model: "llama3".into(),
            messages: vec![Message::user("hi")],
            ..Default::default()
        };
        let mut stream = mock.chat_stream(req).await.unwrap();
        let mut content = String::new();
        while let Some(chunk) = stream.next().await {
            content.push_str(&chunk.unwrap().message.content);
        }
        assert!(content.contains("hello"));
        assert!(content.contains("world"));

        let calls = mock.recorded_calls().await;
        assert!(matches!(&calls[0], MockCall::ChatStream { .. }));
    }

    #[tokio::test]
    async fn test_mock_list_models_records_call() {
        let mock = MockOllamaClient::new("llama3");
        let resp = mock.list_models().await.unwrap();
        assert!(!resp.models.is_empty());
        assert_eq!(resp.models[0].name, "llama3:latest");
        let calls = mock.recorded_calls().await;
        assert!(matches!(calls[0], MockCall::ListModels));
    }

    #[tokio::test]
    async fn test_mock_embeddings_records_prompt() {
        let mock = MockOllamaClient::new("llama3");
        let req = EmbeddingsRequest {
            model: "llama3".into(),
            prompt: "hello world".into(),
            options: None,
        };
        let resp = mock.embeddings(req).await.unwrap();
        assert_eq!(resp.embedding, vec![0.1, 0.2, 0.3]);
        let calls = mock.recorded_calls().await;
        assert!(matches!(
            &calls[0],
            MockCall::Embeddings { prompt, .. } if prompt == "hello world"
        ));
    }

    #[tokio::test]
    async fn test_mock_records_multiple_calls_in_order() {
        let mock = MockOllamaClient::new("llama3");
        mock.list_models().await.unwrap();
        mock.chat(ChatRequest {
            model: "llama3".into(),
            messages: vec![Message::user("a")],
            ..Default::default()
        })
        .await
        .unwrap();
        mock.embeddings(EmbeddingsRequest {
            model: "llama3".into(),
            prompt: "b".into(),
            options: None,
        })
        .await
        .unwrap();

        let calls = mock.recorded_calls().await;
        assert_eq!(calls.len(), 3);
        assert!(matches!(calls[0], MockCall::ListModels));
        assert!(matches!(calls[1], MockCall::Chat { .. }));
        assert!(matches!(calls[2], MockCall::Embeddings { .. }));
    }
}
