use std::sync::Arc;

use tokio::sync::mpsc;

use crate::error::TermiError;
use crate::ollama::OllamaClient;
use crate::workflow::context::WorkflowContext;
use crate::workflow::events::StepEvent;
use crate::workflow::http::{url_encode, HttpStepBuilder};
use crate::workflow::runner::Workflow;
use crate::workflow::step::StepBuilder;

pub struct SearchtorPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl SearchtorPipeline {
    pub fn new(client: Arc<dyn OllamaClient>, model: String) -> Self {
        Self {
            client,
            model,
            events: None,
        }
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    pub async fn run(&self, query: String) -> Result<String, TermiError> {
        let mut b = Workflow::builder();
        if let Some(tx) = self.events.clone() {
            b = b.with_events(tx);
        }

        let search_url = format!(
            "http://192.168.1.54:8080/search?q={}&format=json",
            url_encode(&query)
        );

        let ctx = WorkflowContext::new()
            .with("url", search_url)
            .with("question", query);

        let model = self.model.clone();

        let workflow_result = b
            .http(
                HttpStepBuilder::new("fetch")
                    .url(|ctx| ctx.get_str("url").to_string())
                    .store_as("content")
                    .strip_html()
                    .timeout_secs(20),
            )
            .step(
                StepBuilder::new("answer")
                    .model(model)
                    .prompt(|ctx| {
                        format!(
                            "Using only the search results below, answer the query: {}\n\nSearch Results:\n{}",
                            ctx.get_str("question"),
                            ctx.get_str("content"),
                        )
                    })
                    .output_text()
                    .store_as("answer"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await;

        match workflow_result {
            Ok(ctx) => {
                if let Some(tx) = &self.events {
                    let _ = tx.send(StepEvent::WorkflowComplete).await;
                }
                Ok(ctx.get_str("answer").to_string())
            }
            Err(e) => {
                if let Some(tx) = &self.events {
                    let _ = tx
                        .send(StepEvent::WorkflowFailed {
                            message: e.to_string(),
                        })
                        .await;
                }
                Err(e)
            }
        }
    }
}
