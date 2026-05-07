use std::sync::Arc;

use tokio::sync::mpsc;

use crate::error::TermiError;
use crate::media::{self, MediaSearchOutput, ServiceConfig};
use crate::ollama::OllamaClient;
use crate::workflow::events::StepEvent;

pub struct SonarrPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl SonarrPipeline {
    pub fn new(client: Arc<dyn OllamaClient>, model: String) -> Self {
        Self { client, model, events: None }
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    pub async fn run(
        &self,
        query: &str,
        base_url: &str,
        api_key: &str,
    ) -> Result<MediaSearchOutput, TermiError> {
        let cfg = ServiceConfig {
            name: "Sonarr",
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            search_path: "/api/v3/series/lookup",
            title_field: "title",
        };

        let result = media::run_pipeline(
            Arc::clone(&self.client),
            self.model.clone(),
            &cfg,
            query,
            self.events.clone(),
        )
        .await;

        if let Some(tx) = &self.events {
            match &result {
                Ok(o) => {
                    let msg = if o.results.is_empty() {
                        format!("No results found for \"{}\"", o.corrected_query)
                    } else {
                        format!(
                            "Found {} result(s) for \"{}\" — press q to choose",
                            o.results.len(),
                            o.corrected_query
                        )
                    };
                    let _ = tx.send(StepEvent::StatusUpdate { message: msg }).await;
                }
                Err(e) => {
                    let _ = tx
                        .send(StepEvent::WorkflowFailed { message: e.to_string() })
                        .await;
                }
            }
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        result
    }
}
