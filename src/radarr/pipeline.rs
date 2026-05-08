use std::sync::Arc;

use tokio::sync::mpsc;

use crate::error::TermiError;
use crate::media::{self, ServiceConfig};
use crate::ollama::OllamaClient;
use crate::workflow::events::StepEvent;

pub struct RadarrPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl RadarrPipeline {
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

    pub async fn run(
        &self,
        query: &str,
        base_url: &str,
        api_key: &str,
    ) -> Result<String, TermiError> {
        let cfg = ServiceConfig {
            name: "Radarr",
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            search_path: "/api/v3/movie/lookup",
            title_field: "title",
        };

        let output = media::run_pipeline(
            Arc::clone(&self.client),
            self.model.clone(),
            &cfg,
            query,
            self.events.clone(),
        )
        .await?;

        if output.results.is_empty() {
            let msg = format!("No results found for \"{}\".", output.corrected_query);
            if let Some(tx) = &self.events {
                let _ = tx
                    .send(StepEvent::StatusUpdate {
                        message: msg.clone(),
                    })
                    .await;
                let _ = tx
                    .send(StepEvent::WorkflowComplete(Some(msg.clone())))
                    .await;
            }
            return Ok(msg);
        }

        // Ask for selection (either via TUI or console)
        let selection = output.select(&self.events).await?;

        let result_msg = if let Some(idx) = selection {
            let item = &output.results[idx];
            if item.already_added {
                format!("\"{}\" is already in Radarr.", item.display)
            } else {
                if let Some(tx) = &self.events {
                    let _ = tx
                        .send(StepEvent::StatusUpdate {
                            message: format!("Adding \"{}\" to Radarr...", item.display),
                        })
                        .await;
                }
                media::post_add_media(base_url, api_key, "/api/v3/movie", &item.raw).await?;
                format!("✓  Added \"{}\" to Radarr.", item.display)
            }
        } else {
            "Selection cancelled.".to_string()
        };

        if let Some(tx) = &self.events {
            let _ = tx
                .send(StepEvent::StatusUpdate {
                    message: result_msg.clone(),
                })
                .await;
            let _ = tx
                .send(StepEvent::WorkflowComplete(Some(result_msg.clone())))
                .await;
        }

        Ok(result_msg)
    }
}
