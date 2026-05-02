use std::sync::Arc;

use crate::error::TermiError;
use crate::media::{self, MediaSearchOutput, ServiceConfig};
use crate::ollama::OllamaClient;

pub struct LidarrPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
}

impl LidarrPipeline {
    pub fn new(client: Arc<dyn OllamaClient>, model: String) -> Self {
        Self { client, model }
    }

    pub async fn run(
        &self,
        query: &str,
        base_url: &str,
        api_key: &str,
    ) -> Result<MediaSearchOutput, TermiError> {
        let cfg = ServiceConfig {
            name: "Lidarr",
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            search_path: "/api/v1/artist/lookup",
            title_field: "artistName",
        };
        media::run_pipeline(Arc::clone(&self.client), self.model.clone(), &cfg, query, None).await
    }
}
