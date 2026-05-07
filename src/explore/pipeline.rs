use std::path::Path;
use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::error::TermiError;
use crate::explore::prompts::{format_file_contents, format_file_list};
use crate::explore::walker::{walk_directory, FileEntry};
use crate::ollama::OllamaClient;
use crate::workflow::context::WorkflowContext;
use crate::workflow::events::StepEvent;
use crate::workflow::presets;

pub struct ExploreConfig {
    pub model: String,
    /// Skip files larger than this (bytes) in the read step.
    pub max_file_bytes: u64,
    /// Stop reading once total content reaches this (bytes).
    pub max_total_content_bytes: usize,
}

impl Default for ExploreConfig {
    fn default() -> Self {
        Self {
            model: "gemma4:e4b".to_string(),
            max_file_bytes: 128 * 1024,
            max_total_content_bytes: 512 * 1024,
        }
    }
}

pub struct ExplorePipeline {
    client: Arc<dyn OllamaClient>,
    config: ExploreConfig,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl ExplorePipeline {
    pub fn new(client: Arc<dyn OllamaClient>, config: ExploreConfig) -> Self {
        Self {
            client,
            config,
            events: None,
        }
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    pub async fn run(&self, root: &Path) -> Result<String, TermiError> {
        // ── Step 1: Walk the file tree ────────────────────────────────────────
        info!("explore: walking file tree at {:?}", root);
        let root_buf = root.to_path_buf();
        let entries: Vec<FileEntry> =
            tokio::task::spawn_blocking(move || walk_directory(&root_buf))
                .await
                .map_err(|e| TermiError::Pipeline(format!("spawn_blocking error: {e}")))??;

        info!("explore: found {} files", entries.len());
        let file_list_str = format_file_list(&entries);

        // ── Step 2: LLM identifies interesting files ──────────────────────────
        let mut ctx = WorkflowContext::new();
        ctx.set("file_list", &file_list_str);

        let mut filter_builder = presets::filter_files(&self.config.model);
        if let Some(tx) = self.events.clone() {
            filter_builder = filter_builder.with_events(tx);
        }

        let ctx = filter_builder
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        let interesting_paths: Vec<String> = ctx
            .get_array("interesting_files")
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();

        info!("explore: LLM selected {} files", interesting_paths.len());

        if let Some(tx) = &self.events {
            let _ = tx
                .send(StepEvent::StatusUpdate {
                    message: format!("Reading {} selected files...", interesting_paths.len()),
                })
                .await;
        }

        // ── Step 3: Read the selected files ───────────────────────────────────
        let mut file_contents: Vec<(String, String)> = Vec::new();
        let mut total_bytes = 0usize;

        for path_str in &interesting_paths {
            let Some(entry) = entries
                .iter()
                .find(|e| e.relative_path.to_string_lossy() == path_str.as_str())
            else {
                warn!(
                    "explore: LLM suggested '{}' which is not in the file list; skipping",
                    path_str
                );
                continue;
            };

            if entry.size_bytes > self.config.max_file_bytes {
                warn!(
                    "explore: '{}' is {} bytes, exceeds per-file limit; skipping",
                    path_str, entry.size_bytes
                );
                continue;
            }

            match tokio::fs::read_to_string(&entry.absolute_path).await {
                Ok(content) => {
                    total_bytes += content.len();
                    file_contents.push((path_str.clone(), content));
                    if total_bytes >= self.config.max_total_content_bytes {
                        warn!("explore: total content limit reached; stopping file reads");
                        break;
                    }
                }
                Err(e) => {
                    warn!("explore: could not read '{}': {}; skipping", path_str, e);
                }
            }
        }

        info!(
            "explore: read {} files ({} bytes total)",
            file_contents.len(),
            total_bytes
        );

        // ── Step 4: LLM summarizes the project ────────────────────────────────
        let contents_block = format_file_contents(&file_contents);

        let mut ctx2 = WorkflowContext::new();
        ctx2.set("file_contents", &contents_block);

        let mut summary_builder = presets::summarize_content(&self.config.model);
        if let Some(tx) = self.events.clone() {
            summary_builder = summary_builder.with_events(tx);
        }

        let ctx2 = summary_builder
            .build()
            .run(Arc::clone(&self.client), ctx2)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        Ok(ctx2.get_str("summary").to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use tempfile::TempDir;

    use crate::ollama::mock::{MockCall, MockOllamaClient};

    fn make_project() -> TempDir {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("main.rs"), "fn main() { println!(\"hi\"); }").unwrap();
        fs::write(
            root.join("lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }",
        )
        .unwrap();
        fs::write(root.join("README.md"), "# My Project\nA test project.").unwrap();
        dir
    }

    fn mock_with_filter(files_json: &str) -> Arc<MockOllamaClient> {
        Arc::new(MockOllamaClient::new("gemma4:e4b").with_chat_response(files_json))
    }

    #[tokio::test]
    async fn test_pipeline_makes_exactly_two_chat_calls() {
        let dir = make_project();
        let client = mock_with_filter(r#"["main.rs","lib.rs"]"#);

        let pipeline = ExplorePipeline::new(
            Arc::clone(&client) as Arc<dyn OllamaClient>,
            ExploreConfig {
                model: "gemma4:e4b".into(),
                ..Default::default()
            },
        );

        let result = pipeline.run(dir.path()).await;
        assert!(result.is_ok(), "pipeline failed: {:?}", result.err());

        let calls = client.recorded_calls().await;
        assert_eq!(
            calls.len(),
            2,
            "expected exactly 2 LLM calls, got: {calls:?}"
        );
        assert!(
            matches!(&calls[0], MockCall::ChatStream { .. }),
            "call 0 should be ChatStream (filter)"
        );
        assert!(
            matches!(&calls[1], MockCall::ChatStream { .. }),
            "call 1 should be ChatStream (summarize)"
        );
    }

    #[tokio::test]
    async fn test_pipeline_invalid_filter_json_returns_error() {
        let dir = make_project();
        let client = mock_with_filter("this is not json");

        let pipeline = ExplorePipeline::new(
            Arc::clone(&client) as Arc<dyn OllamaClient>,
            ExploreConfig {
                model: "gemma4:e4b".into(),
                ..Default::default()
            },
        );

        let result = pipeline.run(dir.path()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pipeline_handles_hallucinated_filenames() {
        let dir = make_project();
        // LLM suggests a real file AND a hallucinated one
        let client = mock_with_filter(r#"["main.rs","ghost.rs"]"#);

        let pipeline = ExplorePipeline::new(
            Arc::clone(&client) as Arc<dyn OllamaClient>,
            ExploreConfig {
                model: "gemma4:e4b".into(),
                ..Default::default()
            },
        );

        // Should not error — ghost.rs is skipped with a warning
        let result = pipeline.run(dir.path()).await;
        assert!(
            result.is_ok(),
            "should tolerate hallucinated filenames: {:?}",
            result.err()
        );
        // Still exactly 2 LLM calls
        assert_eq!(client.recorded_calls().await.len(), 2);
    }

    #[tokio::test]
    async fn test_pipeline_empty_file_list_still_completes() {
        let dir = make_project();
        // LLM returns empty array — no files to read, still runs summarize step
        let client = mock_with_filter(r#"[]"#);

        let pipeline = ExplorePipeline::new(
            Arc::clone(&client) as Arc<dyn OllamaClient>,
            ExploreConfig {
                model: "gemma4:e4b".into(),
                ..Default::default()
            },
        );

        let result = pipeline.run(dir.path()).await;
        assert!(result.is_ok());
        assert_eq!(client.recorded_calls().await.len(), 2);
    }
}
