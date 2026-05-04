use std::sync::Arc;

use reqwest::Client;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::error::TermiError;
use crate::ollama::OllamaClient;
use crate::workflow::context::WorkflowContext;
use crate::workflow::events::StepEvent;
use crate::workflow::http::HttpStepBuilder;
use crate::workflow::runner::Workflow;
use crate::workflow::step::StepBuilder;
use crate::workflow::url_encode;

pub struct ServiceConfig {
    pub name: &'static str,
    pub base_url: String,
    pub api_key: String,
    /// e.g. "/api/v3/series/lookup"
    pub search_path: &'static str,
    /// JSON field used as the display title, e.g. "title" or "artistName"
    pub title_field: &'static str,
}

pub struct MediaResult {
    pub display: String,
    pub already_added: bool,
    pub raw: Value,
}

pub struct MediaSearchOutput {
    pub corrected_query: String,
    pub results: Vec<MediaResult>,
}

fn build_display(item: &Value, title_field: &str) -> String {
    let title = item.get(title_field).and_then(|v| v.as_str()).unwrap_or("Unknown");
    let year = item.get("year").and_then(|v| v.as_u64());
    let already = item.get("alreadyAdded").and_then(|v| v.as_bool()).unwrap_or(false);

    let mut s = title.to_string();
    if let Some(y) = year {
        s.push_str(&format!(" ({})", y));
    }
    if already {
        s.push_str(" [already added]");
    }
    s
}

/// Runs the spell-correction + search workflow and returns normalised results.
pub async fn run_pipeline(
    client: Arc<dyn OllamaClient>,
    model: String,
    cfg: &ServiceConfig,
    query: &str,
    events: Option<mpsc::Sender<StepEvent>>,
) -> Result<MediaSearchOutput, TermiError> {
    // Capture &'static str fields by copy — safe to move into closures.
    let search_path = cfg.search_path;
    let title_field = cfg.title_field;

    let mut b = Workflow::builder();
    if let Some(tx) = events {
        b = b.with_events(tx);
    }

    let ctx = WorkflowContext::new()
        .with("raw_query", query)
        .with("base_url", cfg.base_url.as_str());

    let ctx = b
        .step(
            StepBuilder::new("fix_spelling")
                .model(&model)
                .temperature(0.0)
                .system_prompt(
                    "You are a spelling corrector for media titles. \
                     Return ONLY the corrected title, nothing else.",
                )
                .prompt(|ctx| {
                    format!("Correct any misspellings in this title: {}", ctx.get_str("raw_query"))
                })
                .output_text()
                .store_as("corrected_query"),
        )
        .http(
            HttpStepBuilder::new("search")
                .url(move |ctx| {
                    format!(
                        "{}{}?term={}",
                        ctx.get_str("base_url"),
                        search_path,
                        url_encode(ctx.get_str("corrected_query")),
                    )
                })
                .store_as("raw_results")
                .header("X-Api-Key", cfg.api_key.as_str())
                .timeout_secs(15),
        )
        .transform("normalize", move |ctx| {
            // HTTP step stores the body as a JSON string; parse it to an array.
            let raw = ctx.get_str("raw_results").to_string();
            let parsed: Vec<Value> = serde_json::from_str(&raw).unwrap_or_default();
            ctx.set("results", parsed);
        })
        .build()
        .run(client, ctx)
        .await?;

    let corrected = ctx.get_str("corrected_query").to_string();
    let raw_results = ctx.get_array("results").to_vec();

    let results = raw_results
        .into_iter()
        .map(|item| {
            let already =
                item.get("alreadyAdded").and_then(|v| v.as_bool()).unwrap_or(false);
            let display = build_display(&item, title_field);
            MediaResult { display, already_added: already, raw: item }
        })
        .collect();

    Ok(MediaSearchOutput { corrected_query: corrected, results })
}

/// POSTs a media item to a *arr add endpoint.
///
/// Returns an error on non-2xx, surfacing the `{"message":"..."}` field from
/// the response body when present.
pub async fn post_add_media(
    base_url: &str,
    api_key: &str,
    endpoint: &str,
    body: &Value,
) -> Result<(), TermiError> {
    let url = format!("{}{}", base_url, endpoint);

    let resp = Client::new()
        .post(&url)
        .header("X-Api-Key", api_key)
        .json(body)
        .send()
        .await
        .map_err(|e| TermiError::Pipeline(format!("POST {} failed: {}", url, e)))?;

    if resp.status().is_success() {
        return Ok(());
    }

    let status = resp.status().as_u16();
    let body_text = resp.text().await.unwrap_or_default();
    let msg = serde_json::from_str::<Value>(&body_text)
        .ok()
        .and_then(|v| v.get("message").and_then(|m| m.as_str()).map(String::from))
        .unwrap_or(body_text);

    Err(TermiError::Pipeline(format!("POST {} returned {}: {}", url, status, msg)))
}
