use std::sync::Arc;

use serde_json::Value;
use tokio::process::Command;
use tokio::sync::mpsc;

use crate::error::TermiError;
use crate::ollama::OllamaClient;
use crate::workflow::context::WorkflowContext;
use crate::workflow::events::StepEvent;
use crate::workflow::runner::Workflow;
use crate::workflow::shell::ShellStepBuilder;
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

impl MediaSearchOutput {
    /// Select a result from the list. If events are provided, it asks the TUI.
    /// Otherwise, it uses dialoguer in the terminal.
    pub async fn select(
        &self,
        events: &Option<mpsc::Sender<StepEvent>>,
    ) -> Result<Option<usize>, crate::error::TermiError> {
        if self.results.is_empty() {
            return Ok(None);
        }

        let options: Vec<String> = self.results.iter().map(|r| r.display.clone()).collect();

        if let Some(tx) = events {
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            tx.send(StepEvent::SelectRequest {
                prompt: format!("Results for \"{}\"", self.corrected_query),
                options,
                reply: reply_tx,
            })
            .await
            .map_err(|e| {
                crate::error::TermiError::Pipeline(format!("Failed to send select request: {}", e))
            })?;

            let selection = reply_rx.await.map_err(|e| {
                crate::error::TermiError::Pipeline(format!("Failed to receive select reply: {}", e))
            })?;

            Ok(selection)
        } else {
            use dialoguer::{theme::ColorfulTheme, Select};
            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt(format!("Results for \"{}\"", self.corrected_query))
                .items(&options)
                .interact_opt()
                .map_err(|e| {
                    crate::error::TermiError::Pipeline(format!("Selection failed: {}", e))
                })?;

            Ok(selection)
        }
    }
}

fn build_display(item: &Value, title_field: &str) -> String {
    let title = item
        .get(title_field)
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown");
    let year = item.get("year").and_then(|v| v.as_u64());
    let already = item
        .get("alreadyAdded")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

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
/// Uses `curl` shell-outs for searches to avoid connectivity issues with local services.
pub async fn run_pipeline(
    client: Arc<dyn OllamaClient>,
    model: String,
    cfg: &ServiceConfig,
    query: &str,
    events: Option<mpsc::Sender<StepEvent>>,
) -> Result<MediaSearchOutput, TermiError> {
    let search_path = cfg.search_path;
    let title_field = cfg.title_field;

    // Strip any stray double-quotes that CLI tools or config files can introduce.
    let base_url = cfg.base_url.trim_matches('"').to_string();
    let api_key = cfg.api_key.trim_matches('"').to_string();
    let raw_query = query.trim_matches('"');

    // ── Phase 1: LLM spell-correction ─────────────────────────────────────────
    let mut b1 = Workflow::builder();
    if let Some(tx) = events.clone() {
        b1 = b1.with_events(tx);
    }

    let ctx = WorkflowContext::new()
        .with("raw_query", raw_query)
        .with("base_url", base_url.as_str());

    let ctx = b1
        .step(
            StepBuilder::new("fix_spelling")
                .model(&model)
                .temperature(0.0)
                .system_prompt(
                    "You are a spelling corrector for media titles. \
                     Return ONLY the corrected title, nothing else.",
                )
                .prompt(|ctx| {
                    format!(
                        "Correct any misspellings in this title: {}",
                        ctx.get_str("raw_query")
                    )
                })
                .output_text()
                .store_as("corrected_query"),
        )
        .build()
        .run(Arc::clone(&client), ctx)
        .await?;

    // LLMs sometimes wrap their answer in quotes — strip them.
    let corrected = ctx.get_str("corrected_query").trim_matches('"').to_string();

    // Pre-compute the full search URL so it appears in the debug context panel.
    let search_url = format!(
        "{}{}?term={}",
        base_url,
        search_path,
        url_encode(&corrected)
    );

    // Emit a status so the user sees progress between the two phases.
    if let Some(tx) = &events {
        let _ = tx
            .send(StepEvent::StatusUpdate {
                message: format!("Searching {} for \"{}\"…", cfg.name, corrected),
            })
            .await;
    }

    // ── Phase 2: Shell search (curl) + JSON normalisation ──────────────────────
    let mut b2 = Workflow::builder();
    if let Some(tx) = events.clone() {
        b2 = b2.with_events(tx);
    }

    // Seed the second workflow context with the corrected query and computed URL
    // so the debug panel shows exactly what is being fetched.
    let ctx = ctx
        .with("corrected_query", corrected.as_str())
        .with("search_url", search_url.as_str())
        .with("api_key", api_key.as_str());

    let ctx = b2
        .shell(
            ShellStepBuilder::new("search")
                .command(|ctx| {
                    format!(
                        "curl --fail --silent --show-error --max-time 15 -H \"X-Api-Key: {}\" \"{}\"",
                        ctx.get_str("api_key"),
                        ctx.get_str("search_url")
                    )
                })
                .store_stdout_as("raw_results"),
        )
        .transform("normalize", move |ctx| {
            let raw = ctx.get_str("raw_results").to_string();
            let parsed: Vec<Value> = serde_json::from_str(&raw).unwrap_or_default();
            ctx.set("results", parsed);
        })
        .build()
        .run(client, ctx)
        .await?;

    let raw_results = ctx.get_array("results").to_vec();

    let results = raw_results
        .into_iter()
        .map(|item| {
            let already = item
                .get("alreadyAdded")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let display = build_display(&item, title_field);
            MediaResult {
                display,
                already_added: already,
                raw: item,
            }
        })
        .collect();

    Ok(MediaSearchOutput {
        corrected_query: corrected,
        results,
    })
}

async fn curl_get_json(base_url: &str, api_key: &str, path: &str) -> Option<Value> {
    let url = format!("{}{}", base_url, path);
    let output = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--max-time",
            "10",
            "-H",
            &format!("X-Api-Key: {}", api_key),
            &url,
        ])
        .output()
        .await
        .ok()?;

    if !output.status.success() {
        return None;
    }
    serde_json::from_slice(&output.stdout).ok()
}

/// POSTs a media item to a *arr add endpoint via curl.
pub async fn post_add_media(
    base_url: &str,
    api_key: &str,
    endpoint: &str,
    body: &Value,
) -> Result<String, TermiError> {
    // Strip stray quotes for the same reason as run_pipeline above.
    let base_url = base_url.trim_matches('"');
    let api_key = api_key.trim_matches('"');

    // ── Enrich payload with defaults ───────────────────────────────────────
    // *arr services require root folder and profiles to be set when adding.
    // We fetch the first available one for each category if not already set.
    let mut payload = body.clone();
    let mut profile_name = None;

    if payload.get("rootFolderPath").is_none()
        || payload["rootFolderPath"].as_str().unwrap_or("").is_empty()
    {
        let path = if endpoint.contains("v1") {
            "/api/v1/rootfolder"
        } else {
            "/api/v3/rootfolder"
        };
        if let Some(folders) = curl_get_json(base_url, api_key, path).await {
            if let Some(folder) = folders.as_array().and_then(|a| a.first()) {
                if let Some(p) = folder.get("path") {
                    payload["rootFolderPath"] = p.clone();
                }
            }
        }
    }

    let qp_path = if endpoint.contains("v1") {
        "/api/v1/qualityprofile"
    } else {
        "/api/v3/qualityprofile"
    };

    if let Some(profiles) = curl_get_json(base_url, api_key, qp_path).await {
        let profile_list = profiles.as_array();

        if let Some(id_val) = payload.get("qualityProfileId") {
            // Try to find the name for the existing ID
            if let Some(list) = profile_list {
                profile_name = list
                    .iter()
                    .find(|p| p.get("id") == Some(id_val))
                    .and_then(|p| p.get("name"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
            }
        }

        // If still no ID or name, pick the first one
        if payload.get("qualityProfileId").is_none()
            || profile_name.is_none()
            || payload["qualityProfileId"].as_i64().unwrap_or(0) == 0
        {
            if let Some(p) = profile_list.and_then(|list| list.first()) {
                if let Some(id) = p.get("id") {
                    payload["qualityProfileId"] = id.clone();
                    profile_name = p.get("name").and_then(|v| v.as_str()).map(String::from);
                }
            }
        }
    }

    // Lidarr specific metadata profile
    if endpoint.contains("artist") {
        if let Some(profiles) = curl_get_json(base_url, api_key, "/api/v1/metadataprofile").await {
            let current_id = payload
                .get("metadataProfileId")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            if current_id == 0 {
                if let Some(id) = profiles
                    .as_array()
                    .and_then(|list| list.first())
                    .and_then(|p| p.get("id"))
                {
                    payload["metadataProfileId"] = id.clone();
                }
            }
        }

        // Ensure Lidarr specific options
        payload["addOptions"] = serde_json::json!({
            "searchForMissingAlbums": true
        });
        if payload.get("artistType").is_none() {
            payload["artistType"] = Value::String("group".to_string());
        }
    }

    // Force monitored to true so it actually works
    payload["monitored"] = Value::Bool(true);

    let url = format!("{}{}", base_url, endpoint);
    let body_json = serde_json::to_string(&payload)
        .map_err(|e| TermiError::Pipeline(format!("Failed to serialize media item: {}", e)))?;

    let output = Command::new("curl")
        .args([
            "--silent",
            "--show-error",
            "-w",
            "\n%{http_code}",
            "--max-time",
            "15",
            "--request",
            "POST",
            "--header",
            "Content-Type: application/json",
            "--header",
            &format!("X-Api-Key: {}", api_key),
            "--data",
            &body_json,
            &url,
        ])
        .output()
        .await
        .map_err(|e| {
            TermiError::Pipeline(format!("curl POST to {} failed to launch: {}", url, e))
        })?;

    let full_output = String::from_utf8_lossy(&output.stdout);
    let mut out_parts = full_output.rsplitn(2, '\n');
    let http_status_str = out_parts.next().unwrap_or("0").trim();
    let stdout_raw = out_parts.next().unwrap_or("");
    let http_status: u16 = http_status_str.parse().unwrap_or(0);

    if output.status.success() && (200..300).contains(&http_status) {
        let root = payload
            .get("rootFolderPath")
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        let profile = profile_name.as_deref().unwrap_or("default");
        let monitored = payload
            .get("monitored")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        return Ok(format!(
            "Config: Folder={}, Profile={}, Monitored={}",
            root,
            profile,
            if monitored { "Yes" } else { "No" }
        ));
    }

    let status_code = if http_status > 0 {
        http_status as i32
    } else {
        output.status.code().unwrap_or(-1)
    };
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Try to parse message from body if it's JSON
    let msg = serde_json::from_str::<Value>(stdout_raw)
        .ok()
        .and_then(|v| {
            if let Some(m) = v.get("message").and_then(|m| m.as_str()) {
                Some(m.to_string())
            } else if let Some(errors) = v.as_array() {
                let err_msgs: Vec<_> = errors
                    .iter()
                    .filter_map(|e| {
                        e.get("errorMessage")
                            .and_then(|v| v.as_str())
                            .or_else(|| e.get("message").and_then(|v| v.as_str()))
                    })
                    .collect();
                if !err_msgs.is_empty() {
                    Some(err_msgs.join(", "))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            if !stderr.is_empty() {
                stderr.trim().to_string()
            } else {
                stdout_raw.trim().to_string()
            }
        });

    Err(TermiError::Pipeline(format!(
        "POST to {} failed (exit code {}). Details: {}",
        url, status_code, msg
    )))
}
