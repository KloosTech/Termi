use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use chrono::Local;
use tokio::sync::mpsc;
use tracing::info;

use crate::error::TermiError;
use crate::ollama::OllamaClient;
use crate::workflow::context::WorkflowContext;
use crate::workflow::events::StepEvent;
use crate::workflow::runner::Workflow;
use crate::workflow::step::StepBuilder;

use super::prompts;

/// Walk the full error source chain and join each cause with ": ".
fn full_error_chain(e: &dyn std::error::Error) -> String {
    let mut msg = e.to_string();
    let mut src = e.source();
    while let Some(s) = src {
        msg.push_str(&format!(": {s}"));
        src = s.source();
    }
    msg
}

pub struct DeepSearchPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    depth: usize,
    events: Option<mpsc::Sender<StepEvent>>,
    vault_path: Option<String>,
}

/// Keep the original name available so `main.rs` and `mod.rs` require no changes.
pub type SearchtorPipeline = DeepSearchPipeline;

impl DeepSearchPipeline {
    pub fn new(client: Arc<dyn OllamaClient>, model: String) -> Self {
        Self {
            client,
            model,
            depth: 3,
            events: None,
            vault_path: None,
        }
    }

    pub fn with_depth(mut self, n: usize) -> Self {
        self.depth = n;
        self
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    pub fn with_vault(mut self, path: impl Into<String>) -> Self {
        self.vault_path = Some(path.into());
        self
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    async fn emit_status(&self, message: impl Into<String>) {
        if let Some(tx) = &self.events {
            let _ = tx
                .send(StepEvent::StatusUpdate {
                    message: message.into(),
                })
                .await;
        }
    }

    /// Run a single-step LLM workflow and return the resulting context.
    async fn run_llm_step(
        &self,
        step: StepBuilder,
        ctx: WorkflowContext,
    ) -> Result<WorkflowContext, TermiError> {
        let mut b = Workflow::builder();
        if let Some(tx) = self.events.clone() {
            b = b.with_events(tx);
        }
        b.step(step)
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await
    }

    /// Save the final report as a markdown file in the Obsidian vault.
    /// Failures are logged as warnings — the pipeline never errors because of this.
    async fn save_to_vault(&self, vault_path: &str, query: &str, document: &str) {
        let now = chrono::Local::now();
        let date_str = now.format("%Y-%m-%d").to_string();

        // Sanitise the query into a safe filename.
        let safe_name: String = query
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == ' ' {
                    c
                } else {
                    '-'
                }
            })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let safe_name = if safe_name.len() > 60 {
            safe_name.chars().take(60).collect::<String>()
        } else {
            safe_name
        };

        let filename = format!("{safe_name} - {date_str}.md");
        let full_path = Path::new(vault_path).join(&filename);

        let frontmatter = format!(
            "---\ntags:\n  - research\n  - searchtor\ndate: {date_str}\nquery: \"{query}\"\n---\n\n"
        );
        let content = format!("{frontmatter}{document}");

        match tokio::fs::create_dir_all(vault_path).await {
            Err(e) => {
                tracing::warn!("Could not create vault directory '{}': {e}", vault_path);
                return;
            }
            Ok(_) => {}
        }

        match tokio::fs::write(&full_path, &content).await {
            Ok(_) => {
                info!("Report saved to vault: {}", full_path.display());
                self.emit_status(format!("Saved to vault: {filename}"))
                    .await;
            }
            Err(e) => {
                tracing::warn!("Could not write report to '{}': {e}", full_path.display());
            }
        }
    }

    /// Query SearXNG via curl (shell out) to avoid any reqwest connectivity issues.
    /// Retries up to 3 times with exponential back-off.
    async fn fetch_and_format(
        &self,
        searxng_base: &str,
        query: &str,
    ) -> Result<String, TermiError> {
        const MAX_ATTEMPTS: u32 = 3;
        const BASE_DELAY_MS: u64 = 800;

        let endpoint = format!("{}/search", searxng_base.trim_end_matches('/'));
        let mut last_err = String::new();

        for attempt in 0..MAX_ATTEMPTS {
            if attempt > 0 {
                let delay_ms = BASE_DELAY_MS * 2u64.pow(attempt - 1);
                self.emit_status(format!(
                    "SearXNG retry {}/{} (waiting {}ms)…",
                    attempt,
                    MAX_ATTEMPTS - 1,
                    delay_ms
                ))
                .await;
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }

            self.emit_status(format!("Searching: {query}")).await;

            // Shell out to curl — matches exactly what the SearXNG docs show and
            // avoids any reqwest connection-pool / HTTP-version issues.
            let output = tokio::process::Command::new("curl")
                .args([
                    "--silent",
                    "--show-error",
                    "--max-time",
                    "20",
                    "--request",
                    "POST",
                    "--data",
                    &format!("q={}&format=json", urlencoding::encode(query)),
                    "--header",
                    "Content-Type: application/x-www-form-urlencoded",
                    &endpoint,
                ])
                .output()
                .await;

            match output {
                Err(e) => {
                    last_err = format!("curl spawn failed: {}", full_error_chain(&e));
                    tracing::warn!(attempt = attempt + 1, error = %last_err, "curl spawn error");
                }
                Ok(out) => {
                    if !out.status.success() {
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        last_err = format!(
                            "curl exited {} — {}",
                            out.status.code().unwrap_or(-1),
                            stderr.trim()
                        );
                        tracing::warn!(attempt = attempt + 1, error = %last_err, "curl failed");
                        continue;
                    }
                    let body = String::from_utf8_lossy(&out.stdout).into_owned();
                    // Small courtesy delay before the next request.
                    tokio::time::sleep(Duration::from_millis(300)).await;
                    return Ok(prompts::parse_searxng_results(&body));
                }
            }
        }

        Err(TermiError::Pipeline(format!(
            "SearXNG fetch failed after {MAX_ATTEMPTS} attempts: {last_err}"
        )))
    }

    // ── Main entry point ──────────────────────────────────────────────────────

    pub async fn run(&self, query: String) -> Result<String, TermiError> {
        info!("deep_search: starting for '{}'", query);
        let result = self.run_inner(query).await;
        if let Err(ref e) = result {
            if let Some(tx) = &self.events {
                let _ = tx
                    .send(StepEvent::WorkflowFailed {
                        message: e.to_string(),
                    })
                    .await;
            }
        }
        result
    }

    async fn run_inner(&self, query: String) -> Result<String, TermiError> {
        info!("deep_search: run_inner for '{}'", query);

        // ── Phase 1: Query generation ─────────────────────────────────────────
        self.emit_status(format!(
            "Generating {} search queries per section...",
            self.depth
        ))
        .await;

        let schema = serde_json::json!({
            "type": "object",
            "required": [
                "executive_summary","objectives","methodology","findings",
                "conclusions","recommendations","appendices"
            ],
            "properties": {
                "executive_summary": {"type": "array", "items": {"type": "string"}},
                "objectives":        {"type": "array", "items": {"type": "string"}},
                "methodology":       {"type": "array", "items": {"type": "string"}},
                "findings":          {"type": "array", "items": {"type": "string"}},
                "conclusions":       {"type": "array", "items": {"type": "string"}},
                "recommendations":   {"type": "array", "items": {"type": "string"}},
                "appendices":        {"type": "array", "items": {"type": "string"}}
            }
        });

        let gen_prompt = prompts::build_query_generation_prompt(&query, self.depth);
        let ctx = WorkflowContext::new().with("gen_prompt", &gen_prompt);
        let model = self.model.clone();

        let ctx = self
            .run_llm_step(
                StepBuilder::new("generate_queries")
                    .model(model)
                    .prompt(|ctx| ctx.get_str("gen_prompt").to_string())
                    .output_json_schema(schema)
                    .store_as("queries"),
                ctx,
            )
            .await?;

        let queries_val = ctx
            .get("queries")
            .ok_or_else(|| TermiError::Pipeline("query generation produced no output".into()))?
            .clone();

        let queries_map = prompts::parse_query_plan(queries_val)?;

        info!("deep_search: query plan ready");

        // ── Phase 2: Sequential search & findings accumulation ────────────────
        // Findings live in Rust — only the prompt string enters the WorkflowContext.
        let mut findings: HashMap<String, String> = prompts::sections()
            .iter()
            .map(|(k, _)| (k.to_string(), String::new()))
            .collect();

        let total_sections = prompts::sections().len();

        for (si, (section_key, section_label)) in prompts::sections().iter().enumerate() {
            let queries = queries_map.get(*section_key).cloned().unwrap_or_default();
            let total_q = queries.len();

            for (qi, query_str) in queries.iter().enumerate() {
                self.emit_status(format!(
                    "[{}/{}] {section_label} — query {}/{}: {query_str}",
                    si + 1,
                    total_sections,
                    qi + 1,
                    total_q,
                ))
                .await;

                let formatted = self
                    .fetch_and_format("http://192.168.1.54:8080", query_str)
                    .await?;

                let existing = findings.get(*section_key).cloned().unwrap_or_default();
                let analysis_prompt =
                    prompts::build_analysis_prompt(section_label, query_str, &formatted, &existing);

                let ctx = WorkflowContext::new().with("analysis_prompt", &analysis_prompt);
                let model = self.model.clone();
                let ctx = self
                    .run_llm_step(
                        StepBuilder::new("analyze")
                            .model(model)
                            .prompt(|ctx| ctx.get_str("analysis_prompt").to_string())
                            .output_text()
                            .store_as("analysis"),
                        ctx,
                    )
                    .await?;

                let new_text = ctx.get_str("analysis").to_string();
                findings.entry(section_key.to_string()).and_modify(|f| {
                    if !f.is_empty() {
                        f.push_str("\n\n---\n\n");
                    }
                    f.push_str(&new_text);
                });
            }

            info!("deep_search: '{}' analysis complete", section_key);
        }

        // ── Phase 3: Section writing ──────────────────────────────────────────
        self.emit_status("Writing structured sections...").await;
        let mut written_sections: HashMap<String, String> = HashMap::new();

        for (section_key, section_label) in prompts::sections() {
            self.emit_status(format!("Writing: {section_label}")).await;

            let findings_text = findings.get(*section_key).cloned().unwrap_or_default();
            let write_prompt = prompts::build_section_writing_prompt(section_label, &findings_text);
            let ctx = WorkflowContext::new().with("write_prompt", &write_prompt);
            let model = self.model.clone();

            let ctx = self
                .run_llm_step(
                    StepBuilder::new("write_section")
                        .model(model)
                        .prompt(|ctx| ctx.get_str("write_prompt").to_string())
                        .output_text()
                        .store_as("section_text"),
                    ctx,
                )
                .await?;

            written_sections.insert(
                section_key.to_string(),
                ctx.get_str("section_text").to_string(),
            );
        }

        // ── Phase 4: Final synthesis ──────────────────────────────────────────
        self.emit_status("Synthesising final document...").await;

        let mut all_sections = String::new();
        for (section_key, section_label) in prompts::sections() {
            let text = written_sections
                .get(*section_key)
                .cloned()
                .unwrap_or_default();
            all_sections.push_str(&format!("## {section_label}\n\n{text}\n\n"));
        }

        let synthesis_prompt = prompts::build_synthesis_prompt(&query, &all_sections);
        let ctx = WorkflowContext::new().with("synthesis_prompt", &synthesis_prompt);
        let model = self.model.clone();

        let ctx = self
            .run_llm_step(
                StepBuilder::new("synthesize")
                    .model(model)
                    .prompt(|ctx| ctx.get_str("synthesis_prompt").to_string())
                    .output_text()
                    .store_as("document"),
                ctx,
            )
            .await?;

        let document = ctx.get_str("document").to_string();

        // ── Phase 5: Save to Obsidian vault ──────────────────────────────────
        if let Some(ref vault_path) = self.vault_path {
            self.save_to_vault(vault_path, &query, &document).await;
        }

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        info!("deep_search: complete for '{}'", query);
        Ok(document)
    }
}
