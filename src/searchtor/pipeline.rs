use std::sync::Arc;
use std::time::Duration;

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
        // Initialize findings keys in the context for visibility.
        let mut ctx = ctx;
        for (section_key, _) in prompts::sections() {
            ctx.set(&format!("findings_{}", section_key), "");
        }

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

                let findings_key = format!("findings_{}", section_key);
                let existing = ctx.get_str(&findings_key).to_string();
                let analysis_prompt =
                    prompts::build_analysis_prompt(section_label, query_str, &formatted, &existing);

                // Run the analysis step using the current context to maintain visibility.
                let model = self.model.clone();
                ctx = ctx.with("analysis_prompt", &analysis_prompt);
                ctx = self
                    .run_llm_step(
                        StepBuilder::new("analyze")
                            .model(model)
                            .prompt(|ctx| ctx.get_str("analysis_prompt").to_string())
                            .output_text()
                            .store_as("analysis_output"),
                        ctx,
                    )
                    .await?;

                let new_text = ctx.get_str("analysis_output").to_string();
                let mut updated_findings = existing;
                if !updated_findings.is_empty() {
                    updated_findings.push_str("\n\n---\n\n");
                }
                updated_findings.push_str(&new_text);
                ctx.set(&findings_key, updated_findings);
            }

            info!("deep_search: '{}' analysis complete", section_key);
        }

        // ── Phase 3: Section writing ──────────────────────────────────────────
        self.emit_status("Writing structured sections...").await;

        for (section_key, section_label) in prompts::sections() {
            self.emit_status(format!("Writing: {section_label}")).await;

            let findings_key = format!("findings_{}", section_key);
            let findings_text = ctx.get_str(&findings_key).to_string();
            let write_prompt = prompts::build_section_writing_prompt(section_label, &findings_text);

            let model = self.model.clone();
            let section_key_str = section_key.to_string();
            ctx = ctx.with("write_prompt", &write_prompt);
            ctx = self
                .run_llm_step(
                    StepBuilder::new("write_section")
                        .model(model)
                        .prompt(|ctx| ctx.get_str("write_prompt").to_string())
                        .output_text()
                        .store_as(format!("section_{}", section_key_str)),
                    ctx,
                )
                .await?;
        }

        // ── Phase 4: Final synthesis ──────────────────────────────────────────
        self.emit_status("Synthesising final document...").await;

        let mut all_sections = String::new();
        for (section_key, section_label) in prompts::sections() {
            let section_output_key = format!("section_{}", section_key);
            let text = ctx.get_str(&section_output_key).to_string();
            all_sections.push_str(&format!("## {section_label}\n\n{text}\n\n"));
        }

        let synthesis_prompt = prompts::build_synthesis_prompt(&query, &all_sections);
        ctx = ctx.with("synthesis_prompt", &synthesis_prompt);
        let model = self.model.clone();

        ctx = self
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

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        info!("deep_search: complete for '{}'", query);
        Ok(document)
    }
}
