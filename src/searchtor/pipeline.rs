use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::info;

use crate::error::TermiError;
use crate::ollama::OllamaClient;
use crate::workflow::context::WorkflowContext;
use crate::workflow::events::StepEvent;
use crate::workflow::http::{url_encode, HttpStepBuilder};
use crate::workflow::runner::Workflow;
use crate::workflow::step::StepBuilder;

use super::prompts;

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
        Self { client, model, depth: 3, events: None }
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
            let _ = tx.send(StepEvent::StatusUpdate { message: message.into() }).await;
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
        b.step(step).build().run(Arc::clone(&self.client), ctx).await
    }

    /// HTTP-fetch a SearXNG URL and parse the top-10 results into formatted text.
    async fn fetch_and_format(&self, url: String) -> Result<String, TermiError> {
        let ctx = WorkflowContext::new().with("search_url", &url);
        let mut b = Workflow::builder();
        if let Some(tx) = self.events.clone() {
            b = b.with_events(tx);
        }
        let ctx = b
            .http(
                // SearXNG returns JSON — do NOT strip_html
                HttpStepBuilder::new("fetch_raw")
                    .url(|ctx| ctx.get_str("search_url").to_string())
                    .store_as("raw_results")
                    .timeout_secs(20),
            )
            .transform("parse_results", |ctx| {
                // Clone before mutating to satisfy the borrow checker.
                let raw = ctx.get_str("raw_results").to_string();
                let formatted = prompts::parse_searxng_results(&raw);
                ctx.set("formatted_results", formatted);
            })
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;
        Ok(ctx.get_str("formatted_results").to_string())
    }

    // ── Main entry point ──────────────────────────────────────────────────────

    pub async fn run(&self, query: String) -> Result<String, TermiError> {
        info!("deep_search: starting for '{}'", query);

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

                let search_url = format!(
                    "http://192.168.1.54:8080/search?q={}&format=json",
                    url_encode(query_str)
                );

                let formatted = self.fetch_and_format(search_url).await?;

                let existing = findings.get(*section_key).cloned().unwrap_or_default();
                let analysis_prompt = prompts::build_analysis_prompt(
                    section_label,
                    query_str,
                    &formatted,
                    &existing,
                );

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
            let text = written_sections.get(*section_key).cloned().unwrap_or_default();
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

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        info!("deep_search: complete for '{}'", query);
        Ok(ctx.get_str("document").to_string())
    }
}
