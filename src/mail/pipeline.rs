use std::sync::Arc;

use tokio::sync::mpsc;

use crate::error::TermiError;
use crate::mail::imap::{ImapFetcher, ParsedEmail};
use crate::mail::store::MailStore;
use crate::ollama::client::OllamaClient;
use crate::workflow::context::WorkflowContext;
use crate::workflow::events::StepEvent;
use crate::workflow::runner::Workflow;
use crate::workflow::step::StepBuilder;

pub struct MailPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
    username: String,
    password: String,
    limit: usize,
}

impl MailPipeline {
    pub fn new(
        client: Arc<dyn OllamaClient>,
        model: String,
        username: String,
        password: String,
    ) -> Self {
        Self {
            client,
            model,
            events: None,
            username,
            password,
            limit: 50,
        }
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub async fn run(&self) -> Result<String, TermiError> {
        self.emit(StepEvent::StatusUpdate {
            message: "Connecting to IMAP...".to_string(),
        })
        .await;

        let mut store = MailStore::load()?;
        let seen_ids = store.seen_ids_snapshot();
        let host = std::env::var("MAIL_HOST").unwrap_or_else(|_| "imap.ionos.de".to_string());
        let fetcher = ImapFetcher::new(self.username.clone(), self.password.clone(), host);
        let emails = fetcher.fetch_new(seen_ids, self.limit).await?;

        if emails.is_empty() {
            self.emit(StepEvent::WorkflowComplete).await;
            return Ok("No new emails since last run.".to_string());
        }

        // Mark seen before LLM work (idempotent — even if pipeline fails, won't reprocess)
        let new_ids: Vec<String> = emails.iter().map(|e| e.message_id.clone()).collect();
        store.merge_seen(new_ids);
        store.save()?;

        self.emit(StepEvent::StatusUpdate {
            message: format!("Analysing {} new emails...", emails.len()),
        })
        .await;

        let emails_text = format_emails_for_llm(&emails);
        let email_count = emails.len().to_string();

        let mut ctx = WorkflowContext::new();
        ctx.set("emails", &emails_text);
        ctx.set("email_count", &email_count);

        let mut b = Workflow::builder();
        if let Some(tx) = self.events.clone() {
            b = b.with_events(tx);
        }

        let ctx = b
            .step(
                StepBuilder::new("triage")
                    .model(self.model.clone())
                    .system_prompt(
                        "You are an expert email assistant. Analyse emails and categorise each one.\n\
                         Categories:\n\
                         - urgent: requires immediate attention today\n\
                         - action_required: needs a response or action, not urgent\n\
                         - fyi: informational only, no action needed\n\
                         - newsletter: marketing or subscription content\n\
                         - automated: automated notifications, receipts, confirmations\n\
                         - spam: unwanted or suspicious mail\n\n\
                         For each email output: [#N] CATEGORY — one-line reason\n\
                         Be terse. No preamble.",
                    )
                    .prompt(|ctx| {
                        format!(
                            "Triage these {} emails:\n\n{}",
                            ctx.get_str("email_count"),
                            ctx.get_str("emails")
                        )
                    })
                    .output_text()
                    .store_as("triage"),
            )
            .step(
                StepBuilder::new("extract_actions")
                    .model(self.model.clone())
                    .system_prompt(
                        "You are a personal assistant extracting concrete action items from emails.\n\
                         Focus only on urgent and action_required emails.\n\
                         For each action item:\n\
                         - ACTION: what must be done\n\
                         - FROM: who sent it\n\
                         - DEADLINE: specific date/time if mentioned, else 'unspecified'\n\
                         - PRIORITY: high / medium / low\n\n\
                         If no action items exist, output exactly: No action items.\n\
                         Be terse. No preamble.",
                    )
                    .prompt(|ctx| {
                        format!(
                            "Extract action items.\n\nTriage:\n{}\n\nEmails:\n{}",
                            ctx.get_str("triage"),
                            ctx.get_str("emails")
                        )
                    })
                    .output_text()
                    .store_as("action_items"),
            )
            .step(
                StepBuilder::new("report")
                    .model(self.model.clone())
                    .system_prompt(
                        "You are an executive assistant producing a concise email briefing.\n\
                         Format as clean markdown with exactly these sections:\n\n\
                         ## Summary\n\
                         One sentence total count + breakdown by category.\n\n\
                         ## Action Items\n\
                         Numbered list, most urgent first. Include deadline and sender.\n\
                         If none, write: None.\n\n\
                         ## Notable Messages\n\
                         2-3 sentence summaries of important FYI emails only. Skip newsletters/spam/automated.\n\
                         If none, write: None.\n\n\
                         ## Skipped\n\
                         One line: count of newsletters + automated + spam.\n\n\
                         Be concise. This is a digest, not a transcript.",
                    )
                    .prompt(|ctx| {
                        format!(
                            "Produce the email briefing.\n\nTriage:\n{}\n\nAction Items:\n{}\n\nTotal emails: {}",
                            ctx.get_str("triage"),
                            ctx.get_str("action_items"),
                            ctx.get_str("email_count")
                        )
                    })
                    .output_text()
                    .store_as("report"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        self.emit(StepEvent::WorkflowComplete).await;

        Ok(ctx.get_str("report").to_string())
    }

    async fn emit(&self, event: StepEvent) {
        if let Some(tx) = &self.events {
            let _ = tx.send(event).await;
        }
    }
}

fn format_emails_for_llm(emails: &[ParsedEmail]) -> String {
    let total = emails.len();
    let mut out = String::new();
    for (i, email) in emails.iter().enumerate() {
        out.push_str(&format!(
            "---\n[Email {}/{}]\nFrom: {}\nSubject: {}\nDate: {}\n\n{}\n\n",
            i + 1,
            total,
            email.from,
            email.subject,
            email.date,
            email.body.chars().take(1500).collect::<String>()
        ));
    }
    out
}
