use std::sync::Arc;

use tokio::sync::mpsc;

use crate::error::TermiError;
use crate::ollama::client::OllamaClient;
use crate::workflow::context::WorkflowContext;
use crate::workflow::events::StepEvent;
use crate::workflow::runner::Workflow;
use crate::workflow::shell::ShellStepBuilder;
use crate::workflow::step::StepBuilder;

pub struct TechRadarPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl TechRadarPipeline {
    pub fn new(client: Arc<dyn OllamaClient>, model: String) -> Self {
        Self { client, model, events: None }
    }

    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    pub async fn run(&self, topic: &str) -> Result<String, TermiError> {
        let mut b = Workflow::builder();
        if let Some(tx) = self.events.clone() {
            b = b.with_events(tx);
        }

        let topic = topic.to_string();
        let topic2 = topic.clone();

        let ctx = WorkflowContext::new().with("topic", &topic);

        let ctx = b
            .shell(
                ShellStepBuilder::new("fetch_hn_stories")
                    .command(move |_ctx| {
                        format!(
                            "curl -s 'https://hn.algolia.com/api/v1/search?query={}&tags=story&hitsPerPage=15' \
                             -H 'User-Agent: termi/0.1' 2>/dev/null \
                             | grep -o '\"title\":\"[^\"]*\"\\|\"url\":\"[^\"]*\"\\|\"points\":[0-9]*' \
                             | head -60 || echo 'HN fetch failed'",
                            topic
                        )
                    })
                    .store_stdout_as("hn_stories")
                    .timeout_secs(20),
            )
            .shell(
                ShellStepBuilder::new("fetch_popular_crates")
                    .command(move |_ctx| {
                        format!(
                            "curl -s 'https://crates.io/api/v1/crates?sort=recent-downloads&per_page=20&q={topic}' \
                             -H 'User-Agent: termi/0.1' 2>/dev/null \
                             | grep -o '\"name\":\"[^\"]*\"\\|\"downloads\":[0-9]*\\|\"description\":\"[^\"]*\"' \
                             | head -80 || echo 'crates.io fetch failed'",
                            topic = topic2
                        )
                    })
                    .store_stdout_as("crates_data")
                    .timeout_secs(20),
            )
            .shell(
                ShellStepBuilder::new("fetch_rust_blog")
                    .command(move |_ctx| {
                        "curl -s 'https://blog.rust-lang.org/feed.xml' -H 'User-Agent: termi/0.1' 2>/dev/null \
                         | grep -o '<title>[^<]*</title>' \
                         | sed 's/<[^>]*>//g' \
                         | head -20 || echo 'Rust blog fetch failed'"
                            .to_string()
                    })
                    .store_stdout_as("blog_titles")
                    .timeout_secs(20),
            )
            .step(
                StepBuilder::new("analyze_trends")
                    .model(self.model.clone())
                    .system_prompt("You are a technology analyst. Respond only with valid JSON.")
                    .prompt(|ctx| {
                        format!(
                            "Analyse the following data sources and identify technology trends for the topic: '{}'. \
                            Return a JSON object with key 'trends' containing an array of objects with fields: \
                            name (string, the technology/tool/approach), \
                            ring (adopt|trial|assess|hold), \
                            quadrant (languages-frameworks|tools|platforms|techniques), \
                            confidence (high|medium|low), \
                            summary (string, 1-2 sentence description), \
                            signals (array of strings, evidence from the data).\n\n\
                            HN stories:\n{}\n\nPopular crates:\n{}\n\nRust blog:\n{}",
                            ctx.get_str("topic"),
                            ctx.get_str("hn_stories"),
                            ctx.get_str("crates_data"),
                            ctx.get_str("blog_titles"),
                        )
                    })
                    .output_json()
                    .store_as("trends"),
            )
            .step(
                StepBuilder::new("write_radar")
                    .model(self.model.clone())
                    .system_prompt("You are a technology strategist writing a technology radar report.")
                    .prompt(|ctx| {
                        format!(
                            "Write a technology radar report for the topic: '{}'. \
                            Use the classic ThoughtWorks radar format with four rings: \
                            Adopt (use now), Trial (worth pursuing), Assess (worth exploring), Hold (proceed with caution). \
                            For each item explain WHY it belongs in that ring. \
                            End with a 'Key Takeaways' section of 3-5 actionable insights.\n\n\
                            Trends identified:\n{}",
                            ctx.get_str("topic"),
                            ctx.get_str("trends"),
                        )
                    })
                    .output_text()
                    .store_as("radar"),
            )
            .build()
            .run(Arc::clone(&self.client), ctx)
            .await?;

        if let Some(tx) = &self.events {
            let _ = tx.send(StepEvent::WorkflowComplete).await;
        }

        Ok(ctx.get_str("radar").to_string())
    }
}
