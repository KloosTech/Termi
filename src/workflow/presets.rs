/// Reusable `WorkflowBuilder` fragments.
///
/// Each function returns a builder pre-loaded with one or more steps. Compose
/// them with [`WorkflowBuilder::extend`] or call `.build().run(…)` directly.
///
/// Context key contracts are documented per function.
use serde_json::json;

use crate::workflow::runner::WorkflowBuilder;
use crate::workflow::step::StepBuilder;

// ── filter_files ──────────────────────────────────────────────────────────────

/// Select the most architecturally relevant files from a directory listing.
///
/// **Input context key:** `"file_list"` — newline-separated relative paths.
/// **Output context key:** `"interesting_files"` — JSON array of path strings.
pub fn filter_files(model: impl Into<String>) -> WorkflowBuilder {
    let model = model.into();
    let schema = json!({"type": "array", "items": {"type": "string"}});
    WorkflowBuilder::new().step(
        StepBuilder::new("filter_files")
            .model(model)
            .prompt(|ctx| {
                let file_list = ctx.get_str("file_list");
                format!(
                    r#"You are a senior software engineer reviewing a codebase.

Below is the complete list of files in this project:

{file_list}

Your task: identify the 5–15 files that are MOST IMPORTANT for understanding:
- The overall architecture
- Core business logic
- Key data structures
- Entry points

Return ONLY a JSON array of relative file paths (strings), exactly as they appear in the list above.
No explanation. No markdown fences. Just a raw JSON array.

Example: ["src/main.rs", "src/lib.rs", "README.md"]"#
                )
            })
            .output_json_schema(schema)
            .store_as("interesting_files"),
    )
}

// ── summarize_content ─────────────────────────────────────────────────────────

/// Produce a technical prose summary of file contents.
///
/// **Input context key:** `"file_contents"` — formatted file contents block.
/// **Output context key:** `"summary"` — plain text.
pub fn summarize_content(model: impl Into<String>) -> WorkflowBuilder {
    let model = model.into();
    WorkflowBuilder::new().step(
        StepBuilder::new("summarize_content")
            .model(model)
            .prompt(|ctx| {
                let contents = ctx.get_str("file_contents");
                format!(
                    r#"You are a senior software engineer. Below are the contents of the most important files in a software project.

{contents}

Write a clear, thorough summary of this project covering:
1. Purpose and goals of the project
2. Overall architecture and key design patterns
3. Main components and how they interact
4. Notable implementation details or technology choices
5. Any interesting or unusual aspects

Be specific and technical. Assume the audience is an experienced developer."#
                )
            })
            .output_text()
            .store_as("summary"),
    )
}

// ── classify ──────────────────────────────────────────────────────────────────

/// Classify free-form input into one of the provided categories.
///
/// **Input context key:** `"input"` — text to classify.
/// **Output context key:** `"classification"` — JSON object `{label, confidence}`.
pub fn classify(model: impl Into<String>, categories: &[&str]) -> WorkflowBuilder {
    let model = model.into();
    let cats = categories.join(", ");
    let schema = json!({
        "type": "object",
        "required": ["label", "confidence"]
    });
    WorkflowBuilder::new().step(
        StepBuilder::new("classify")
            .model(model)
            .prompt(move |ctx| {
                let input = ctx.get_str("input");
                format!(
                    r#"Classify the following text into exactly one of these categories: {cats}

Text:
{input}

Respond with a JSON object only — no markdown, no explanation.
Format: {{"label": "<chosen category>", "confidence": <0.0-1.0>}}"#
                )
            })
            .output_json_schema(schema)
            .store_as("classification"),
    )
}

// ── qa ────────────────────────────────────────────────────────────────────────

/// Answer a question given a context passage.
///
/// **Input context keys:** `"context"` — background text; `"question"` — query.
/// **Output context key:** `"answer"` — plain text.
pub fn qa(model: impl Into<String>) -> WorkflowBuilder {
    let model = model.into();
    WorkflowBuilder::new().step(
        StepBuilder::new("qa")
            .model(model)
            .prompt(|ctx| {
                let background = ctx.get_str("context");
                let question = ctx.get_str("question");
                format!(
                    r#"Use the following context to answer the question as accurately and concisely as possible.

Context:
{background}

Question: {question}

Answer:"#
                )
            })
            .output_text()
            .store_as("answer"),
    )
}

// ── chunk_and_summarize ───────────────────────────────────────────────────────

/// Summarize each chunk in a JSON array of text chunks.
///
/// **Input context key:** `"chunks"` — JSON array of text strings.
/// **Output context key:** `"chunk_summaries"` — JSON array of summary strings.
pub fn chunk_and_summarize(model: impl Into<String>) -> WorkflowBuilder {
    let model = model.into();
    let schema = json!({"type": "array", "items": {"type": "string"}});
    WorkflowBuilder::new().step(
        StepBuilder::new("chunk_and_summarize")
            .model(model)
            .prompt(|ctx| {
                let chunks = ctx.get_array("chunks");
                let numbered: String = chunks
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        format!(
                            "--- Chunk {} ---\n{}",
                            i + 1,
                            c.as_str().unwrap_or("")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
                format!(
                    r#"Summarize each of the following chunks in one or two sentences. Return a JSON array of strings — one summary per chunk, in the same order.

{numbered}

Return ONLY the JSON array. No markdown. No explanation."#
                )
            })
            .output_json_schema(schema)
            .store_as("chunk_summaries"),
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::*;
    use crate::ollama::client::OllamaClient;
    use crate::ollama::mock::MockOllamaClient;
    use crate::workflow::context::WorkflowContext;

    #[tokio::test]
    async fn test_filter_files_preset() {
        let client = Arc::new(
            MockOllamaClient::new("llama3")
                .with_chat_response(r#"["src/main.rs","src/lib.rs"]"#),
        );
        let mut ctx = WorkflowContext::new();
        ctx.set("file_list", "src/main.rs\nsrc/lib.rs\nsrc/foo.rs");

        let result = filter_files("llama3")
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();

        let files = result.get_array("interesting_files");
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].as_str().unwrap(), "src/main.rs");
    }

    #[tokio::test]
    async fn test_summarize_content_preset() {
        let client = Arc::new(
            MockOllamaClient::new("llama3").with_chat_response("This is a great project."),
        );
        let mut ctx = WorkflowContext::new();
        ctx.set("file_contents", "--- src/main.rs ---\nfn main() {}");

        let result = summarize_content("llama3")
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();

        assert_eq!(result.get_str("summary"), "This is a great project.");
    }

    #[tokio::test]
    async fn test_classify_preset() {
        let client = Arc::new(
            MockOllamaClient::new("llama3")
                .with_chat_response(r#"{"label":"positive","confidence":0.95}"#),
        );
        let mut ctx = WorkflowContext::new();
        ctx.set("input", "I love this product!");

        let result = classify("llama3", &["positive", "negative", "neutral"])
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();

        let cls = result.get("classification").unwrap();
        assert_eq!(cls["label"].as_str().unwrap(), "positive");
    }

    #[tokio::test]
    async fn test_qa_preset() {
        let client = Arc::new(
            MockOllamaClient::new("llama3").with_chat_response("Paris."),
        );
        let mut ctx = WorkflowContext::new();
        ctx.set("context", "France is a country in Europe. Its capital is Paris.");
        ctx.set("question", "What is the capital of France?");

        let result = qa("llama3")
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();

        assert_eq!(result.get_str("answer"), "Paris.");
    }

    #[tokio::test]
    async fn test_chunk_and_summarize_preset() {
        let client = Arc::new(
            MockOllamaClient::new("llama3")
                .with_chat_response(r#"["Summary of chunk 1.","Summary of chunk 2."]"#),
        );
        let mut ctx = WorkflowContext::new();
        ctx.set("chunks", json!(["Long text about topic A.", "Long text about topic B."]));

        let result = chunk_and_summarize("llama3")
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();

        let summaries = result.get_array("chunk_summaries");
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].as_str().unwrap(), "Summary of chunk 1.");
    }

    #[tokio::test]
    async fn test_extend_composes_presets() {
        let client = Arc::new(
            MockOllamaClient::new("llama3").with_responses([
                r#"["src/main.rs"]"#,
                "A great Rust project.",
            ]),
        );

        let mut ctx = WorkflowContext::new();
        ctx.set("file_list", "src/main.rs");
        ctx.set("file_contents", "--- src/main.rs ---\nfn main() {}");

        let wf = filter_files("llama3")
            .extend(summarize_content("llama3"))
            .build();

        let result = wf
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();

        assert!(result.contains("interesting_files"));
        assert_eq!(result.get_str("summary"), "A great Rust project.");
        assert_eq!(client.recorded_calls().await.len(), 2);
    }
}
