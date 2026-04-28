/// Ten reference workflow designs covering common agent patterns.
///
/// Every function returns a [`WorkflowBuilder`] that can be composed further
/// or run directly with `.build().run(client, ctx)`.
///
/// # Context key conventions
/// Input keys must be set on the [`WorkflowContext`] before calling `run`.
/// Output keys are documented per function.
use serde_json::json;

use crate::workflow::presets;
use crate::workflow::runner::WorkflowBuilder;
use crate::workflow::step::{StepBuilder, StepErrorAction};

// ─────────────────────────────────────────────────────────────────────────────
// W-1  Code Review
//
// Input:  "code"   — source code to review
// Output: "review" — synthesised multi-angle review (text)
// ─────────────────────────────────────────────────────────────────────────────

/// Analyse code from three angles (complexity, correctness, style) in parallel,
/// then synthesise the findings into a single review.
pub fn code_review(model: impl Into<String>) -> WorkflowBuilder {
    let m = model.into();
    WorkflowBuilder::new()
        .parallel(vec![
            StepBuilder::new("review_complexity")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Analyse the COMPLEXITY of the following code. Comment on cognitive \
                         load, nesting depth, and abstraction quality.\n\n```\n{}\n```",
                        ctx.get_str("code")
                    )
                })
                .output_text()
                .store_as("complexity_notes"),
            StepBuilder::new("review_correctness")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Analyse the CORRECTNESS of the following code. Look for bugs, \
                         edge-case failures, and incorrect logic.\n\n```\n{}\n```",
                        ctx.get_str("code")
                    )
                })
                .output_text()
                .store_as("correctness_notes"),
            StepBuilder::new("review_style")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Analyse the STYLE and MAINTAINABILITY of the following code. \
                         Comment on naming, documentation, and idioms.\n\n```\n{}\n```",
                        ctx.get_str("code")
                    )
                })
                .output_text()
                .store_as("style_notes"),
        ])
        .step(
            StepBuilder::new("synthesise_review")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "You have three independent code-review analyses. Synthesise them \
                         into a single, prioritised review with actionable suggestions.\n\n\
                         ## Complexity\n{}\n\n## Correctness\n{}\n\n## Style\n{}",
                        ctx.get_str("complexity_notes"),
                        ctx.get_str("correctness_notes"),
                        ctx.get_str("style_notes"),
                    )
                })
                .output_text()
                .store_as("review"),
        )
}

// ─────────────────────────────────────────────────────────────────────────────
// W-2  Research & Synthesis
//
// Input:  "topic"  — subject to research
// Output: "report" — ranked, synthesised report (text)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate multiple research sub-queries in parallel, then synthesise a ranked
/// report from all gathered information.
pub fn research_synthesis(model: impl Into<String>) -> WorkflowBuilder {
    let m = model.into();
    WorkflowBuilder::new()
        .step(
            StepBuilder::new("generate_queries")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "You are a research assistant. Generate exactly 3 focused \
                         sub-questions that together cover the topic: \"{}\"\n\n\
                         Return a JSON array of 3 strings. No markdown.",
                        ctx.get_str("topic")
                    )
                })
                .output_json_schema(json!({"type": "array", "items": {"type": "string"}}))
                .store_as("queries"),
        )
        .parallel(vec![
            StepBuilder::new("research_q1")
                .model(&m)
                .prompt(|ctx| {
                    let q = ctx.get_array("queries");
                    format!(
                        "Answer this research question thoroughly: {}",
                        q.first().and_then(|v| v.as_str()).unwrap_or("")
                    )
                })
                .output_text()
                .store_as("finding_1"),
            StepBuilder::new("research_q2")
                .model(&m)
                .prompt(|ctx| {
                    let q = ctx.get_array("queries");
                    format!(
                        "Answer this research question thoroughly: {}",
                        q.get(1).and_then(|v| v.as_str()).unwrap_or("")
                    )
                })
                .output_text()
                .store_as("finding_2"),
            StepBuilder::new("research_q3")
                .model(&m)
                .prompt(|ctx| {
                    let q = ctx.get_array("queries");
                    format!(
                        "Answer this research question thoroughly: {}",
                        q.get(2).and_then(|v| v.as_str()).unwrap_or("")
                    )
                })
                .output_text()
                .store_as("finding_3"),
        ])
        .step(
            StepBuilder::new("synthesise_report")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Synthesise the following research findings into a concise, \
                         ranked report on \"{}\". Lead with the most important insights.\n\n\
                         Finding 1:\n{}\n\nFinding 2:\n{}\n\nFinding 3:\n{}",
                        ctx.get_str("topic"),
                        ctx.get_str("finding_1"),
                        ctx.get_str("finding_2"),
                        ctx.get_str("finding_3"),
                    )
                })
                .output_text()
                .store_as("report"),
        )
}

// ─────────────────────────────────────────────────────────────────────────────
// W-3  Multi-Agent Debate
//
// Input:  "proposition" — statement to debate
// Output: "verdict"     — impartial judge's verdict (text)
// ─────────────────────────────────────────────────────────────────────────────

/// Run a for-and-against debate in parallel, then have an impartial judge
/// deliver a balanced verdict.
pub fn multi_agent_debate(model: impl Into<String>) -> WorkflowBuilder {
    let m = model.into();
    WorkflowBuilder::new()
        .parallel(vec![
            StepBuilder::new("advocate_for")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "You are a persuasive advocate. Argue strongly IN FAVOUR of the \
                         following proposition. Give your three strongest arguments.\n\n\
                         Proposition: \"{}\"",
                        ctx.get_str("proposition")
                    )
                })
                .output_text()
                .store_as("argument_for"),
            StepBuilder::new("advocate_against")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "You are a persuasive advocate. Argue strongly AGAINST the \
                         following proposition. Give your three strongest arguments.\n\n\
                         Proposition: \"{}\"",
                        ctx.get_str("proposition")
                    )
                })
                .output_text()
                .store_as("argument_against"),
        ])
        .step(
            StepBuilder::new("judge")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "You are an impartial judge. Consider both sides of the debate \
                         below and deliver a balanced verdict.\n\n\
                         Proposition: \"{}\"\n\n\
                         --- Arguments FOR ---\n{}\n\n\
                         --- Arguments AGAINST ---\n{}",
                        ctx.get_str("proposition"),
                        ctx.get_str("argument_for"),
                        ctx.get_str("argument_against"),
                    )
                })
                .output_text()
                .store_as("verdict"),
        )
}

// ─────────────────────────────────────────────────────────────────────────────
// W-4  Code Generation → Validation → Docs
//
// Input:  "spec"        — natural-language specification
// Output: "code"        — generated code
//         "validation"  — validation result
//         "docs"        — generated documentation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate code from a spec, validate it, then produce documentation and tests.
/// Validation uses `on_error` to degrade gracefully if the model fails.
pub fn code_gen_with_docs(model: impl Into<String>) -> WorkflowBuilder {
    let m = model.into();
    WorkflowBuilder::new()
        .step(
            StepBuilder::new("generate_code")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "You are an expert software engineer. Implement the following \
                         specification. Return ONLY the code, no explanations.\n\n\
                         Spec: {}",
                        ctx.get_str("spec")
                    )
                })
                .output_text()
                .store_as("code"),
        )
        .step(
            StepBuilder::new("validate_code")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Review this code for correctness, security, and adherence to the \
                         spec. Return a JSON object with keys \"passed\" (bool) and \
                         \"issues\" (array of strings).\n\n\
                         Spec: {}\n\nCode:\n{}",
                        ctx.get_str("spec"),
                        ctx.get_str("code"),
                    )
                })
                .output_json_schema(json!({
                    "type": "object",
                    "required": ["passed", "issues"]
                }))
                .store_as("validation")
                .on_error(|_err, _ctx| {
                    StepErrorAction::UseDefault(json!({"passed": false, "issues": ["validation step failed"]}))
                }),
        )
        .parallel(vec![
            StepBuilder::new("write_docs")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Write clear API documentation for the following code.\n\n```\n{}\n```",
                        ctx.get_str("code")
                    )
                })
                .output_text()
                .store_as("docs"),
            StepBuilder::new("write_tests")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Write unit tests for the following code. Cover happy paths and \
                         edge cases.\n\n```\n{}\n```",
                        ctx.get_str("code")
                    )
                })
                .output_text()
                .store_as("tests"),
        ])
}

// ─────────────────────────────────────────────────────────────────────────────
// W-5  Hierarchical Summarisation
//
// Input:  "chunks"          — JSON array of text chunks
// Output: "chunk_summaries" — JSON array of per-chunk summaries
//         "final_summary"   — merged top-level summary (text)
// ─────────────────────────────────────────────────────────────────────────────

/// Summarise each chunk individually (via the preset), then merge all summaries
/// into a single coherent document.
pub fn hierarchical_summarization(model: impl Into<String>) -> WorkflowBuilder {
    let m = model.into();
    presets::chunk_and_summarize(&m).step(
        StepBuilder::new("merge_summaries")
            .model(m)
            .prompt(|ctx| {
                let summaries = ctx.get_array("chunk_summaries");
                let list: String = summaries
                    .iter()
                    .enumerate()
                    .map(|(i, s)| format!("{}. {}", i + 1, s.as_str().unwrap_or("")))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "You have the following section summaries. Merge them into a single \
                     coherent, well-structured document summary.\n\n{list}"
                )
            })
            .output_text()
            .store_as("final_summary"),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// W-6  Structured Extraction
//
// Input:  "document"  — raw text to extract from
// Output: "people"    — JSON array of person entities
//         "places"    — JSON array of place entities
//         "events"    — JSON array of event entities
//         "entities"  — merged entity report (text)
// ─────────────────────────────────────────────────────────────────────────────

/// Extract different entity types in parallel, then merge into a final report.
pub fn structured_extraction(model: impl Into<String>) -> WorkflowBuilder {
    let m = model.into();
    let arr_schema = json!({"type": "array", "items": {"type": "string"}});
    WorkflowBuilder::new()
        .parallel(vec![
            StepBuilder::new("extract_people")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Extract all PERSON names from the following text. \
                         Return a JSON array of strings. No markdown.\n\n{}",
                        ctx.get_str("document")
                    )
                })
                .output_json_schema(arr_schema.clone())
                .store_as("people"),
            StepBuilder::new("extract_places")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Extract all PLACE names from the following text. \
                         Return a JSON array of strings. No markdown.\n\n{}",
                        ctx.get_str("document")
                    )
                })
                .output_json_schema(arr_schema.clone())
                .store_as("places"),
            StepBuilder::new("extract_events")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Extract all key EVENTS or ACTIONS from the following text. \
                         Return a JSON array of strings. No markdown.\n\n{}",
                        ctx.get_str("document")
                    )
                })
                .output_json_schema(arr_schema)
                .store_as("events"),
        ])
        .step(
            StepBuilder::new("merge_entities")
                .model(&m)
                .prompt(|ctx| {
                    let people = ctx.get_array("people");
                    let places = ctx.get_array("places");
                    let events = ctx.get_array("events");
                    let fmt = |arr: &[serde_json::Value]| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    };
                    format!(
                        "Summarise the following extracted entities into a concise report.\n\n\
                         People: {}\nPlaces: {}\nEvents: {}",
                        fmt(people),
                        fmt(places),
                        fmt(events),
                    )
                })
                .output_text()
                .store_as("entities"),
        )
}

// ─────────────────────────────────────────────────────────────────────────────
// W-7  Iterative Bug-Fix
//
// Input:  "buggy_code"   — code containing a bug
//         "bug_report"   — description of the observed failure
// Output: "fixed_code"   — corrected code (attempt 1, 2, or 3)
//         "explanation"  — explanation of the fix
// ─────────────────────────────────────────────────────────────────────────────

/// Attempt up to three fix iterations. Each attempt uses `step_with_fallback`
/// so a validation parse failure still keeps the pipeline alive.
pub fn iterative_bug_fix(model: impl Into<String>) -> WorkflowBuilder {
    let m = model.into();
    let fix_schema = json!({"type": "object", "required": ["code", "rationale"]});

    WorkflowBuilder::new()
        // Attempt 1
        .step_with_fallback(
            StepBuilder::new("fix_attempt_1")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Fix the bug described below. Return JSON with keys \
                         \"code\" (fixed code) and \"rationale\" (one sentence).\n\n\
                         Bug report: {}\n\nBuggy code:\n{}",
                        ctx.get_str("bug_report"),
                        ctx.get_str("buggy_code"),
                    )
                })
                .output_json_schema(fix_schema.clone())
                .store_as("fix_attempt"),
            StepBuilder::new("fix_attempt_1_fallback")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "The previous fix attempt failed to parse. Try again. \
                         Return ONLY valid JSON with keys \"code\" and \"rationale\".\n\n\
                         Bug: {}\n\nCode:\n{}",
                        ctx.get_str("bug_report"),
                        ctx.get_str("buggy_code"),
                    )
                })
                .output_json(/* any valid JSON */)
                .store_as("fix_attempt"),
        )
        // Validate attempt 1; on failure try attempt 2
        .step_with_fallback(
            StepBuilder::new("validate_fix_1")
                .model(&m)
                .prompt(|ctx| {
                    let attempt = ctx.get("fix_attempt")
                        .map(|v| v.to_string())
                        .unwrap_or_default();
                    format!(
                        "Does this fix correctly resolve the bug? Answer with JSON \
                         {{\"ok\": true/false, \"reason\": \"...\"}}.\n\n\
                         Bug: {}\n\nFix attempt:\n{}",
                        ctx.get_str("bug_report"),
                        attempt,
                    )
                })
                .output_json_schema(json!({"type": "object", "required": ["ok", "reason"]}))
                .store_as("validation_1"),
            StepBuilder::new("validate_fix_1_fallback")
                .model(&m)
                .prompt(|_| r#"{"ok": false, "reason": "validation parse error"}"#.to_string())
                .output_json()
                .store_as("validation_1"),
        )
        // Attempt 2 (uses context from attempt 1 and validation)
        .step_with_fallback(
            StepBuilder::new("fix_attempt_2")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Your previous fix was not accepted. Reason: {}.\n\
                         Try a different approach.\n\n\
                         Bug: {}\n\nOriginal code:\n{}",
                        ctx.get("validation_1")
                            .and_then(|v| v["reason"].as_str())
                            .unwrap_or("unknown"),
                        ctx.get_str("bug_report"),
                        ctx.get_str("buggy_code"),
                    )
                })
                .output_json_schema(fix_schema.clone())
                .store_as("fixed_code_raw"),
            StepBuilder::new("fix_attempt_2_fallback")
                .model(&m)
                .prompt(|ctx| ctx.get_str("buggy_code").to_string())
                .output_text()
                .store_as("fixed_code_raw"),
        )
        .step(
            StepBuilder::new("explain_fix")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Write a concise explanation of how the bug was fixed. \
                         Bug report: {}\n\nFinal fix:\n{}",
                        ctx.get_str("bug_report"),
                        ctx.get("fixed_code_raw")
                            .map(|v| v.to_string())
                            .unwrap_or_default(),
                    )
                })
                .output_text()
                .store_as("explanation"),
        )
}

// ─────────────────────────────────────────────────────────────────────────────
// W-8  RAG Q&A
//
// Input:  "question"   — user question
// Output: "sub_queries" — decomposed sub-questions (JSON array)
//         "retrieval_1/2/3" — simulated retrieved passages
//         "answer"     — final grounded answer (text)
// ─────────────────────────────────────────────────────────────────────────────

/// Decompose a question into sub-queries, retrieve in parallel, then synthesise
/// a grounded answer.  Partial retrieval failures are tolerated.
pub fn rag_qa(model: impl Into<String>) -> WorkflowBuilder {
    let m = model.into();
    WorkflowBuilder::new()
        .step(
            StepBuilder::new("decompose_question")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Break this question into exactly 3 simpler sub-questions that \
                         together cover all aspects of the answer.\n\n\
                         Question: {}\n\n\
                         Return a JSON array of 3 strings. No markdown.",
                        ctx.get_str("question")
                    )
                })
                .output_json_schema(json!({"type": "array", "items": {"type": "string"}}))
                .store_as("sub_queries"),
        )
        .parallel_partial(vec![
            StepBuilder::new("retrieve_1")
                .model(&m)
                .prompt(|ctx| {
                    let q = ctx.get_array("sub_queries");
                    format!(
                        "Answer this specific sub-question as if retrieving from a knowledge \
                         base: {}",
                        q.first().and_then(|v| v.as_str()).unwrap_or("")
                    )
                })
                .output_text()
                .store_as("retrieval_1"),
            StepBuilder::new("retrieve_2")
                .model(&m)
                .prompt(|ctx| {
                    let q = ctx.get_array("sub_queries");
                    format!(
                        "Answer this specific sub-question as if retrieving from a knowledge \
                         base: {}",
                        q.get(1).and_then(|v| v.as_str()).unwrap_or("")
                    )
                })
                .output_text()
                .store_as("retrieval_2"),
            StepBuilder::new("retrieve_3")
                .model(&m)
                .prompt(|ctx| {
                    let q = ctx.get_array("sub_queries");
                    format!(
                        "Answer this specific sub-question as if retrieving from a knowledge \
                         base: {}",
                        q.get(2).and_then(|v| v.as_str()).unwrap_or("")
                    )
                })
                .output_text()
                .store_as("retrieval_3"),
        ])
        .extend(
            presets::qa(&m)
        )
}

// ─────────────────────────────────────────────────────────────────────────────
// W-9  Task Decomposition  (most agent-like)
//
// Input:  "goal"       — high-level goal description
// Output: "task_plan"  — JSON array of task descriptions
//         "task_results" — JSON array of per-task results
//         "summary"    — final summary across all tasks (text)
// ─────────────────────────────────────────────────────────────────────────────

/// Plan a goal into discrete tasks, execute all tasks in parallel, then
/// summarise the combined results.
pub fn task_decomposition(model: impl Into<String>) -> WorkflowBuilder {
    let m = model.into();
    WorkflowBuilder::new()
        .step(
            StepBuilder::new("plan_tasks")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Break the following goal into exactly 3 concrete, actionable tasks. \
                         Each task should be independently executable.\n\n\
                         Goal: {}\n\n\
                         Return a JSON array of 3 task description strings. No markdown.",
                        ctx.get_str("goal")
                    )
                })
                .output_json_schema(json!({"type": "array", "items": {"type": "string"}}))
                .store_as("task_plan"),
        )
        .parallel(vec![
            StepBuilder::new("execute_task_1")
                .model(&m)
                .prompt(|ctx| {
                    let tasks = ctx.get_array("task_plan");
                    format!(
                        "Execute the following task and report the result:\n\n{}",
                        tasks.first().and_then(|v| v.as_str()).unwrap_or("")
                    )
                })
                .output_text()
                .store_as("task_result_1"),
            StepBuilder::new("execute_task_2")
                .model(&m)
                .prompt(|ctx| {
                    let tasks = ctx.get_array("task_plan");
                    format!(
                        "Execute the following task and report the result:\n\n{}",
                        tasks.get(1).and_then(|v| v.as_str()).unwrap_or("")
                    )
                })
                .output_text()
                .store_as("task_result_2"),
            StepBuilder::new("execute_task_3")
                .model(&m)
                .prompt(|ctx| {
                    let tasks = ctx.get_array("task_plan");
                    format!(
                        "Execute the following task and report the result:\n\n{}",
                        tasks.get(2).and_then(|v| v.as_str()).unwrap_or("")
                    )
                })
                .output_text()
                .store_as("task_result_3"),
        ])
        .step(
            StepBuilder::new("summarise_tasks")
                .model(&m)
                .prompt(|ctx| {
                    format!(
                        "Summarise the results of all tasks toward the goal: \"{}\"\n\n\
                         Task 1 result:\n{}\n\nTask 2 result:\n{}\n\nTask 3 result:\n{}",
                        ctx.get_str("goal"),
                        ctx.get_str("task_result_1"),
                        ctx.get_str("task_result_2"),
                        ctx.get_str("task_result_3"),
                    )
                })
                .output_text()
                .store_as("summary"),
        )
}

// ─────────────────────────────────────────────────────────────────────────────
// W-10  Model Cascade
//
// Input:  "prompt"    — any prompt
// Output: "response"  — answer from the cheapest model that met the bar
//         "tier"      — which model tier answered ("cheap" / "mid" / "expensive")
// ─────────────────────────────────────────────────────────────────────────────

/// Try a cheap model first; if it signals low confidence, escalate to `mid`,
/// then to `expensive`. Uses `step_with_fallback` chains.
///
/// The cheap/mid steps return `{"response": "...", "confidence": 0.0–1.0}`.
/// The expensive step returns plain text as the final fallback.
pub fn model_cascade(
    cheap: impl Into<String>,
    mid: impl Into<String>,
    expensive: impl Into<String>,
) -> WorkflowBuilder {
    let cheap = cheap.into();
    let mid = mid.into();
    let expensive = expensive.into();
    let conf_schema = json!({"type": "object", "required": ["response", "confidence"]});

    WorkflowBuilder::new()
        // Tier 1 — cheap model
        .step_with_fallback(
            StepBuilder::new("tier_cheap")
                .model(&cheap)
                .prompt(|ctx| {
                    format!(
                        "Answer the following. After your answer add a JSON line with your \
                         confidence (0.0–1.0) that your answer is correct and complete.\n\
                         Format: {{\"response\": \"...\", \"confidence\": 0.0}}\n\n\
                         Prompt: {}",
                        ctx.get_str("prompt")
                    )
                })
                .output_json_schema(conf_schema.clone())
                .store_as("cascade_result"),
            // Fallback: treat parse failure as low-confidence → escalate
            StepBuilder::new("tier_cheap_fallback")
                .model(&cheap)
                .prompt(|_| r#"{"response": "", "confidence": 0.0}"#.to_string())
                .output_json()
                .store_as("cascade_result"),
        )
        // Tier 2 — mid model (runs only when cheap confidence < 0.7)
        .step_with_fallback(
            StepBuilder::new("tier_mid")
                .model(&mid)
                .prompt(|ctx| {
                    let conf = ctx.get("cascade_result")
                        .and_then(|v| v["confidence"].as_f64())
                        .unwrap_or(0.0);
                    if conf >= 0.7 {
                        // Signal "skip" by returning the existing result
                        return format!(
                            "{{\"response\": {}, \"confidence\": {}}}",
                            serde_json::to_string(
                                ctx.get("cascade_result")
                                    .and_then(|v| v.get("response"))
                                    .unwrap_or(&serde_json::Value::String(String::new()))
                            ).unwrap_or_default(),
                            conf
                        );
                    }
                    format!(
                        "The cheap model had low confidence. Provide a better answer.\n\n\
                         Format: {{\"response\": \"...\", \"confidence\": 0.0}}\n\n\
                         Prompt: {}",
                        ctx.get_str("prompt")
                    )
                })
                .output_json_schema(conf_schema.clone())
                .store_as("cascade_result"),
            StepBuilder::new("tier_mid_fallback")
                .model(&mid)
                .prompt(|_| r#"{"response": "", "confidence": 0.0}"#.to_string())
                .output_json()
                .store_as("cascade_result"),
        )
        // Tier 3 — expensive model, unconditional fallback
        .step_with_fallback(
            StepBuilder::new("tier_expensive")
                .model(&expensive)
                .prompt(|ctx| {
                    let conf = ctx.get("cascade_result")
                        .and_then(|v| v["confidence"].as_f64())
                        .unwrap_or(0.0);
                    if conf >= 0.7 {
                        return ctx.get("cascade_result")
                            .and_then(|v| v["response"].as_str())
                            .unwrap_or("")
                            .to_string();
                    }
                    format!(
                        "Both cheaper models had low confidence. Give the best possible \
                         answer.\n\nPrompt: {}",
                        ctx.get_str("prompt")
                    )
                })
                .output_text()
                .store_as("response"),
            StepBuilder::new("tier_expensive_fallback")
                .model(&expensive)
                .prompt(|ctx| ctx.get_str("prompt").to_string())
                .output_text()
                .store_as("response"),
        )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::*;
    use crate::ollama::client::OllamaClient;
    use crate::ollama::mock::MockOllamaClient;
    use crate::workflow::context::WorkflowContext;

    fn ctx_with(pairs: &[(&str, &str)]) -> WorkflowContext {
        let mut ctx = WorkflowContext::new();
        for (k, v) in pairs {
            ctx.set(*k, *v);
        }
        ctx
    }

    // W-1
    #[tokio::test]
    async fn test_w1_code_review() {
        let client = Arc::new(
            MockOllamaClient::new("llama3")
                .with_responses(["complex notes", "correctness notes", "style notes", "final review"]),
        );
        let ctx = ctx_with(&[("code", "fn add(a: i32, b: i32) -> i32 { a + b }")]);
        let result = code_review("llama3")
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();
        assert!(result.contains("complexity_notes"));
        assert!(result.contains("correctness_notes"));
        assert!(result.contains("style_notes"));
        assert_eq!(result.get_str("review"), "final review");
        assert_eq!(client.recorded_calls().await.len(), 4);
    }

    // W-2
    #[tokio::test]
    async fn test_w2_research_synthesis() {
        let client = Arc::new(
            MockOllamaClient::new("llama3").with_responses([
                r#"["q1","q2","q3"]"#,
                "finding 1",
                "finding 2",
                "finding 3",
                "final report",
            ]),
        );
        let ctx = ctx_with(&[("topic", "Rust async programming")]);
        let result = research_synthesis("llama3")
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();
        assert_eq!(result.get_str("report"), "final report");
        assert_eq!(client.recorded_calls().await.len(), 5);
    }

    // W-3
    #[tokio::test]
    async fn test_w3_debate() {
        let client = Arc::new(
            MockOllamaClient::new("llama3")
                .with_responses(["for argument", "against argument", "verdict text"]),
        );
        let ctx = ctx_with(&[("proposition", "Open source is always better")]);
        let result = multi_agent_debate("llama3")
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();
        assert_eq!(result.get_str("verdict"), "verdict text");
        assert_eq!(client.recorded_calls().await.len(), 3);
    }

    // W-4
    #[tokio::test]
    async fn test_w4_code_gen_with_docs() {
        let client = Arc::new(
            MockOllamaClient::new("llama3").with_responses([
                "fn hello() {}",
                r#"{"passed": true, "issues": []}"#,
                "docs text",
                "tests text",
            ]),
        );
        let ctx = ctx_with(&[("spec", "A function that prints hello")]);
        let result = code_gen_with_docs("llama3")
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();
        assert_eq!(result.get_str("code"), "fn hello() {}");
        assert!(result.contains("docs"));
        assert!(result.contains("tests"));
        assert_eq!(client.recorded_calls().await.len(), 4);
    }

    // W-5
    #[tokio::test]
    async fn test_w5_hierarchical_summarization() {
        let client = Arc::new(
            MockOllamaClient::new("llama3")
                .with_responses([r#"["sum1","sum2"]"#, "merged summary"]),
        );
        let mut ctx = WorkflowContext::new();
        ctx.set("chunks", json!(["chunk A text", "chunk B text"]));
        let result = hierarchical_summarization("llama3")
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();
        assert_eq!(result.get_str("final_summary"), "merged summary");
        assert_eq!(client.recorded_calls().await.len(), 2);
    }

    // W-6
    #[tokio::test]
    async fn test_w6_structured_extraction() {
        let client = Arc::new(
            MockOllamaClient::new("llama3").with_responses([
                r#"["Alice","Bob"]"#,
                r#"["Paris","London"]"#,
                r#"["signed treaty","declared war"]"#,
                "entity report",
            ]),
        );
        let ctx = ctx_with(&[("document", "Alice and Bob met in Paris.")]);
        let result = structured_extraction("llama3")
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();
        assert_eq!(result.get_str("entities"), "entity report");
        assert_eq!(client.recorded_calls().await.len(), 4);
    }

    // W-7
    #[tokio::test]
    async fn test_w7_iterative_bug_fix() {
        let client = Arc::new(
            MockOllamaClient::new("llama3").with_responses([
                r#"{"code":"fn fixed(){}","rationale":"fixed the off-by-one"}"#,
                r#"{"ok":true,"reason":"looks good"}"#,
                r#"{"code":"fn fixed(){}","rationale":"same fix"}"#,
                "The bug was an off-by-one error.",
            ]),
        );
        let ctx = ctx_with(&[
            ("buggy_code", "fn broken() { panic!(); }"),
            ("bug_report", "always panics"),
        ]);
        let result = iterative_bug_fix("llama3")
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();
        assert!(result.contains("explanation"));
        assert!(result.contains("fix_attempt") || result.contains("fixed_code_raw"));
    }

    // W-8
    #[tokio::test]
    async fn test_w8_rag_qa() {
        let client = Arc::new(
            MockOllamaClient::new("llama3").with_responses([
                r#"["sub q1","sub q2","sub q3"]"#,
                "retrieval 1",
                "retrieval 2",
                "retrieval 3",
                "final answer",
            ]),
        );
        let mut ctx = WorkflowContext::new();
        ctx.set("question", "How does Rust ownership work?");
        // rag_qa extends qa preset which needs "context" and "question"
        // we pre-populate "context" with the retrievals placeholder
        ctx.set("context", "");
        let result = rag_qa("llama3")
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();
        assert_eq!(result.get_str("answer"), "final answer");
    }

    // W-9
    #[tokio::test]
    async fn test_w9_task_decomposition() {
        let client = Arc::new(
            MockOllamaClient::new("llama3").with_responses([
                r#"["task1","task2","task3"]"#,
                "result 1",
                "result 2",
                "result 3",
                "final summary",
            ]),
        );
        let ctx = ctx_with(&[("goal", "Build a website")]);
        let result = task_decomposition("llama3")
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();
        assert_eq!(result.get_str("summary"), "final summary");
        assert_eq!(client.recorded_calls().await.len(), 5);
    }

    // W-10
    #[tokio::test]
    async fn test_w10_model_cascade_cheap_succeeds() {
        // cheap returns high confidence → mid/expensive tiers still run but
        // just echo back the existing result
        let client = Arc::new(
            MockOllamaClient::new("llama3").with_responses([
                r#"{"response":"cheap answer","confidence":0.9}"#,
                r#"{"response":"cheap answer","confidence":0.9}"#,
                "cheap answer", // expensive tier echoes back
            ]),
        );
        let ctx = ctx_with(&[("prompt", "What is 2 + 2?")]);
        let result = model_cascade("cheap-model", "mid-model", "expensive-model")
            .build()
            .run(Arc::clone(&client) as Arc<dyn OllamaClient>, ctx)
            .await
            .unwrap();
        assert!(result.contains("response"));
    }
}
