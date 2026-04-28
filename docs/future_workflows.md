# Future Workflow Plan

This document describes three areas of planned investment:

1. **Reusable preset workflows** — extracting known pipelines into composable builder functions
2. **Error state handling** — first-class recovery, fallbacks, and partial failure
3. **Interesting new workflows** — inspired by multi-agent tool use patterns

Nothing here is implemented yet. This is a design reference.

---

## 1. Reusable Preset Workflows

### The Problem Today

The `ExplorePipeline` constructs two separate `Workflow::builder()` calls inline inside an imperative
`async fn run()`. There is no way to share those workflows, compose them with other steps, or test
them in isolation. The same pattern will recur with every new feature.

### Proposed: `WorkflowBuilder::extend()` + a `presets` module

Add a method to `WorkflowBuilder` that absorbs all nodes from another builder:

```rust
pub fn extend(mut self, other: WorkflowBuilder) -> Self {
    self.nodes.extend(other.nodes);
    self
}
```

This lets pre-built workflow fragments be composed freely:

```rust
let wf = Workflow::builder()
    .extend(Presets::filter_files(&model))
    .step(my_custom_step)
    .extend(Presets::summarize_content(&model))
    .build();
```

### Preset factory functions to build

Each function returns a `WorkflowBuilder` (not a finished `Workflow`) so it stays composable.

| Function | Context in | Context out | Description |
|---|---|---|---|
| `Presets::filter_files(model)` | `"file_list": String` | `"interesting_files": [String]` | Existing explore filter step, extracted |
| `Presets::summarize_content(model)` | `"file_contents": String` | `"summary": String` | Existing summarize step, extracted |
| `Presets::classify(model, categories)` | `"input": String` | `"label": String` | Classifies input into one of N categories |
| `Presets::extract_json(model, schema)` | `"input": String` | `"extracted": Value` | Structured extraction against a JSON schema |
| `Presets::qa(model)` | `"context": String`, `"question": String` | `"answer": String` | Single-turn question answering with context |
| `Presets::score_and_improve(model, threshold, max_iter)` | `"draft": String` | `"draft": String`, `"score": i64` | Self-improvement loop until score ≥ threshold |
| `Presets::translate(model, target_language)` | `"text": String` | `"translation": String` | Translation step |
| `Presets::chunk_and_summarize(model, chunk_size)` | `"document": String` | `"summaries": [String]` | Splits document, summarizes each chunk in parallel |

### Migrating `ExplorePipeline`

`ExplorePipeline::run()` would shrink from ~100 imperative lines to roughly:

```rust
pub async fn run(&self, root: &Path) -> Result<String, TermiError> {
    let entries = self.walk(root).await?;
    let file_list = format_file_list(&entries);

    let ctx = Workflow::builder()
        .extend(Presets::filter_files(&self.config.model))
        .build()
        .run(Arc::clone(&self.client), WorkflowContext::new().with("file_list", file_list))
        .await?;

    let contents = self.read_selected_files(&entries, &ctx).await?;

    let ctx2 = Workflow::builder()
        .extend(Presets::summarize_content(&self.config.model))
        .build()
        .run(Arc::clone(&self.client), WorkflowContext::new().with("file_contents", contents))
        .await?;

    Ok(ctx2.get_str("summary").to_string())
}
```

The file-reading step between the two workflows stays imperative because it involves disk I/O, size
limits, and hallucination filtering — none of which belong inside an LLM step.

---

## 2. Error State Handling

### The Problem Today

All errors immediately abort the workflow and bubble up as `TermiError::Pipeline`. There is no way
to recover, fall back, inspect what failed, or continue past a non-critical step. Parallel failure
is all-or-nothing. The `loop_step` guard returns a generic pipeline error.

### 2a. Per-step error recovery

Add an `.on_error()` method to `StepBuilder`:

```rust
pub fn on_error<F>(mut self, f: F) -> Self
where
    F: Fn(&TermiError, &WorkflowContext) -> StepErrorAction + Send + Sync + 'static,
```

`StepErrorAction` is a small enum:

```rust
pub enum StepErrorAction {
    /// Store `value` under `output_key` and continue as if the step succeeded.
    UseDefault(Value),
    /// Leave the output key unset and continue.
    Skip,
    /// Abort with the original error.
    Fail,
    /// Abort with a different error.
    FailWith(TermiError),
}
```

Usage:

```rust
StepBuilder::new("risky_extraction")
    .model("llama3")
    .prompt(|ctx| format!("Extract entities from:\n{}", ctx.get_str("text")))
    .output_json()
    .store_as("entities")
    .on_error(|_err, _ctx| StepErrorAction::UseDefault(json!([])));
```

This step never fails — on any error it stores an empty array and the workflow continues.

### 2b. Error context keys

On any step failure (whether recovered or not) write diagnostic keys automatically:

| Key | Type | Value |
|---|---|---|
| `__last_error_step` | String | Name of the step that failed |
| `__last_error_msg` | String | `err.to_string()` |
| `__error_count` | i64 | Total errors so far in this run |

This lets downstream steps react:

```rust
.if_step(
    |ctx| ctx.get_i64("__error_count").unwrap_or(0) > 0,
    StepBuilder::new("error_report")
        .model("llama3")
        .prompt(|ctx| format!(
            "Step '{}' failed: {}. Suggest a recovery approach.",
            ctx.get_str("__last_error_step"),
            ctx.get_str("__last_error_msg")
        ))
        .output_text()
        .store_as("recovery_suggestion"),
)
```

### 2c. Fallback step

A convenience builder method for the common "try A, use B if A fails" pattern:

```rust
pub fn step_with_fallback(mut self, primary: StepBuilder, fallback: StepBuilder) -> Self
```

Semantics: run `primary`; if it errors, run `fallback` with the same context instead.

```rust
Workflow::builder()
    .step_with_fallback(
        StepBuilder::new("fast_model")
            .model("llama3:8b")
            .prompt(|ctx| ctx.get_str("prompt").to_string())
            .output_json()
            .store_as("result"),
        StepBuilder::new("fallback_model")
            .model("llama3:70b")
            .prompt(|ctx| format!(
                "The smaller model failed. Try again:\n{}",
                ctx.get_str("prompt")
            ))
            .output_text()
            .store_as("result"),
    )
```

### 2d. Partial-failure parallel

A new `parallel_partial()` method runs all steps and collects both successes and errors.
Successes are merged into context normally. Failed steps write their error to
`__error_<step_name>` instead of aborting the whole block.

```rust
pub fn parallel_partial(mut self, steps: Vec<StepBuilder>) -> Self
```

```rust
Workflow::builder()
    .parallel_partial(vec![
        StepBuilder::new("translate_fr").model("llama3")...store_as("french"),
        StepBuilder::new("translate_de").model("llama3")...store_as("german"),
        StepBuilder::new("translate_ja").model("llama3")...store_as("japanese"),
    ])
    // japanese translation may have failed — check before using it
    .if_step(
        |ctx| !ctx.contains("__error_translate_ja"),
        StepBuilder::new("use_japanese")...
    )
```

### 2e. Richer `TermiError` variants

Replace the single `Pipeline(String)` catch-all with targeted variants:

```rust
pub enum TermiError {
    // existing
    Http(reqwest::Error),
    Json(serde_json::Error),
    Io(std::io::Error),
    Walk(walkdir::Error),
    OllamaApi { status: u16, body: String },
    Stream(String),

    // new
    Pipeline(String),                          // keep for general use
    StepFailed { step: String, source: Box<TermiError> },
    ValidationFailed { step: String, field: String, details: String },
    LoopLimitExceeded { step: String, iterations: usize, max: usize },
    Timeout { step: String, elapsed_ms: u128 },
    AllBranchesFailed { attempted: Vec<String> },
}
```

`StepFailed` wraps the original error with the step name, giving stack-trace-like context without
losing type information:

```rust
Err(TermiError::StepFailed {
    step: "filter_files".to_string(),
    source: Box::new(original_err),
})
```

### 2f. Step timeout

Add `.timeout_ms(u64)` to `StepBuilder`. The runner wraps the LLM call with
`tokio::time::timeout`. On expiry it returns `TermiError::Timeout` (or triggers
`.on_error()` recovery if set).

```rust
StepBuilder::new("slow_analysis")
    .model("llama3:70b")
    .prompt(|ctx| format!("Deep analysis of:\n{}", ctx.get_str("code")))
    .output_text()
    .store_as("analysis")
    .timeout_ms(30_000)
    .on_error(|_, _| StepErrorAction::UseDefault(json!("analysis timed out")));
```

---

## 3. Interesting New Workflows

These are inspired by the tool surface that a capable agent has: the ability to search, read files,
run code, spawn sub-agents, plan, and review. Each workflow is described in terms of the builder
API it would use (once presets and error handling are in place).

---

### W-1: Code Review Pipeline

**Inspiration:** The review sub-agent — reads a diff, analyses it from multiple angles, synthesizes findings.

**Shape:** sequential → parallel fan-out → sequential fan-in

```
[ingest diff]
    │
    ├── [security scan]   ─┐
    ├── [style check]      ├── parallel_partial
    ├── [logic review]    ─┘
    │
[synthesize review]
    │
[if critical findings → generate fix suggestions]
```

**Context flow:**
```
in:  "diff": String
out: "security_findings": String
     "style_findings":    String
     "logic_findings":    String
     "review_summary":    String
     "fix_suggestions":   String  (conditional)
```

**Key builder features used:** `parallel_partial`, `if_step`, `system_prompt`, `on_error`

---

### W-2: Research & Synthesis Pipeline

**Inspiration:** WebSearch → multiple WebFetch → synthesis, mirroring how a research sub-agent works.

**Shape:** sequential → transform → parallel fan-out → sequential

```
[generate N search queries from topic]
    │
[transform: split queries into context array]
    │
[parallel: for each query, fetch + extract key facts]
    │
[deduplicate and rank facts]
    │
[synthesize final report]
```

**Context flow:**
```
in:  "topic": String
out: "queries": [String]
     "facts_0"..."facts_n": String   (one per query, from parallel block)
     "ranked_facts": String
     "report": String
```

**Key builder features used:** `parallel`, `transform` (to split/join), `loop_step` for
query expansion if initial results are thin

---

### W-3: Multi-Agent Debate

**Inspiration:** The AskUserQuestion + multiple agent invocations pattern — perspectives are
collected independently, then judged.

**Shape:** parallel fan-out → sequential fan-in

```
[parallel]
    ├── [advocate_a: argue FOR the proposition]
    └── [advocate_b: argue AGAINST the proposition]
        │
[judge: read both arguments, give verdict + reasoning]
    │
[summarize: distill the key points of agreement and disagreement]
```

**Context flow:**
```
in:  "proposition": String
out: "argument_for":     String
     "argument_against": String
     "verdict":          String
     "key_points":       String
```

**Key builder features used:** `parallel`, `system_prompt` (to set each advocate's role),
`temperature` (higher for advocates, lower for judge)

**Example sketch:**
```rust
Workflow::builder()
    .parallel(vec![
        StepBuilder::new("advocate_for")
            .model("llama3")
            .system_prompt("You argue strongly FOR the proposition. Do not hedge.")
            .temperature(0.8)
            .prompt(|ctx| format!("Argue for: {}", ctx.get_str("proposition")))
            .output_text()
            .store_as("argument_for"),
        StepBuilder::new("advocate_against")
            .model("llama3")
            .system_prompt("You argue strongly AGAINST the proposition. Do not hedge.")
            .temperature(0.8)
            .prompt(|ctx| format!("Argue against: {}", ctx.get_str("proposition")))
            .output_text()
            .store_as("argument_against"),
    ])
    .step(
        StepBuilder::new("judge")
            .model("llama3")
            .system_prompt("You are an impartial judge. Evaluate both arguments fairly.")
            .temperature(0.0)
            .prompt(|ctx| format!(
                "FOR:\n{}\n\nAGAINST:\n{}\n\nVerdict:",
                ctx.get_str("argument_for"),
                ctx.get_str("argument_against"),
            ))
            .output_text()
            .store_as("verdict"),
    )
```

---

### W-4: Code Generation → Validation → Documentation

**Inspiration:** The session-start-hook skill — generate code, verify it meets criteria, document it.
Mirrors the Plan → Implement → Review agent cycle.

**Shape:** sequential → conditional loop → sequential

```
[generate initial implementation]
    │
[loop while quality_score < threshold]
    ├── [score and critique the code]
    └── [if score low: rewrite with feedback]
        │
[generate tests for the final implementation]
    │
[generate documentation]
```

**Context flow:**
```
in:  "spec": String, "language": String
out: "implementation": String
     "quality_score":  i64
     "tests":          String
     "docs":           String
```

**Key builder features used:** `loop_step`, `if_step`, `seed` (for reproducible doc generation),
`transform_output` (to unpack score + code from a single JSON response)

---

### W-5: Document Chunking & Hierarchical Summarization

**Inspiration:** How a large-context reading task would be handled by spawning multiple Explore
agents over document sections.

**Shape:** transform (chunk) → parallel summarize chunks → sequential final merge

```
[transform: split document into N chunks, store as array]
    │
[parallel: summarize each chunk]
    │
[transform: join chunk summaries into single block]
    │
[final summary: synthesize from chunk summaries]
    │
[optional: extract key entities / action items in parallel]
```

**Context flow:**
```
in:  "document": String, "chunk_size": i64
out: "chunk_0_summary"..."chunk_n_summary": String
     "combined_summaries": String
     "final_summary":      String
     "entities":           [String]   (optional)
     "action_items":       [String]   (optional)
```

**Key builder features used:** `transform` (chunking logic), `parallel`, `parallel_partial`
(for optional extractions), `on_error` (if a chunk fails, store empty string and continue)

**This workflow would be the first to motivate a `map_array` convenience method:**

```rust
// Future API idea — auto-fans-out over an array key:
.map_step(
    "chunks",          // context key holding the array
    |item, idx| StepBuilder::new(/* dynamic name */)
        .model("llama3")
        .prompt(move |_| format!("Summarise chunk {}:\n{}", idx, item))
        .output_text()
        .store_as_indexed("chunk_summary", idx),  // stores "chunk_summary_0", etc.
)
```

---

### W-6: Structured Data Extraction Pipeline

**Inspiration:** The MCP GitHub tools — read structured data from unstructured text, validate,
and store in typed fields.

**Shape:** sequential → parallel fan-out (one step per entity type) → transform (merge/validate)

```
[identify entity types present in the text]
    │
[parallel]
    ├── [extract: people]
    ├── [extract: organisations]
    ├── [extract: dates]
    └── [extract: locations]
        │
[transform: merge all into a single entities object]
    │
[validate: cross-reference entities for consistency]
```

**Context flow:**
```
in:  "text": String
out: "entity_types":   [String]
     "people":         [String]
     "organisations":  [String]
     "dates":          [String]
     "locations":      [String]
     "entities":       Object
     "validation":     String
```

**Key builder features used:** `parallel`, `output_json_schema`, `transform`, `on_error`

---

### W-7: Iterative Bug-Fix Pipeline

**Inspiration:** The general-purpose agent loop — attempt → observe failure → fix → retry.

**Shape:** sequential → loop (attempt + validate) → sequential

```
[generate initial fix for the described bug]
    │
[loop while not_valid AND attempts < max]
    ├── [validate: does the fix address the root cause?]
    │     └── [transform: extract score and issues from validation JSON]
    └── [if issues found: refine the fix addressing specific feedback]
        │
[generate explanation of what was changed and why]
    │
[generate regression test for this specific bug]
```

**Context flow:**
```
in:  "bug_description": String, "affected_code": String
out: "fix":          String
     "valid":        bool
     "explanation":  String
     "test":         String
```

**Key builder features used:** `loop_step`, `if_step`, `transform` (unpack validation JSON),
`with_retries` (on the validation step), `LoopLimitExceeded` error variant

---

### W-8: Q&A with Context Retrieval (RAG-style)

**Inspiration:** How the WebFetch + Explore agents are used together — gather context first,
then answer.

**Shape:** sequential → parallel retrieval → sequential answer

```
[decompose question into N sub-queries]
    │
[parallel: for each sub-query, retrieve + extract relevant passages]
    │
[transform: rank and deduplicate passages by relevance]
    │
[answer: synthesize from ranked passages]
    │
[verify: does the answer actually address the original question?]
    │
[if verification fails → loop back to retrieval with refined queries]
```

**Context flow:**
```
in:  "question": String, "corpus": [String]
out: "sub_queries":  [String]
     "passages":     [String]
     "answer":       String
     "verified":     bool
```

**Key builder features used:** `parallel`, `transform`, `loop_step`, `if_step`,
`output_json_schema` for structured passage extraction

---

### W-9: Autonomous Task Decomposition

**Inspiration:** The Plan agent + TodoWrite pattern — break a goal into tasks, execute them,
track completion.

**Shape:** sequential → transform → loop (one task at a time) → sequential

```
[decompose: given a goal, produce an ordered task list as JSON]
    │
[transform: initialise task index = 0]
    │
[loop while incomplete tasks remain]
    ├── [execute current task using its description as prompt]
    ├── [transform: mark task done, increment index, store result]
    └── [if task failed → on_error: mark as failed, continue to next]
        │
[summarize: given all results (including failures), produce final output]
```

**Context flow:**
```
in:  "goal": String
out: "tasks":       [{title, description, status, result}]
     "task_index":  i64
     "final_output": String
```

**Key builder features used:** `loop_step`, `transform`, `on_error` with `Skip`,
`output_json_schema`, error context keys (`__error_count`)

This is the most agent-like workflow in the list and motivates most of the error-handling
features in section 2.

---

### W-10: Model Cascade (cost-aware routing)

**Inspiration:** The haiku/sonnet/opus model tiers — use the cheapest model that can handle the
task; escalate only on failure or low confidence.

**Shape:** sequential cascade with fallback chain

```
[attempt with small/fast model]
    │
[if result confidence < threshold OR step failed]
    └── [attempt with medium model]
            │
            [if still unsatisfactory]
                └── [attempt with large model, high quality mode]
```

**Context flow:**
```
in:  "prompt": String, "confidence_threshold": f64
out: "result":          String
     "model_used":      String
     "confidence":      f64
```

**Builder API this motivates:**

```rust
// step_with_fallback chained
Workflow::builder()
    .step_with_fallback(
        StepBuilder::new("tier1").model("llama3:8b")
            .prompt(|ctx| ctx.get_str("prompt").to_string())
            .output_json_schema(confidence_schema())
            .store_as("_tier1"),
        StepBuilder::new("tier2").model("llama3:70b")
            .prompt(|ctx| ctx.get_str("prompt").to_string())
            .output_json_schema(confidence_schema())
            .store_as("_tier2"),
    )
```

---

## Implementation Priority

Based on dependencies and value delivered:

| Priority | Item | Unlocks |
|---|---|---|
| 1 | `WorkflowBuilder::extend()` | Preset composition, migrate ExplorePipeline |
| 2 | `Presets::filter_files`, `Presets::summarize_content` | Cleaned-up ExplorePipeline |
| 3 | `StepErrorAction` + `.on_error()` | W-5, W-7, W-9 |
| 4 | Error context keys (`__last_error_*`) | W-9, conditional recovery |
| 5 | `step_with_fallback()` | W-10 |
| 6 | Richer `TermiError` variants | Better diagnostics everywhere |
| 7 | `.timeout_ms()` | Robustness in production use |
| 8 | `parallel_partial()` | W-2, W-5, W-6 |
| 9 | `Presets::classify`, `extract_json`, `qa` | W-6, W-8 |
| 10 | W-1 Code Review | First real multi-angle pipeline |
| 11 | W-3 Debate | Demonstrates parallel + roles |
| 12 | W-4 Code Gen+Validate | Demonstrates quality loop |
| 13 | W-9 Task Decomposition | Most agent-like, needs all of the above |
