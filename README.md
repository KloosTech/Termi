# Termi

An Ollama-powered CLI for exploring codebases and running multi-step LLM workflows.

---

## Workflow Builder

The workflow module provides a fluent builder API for composing multi-step LLM pipelines. Each workflow is a sequence of **nodes** — LLM calls, conditional branches, parallel batches, loops, and pure context transforms — all communicating through a shared `WorkflowContext`.

### Quick Start

```rust
use std::sync::Arc;
use termi::workflow::{runner::{Workflow, WorkflowBuilder}, step::StepBuilder, context::WorkflowContext};

let wf = Workflow::builder()
    .step(
        StepBuilder::new("summarise")
            .model("llama3:8b")
            .prompt(|ctx| format!("Summarise this code:\n{}", ctx.get_str("source")))
            .output_text()
            .store_as("summary"),
    )
    .build();

let ctx = WorkflowContext::new().with("source", include_str!("main.rs"));
let result = wf.run(Arc::clone(&client), ctx).await?;
println!("{}", result.get_str("summary"));
```

---

## WorkflowBuilder

Create a builder with `Workflow::builder()`, chain nodes, then call `.build()`.

### `.step(StepBuilder) -> Self`

Adds a single sequential LLM call.

```rust
Workflow::builder()
    .step(
        StepBuilder::new("classify")
            .model("llama3")
            .prompt(|_| "Classify this issue as bug/feature/question.".to_string())
            .output_text()
            .store_as("label"),
    )
    .build();
```

---

### `.parallel(Vec<StepBuilder>) -> Self`

Runs multiple LLM steps **concurrently**. All results are merged into the context once every step finishes. Use this when steps are independent of each other.

```rust
Workflow::builder()
    .parallel(vec![
        StepBuilder::new("translate_fr")
            .model("llama3")
            .prompt(|ctx| format!("Translate to French: {}", ctx.get_str("text")))
            .output_text()
            .store_as("french"),
        StepBuilder::new("translate_de")
            .model("llama3")
            .prompt(|ctx| format!("Translate to German: {}", ctx.get_str("text")))
            .output_text()
            .store_as("german"),
        StepBuilder::new("translate_es")
            .model("llama3")
            .prompt(|ctx| format!("Translate to Spanish: {}", ctx.get_str("text")))
            .output_text()
            .store_as("spanish"),
    ])
    .build();
```

---

### `.if_step(condition, StepBuilder) -> Self`

Runs `step` only when `condition` returns `true` at execution time. If the condition is `false` the step is silently skipped.

```rust
Workflow::builder()
    .if_step(
        |ctx| ctx.get_bool("needs_review"),
        StepBuilder::new("review")
            .model("llama3")
            .prompt(|ctx| format!("Review this code for bugs:\n{}", ctx.get_str("diff")))
            .output_text()
            .store_as("review_notes"),
    )
    .build();
```

---

### `.if_else_step(condition, if_step, else_step) -> Self`

Runs `if_step` when `condition` is `true`, otherwise runs `else_step`. Exactly one branch always executes.

```rust
Workflow::builder()
    .if_else_step(
        |ctx| ctx.get_str("language") == "rust",
        StepBuilder::new("rust_lint")
            .model("llama3")
            .prompt(|ctx| format!("Check Rust idioms:\n{}", ctx.get_str("code")))
            .output_text()
            .store_as("lint_result"),
        StepBuilder::new("generic_lint")
            .model("llama3")
            .prompt(|ctx| format!("Check code quality:\n{}", ctx.get_str("code")))
            .output_text()
            .store_as("lint_result"),
    )
    .build();
```

---

### `.transform(name, |ctx: &mut WorkflowContext|) -> Self`

Inserts a **pure context transformation** — no LLM is called. The closure receives `&mut WorkflowContext` and can read, write, or delete any key. Use this to reshape data between LLM steps.

```rust
Workflow::builder()
    .step(
        StepBuilder::new("extract")
            .model("llama3")
            .prompt(|_| "List all function names as JSON array.".to_string())
            .output_json()
            .store_as("raw_functions"),
    )
    // Normalise to lowercase before passing on
    .transform("normalise", |ctx| {
        let fns: Vec<String> = ctx
            .get_array("raw_functions")
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_lowercase())
            .collect();
        ctx.set("functions", fns);
        ctx.remove("raw_functions");
    })
    .step(
        StepBuilder::new("document")
            .model("llama3")
            .prompt(|ctx| format!("Document these functions: {:?}", ctx.get_array("functions")))
            .output_text()
            .store_as("docs"),
    )
    .build();
```

---

### `.loop_step(condition, StepBuilder, max_iterations) -> Self`

Repeats `step` while `condition` returns `true`. The `max_iterations` guard prevents infinite loops — the workflow returns an error if the limit is exceeded.

```rust
Workflow::builder()
    // Seed the iteration counter
    .transform("init", |ctx| ctx.set("attempts", 0i64))
    // Keep asking the LLM to improve the answer until it scores >= 8
    .loop_step(
        |ctx| ctx.get_i64("score").unwrap_or(0) < 8,
        StepBuilder::new("refine")
            .model("llama3")
            .system_prompt("You are a strict quality judge. Respond with JSON: {\"answer\": \"...\", \"score\": 0-10}")
            .prompt(|ctx| {
                let prev = ctx.get_str("answer");
                if prev.is_empty() {
                    "Write a haiku about Rust.".to_string()
                } else {
                    format!("Improve this haiku (current score {}):\n{}", ctx.get_i64("score").unwrap_or(0), prev)
                }
            })
            .output_json_schema(serde_json::json!({
                "type": "object",
                "required": ["answer", "score"]
            }))
            .store_as("_result")
            .transform_output(|v, _| {
                // Unwrap the answer and score into separate context keys
                // by returning the full object — we'll split it in a transform node
                v
            }),
        5, // give up after 5 iterations
    )
    .transform("unpack", |ctx| {
        if let Some(obj) = ctx.get_object("_result").cloned() {
            if let Some(answer) = obj.get("answer").and_then(|v| v.as_str()) {
                ctx.set("answer", answer.to_string());
            }
            if let Some(score) = obj.get("score").and_then(|v| v.as_i64()) {
                ctx.set("score", score);
            }
        }
    })
    .build();
```

---

## StepBuilder

Create a step with `StepBuilder::new("name")` and chain the methods below. Every step **requires** `.model()`, `.prompt()`, and `.store_as()` before the workflow is built — missing any of them panics at build time.

### Required methods

| Method | Description |
|--------|-------------|
| `.model(impl Into<String>)` | Ollama model to call (e.g. `"llama3:8b"`, `"mistral:latest"`) |
| `.prompt(\|ctx\| String)` | Closure that builds the user prompt from the current context |
| `.store_as(&'static str)` | Context key where the parsed output is stored |

### Output format

Exactly one of these should be called (default is `.output_text()`):

| Method | Description |
|--------|-------------|
| `.output_text()` | Store raw LLM text as a string (default) |
| `.output_json()` | Parse LLM output as any valid JSON |
| `.output_json_schema(Value)` | Parse JSON and validate against a schema |

**Schema validation** checks `type`, required `properties` for objects, and `items.type` for arrays:

```rust
let schema = serde_json::json!({
    "type": "object",
    "required": ["title", "tags"],
    "properties": {
        "title": { "type": "string" },
        "tags":  { "type": "array", "items": { "type": "string" } }
    }
});

StepBuilder::new("meta")
    .model("llama3")
    .prompt(|_| "Extract title and tags as JSON.".to_string())
    .output_json_schema(schema)
    .store_as("metadata");
```

---

### System prompt

#### `.system_prompt(impl Into<String>)`

Prepends a system message before the user prompt. Use it to set the model's persona, output format, or constraints.

```rust
StepBuilder::new("translate")
    .model("llama3")
    .system_prompt("You are a professional translator. Output only the translated text, nothing else.")
    .prompt(|ctx| format!("Translate to Japanese: {}", ctx.get_str("text")))
    .output_text()
    .store_as("translation");
```

---

### Inference options

These map directly to Ollama's `ModelOptions`:

| Method | Type | Description |
|--------|------|-------------|
| `.temperature(f32)` | 0.0–1.0 | Lower = more deterministic, higher = more creative |
| `.max_tokens(i32)` | positive int | Maximum tokens the model may generate |
| `.top_p(f32)` | 0.0–1.0 | Nucleus sampling probability |
| `.seed(u32)` | any | Fixed seed for reproducible outputs |

```rust
StepBuilder::new("creative_story")
    .model("llama3")
    .temperature(0.9)
    .max_tokens(500)
    .prompt(|_| "Write an unexpected plot twist.".to_string())
    .output_text()
    .store_as("twist");

StepBuilder::new("deterministic_summary")
    .model("llama3")
    .temperature(0.0)
    .seed(42)
    .prompt(|ctx| format!("Summarise: {}", ctx.get_str("article")))
    .output_text()
    .store_as("summary");
```

---

### Retries

#### `.with_retries(u32)`

Retries the step up to `n` additional times on any error before propagating the failure. The total number of attempts is `1 + n`.

```rust
StepBuilder::new("flaky_extraction")
    .model("llama3")
    .prompt(|_| "Extract the version number as plain text.".to_string())
    .output_text()
    .store_as("version")
    .with_retries(3); // try up to 4 times total
```

---

### Conditional skip

#### `.skip_if(|ctx: &WorkflowContext| -> bool)`

Skips the step entirely (no LLM call) when the closure returns `true`. The context is not modified; execution continues with the next node.

```rust
StepBuilder::new("expensive_step")
    .model("llama3")
    .prompt(|_| "Do something costly.".to_string())
    .output_text()
    .store_as("result")
    .skip_if(|ctx| ctx.get_bool("already_done"));
```

---

### Output post-processing

#### `.transform_output(|value: Value, ctx: &WorkflowContext| -> Value)`

Applies a transformation to the parsed LLM output **before** it is stored in the context. Receives the fully parsed `serde_json::Value` and the current context (read-only). Return the value you want stored.

```rust
// Extract a nested field from a JSON response
StepBuilder::new("user_info")
    .model("llama3")
    .prompt(|_| "Return a JSON object with user details.".to_string())
    .output_json()
    .store_as("username")
    .transform_output(|v, _| {
        v.get("username").cloned().unwrap_or(serde_json::Value::Null)
    });

// Convert a comma-separated string into a JSON array
StepBuilder::new("keywords")
    .model("llama3")
    .prompt(|_| "List 5 keywords separated by commas.".to_string())
    .output_text()
    .store_as("keyword_list")
    .transform_output(|v, _| {
        let words: Vec<&str> = v.as_str()
            .unwrap_or("")
            .split(',')
            .map(str::trim)
            .collect();
        serde_json::to_value(words).unwrap()
    });
```

---

## WorkflowContext

`WorkflowContext` is the shared state bag passed through every node. Pre-populate it before calling `run`, and read results from it afterwards.

### Construction

```rust
// Empty context
let ctx = WorkflowContext::new();

// Builder pattern — chain multiple initial values
let ctx = WorkflowContext::new()
    .with("language", "rust")
    .with("source", source_code)
    .with("max_issues", 10u32);
```

### Writing

```rust
ctx.set("key", value);   // any Serialize value
ctx.remove("key");       // remove and return the value
```

### Reading

| Method | Returns | Notes |
|--------|---------|-------|
| `get_str("key")` | `&str` | `""` if absent or not a string |
| `get_bool("key")` | `bool` | `false` if absent or not a boolean |
| `get_i64("key")` | `Option<i64>` | `None` if absent or not an integer |
| `get_f64("key")` | `Option<f64>` | `None` if absent or not a number |
| `get_array("key")` | `&[Value]` | `&[]` if absent or not an array |
| `get_object("key")` | `Option<&Map<String, Value>>` | `None` if absent or not an object |
| `get("key")` | `Option<&Value>` | Raw JSON value |

### Introspection

```rust
ctx.contains("key");             // bool
ctx.keys()                       // impl Iterator<Item = &str>
    .collect::<Vec<_>>();
```

---

## OutputFormat reference

| Variant / Method | Behaviour |
|-----------------|-----------|
| `output_text()` | Raw text stored as `Value::String`. No validation. |
| `output_json()` | Parses any valid JSON. Fails if the model returns non-JSON. |
| `output_json_schema(schema)` | Parses JSON and validates `type`, `required` properties, and `items.type` for arrays. |

When using `output_json()` or `output_json_schema()`, Ollama is instructed to produce JSON output via the `format: "json"` request field.

---

## Advanced Recipes

### 1. Retry with exponential back-off (application-level)

`with_retries` retries immediately. For back-off, wrap the workflow call in your own retry loop:

```rust
for attempt in 0..3 {
    match wf.run(Arc::clone(&client), ctx.clone()).await {
        Ok(result) => return Ok(result),
        Err(e) if attempt < 2 => {
            tokio::time::sleep(Duration::from_secs(2u64.pow(attempt))).await;
        }
        Err(e) => return Err(e),
    }
}
```

### 2. Multi-stage pipeline with fan-out / fan-in

```rust
let wf = Workflow::builder()
    // Stage 1: analyse the input
    .step(
        StepBuilder::new("analyse")
            .model("llama3")
            .prompt(|ctx| format!("Analyse this text:\n{}", ctx.get_str("input")))
            .output_json_schema(serde_json::json!({"type": "object", "required": ["topics", "sentiment"]}))
            .store_as("analysis"),
    )
    // Stage 2: fan-out — process topics and sentiment in parallel
    .parallel(vec![
        StepBuilder::new("expand_topics")
            .model("llama3")
            .prompt(|ctx| {
                let topics = ctx.get_object("analysis")
                    .and_then(|o| o.get("topics"))
                    .map(|v| v.to_string())
                    .unwrap_or_default();
                format!("Expand on these topics: {topics}")
            })
            .output_text()
            .store_as("topic_expansion"),
        StepBuilder::new("sentiment_report")
            .model("llama3")
            .prompt(|ctx| {
                let sentiment = ctx.get_object("analysis")
                    .and_then(|o| o.get("sentiment"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                format!("Write a sentiment report for: {sentiment}")
            })
            .output_text()
            .store_as("sentiment_report"),
    ])
    // Stage 3: fan-in — combine into a final summary
    .step(
        StepBuilder::new("synthesise")
            .model("llama3")
            .prompt(|ctx| format!(
                "Combine into a final report:\n\nTopics:\n{}\n\nSentiment:\n{}",
                ctx.get_str("topic_expansion"),
                ctx.get_str("sentiment_report"),
            ))
            .output_text()
            .store_as("final_report"),
    )
    .build();
```

### 3. Self-improving loop

```rust
let wf = Workflow::builder()
    .transform("init", |ctx| {
        ctx.set("score", 0i64);
        ctx.set("draft", "");
    })
    .loop_step(
        |ctx| ctx.get_i64("score").unwrap_or(0) < 8,
        StepBuilder::new("write_and_score")
            .model("llama3")
            .system_prompt("Respond with JSON: {\"draft\": \"...\", \"score\": 0-10}")
            .prompt(|ctx| {
                let draft = ctx.get_str("draft");
                if draft.is_empty() {
                    format!("Write a short description of: {}", ctx.get_str("topic"))
                } else {
                    format!(
                        "Improve this text (score was {}/10):\n{}",
                        ctx.get_i64("score").unwrap_or(0),
                        draft
                    )
                }
            })
            .output_json_schema(serde_json::json!({"type": "object", "required": ["draft", "score"]}))
            .store_as("_raw"),
        6,
    )
    .transform("unpack", |ctx| {
        if let Some(obj) = ctx.get_object("_raw").cloned() {
            if let Some(d) = obj.get("draft").and_then(|v| v.as_str()) {
                ctx.set("draft", d.to_string());
            }
            if let Some(s) = obj.get("score").and_then(|v| v.as_i64()) {
                ctx.set("score", s);
            }
            ctx.remove("_raw");
        }
    })
    .build();
```

### 4. Conditional model routing

Choose a larger or smaller model based on task complexity:

```rust
let wf = Workflow::builder()
    // First, classify the complexity
    .step(
        StepBuilder::new("classify_complexity")
            .model("llama3:8b") // cheap model for routing
            .temperature(0.0)
            .prompt(|ctx| format!(
                "Is this task simple or complex? Reply with one word.\nTask: {}",
                ctx.get_str("task")
            ))
            .output_text()
            .store_as("complexity"),
    )
    // Route to appropriate model
    .if_else_step(
        |ctx| ctx.get_str("complexity").to_lowercase().contains("simple"),
        StepBuilder::new("fast_answer")
            .model("llama3:8b")
            .prompt(|ctx| ctx.get_str("task").to_string())
            .output_text()
            .store_as("answer"),
        StepBuilder::new("thorough_answer")
            .model("llama3:70b")
            .temperature(0.3)
            .max_tokens(2000)
            .prompt(|ctx| format!(
                "Provide a thorough, detailed answer:\n{}",
                ctx.get_str("task")
            ))
            .output_text()
            .store_as("answer"),
    )
    .build();
```

### 5. Processing a list of items

Use a `transform` node to set up a counter, then loop over items with `loop_step`:

```rust
let wf = Workflow::builder()
    .transform("init_counter", |ctx| ctx.set("idx", 0i64))
    .loop_step(
        |ctx| {
            let idx = ctx.get_i64("idx").unwrap_or(0) as usize;
            idx < ctx.get_array("items").len()
        },
        StepBuilder::new("process_item")
            .model("llama3")
            .prompt(|ctx| {
                let idx = ctx.get_i64("idx").unwrap_or(0) as usize;
                let item = ctx.get_array("items")
                    .get(idx)
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                format!("Summarise in one sentence: {item}")
            })
            .output_text()
            .store_as("_item_result")
            .transform_output(|v, ctx| {
                // Append to results array
                let mut results = ctx.get_array("results").to_vec();
                results.push(v);
                serde_json::to_value(results).unwrap()
            }),
        100,
    )
    .build();

let ctx = WorkflowContext::new()
    .with("items", vec!["article 1 text", "article 2 text", "article 3 text"])
    .with("results", Vec::<String>::new());
```

> **Note:** In `transform_output`, `ctx` still holds the value from the _previous_ iteration when the current step began. Update `results` in the step's `store_as` key to accumulate properly, or use a separate `transform` node after the loop to reshape accumulated data.

---

## Running the CLI

```bash
# Explore a codebase and generate a summary
cargo run -- explore --path ./src --model llama3:8b
```

Set the Ollama base URL via environment variable (defaults to `http://localhost:11434`):

```bash
OLLAMA_BASE_URL=http://my-server:11434 cargo run -- explore --path .
```
