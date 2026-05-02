# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build
cargo build
cargo build --features js-render   # enables headless Chromium via Playwright

# Run
cargo run -- explore --path ./src --model llama3:8b
cargo run -- searchtor --query some search terms
cargo run -- list-models
cargo run -- embed "some text"
cargo run -- new <workflow-name>    # interactive scaffold wizard
cargo run -- --mock explore .       # no Ollama server; plain stdout for testing

# Test
cargo test                          # all tests
cargo test <test_name>              # single test by name
cargo test -p termi workflow        # filter by module

# Lint / format
cargo clippy
cargo fmt
```

Ollama base URL defaults to `http://localhost:11434` and can be overridden:
```bash
OLLAMA_URL=http://my-server:11434 cargo run -- explore .
OLLAMA_MODEL=mistral:latest cargo run -- explore .
```

## Architecture

The project is a Rust CLI (`termi`) for running Ollama-powered LLM pipelines. The binary entry point is `src/main.rs`. Everything flows through three coordinated layers:

### 1. Workflow engine (`src/workflow/`)

The core abstraction. A `Workflow` is a `Vec<WorkflowNode>` executed sequentially. `WorkflowContext` is the only communication channel between nodes — a `HashMap<String, serde_json::Value>` passed by value through every step.

Node types (`WorkflowNode` enum in `runner.rs`):
- `Step` — a streaming Ollama LLM call built with `StepBuilder`
- `Shell` — runs `sh -c "…"` capturing stdout/stderr/exit code
- `Http` — fetches a URL (optionally with headless Chromium and/or HTML→Markdown via `htmd`)
- `Parallel` — runs a `Vec<Step>` concurrently with `join_all`, then merges context
- `Conditional` — if/if-else branching on a closure over `&WorkflowContext`
- `LoopWhile` — repeats a node up to `max_iterations` while a condition holds
- `Transform` — a pure `FnMut(&mut WorkflowContext)` with no LLM call

`WorkflowBuilder` uses a fluent API: `.step()`, `.shell()`, `.http()`, `.parallel()`, `.if_step()`, `.if_else_step()`, `.transform()`, `.loop_step()`, `.with_events()`, `.build()`.

Events flow out through an `Option<mpsc::Sender<StepEvent>>`. When `None`, the workflow runs silently (used in `--mock` mode and tests). When `Some`, every token, step transition, context snapshot, and `WorkflowComplete`/`WorkflowFailed` is sent to the TUI.

### 2. TUI (`src/tui/mod.rs`)

Built with `ratatui` + `crossterm`. Launched from `main.rs` when a command is not `--mock`. Runs on the main task; the pipeline runs in a `tokio::spawn` background task. Communicates via:
- `mpsc::Receiver<StepEvent>` from the pipeline → drives the "Working" phase (streaming tokens, completed step list)
- `mpsc::Receiver<AnswerChunk>` from an internal Q&A task → drives the "Answering" phase

TUI phases: `Working → Reading → Answering ↔ AnswerReady`. On `WorkflowComplete`, the last streamed text becomes the summary and the TUI switches to `Reading`, where the user can scroll and ask follow-up questions against the Ollama model.

The `--debug` flag adds a right-side context inspector panel showing live `WorkflowContext` snapshots.

### 3. Pipeline modules (`src/explore/`, `src/searchtor/`)

Each command gets its own module with a `Pipeline` struct that owns `client`, `model`, and `events`. The pattern is always:

```rust
pub struct FooPipeline {
    client: Arc<dyn OllamaClient>,
    model: String,
    events: Option<mpsc::Sender<StepEvent>>,
}

impl FooPipeline {
    pub fn new(client, model) -> Self { ... }
    pub fn with_events(mut self, tx) -> Self { ... }
    pub async fn run(&self, /* args */) -> Result<String, TermiError> {
        // build Workflow, optionally pass events, run, send WorkflowComplete
    }
}
```

`main.rs` dispatches each `Command` variant with two branches: `--mock` (plain stdout, no TUI) and real (spawns pipeline, launches TUI, awaits handle).

### 4. Ollama client (`src/ollama/`)

`OllamaClient` is an `async_trait`. `RealOllamaClient` calls the Ollama HTTP API via `reqwest`. `MockOllamaClient` records calls in an `Arc<Mutex<Vec<MockCall>>>` and returns configurable canned responses — used in all tests, accessible with `--mock` on the CLI.

### 5. Scaffold wizard (`src/wizard/mod.rs`)

`cargo run -- new <name>` runs an interactive `dialoguer` wizard that generates `src/<name>/mod.rs` and `src/<name>/pipeline.rs`, then patches `src/cli.rs` and `src/main.rs` in-place. After generation, fill in the `TODO` comments in `pipeline.rs`.

## Adding a new workflow (manual path)

Four files to touch, in order:

1. **Create** `src/<name>/mod.rs` + `src/<name>/pipeline.rs` — implement `Pipeline::new()`, `with_events()`, `run()`. Always send `StepEvent::WorkflowComplete` at the end of `run()`.
2. **`src/main.rs`** — add `mod <name>;` + `use <name>::<Name>Pipeline;`, expand the `will_run_tui` match pattern, add a match arm with `--mock` and TUI branches.
3. **`src/cli.rs`** — add a `Command` variant with doc-comment (becomes `--help` text).

The wizard (`cargo run -- new <name>`) automates all four steps; prefer it for new workflows.

## Key conventions

- `StepBuilder` requires `.model()`, `.prompt()`, and `.store_as()` — missing any panics at `finish()`.
- Pass `events` into every `Workflow::builder()` call with `if let Some(tx) = self.events.clone() { b = b.with_events(tx); }` — one `Workflow` per `with_events` call.
- Non-zero shell exit codes are **not** errors — they are stored in the context. A launch failure or timeout is a `TermiError::Pipeline`.
- Non-2xx HTTP responses are errors unless `.store_status_as()` is set.
- Use `.strip_html()` when passing web content to an LLM; it reduces prompt size by 80–95%.
- Tests use `MockOllamaClient` — create one with `MockOllamaClient::new("model").with_chat_response("…")`, cast to `Arc<dyn OllamaClient>`, and call `client.recorded_calls().await` to assert call ordering.
- The `js-render` Cargo feature gates Playwright/headless Chromium — requires `npx playwright@1.59.1 install chromium` at runtime.
