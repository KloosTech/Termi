use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context};
use dialoguer::{theme::ColorfulTheme, Confirm, Input, MultiSelect, Select};

// ── Configuration types ───────────────────────────────────────────────────────

struct WizardConfig {
    /// snake_case module name, e.g. "my_review"
    name: String,
    /// PascalCase struct prefix, e.g. "MyReview"
    name_pascal: String,
    description: String,
    args: Vec<CliArg>,
    steps: Vec<StepKind>,
}

struct CliArg {
    name: String,
    description: String,
    kind: ArgKind,
    /// None means the argument is required (no default).
    default: Option<String>,
}

#[derive(Clone, Copy)]
enum ArgKind {
    Str,
    Path,
    Bool,
}

#[derive(Clone, Copy, PartialEq)]
enum StepKind {
    Llm,
    Shell,
    Http,
}

// ── Public entry point ────────────────────────────────────────────────────────

pub fn run(name: String) -> anyhow::Result<()> {
    let name = normalize_name(&name);
    let cfg = prompt_config(name)?;

    println!();

    let dir = format!("src/{}", cfg.name);
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create {dir}"))?;

    write_file(&format!("{dir}/mod.rs"), gen_mod_rs(&cfg))?;
    write_file(&format!("{dir}/pipeline.rs"), gen_pipeline_rs(&cfg))?;

    let cli_ok = patch_cli_rs(&cfg);
    let main_ok = patch_main_rs(&cfg);

    let arg_example: String = cfg
        .args
        .iter()
        .map(|a| {
            let flag = a.name.replace('_', "-");
            match a.kind {
                ArgKind::Bool => format!(" --{flag}"),
                _ => {
                    let val = a.default.as_deref().unwrap_or("<value>");
                    format!(" --{flag} {val}")
                }
            }
        })
        .collect();

    println!();
    println!(
        "Edit src/{}/pipeline.rs to fill in the TODO comments, then run:",
        cfg.name
    );
    println!();
    println!(
        "  cargo run -- {}{arg_example}",
        cfg.name.replace('_', "-")
    );

    if cli_ok.is_err() || main_ok.is_err() {
        println!();
        println!("Some files could not be patched automatically.");
        println!("See the README section 'Adding a new workflow to the CLI' for the manual steps.");
    }

    Ok(())
}

// ── Interactive prompts ───────────────────────────────────────────────────────

fn prompt_config(name: String) -> anyhow::Result<WizardConfig> {
    let theme = ColorfulTheme::default();
    let name_pascal = to_pascal_case(&name);

    println!();
    println!("Creating workflow: {name}  ({name_pascal}Pipeline)");
    println!();

    let description: String = Input::with_theme(&theme)
        .with_prompt("Short description (shown in --help)")
        .interact_text()?;

    // ── CLI arguments ──────────────────────────────────────────────────────
    let mut args: Vec<CliArg> = Vec::new();
    loop {
        let prompt = if args.is_empty() {
            "Add a CLI argument?"
        } else {
            "Add another CLI argument?"
        };
        let add = Confirm::with_theme(&theme)
            .with_prompt(prompt)
            .default(args.is_empty())
            .interact()?;
        if !add {
            break;
        }

        let arg_name: String = Input::with_theme(&theme)
            .with_prompt("  Name (e.g. path, query, base)")
            .interact_text()?;

        let arg_desc: String = Input::with_theme(&theme)
            .with_prompt("  Description")
            .interact_text()?;

        let kind_idx = Select::with_theme(&theme)
            .with_prompt("  Type")
            .items(&[
                "String                 — text value (--flag value)",
                "PathBuf                — file or directory path",
                "bool                   — flag with no value (--flag)",
            ])
            .default(0)
            .interact()?;

        let kind = match kind_idx {
            1 => ArgKind::Path,
            2 => ArgKind::Bool,
            _ => ArgKind::Str,
        };

        let default = if matches!(kind, ArgKind::Bool) {
            None
        } else {
            let d: String = Input::with_theme(&theme)
                .with_prompt("  Default value (empty = required argument)")
                .allow_empty(true)
                .interact_text()?;
            if d.is_empty() { None } else { Some(d) }
        };

        args.push(CliArg {
            name: normalize_name(&arg_name),
            description: arg_desc,
            kind,
            default,
        });
    }

    // ── Step selection ─────────────────────────────────────────────────────
    let selections = MultiSelect::with_theme(&theme)
        .with_prompt("Steps to include  (Space to toggle, Enter to confirm)")
        .items(&[
            "LLM call    — prompt an Ollama model",
            "Shell step  — run a terminal command and capture output",
            "HTTP step   — fetch a URL, optionally convert HTML to Markdown",
        ])
        .defaults(&[true, false, false])
        .interact()?;

    let steps: Vec<StepKind> = if selections.is_empty() {
        vec![StepKind::Llm]
    } else {
        selections
            .iter()
            .map(|&i| match i {
                1 => StepKind::Shell,
                2 => StepKind::Http,
                _ => StepKind::Llm,
            })
            .collect()
    };

    Ok(WizardConfig { name, name_pascal, description, args, steps })
}

// ── Code generation ───────────────────────────────────────────────────────────

fn gen_mod_rs(cfg: &WizardConfig) -> String {
    format!(
        "pub mod pipeline;\n\npub use pipeline::{}Pipeline;\n",
        cfg.name_pascal
    )
}

fn gen_pipeline_rs(cfg: &WizardConfig) -> String {
    let mut o = String::new();
    let pascal = &cfg.name_pascal;
    let has_llm = cfg.steps.contains(&StepKind::Llm);
    let has_shell = cfg.steps.contains(&StepKind::Shell);
    let has_http = cfg.steps.contains(&StepKind::Http);

    // The key that holds the final output returned by run().
    let result_key = if has_llm {
        "result"
    } else if has_http {
        "http_content"
    } else {
        "shell_output"
    };

    // ── Imports ────────────────────────────────────────────────────────────
    writeln!(o, "use std::sync::Arc;\n").unwrap();
    writeln!(o, "use tokio::sync::mpsc;\n").unwrap();
    writeln!(o, "use crate::error::TermiError;").unwrap();
    writeln!(o, "use crate::ollama::OllamaClient;").unwrap();
    writeln!(o, "use crate::workflow::context::WorkflowContext;").unwrap();
    writeln!(o, "use crate::workflow::events::StepEvent;").unwrap();
    writeln!(o, "use crate::workflow::runner::Workflow;").unwrap();
    if has_shell {
        writeln!(o, "use crate::workflow::shell::ShellStepBuilder;").unwrap();
    }
    if has_http {
        writeln!(o, "use crate::workflow::http::{{HttpStepBuilder, url_encode}};").unwrap();
    }
    if has_llm {
        writeln!(o, "use crate::workflow::step::StepBuilder;").unwrap();
    }
    writeln!(o).unwrap();

    // ── Struct ─────────────────────────────────────────────────────────────
    writeln!(o, "pub struct {pascal}Pipeline {{").unwrap();
    writeln!(o, "    client: Arc<dyn OllamaClient>,").unwrap();
    writeln!(o, "    model: String,").unwrap();
    writeln!(o, "    events: Option<mpsc::Sender<StepEvent>>,").unwrap();
    writeln!(o, "}}\n").unwrap();

    // ── impl ───────────────────────────────────────────────────────────────
    writeln!(o, "impl {pascal}Pipeline {{").unwrap();
    writeln!(
        o,
        "    pub fn new(client: Arc<dyn OllamaClient>, model: String) -> Self {{"
    )
    .unwrap();
    writeln!(o, "        Self {{ client, model, events: None }}").unwrap();
    writeln!(o, "    }}\n").unwrap();

    writeln!(
        o,
        "    pub fn with_events(mut self, tx: mpsc::Sender<StepEvent>) -> Self {{"
    )
    .unwrap();
    writeln!(o, "        self.events = Some(tx);").unwrap();
    writeln!(o, "        self").unwrap();
    writeln!(o, "    }}\n").unwrap();

    // ── run() signature ────────────────────────────────────────────────────
    if cfg.args.is_empty() {
        writeln!(o, "    pub async fn run(&self) -> Result<String, TermiError> {{").unwrap();
    } else {
        writeln!(o, "    pub async fn run(").unwrap();
        writeln!(o, "        &self,").unwrap();
        for arg in &cfg.args {
            let ty = rust_type_owned(arg.kind);
            writeln!(o, "        {}: {ty},", arg.name).unwrap();
        }
        writeln!(o, "    ) -> Result<String, TermiError> {{").unwrap();
    }

    // ── Build the workflow ─────────────────────────────────────────────────
    writeln!(o, "        let mut b = Workflow::builder();").unwrap();
    writeln!(o, "        if let Some(tx) = self.events.clone() {{").unwrap();
    writeln!(o, "            b = b.with_events(tx);").unwrap();
    writeln!(o, "        }}\n").unwrap();

    // Seed the context with CLI arg values.
    writeln!(o, "        let ctx = WorkflowContext::new();").unwrap();
    if !cfg.args.is_empty() {
        writeln!(o, "        // TODO: pass CLI arguments into the context if steps need them, e.g.:").unwrap();
        for arg in &cfg.args {
            let val = match arg.kind {
                // Vec<String> args — join words back into one string.
                ArgKind::Str => format!("{}.join(\" \")", arg.name),
                ArgKind::Bool => arg.name.clone(),
                ArgKind::Path => format!("{}.to_string_lossy().into_owned()", arg.name),
            };
            writeln!(o, "        // let ctx = ctx.with(\"{}\", {val});", arg.name).unwrap();
        }
    }
    writeln!(o).unwrap();

    // ── Step chain ─────────────────────────────────────────────────────────
    writeln!(o, "        let ctx = b").unwrap();

    if has_shell {
        writeln!(o, "            .shell(").unwrap();
        writeln!(o, "                ShellStepBuilder::new(\"shell_step\")").unwrap();
        writeln!(o, "                    // TODO: replace with your command").unwrap();
        writeln!(o, "                    .command(|_ctx| \"echo hello\".to_string())").unwrap();
        writeln!(o, "                    .store_stdout_as(\"shell_output\")").unwrap();
        writeln!(o, "                    .store_exit_code_as(\"shell_exit\")").unwrap();
        writeln!(o, "                    .timeout_secs(60),").unwrap();
        writeln!(o, "            )").unwrap();
    }

    if has_http {
        let has_str_args = cfg.args.iter().any(|a| matches!(a.kind, ArgKind::Str));
        writeln!(o, "            .http(").unwrap();
        writeln!(o, "                HttpStepBuilder::new(\"http_step\")").unwrap();
        writeln!(o, "                    // TODO: replace with your URL.").unwrap();
        if has_str_args {
            // Show url_encode pattern when the workflow has string CLI args.
            let first_str = cfg.args.iter().find(|a| matches!(a.kind, ArgKind::Str)).unwrap();
            writeln!(o, "                    // String arguments must be URL-encoded:").unwrap();
            writeln!(o,
                "                    // .url(|ctx| format!(\"https://example.com/search?q={{}}\", url_encode(ctx.get_str(\"{}\"))))",
                first_str.name
            ).unwrap();
        }
        writeln!(o, "                    .url(|_ctx| \"https://example.com\".to_string())").unwrap();
        writeln!(o, "                    .store_as(\"http_content\")").unwrap();
        writeln!(o, "                    .strip_html()").unwrap();
        writeln!(o, "                    .timeout_secs(30),").unwrap();
        writeln!(o, "            )").unwrap();
    }

    if has_llm {
        writeln!(o, "            .step(").unwrap();
        writeln!(o, "                StepBuilder::new(\"llm_step\")").unwrap();
        writeln!(o, "                    .model(&self.model)").unwrap();
        writeln!(o, "                    .prompt(|_ctx| {{").unwrap();
        writeln!(o, "                        // TODO: build your prompt from context values, e.g.:").unwrap();
        if has_shell {
            writeln!(o, "                        // format!(\"Summarise:\\n{{}}\", _ctx.get_str(\"shell_output\"))").unwrap();
        } else if has_http {
            writeln!(o, "                        // format!(\"Summarise:\\n{{}}\", _ctx.get_str(\"http_content\"))").unwrap();
        }
        writeln!(o, "                        \"TODO: write your prompt\".to_string()").unwrap();
        writeln!(o, "                    }})").unwrap();
        writeln!(o, "                    .output_text()").unwrap();
        writeln!(o, "                    .store_as(\"result\"),").unwrap();
        writeln!(o, "            )").unwrap();
    }

    writeln!(o, "            .build()").unwrap();
    writeln!(o, "            .run(Arc::clone(&self.client), ctx)").unwrap();
    writeln!(o, "            .await?;\n").unwrap();

    // ── WorkflowComplete + return ──────────────────────────────────────────
    writeln!(o, "        if let Some(tx) = &self.events {{").unwrap();
    writeln!(o, "            let _ = tx.send(StepEvent::WorkflowComplete(None)).await;").unwrap();
    writeln!(o, "        }}\n").unwrap();
    writeln!(o, "        Ok(ctx.get_str(\"{result_key}\").to_string())").unwrap();
    writeln!(o, "    }}").unwrap();
    writeln!(o, "}}").unwrap();

    o
}

/// Generate the new `Command` variant block to insert into cli.rs.
fn gen_cli_variant(cfg: &WizardConfig) -> String {
    let mut o = String::new();
    let pascal = &cfg.name_pascal;

    writeln!(o, "    /// {}", cfg.description).unwrap();
    if cfg.args.is_empty() {
        writeln!(o, "    {pascal},").unwrap();
    } else {
        writeln!(o, "    {pascal} {{").unwrap();
        for arg in &cfg.args {
            let flag = arg.name.replace('_', "-");
            writeln!(o, "        /// {}", arg.description).unwrap();
            match arg.kind {
                ArgKind::Bool => {
                    writeln!(o, "        #[arg(long = \"{flag}\")]").unwrap();
                    writeln!(o, "        {}: bool,", arg.name).unwrap();
                }
                ArgKind::Path => {
                    if let Some(ref d) = arg.default {
                        writeln!(o, "        #[arg(long = \"{flag}\", default_value = \"{d}\")]").unwrap();
                    } else {
                        writeln!(o, "        #[arg(long = \"{flag}\")]").unwrap();
                    }
                    writeln!(o, "        {}: std::path::PathBuf,", arg.name).unwrap();
                }
                ArgKind::Str => {
                    // num_args = 1.. lets the user write --flag word1 word2 word3
                    // without needing shell quotes, which is unreliable in run files.
                    if let Some(ref d) = arg.default {
                        writeln!(o, "        #[arg(long = \"{flag}\", num_args = 1.., default_values_t = [\"{d}\".to_string()])]").unwrap();
                    } else {
                        writeln!(o, "        #[arg(long = \"{flag}\", num_args = 1..)]").unwrap();
                    }
                    writeln!(o, "        {}: Vec<String>,", arg.name).unwrap();
                }
            }
        }
        writeln!(o, "    }},").unwrap();
    }

    o
}

/// Generate the match arm to insert into main.rs.
fn gen_main_arm(cfg: &WizardConfig) -> String {
    let mut o = String::new();
    let pascal = &cfg.name_pascal;
    let name = &cfg.name;

    // Pattern: `Command::Review { base }` or `Command::Review`
    let arm_pat = if cfg.args.is_empty() {
        format!("Command::{pascal}")
    } else {
        let fields: Vec<String> = cfg.args.iter().map(|a| a.name.clone()).collect();
        format!("Command::{pascal} {{ {} }}", fields.join(", "))
    };

    // run() call args — Vec<String> fields are joined back into a single String.
    let run_args: String = cfg
        .args
        .iter()
        .map(|a| match a.kind {
            ArgKind::Str => format!("{}.join(\" \")", a.name),
            _ => a.name.clone(),
        })
        .collect::<Vec<_>>()
        .join(", ");
    let run_call = if run_args.is_empty() {
        "pipeline.run()".to_string()
    } else {
        format!("pipeline.run({run_args})")
    };

    writeln!(o, "        {arm_pat} => {{").unwrap();
    writeln!(o, "            if cli.mock {{").unwrap();
    writeln!(o, "                let pipeline =").unwrap();
    writeln!(o, "                    {pascal}Pipeline::new(Arc::clone(&client), cli.model.clone());").unwrap();
    writeln!(
        o,
        "                let result = {run_call}.await.context(\"{name} pipeline failed\")?;"
    )
    .unwrap();
    writeln!(o, "                println!(\"\\n=== {pascal} ===\\n\");").unwrap();
    writeln!(o, "                println!(\"{{}}\", result);").unwrap();
    writeln!(o, "            }} else {{").unwrap();
    writeln!(
        o,
        "                let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(1024);"
    )
    .unwrap();
    writeln!(o, "                let pipeline =").unwrap();
    writeln!(
        o,
        "                    {pascal}Pipeline::new(Arc::clone(&client), cli.model.clone())"
    )
    .unwrap();
    writeln!(o, "                        .with_events(tx);").unwrap();
    writeln!(o).unwrap();
    writeln!(
        o,
        "                let handle = tokio::spawn(async move {{ {run_call}.await }});"
    )
    .unwrap();
    writeln!(o).unwrap();
    writeln!(
        o,
        "                tui::run(rx, cli.model.clone(), \"{name}\".to_string(), Arc::clone(&client), cli.debug)"
    )
    .unwrap();
    writeln!(o, "                    .await.context(\"TUI error\")?;").unwrap();
    writeln!(o).unwrap();
    writeln!(
        o,
        "                let result = handle.await.context(\"pipeline task panicked\")??;"
    )
    .unwrap();
    writeln!(o, "                println!(\"\\n=== {pascal} ===\\n\");").unwrap();
    writeln!(o, "                println!(\"{{}}\", result);").unwrap();
    writeln!(o, "            }}").unwrap();
    writeln!(o, "        }}").unwrap();

    o
}

// ── File patching ─────────────────────────────────────────────────────────────

fn patch_cli_rs(cfg: &WizardConfig) -> anyhow::Result<()> {
    let path = Path::new("src/cli.rs");
    let src = fs::read_to_string(path).context("failed to read src/cli.rs")?;

    // Insert the new variant before the `New` scaffolding command so user
    // workflows always appear before the tool-level commands.
    let marker = "    /// Scaffold a new workflow interactively";
    if !src.contains(marker) {
        bail!("insertion marker not found in src/cli.rs");
    }

    let variant = gen_cli_variant(cfg);
    let patched = src.replacen(marker, &format!("{variant}{marker}"), 1);
    fs::write(path, &patched).context("failed to write src/cli.rs")?;
    println!("  ✓ patched src/cli.rs");
    Ok(())
}

fn patch_main_rs(cfg: &WizardConfig) -> anyhow::Result<()> {
    let path = Path::new("src/main.rs");
    let src = fs::read_to_string(path).context("failed to read src/main.rs")?;
    let name = &cfg.name;
    let pascal = &cfg.name_pascal;

    // 1. Add `mod <name>;` after `mod explore;`
    let mod_marker = "mod explore;";
    if !src.contains(mod_marker) {
        bail!("could not find `mod explore;` in src/main.rs");
    }
    let src = src.replacen(mod_marker, &format!("{mod_marker}\nmod {name};"), 1);

    // 2. Add use statement after the explore use line
    let use_marker = "use explore::{ExploreConfig, ExplorePipeline};";
    if !src.contains(use_marker) {
        bail!("could not find explore use statement in src/main.rs");
    }
    let src = src.replacen(
        use_marker,
        &format!("{use_marker}\nuse {name}::{pascal}Pipeline;"),
        1,
    );

    // 3. Expand will_run_tui to include the new command
    let tui_marker = "Command::Explore { .. }";
    if !src.contains(tui_marker) {
        bail!("could not find will_run_tui check in src/main.rs");
    }
    let src = src.replacen(
        tui_marker,
        &format!("{tui_marker} | Command::{pascal} {{ .. }}"),
        1,
    );

    // 4. Insert match arm before `Command::New { name } =>`
    let arm_marker = "        Command::New { name } => {";
    if !src.contains(arm_marker) {
        bail!("could not find Command::New arm in src/main.rs");
    }
    let new_arm = gen_main_arm(cfg);
    let src = src.replacen(arm_marker, &format!("{new_arm}{arm_marker}"), 1);

    fs::write(path, &src).context("failed to write src/main.rs")?;
    println!("  ✓ patched src/main.rs");
    Ok(())
}

// ── Utilities ─────────────────────────────────────────────────────────────────

fn normalize_name(s: &str) -> String {
    s.to_lowercase().replace('-', "_")
}

fn to_pascal_case(s: &str) -> String {
    s.split(|c| c == '_' || c == '-')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}

fn rust_type_owned(kind: ArgKind) -> &'static str {
    match kind {
        ArgKind::Str => "String",
        ArgKind::Path => "std::path::PathBuf",
        ArgKind::Bool => "bool",
    }
}

fn write_file(path: &str, content: String) -> anyhow::Result<()> {
    fs::write(path, &content).with_context(|| format!("failed to write {path}"))?;
    println!("  ✓ created {path}");
    Ok(())
}
