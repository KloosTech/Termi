use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "termi", about = "Ollama-powered terminal tools", version)]
pub struct Cli {
    /// Ollama base URL
    #[arg(long, env = "OLLAMA_URL", default_value = "http://localhost:11434")]
    pub ollama_url: String,

    /// Model to use for LLM calls
    #[arg(long, env = "OLLAMA_MODEL", default_value = "llama3:latest")]
    pub model: String,

    /// Use mock Ollama client (no network calls; for testing)
    #[arg(long)]
    pub mock: bool,

    /// Show a live WorkflowContext inspector on the right side of the TUI
    #[arg(long)]
    pub debug: bool,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Explore a directory and produce a project summary
    Explore {
        /// Path to the project directory to analyse
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,
    },
    /// Searches different Index Sites for Torrent releases
    Searchtor {
        /// Search query — multiple words accepted without quotes:
        ///   --query Donald Trump Latest News
        #[arg(long = "query", num_args = 1..)]
        query: Vec<String>,
    },
    /// Review git changes between two refs and produce a code review
    Review {
        /// Base ref to compare against
        #[arg(long, default_value = "main")]
        base: String,
        /// Head ref to review
        #[arg(long, default_value = "HEAD")]
        head: String,
    },
    /// Hunt for dead/unused code in a Rust project
    DeadCode {
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,
    },
    /// Analyse a Rust project and produce a prioritised refactoring roadmap
    Refactor {
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,
    },
    /// Generate markdown API documentation from source code
    AutoDocs {
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,
    },
    /// Generate a developer onboarding guide for a codebase
    Onboard {
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,
    },
    /// Generate a CHANGELOG from git history
    Changelog {
        /// Start ref (defaults to previous tag or first commit)
        #[arg(long)]
        from: Option<String>,
        /// End ref
        #[arg(long, default_value = "HEAD")]
        to: String,
    },
    /// Fetch recent tech news and produce a technology radar
    TechRadar {
        /// Topic to focus on
        #[arg(long, default_value = "rust")]
        topic: String,
    },
    /// Audit Rust dependencies for security issues and outdated packages
    DepAudit {
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,
    },
    /// Compare a crate against alternatives on crates.io
    Competitive {
        /// Primary crate to evaluate
        #[arg(long = "crate")]
        crate_name: String,
        /// Alternative crates to compare against
        #[arg(long, num_args = 0..)]
        vs: Vec<String>,
    },
    /// Generate test stubs for untested Rust code
    GenTests {
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,
    },
    /// Produce a step-by-step Rust edition migration guide
    Migration {
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,
        /// Target Rust edition
        #[arg(long, default_value = "2024")]
        to: String,
    },
    /// Diagnose build errors and panics in a Rust project
    ErrorDetective {
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,
        /// Optional log file to analyse alongside build errors
        #[arg(long)]
        log: Option<PathBuf>,
    },
    /// Detect anomalies in a log file
    LogAnomaly {
        /// Path to the log file
        #[arg(long)]
        log: PathBuf,
        /// Number of lines to read from the tail of the file
        #[arg(long, default_value = "1000")]
        lines: u64,
    },
    /// Run pre-deployment readiness checks and produce a go/no-go decision
    DeployCheck {
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,
    },
    /// Scaffold a new workflow interactively
    New {
        /// Name of the new workflow module (e.g. "review", "summarise")
        name: String,
    },
    /// List models available in Ollama
    ListModels,
    /// Generate embeddings for TEXT and print the vector
    Embed {
        #[arg(value_name = "TEXT")]
        text: String,
    },
}
