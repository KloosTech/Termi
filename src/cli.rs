use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "termi", about = "Ollama-powered terminal tools", version)]
pub struct Cli {
    /// Ollama base URL
    #[arg(long, env = "OLLAMA_URL", default_value = "http://localhost:11434")]
    pub ollama_url: String,

    /// Model to use for LLM calls
    #[arg(
        long,
        env = "OLLAMA_MODEL",
        default_value = "gemma4:e4b",
        global = true
    )]
    pub model: String,

    /// Use mock Ollama client (no network calls; for testing)
    #[arg(long, global = true)]
    pub mock: bool,

    /// Show a live WorkflowContext inspector on the right side of the TUI
    #[arg(long, global = true)]
    pub debug: bool,

    /// Synthesise the final report to a WAV audio file using Qwen3-TTS.
    /// Requires: cargo run --features tts
    #[arg(long, global = true)]
    pub audio: bool,

    /// Path to an Obsidian vault to save results (markdown)
    #[arg(long, global = true)]
    pub vault: Option<String>,

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
    /// Deep analysis workflow: searches the web and produces a structured 7-part report
    Searchtor {
        /// Search query — multiple words accepted without quotes:
        ///   --query Donald Trump Latest News
        #[arg(long = "query", num_args = 1..)]
        query: Vec<String>,

        /// Number of search queries to generate per analysis section (1–10)
        #[arg(long = "depth", default_value_t = 3)]
        depth: usize,
    },
    /// Search for a TV series and add it to Sonarr
    Sonarr {
        /// Search query — multiple words accepted without quotes
        #[arg(long = "query", num_args = 1..)]
        query: Vec<String>,
        /// Sonarr base URL
        #[arg(long = "url", env = "SONARR_URL")]
        url: String,
        /// Sonarr API key
        #[arg(long = "api-key", env = "SONARR_API_KEY")]
        api_key: String,
    },
    /// Search for a movie and add it to Radarr
    Radarr {
        /// Search query — multiple words accepted without quotes
        #[arg(long = "query", num_args = 1..)]
        query: Vec<String>,
        /// Radarr base URL
        #[arg(long = "url", env = "RADARR_URL")]
        url: String,
        /// Radarr API key
        #[arg(long = "api-key", env = "RADARR_API_KEY")]
        api_key: String,
    },
    /// Search for an artist and add them to Lidarr
    Lidarr {
        /// Search query — multiple words accepted without quotes
        #[arg(long = "query", num_args = 1..)]
        query: Vec<String>,
        /// Lidarr base URL
        #[arg(
            long = "url",
            env = "LIDARR_URL",
            default_value = "http://192.168.1.54:8686"
        )]
        url: String,
        /// Lidarr API key
        #[arg(
            long = "api-key",
            env = "LIDARR_API_KEY",
            default_value = "9a02699fc04c4226b18b4e4eb118343b"
        )]
        api_key: String,
    },
    /// Review git changes between two refs and produce a code review
    Review {
        /// Base ref to compare against
        #[arg(long, default_value = "origin/main")]
        base: String,
        /// Head ref to review
        #[arg(long, default_value = "HEAD")]
        head: String,
    },
    /// Summarise latest git changes and suggest a commit message
    CommitGen,
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
    /// Fetch new emails from IONOS IMAP and produce an AI-powered briefing
    Mail {
        /// Maximum number of recent inbox messages to inspect
        #[arg(long, default_value_t = 50)]
        limit: usize,
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
