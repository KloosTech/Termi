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
