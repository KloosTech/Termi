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
        #[arg(long = "url", env = "LIDARR_URL")]
        url: String,
        /// Lidarr API key
        #[arg(long = "api-key", env = "LIDARR_API_KEY")]
        api_key: String,
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
