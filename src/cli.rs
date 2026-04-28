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
    /// List models available in Ollama
    ListModels,
    /// Generate embeddings for TEXT and print the vector
    Embed {
        #[arg(value_name = "TEXT")]
        text: String,
    },
}
