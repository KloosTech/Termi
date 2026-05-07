use anyhow::Result;

/// Synthesise `text` to a WAV file using Qwen3-TTS 0.6B on Apple Silicon.
///
/// The report is split into paragraphs so each chunk fits the model's context
/// window. All chunks are concatenated into a single WAV file.
///
/// Requires: `cargo run --features tts -- --audio …`
/// First run downloads ≈1.5 GB of model weights to `~/.cache/huggingface/`.
pub async fn generate(text: String, output_path: String) -> Result<()> {
    #[cfg(not(feature = "tts"))]
    {
        let _ = (text, output_path);
        anyhow::bail!(
            "Audio generation requires the `tts` Cargo feature.\n\
             Rebuild with: cargo run --features tts -- --audio [args]"
        );
    }

    #[cfg(feature = "tts")]
    synthesise(text, output_path).await
}

/// Returns a filesystem-safe filename for the audio output.
pub fn output_filename(query: &str) -> String {
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let safe: String = query
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' { c } else { '-' })
        .collect::<String>()
        .split_whitespace()
        .take(6)
        .collect::<Vec<_>>()
        .join("-")
        .to_lowercase();
    format!("{date}-{safe}.wav")
}

// ── TTS implementation (compiled only with --features tts) ───────────────────

#[cfg(feature = "tts")]
async fn synthesise(text: String, output_path: String) -> Result<()> {
    use anyhow::Context;
    use qwen3_tts::{audio::AudioBuffer, auto_device, hub::ModelPaths, Qwen3TTS};

    println!("\n Qwen3-TTS 0.6B — Apple Silicon / Metal");
    println!(" First run downloads ≈1.5 GB to ~/.cache/huggingface/\n");

    tokio::task::spawn_blocking(move || -> Result<()> {
        // ── Load model ─────────────────────────────────────────────────────
        let device = auto_device().context(
            "Failed to detect compute device — is Metal available?",
        )?;

        print!(" Downloading / loading model…");
        let paths = ModelPaths::download(None)
            .context("Failed to download Qwen3-TTS model weights from HuggingFace")?;

        let tts = Qwen3TTS::from_paths(&paths, device)
            .context("Failed to initialise Qwen3-TTS")?;
        println!(" done.\n");

        // ── Split report into paragraphs ───────────────────────────────────
        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .map(str::trim)
            .filter(|s| s.len() >= 15)
            .collect();

        let total = paragraphs.len();
        println!(" Synthesising {total} paragraph(s)…\n");

        let mut all_samples: Vec<f32> = Vec::new();
        let mut sample_rate = 24_000u32;

        for (i, para) in paragraphs.iter().enumerate() {
            let preview: String = para.chars().take(72).collect();
            println!(" [{:>2}/{}] {}…", i + 1, total, preview);

            let audio = tts
                .synthesize(para, None)
                .with_context(|| format!("Synthesis failed on paragraph {}", i + 1))?;

            all_samples.extend_from_slice(&audio.samples);
            sample_rate = audio.sample_rate;
        }

        // ── Concatenate and save ───────────────────────────────────────────
        let combined = AudioBuffer { samples: all_samples, sample_rate };
        let duration = combined.duration();

        combined
            .save(&output_path)
            .with_context(|| format!("Could not write WAV to {output_path}"))?;

        println!("\n Audio saved → {output_path}  ({duration:.1}s / {:.1} min)",
            duration / 60.0);

        Ok(())
    })
    .await
    .context("TTS worker thread panicked")?
}
