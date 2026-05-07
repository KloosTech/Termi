use crate::workflow::events::StepEvent;
use std::path::Path;
use tokio::sync::mpsc;
use tracing::info;

/// Save a document as a markdown file in an Obsidian vault.
/// This includes basic sanitization of the filename and adding simple frontmatter.
pub async fn save(
    vault_path: &str,
    title: &str,
    content_body: &str,
    tags: &[&str],
    events: &Option<mpsc::Sender<StepEvent>>,
) {
    let now = chrono::Local::now();
    let date_str = now.format("%Y-%m-%d").to_string();

    // Sanitise the title into a safe filename.
    let safe_name: String = title
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    let safe_name = if safe_name.len() > 60 {
        safe_name.chars().take(60).collect::<String>()
    } else {
        safe_name
    };

    let filename = format!("{safe_name} - {date_str}.md");
    let full_path = Path::new(vault_path).join(&filename);

    let mut frontmatter = String::from("---\n");
    if !tags.is_empty() {
        frontmatter.push_str("tags:\n");
        for tag in tags {
            frontmatter.push_str(&format!("  - {}\n", tag));
        }
    }
    frontmatter.push_str(&format!(
        "date: {}\ntitle: \"{}\"\n---\n\n",
        date_str,
        title.replace('"', "\\\"")
    ));

    let content = format!("{frontmatter}{content_body}");

    if let Err(e) = tokio::fs::create_dir_all(vault_path).await {
        tracing::warn!("Could not create vault directory '{}': {e}", vault_path);
        return;
    }

    match tokio::fs::write(&full_path, &content).await {
        Ok(_) => {
            info!("Document saved to vault: {}", full_path.display());
            if let Some(tx) = events {
                let _ = tx
                    .send(StepEvent::StatusUpdate {
                        message: format!("Saved to vault: {filename}"),
                    })
                    .await;
            }
        }
        Err(e) => {
            tracing::warn!("Could not write document to '{}': {e}", full_path.display());
        }
    }
}
