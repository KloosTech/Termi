use std::collections::HashSet;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::TermiError;

#[derive(Debug, Default, Serialize, Deserialize)]
struct StoreData {
    seen_message_ids: HashSet<String>,
}

pub struct MailStore {
    path: PathBuf,
    data: StoreData,
}

impl MailStore {
    pub fn load() -> Result<Self, TermiError> {
        let path = store_path()?;
        let data = if path.exists() {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| TermiError::Pipeline(format!("mail store read: {e}")))?;
            serde_json::from_str(&text)
                .map_err(|e| TermiError::Pipeline(format!("mail store parse: {e}")))?
        } else {
            StoreData::default()
        };
        Ok(Self { path, data })
    }

    pub fn is_seen(&self, message_id: &str) -> bool {
        self.data.seen_message_ids.contains(message_id)
    }

    pub fn mark_seen(&mut self, message_id: &str) {
        self.data.seen_message_ids.insert(message_id.to_string());
    }

    pub fn seen_ids_snapshot(&self) -> HashSet<String> {
        self.data.seen_message_ids.clone()
    }

    pub fn merge_seen(&mut self, ids: Vec<String>) {
        for id in ids {
            self.data.seen_message_ids.insert(id);
        }
    }

    pub fn save(&self) -> Result<(), TermiError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| TermiError::Pipeline(format!("mail store dir create: {e}")))?;
        }
        let text = serde_json::to_string_pretty(&self.data)
            .map_err(|e| TermiError::Pipeline(format!("mail store serialize: {e}")))?;
        std::fs::write(&self.path, text)
            .map_err(|e| TermiError::Pipeline(format!("mail store write: {e}")))?;
        Ok(())
    }
}

fn store_path() -> Result<PathBuf, TermiError> {
    let home = std::env::var("HOME")
        .map_err(|_| TermiError::Pipeline("HOME env not set".to_string()))?;
    Ok(PathBuf::from(home).join(".termi").join("mail_seen.json"))
}
