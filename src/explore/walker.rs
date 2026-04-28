use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::error::TermiError;

#[derive(Debug, Clone)]
pub struct FileEntry {
    pub relative_path: PathBuf,
    pub absolute_path: PathBuf,
    pub size_bytes: u64,
}

/// Walk `root` recursively and return all files sorted by relative path.
///
/// Skips hidden directories, build artefacts, and other known noise.
pub fn walk_directory(root: &Path) -> Result<Vec<FileEntry>, TermiError> {
    // Canonicalize so that strip_prefix works even when root contains symlinks
    // (e.g. /tmp → /private/tmp on some systems).
    let root = root.canonicalize()?;
    let mut entries = Vec::new();

    for result in WalkDir::new(&root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| !is_excluded(e))
    {
        let entry = result?;
        if entry.file_type().is_file() {
            let metadata = entry.metadata()?;
            let relative_path = entry
                .path()
                .strip_prefix(&root)
                .unwrap_or(entry.path())
                .to_path_buf();
            entries.push(FileEntry {
                relative_path,
                absolute_path: entry.path().to_path_buf(),
                size_bytes: metadata.len(),
            });
        }
    }

    entries.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    Ok(entries)
}

fn is_excluded(entry: &walkdir::DirEntry) -> bool {
    if !entry.file_type().is_dir() {
        return false;
    }
    if entry.depth() == 0 {
        return false; // never exclude the root itself
    }
    let name = entry.file_name().to_string_lossy();
    matches!(
        name.as_ref(),
        ".git" | "target" | "node_modules" | ".next" | "__pycache__" | ".venv" | "dist" | "build"
    ) || name.starts_with('.')
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn make_tree() -> TempDir {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("main.rs"), "fn main(){}").unwrap();
        fs::write(root.join("lib.rs"), "pub fn f(){}").unwrap();
        fs::create_dir(root.join(".git")).unwrap();
        fs::write(root.join(".git").join("config"), "").unwrap();
        fs::create_dir(root.join("target")).unwrap();
        fs::write(root.join("target").join("debug"), "").unwrap();
        dir
    }

    #[test]
    fn finds_source_files_and_excludes_noise() {
        let dir = make_tree();
        let entries = walk_directory(dir.path()).unwrap();
        let names: Vec<_> = entries
            .iter()
            .map(|e| e.relative_path.display().to_string())
            .collect();
        assert!(names.contains(&"main.rs".to_string()));
        assert!(names.contains(&"lib.rs".to_string()));
        // .git and target must be excluded
        assert!(!names.iter().any(|n| n.starts_with(".git")));
        assert!(!names.iter().any(|n| n.starts_with("target")));
    }

    #[test]
    fn entries_are_sorted() {
        let dir = make_tree();
        let entries = walk_directory(dir.path()).unwrap();
        let paths: Vec<_> = entries.iter().map(|e| e.relative_path.clone()).collect();
        let mut sorted = paths.clone();
        sorted.sort();
        assert_eq!(paths, sorted);
    }
}
