use crate::explore::walker::FileEntry;

pub fn build_filter_prompt(file_list: &str) -> String {
    format!(
        r#"You are a senior software engineer reviewing a codebase.

Below is the complete list of files in this project:

{file_list}

Your task: identify the 5–15 files that are MOST IMPORTANT for understanding:
- The overall architecture
- Core business logic
- Key data structures
- Entry points

Return ONLY a JSON array of relative file paths (strings), exactly as they appear in the list above.
No explanation. No markdown fences. Just a raw JSON array.

Example: ["src/main.rs", "src/lib.rs", "README.md"]"#
    )
}

pub fn build_summary_prompt(file_contents: &str) -> String {
    format!(
        r#"You are a senior software engineer. Below are the contents of the most important files in a software project.

{file_contents}

Write a clear, thorough summary of this project covering:
1. Purpose and goals of the project
2. Overall architecture and key design patterns
3. Main components and how they interact
4. Notable implementation details or technology choices
5. Any interesting or unusual aspects

Be specific and technical. Assume the audience is an experienced developer."#
    )
}

pub fn format_file_list(entries: &[FileEntry]) -> String {
    entries
        .iter()
        .map(|e| e.relative_path.display().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn format_file_contents(files: &[(String, String)]) -> String {
    files
        .iter()
        .map(|(path, content)| format!("--- {} ---\n{}", path, content))
        .collect::<Vec<_>>()
        .join("\n\n")
}
