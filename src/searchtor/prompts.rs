use std::collections::HashMap;

pub const SECTIONS: &[(&str, &str)] = &[
    ("executive_summary", "Executive Summary"),
    ("objectives", "Objectives & Problem Statement"),
    ("methodology", "Methodology"),
    ("findings", "Findings & Discussion"),
    ("conclusions", "Conclusions"),
    ("recommendations", "Recommendations"),
    ("appendices", "Appendices"),
];

pub fn sections() -> &'static [(&'static str, &'static str)] {
    SECTIONS
}

pub fn build_query_generation_prompt(topic: &str, depth: usize) -> String {
    let section_list: Vec<String> = SECTIONS
        .iter()
        .map(|(k, l)| format!("  \"{k}\" ({l})"))
        .collect();

    format!(
        r#"You are a research analyst. Generate search queries for a deep analysis of:

TOPIC: {topic}

Generate exactly {depth} targeted search queries for each of these 7 sections:
{sections}

Return ONLY a JSON object with these exact keys. No explanation, no markdown fences.
Each key maps to an array of {depth} distinct, specific search query strings.

{{
  "executive_summary": ["query1", "query2", ...],
  "objectives": ["query1", ...],
  "methodology": ["query1", ...],
  "findings": ["query1", ...],
  "conclusions": ["query1", ...],
  "recommendations": ["query1", ...],
  "appendices": ["query1", ...]
}}"#,
        topic = topic,
        depth = depth,
        sections = section_list.join(",\n"),
    )
}

/// Parse raw SearXNG JSON response into a readable numbered list.
/// Pure function — safe to call from a WorkflowContext transform closure.
pub fn parse_searxng_results(raw_json: &str) -> String {
    let Ok(val) = serde_json::from_str::<serde_json::Value>(raw_json) else {
        let preview = &raw_json[..raw_json.len().min(200)];
        return format!("[could not parse SearXNG response]\nRaw: {preview}");
    };

    let results = val
        .get("results")
        .and_then(|r| r.as_array())
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    if results.is_empty() {
        return "[no results returned by SearXNG]".to_string();
    }

    results
        .iter()
        .take(10)
        .enumerate()
        .map(|(i, r)| {
            let title = r.get("title").and_then(|v| v.as_str()).unwrap_or("(no title)");
            let url = r.get("url").and_then(|v| v.as_str()).unwrap_or("(no url)");
            let snippet = r.get("content").and_then(|v| v.as_str()).unwrap_or("(no snippet)");
            format!(
                "[{}] {}\n    URL: {}\n    {}",
                i + 1,
                title,
                url,
                snippet
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub fn build_analysis_prompt(
    section_label: &str,
    query: &str,
    search_results: &str,
    existing_findings: &str,
) -> String {
    let existing_block = if existing_findings.trim().is_empty() {
        "(none yet)".to_string()
    } else {
        existing_findings.to_string()
    };

    format!(
        r#"You are building findings for the "{section_label}" section of a deep analysis report.

Search query used: {query}

Search Results:
{search_results}

Existing findings for this section:
{existing_block}

Extract key insights, facts, and evidence relevant to "{section_label}" from the search results.
Append NEW findings only — do not repeat what is already captured above.
Be specific. Cite sources (title + URL) where helpful.
If the results add nothing new, state that briefly.

Output ONLY the updated findings text:"#,
        section_label = section_label,
        query = query,
        search_results = search_results,
        existing_block = existing_block,
    )
}

pub fn build_section_writing_prompt(section_label: &str, findings: &str) -> String {
    let findings_block = if findings.trim().is_empty() {
        "(no findings gathered — write a placeholder noting insufficient data)".to_string()
    } else {
        findings.to_string()
    };

    format!(
        r#"You are a professional technical writer producing the "{section_label}" section of a deep analysis report.

Accumulated research findings:
{findings_block}

Write a cohesive, well-structured markdown section for "{section_label}".
Use ## and ### headings, bullet points, and tables where helpful.
Synthesise the findings into a clear narrative — do not just list raw notes.
Be analytical and precise. Aim for 3–6 substantive paragraphs.
Start directly with the section content — no preamble.

Write the section now:"#,
        section_label = section_label,
        findings_block = findings_block,
    )
}

pub fn build_synthesis_prompt(topic: &str, sections_markdown: &str) -> String {
    format!(
        r#"You are a senior analyst producing a final publication-ready deep analysis document.

Topic: {topic}

The following 7 sections have been individually researched and written:

{sections_markdown}

Synthesise all sections into a single, coherent analysis document:
- Ensure smooth narrative flow and logical cross-references between sections.
- Add a brief introduction (2–3 sentences) before the Executive Summary.
- Maintain all section headings but eliminate redundancy and improve transitions.
- The document should read as a unified whole, not 7 disconnected pieces.
- Use markdown formatting throughout.

Output ONLY the complete synthesised document:"#,
        topic = topic,
        sections_markdown = sections_markdown,
    )
}

/// Parse the LLM's JSON query plan into a section-key → queries map.
pub fn parse_query_plan(
    value: serde_json::Value,
) -> Result<HashMap<String, Vec<String>>, crate::error::TermiError> {
    let obj = value.as_object().ok_or_else(|| {
        crate::error::TermiError::Pipeline("query plan is not a JSON object".into())
    })?;

    let mut result = HashMap::new();
    for (key, val) in obj {
        if let Some(arr) = val.as_array() {
            let queries: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            result.insert(key.clone(), queries);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_searxng_results_limits_to_ten() {
        let results: Vec<serde_json::Value> = (0..15)
            .map(|i| {
                serde_json::json!({
                    "title": format!("Title {i}"),
                    "url": format!("https://example.com/{i}"),
                    "content": format!("Snippet {i}")
                })
            })
            .collect();
        let json = serde_json::json!({ "results": results }).to_string();
        let out = parse_searxng_results(&json);
        assert!(out.contains("[10]"), "should have result 10");
        assert!(!out.contains("[11]"), "should not have result 11");
    }

    #[test]
    fn parse_searxng_results_handles_empty() {
        let json = serde_json::json!({ "results": [] }).to_string();
        let out = parse_searxng_results(&json);
        assert!(out.contains("no results"));
    }

    #[test]
    fn parse_searxng_results_handles_bad_json() {
        let out = parse_searxng_results("not json");
        assert!(out.contains("could not parse"));
    }

    #[test]
    fn parse_query_plan_extracts_all_sections() {
        let val = serde_json::json!({
            "executive_summary": ["q1", "q2"],
            "objectives": ["q3"],
        });
        let map = parse_query_plan(val).unwrap();
        assert_eq!(map["executive_summary"], vec!["q1", "q2"]);
        assert_eq!(map["objectives"], vec!["q3"]);
    }
}
