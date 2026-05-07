use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
};

/// A simple markdown-to-ratatui parser for terminal rendering.
///
/// Supported elements:
/// - Headers (#, ##, ###)
/// - Code blocks (```)
/// - Bold (**text**)
/// - Inline code (`text`)
/// - Lists (- or *)
pub fn parse(text: &str) -> Text<'_> {
    let mut lines = Vec::new();
    let mut in_code_block = false;

    for line in text.lines() {
        // Toggle code block
        if line.starts_with("```") {
            in_code_block = !in_code_block;
            lines.push(Line::from(Span::styled(
                line,
                Style::default().fg(Color::DarkGray),
            )));
            continue;
        }

        // Inside a code block, use a specific color and no extra parsing
        if in_code_block {
            lines.push(Line::from(Span::styled(
                line,
                Style::default().fg(Color::Cyan),
            )));
            continue;
        }

        // Header level 1
        if let Some(content) = line.strip_prefix("# ") {
            lines.push(Line::from(Span::styled(
                format!(" # {} ", content.to_uppercase()),
                Style::default()
                    .bg(Color::Blue)
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )));
            lines.push(Line::from("")); // spacer
            continue;
        }

        // Header level 2
        if let Some(content) = line.strip_prefix("## ") {
            lines.push(Line::from(Span::styled(
                content,
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            )));
            continue;
        }

        // Header level 3
        if let Some(content) = line.strip_prefix("### ") {
            lines.push(Line::from(Span::styled(
                content,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )));
            continue;
        }

        // List items
        if line.starts_with("- ") || line.starts_with("* ") {
            lines.push(parse_inline(line, Style::default()));
            continue;
        }

        // Empty line
        if line.trim().is_empty() {
            lines.push(Line::from(""));
            continue;
        }

        // Normal text line
        lines.push(parse_inline(line, Style::default()));
    }

    Text::from(lines)
}

/// Parses bold and inline code within a line.
fn parse_inline(text: &str, base_style: Style) -> Line<'_> {
    let mut spans = Vec::new();
    let mut last_pos = 0;
    let mut i = 0;
    let bytes = text.as_bytes();

    while i < bytes.len() {
        // Bold: **
        if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'*' {
            if i > last_pos {
                spans.push(Span::styled(&text[last_pos..i], base_style));
            }

            let start = i + 2;
            let mut found = false;
            for j in start..(bytes.len() - 1) {
                if bytes[j] == b'*' && bytes[j + 1] == b'*' {
                    spans.push(Span::styled(
                        &text[start..j],
                        base_style.add_modifier(Modifier::BOLD),
                    ));
                    i = j + 1;
                    last_pos = i + 1;
                    found = true;
                    break;
                }
            }
            if !found {
                i += 1;
            }
        }
        // Inline code: `
        else if bytes[i] == b'`' {
            if i > last_pos {
                spans.push(Span::styled(&text[last_pos..i], base_style));
            }

            let start = i + 1;
            let mut found = false;
            for j in start..bytes.len() {
                if bytes[j] == b'`' {
                    spans.push(Span::styled(
                        &text[start..j],
                        base_style.fg(Color::LightCyan),
                    ));
                    i = j;
                    last_pos = i + 1;
                    found = true;
                    break;
                }
            }
            if !found {
                // If no closing backtick found, just treat it as normal text
            }
        }
        i += 1;
    }

    if last_pos < text.len() {
        spans.push(Span::styled(&text[last_pos..], base_style));
    }

    if spans.is_empty() {
        Line::from(Span::styled(text, base_style))
    } else {
        Line::from(spans)
    }
}
