use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
        MouseButton, MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures_util::StreamExt;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use tokio::sync::mpsc;

use crate::ollama::client::OllamaClient;
use crate::ollama::types::{ChatRequest, Message};
use crate::workflow::events::StepEvent;

// ── Types ─────────────────────────────────────────────────────────────────────

mod markdown;

enum AnswerChunk {
    Token(String),
    Done,
    Error(String),
}

#[derive(PartialEq)]
enum Phase {
    Working,
    Reading,
    Answering,
    AnswerReady,
    Selecting,
}

// ── App state ─────────────────────────────────────────────────────────────────

struct CompletedStep {
    name: &'static str,
    tokens: u32,
    elapsed_ms: u128,
}

struct App {
    model: String,
    command: String,
    client: Arc<dyn OllamaClient>,

    // ── Working phase ──────────────────────────────────────────────────────
    current_step: Option<&'static str>,
    current_model: Option<String>,
    stream_text: String,
    live_tokens: u32,
    step_start: Option<Instant>,
    completed: Vec<CompletedStep>,
    status_message: String,
    workflow_start: Instant,

    // ── Reading / Q&A phase ────────────────────────────────────────────────
    phase: Phase,
    summary_text: String,
    summary_scroll: u16,
    input: String,

    // ── Answer streaming ───────────────────────────────────────────────────
    answer_text: String,
    answer_auto_scroll: bool,
    answer_scroll: u16,
    answer_error: Option<String>,
    answer_rx: Option<mpsc::Receiver<AnswerChunk>>,
    conversation: Vec<(String, String)>,
    pending_question: String,

    // ── Error state ────────────────────────────────────────────────────────
    error_message: Option<String>,

    // ── Debug panel ────────────────────────────────────────────────────────
    debug: bool,
    ctx_entries: Vec<(String, serde_json::Value)>,
    /// Scroll offset for the list view.
    ctx_scroll: u16,
    /// Currently highlighted entry index (list navigation).
    ctx_selected: Option<usize>,
    /// Whether the detail view is open for the selected entry.
    ctx_detail: bool,
    /// Scroll offset inside the detail view.
    ctx_detail_scroll: u16,
    /// Cached terminal size — updated each frame for mouse hit-testing.
    term_cols: u16,
    term_rows: u16,

    // ── Selection phase ────────────────────────────────────────────────────
    select_prompt: String,
    select_options: Vec<String>,
    select_selected: usize,
    select_reply: Option<tokio::sync::oneshot::Sender<Option<usize>>>,
}

impl App {
    fn new(model: String, command: String, client: Arc<dyn OllamaClient>, debug: bool) -> Self {
        Self {
            model,
            command,
            client,
            current_step: None,
            current_model: None,
            stream_text: String::new(),
            live_tokens: 0,
            step_start: None,
            completed: Vec::new(),
            status_message: String::new(),
            workflow_start: Instant::now(),
            phase: Phase::Working,
            summary_text: String::new(),
            summary_scroll: 0,
            input: String::new(),
            answer_text: String::new(),
            answer_auto_scroll: true,
            answer_scroll: 0,
            answer_error: None,
            answer_rx: None,
            conversation: Vec::new(),
            pending_question: String::new(),
            error_message: None,
            debug,
            ctx_entries: Vec::new(),
            ctx_scroll: 0,
            ctx_selected: None,
            ctx_detail: false,
            ctx_detail_scroll: 0,
            term_cols: 80,
            term_rows: 24,
            select_prompt: String::new(),
            select_options: Vec::new(),
            select_selected: 0,
            select_reply: None,
        }
    }

    // ── Workflow event handler ─────────────────────────────────────────────

    fn handle_workflow_event(&mut self, event: StepEvent) {
        match event {
            StepEvent::StepStarted { name, model } => {
                self.current_step = Some(name);
                self.current_model = Some(model);
                self.stream_text.clear();
                self.live_tokens = 0;
                self.step_start = Some(Instant::now());
                self.status_message.clear();
            }
            StepEvent::Token { text, .. } => {
                self.stream_text.push_str(&text);
                self.live_tokens += 1;
            }
            StepEvent::StepCompleted {
                name,
                total_tokens,
                elapsed_ms,
            } => {
                self.completed.push(CompletedStep {
                    name,
                    tokens: total_tokens,
                    elapsed_ms,
                });
                self.current_step = None;
                self.current_model = None;
                self.live_tokens = 0;
                self.step_start = None;
            }
            StepEvent::StepSkipped { name } => {
                self.completed.push(CompletedStep {
                    name,
                    tokens: 0,
                    elapsed_ms: 0,
                });
                self.current_step = None;
            }
            StepEvent::StatusUpdate { message } => {
                self.status_message = message;
                self.current_step = None;
            }
            StepEvent::ContextSnapshot { entries } => {
                self.ctx_entries = entries;
                // Keep the selection valid; don't reset scroll so the user
                // can stay focused on the entry they were reading.
                if let Some(i) = self.ctx_selected {
                    if i >= self.ctx_entries.len() {
                        self.ctx_selected = None;
                        self.ctx_detail = false;
                    }
                }
            }
            StepEvent::WorkflowComplete => {
                self.summary_text = std::mem::take(&mut self.stream_text);
                self.phase = Phase::Reading;
            }
            StepEvent::WorkflowFailed { message } => {
                self.error_message = Some(message);
                self.phase = Phase::Reading;
            }
            StepEvent::SelectRequest {
                prompt,
                options,
                reply,
            } => {
                self.select_prompt = prompt;
                self.select_options = options;
                self.select_selected = 0;
                self.select_reply = Some(reply);
                self.phase = Phase::Selecting;
            }
        }
    }

    // ── Answer chunk handler ───────────────────────────────────────────────

    fn handle_answer_chunk(&mut self, chunk: AnswerChunk) {
        match chunk {
            AnswerChunk::Token(text) => {
                self.answer_text.push_str(&text);
            }
            AnswerChunk::Done => {
                self.conversation
                    .push((self.pending_question.clone(), self.answer_text.clone()));
                self.answer_rx = None;
                self.answer_auto_scroll = false;
                self.answer_scroll = 0;
                self.phase = Phase::AnswerReady;
            }
            AnswerChunk::Error(e) => {
                self.answer_error = Some(e);
                self.answer_rx = None;
                self.phase = Phase::AnswerReady;
            }
        }
    }

    // ── Keyboard handler ───────────────────────────────────────────────────

    /// Returns `true` if the loop should exit.
    fn handle_key(&mut self, code: KeyCode, mods: KeyModifiers) -> bool {
        // ── Context panel controls (all use Ctrl modifier) ─────────────────
        if self.debug && mods.contains(KeyModifiers::CONTROL) {
            match code {
                KeyCode::Up => {
                    if self.ctx_detail {
                        self.ctx_detail_scroll = self.ctx_detail_scroll.saturating_sub(1);
                    } else {
                        self.ctx_selected = Some(match self.ctx_selected {
                            None | Some(0) => 0,
                            Some(i) => i - 1,
                        });
                        self.ensure_selection_visible();
                    }
                    return false;
                }
                KeyCode::Down => {
                    if self.ctx_detail {
                        self.ctx_detail_scroll += 1;
                    } else {
                        let max = self.ctx_entries.len().saturating_sub(1);
                        self.ctx_selected = Some(self.ctx_selected.map_or(0, |i| (i + 1).min(max)));
                        self.ensure_selection_visible();
                    }
                    return false;
                }
                // Ctrl+Enter: open detail view for selected entry, or close it.
                KeyCode::Enter => {
                    if self.ctx_detail {
                        self.ctx_detail = false;
                    } else if self.ctx_selected.is_some() {
                        self.ctx_detail = true;
                        self.ctx_detail_scroll = 0;
                    } else if !self.ctx_entries.is_empty() {
                        self.ctx_selected = Some(0);
                        self.ensure_selection_visible();
                    }
                    return false;
                }
                // Ctrl+R: reset context panel to default state.
                KeyCode::Char('r') => {
                    self.ctx_reset();
                    return false;
                }
                _ => {}
            }
        }

        // ── Phase-specific handlers ────────────────────────────────────────
        match self.phase {
            Phase::Working => matches!(code, KeyCode::Char('q') | KeyCode::Esc),
            Phase::Reading => self.handle_key_reading(code),
            Phase::Answering => matches!(code, KeyCode::Char('q') | KeyCode::Esc),
            Phase::AnswerReady => self.handle_key_answer_ready(code),
            Phase::Selecting => self.handle_key_selecting(code),
        }
    }

    fn handle_key_reading(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::Up | KeyCode::Char('k') if self.input.is_empty() => {
                self.summary_scroll = self.summary_scroll.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') if self.input.is_empty() => {
                self.summary_scroll += 1;
            }
            KeyCode::PageUp => self.summary_scroll = self.summary_scroll.saturating_sub(20),
            KeyCode::PageDown => self.summary_scroll += 20,
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Esc => {
                if self.input.is_empty() {
                    return true;
                }
                self.input.clear();
            }
            KeyCode::Char('q') if self.input.is_empty() => return true,
            KeyCode::Char(c) => self.input.push(c),
            KeyCode::Enter if !self.input.is_empty() => self.submit_question(),
            _ => {}
        }
        false
    }

    fn handle_key_answer_ready(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::Up | KeyCode::Char('k') if self.input.is_empty() => {
                self.answer_scroll = self.answer_scroll.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') if self.input.is_empty() => {
                self.answer_scroll += 1;
            }
            KeyCode::PageUp => self.answer_scroll = self.answer_scroll.saturating_sub(20),
            KeyCode::PageDown => self.answer_scroll += 20,
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Esc => {
                if self.input.is_empty() {
                    return true;
                }
                self.input.clear();
            }
            KeyCode::Char('q') if self.input.is_empty() => return true,
            KeyCode::Char(c) => self.input.push(c),
            KeyCode::Enter if !self.input.is_empty() => self.submit_question(),
            _ => {}
        }
        false
    }

    // ── Mouse handler ──────────────────────────────────────────────────────

    fn handle_key_selecting(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::Up => {
                if self.select_selected > 0 {
                    self.select_selected -= 1;
                }
            }
            KeyCode::Down => {
                if !self.select_options.is_empty()
                    && self.select_selected + 1 < self.select_options.len()
                {
                    self.select_selected += 1;
                }
            }
            KeyCode::Enter => {
                if let Some(reply) = self.select_reply.take() {
                    let _ = reply.send(Some(self.select_selected));
                }
                self.phase = Phase::Working;
            }
            KeyCode::Esc | KeyCode::Char('q') => {
                if let Some(reply) = self.select_reply.take() {
                    let _ = reply.send(None);
                }
                self.phase = Phase::Working;
            }
            _ => {}
        }
        false
    }

    fn handle_mouse_click(&mut self, col: u16, row: u16) {
        if !self.debug {
            return;
        }

        // Context panel occupies the right 35% of the screen.
        let ctx_start_col = self.term_cols * 65 / 100;
        if col < ctx_start_col {
            return;
        } // click is in the main workflow panel

        if self.ctx_detail {
            // Any click in the detail view collapses it back to the list.
            self.ctx_detail = false;
            return;
        }

        // Row 0 is the top border — treat as a reset click.
        if row == 0 {
            self.ctx_reset();
            return;
        }

        // Rows 1..N are entries (offset by the border and list scroll).
        let entry_idx = (row as usize).saturating_sub(1) + self.ctx_scroll as usize;
        if entry_idx < self.ctx_entries.len() {
            self.ctx_selected = Some(entry_idx);
            self.ctx_detail = true;
            self.ctx_detail_scroll = 0;
        }
    }

    // ── Context panel helpers ──────────────────────────────────────────────

    fn ctx_reset(&mut self) {
        self.ctx_selected = None;
        self.ctx_detail = false;
        self.ctx_scroll = 0;
        self.ctx_detail_scroll = 0;
    }

    /// Adjusts `ctx_scroll` so the selected entry stays within the visible rows.
    fn ensure_selection_visible(&mut self) {
        let Some(i) = self.ctx_selected else { return };
        // Approximate visible rows: full terminal height minus borders (2).
        let visible = self.term_rows.saturating_sub(2) as usize;
        let scroll = self.ctx_scroll as usize;
        if i < scroll {
            self.ctx_scroll = i as u16;
        } else if i >= scroll + visible {
            self.ctx_scroll = (i + 1).saturating_sub(visible) as u16;
        }
    }

    // ── Q&A ───────────────────────────────────────────────────────────────

    fn submit_question(&mut self) {
        let question = std::mem::take(&mut self.input);
        self.pending_question = question.clone();
        self.answer_text.clear();
        self.answer_error = None;
        self.answer_auto_scroll = true;
        self.answer_scroll = 0;
        self.phase = Phase::Answering;

        let (tx, rx) = mpsc::channel::<AnswerChunk>(256);
        self.answer_rx = Some(rx);

        let client = Arc::clone(&self.client);
        let model = self.model.clone();
        let summary = self.summary_text.clone();
        let ctx_entries = self.ctx_entries.clone();
        let conversation = self.conversation.clone();

        tokio::spawn(async move {
            fire_question(
                client,
                model,
                summary,
                ctx_entries,
                conversation,
                question,
                tx,
            )
            .await;
        });
    }
}

// ── LLM answer task ───────────────────────────────────────────────────────────

fn build_system_prompt(summary: &str, ctx_entries: &[(String, serde_json::Value)]) -> String {
    const MAX_PER_ENTRY: usize = 4000;

    let mut sections = String::new();
    for (key, val) in ctx_entries {
        let text = match val {
            serde_json::Value::String(s) => s.clone(),
            _ => serde_json::to_string_pretty(val).unwrap_or_default(),
        };
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }
        let truncated: String = trimmed.chars().take(MAX_PER_ENTRY).collect();
        sections.push_str(&format!("\n\n### {key}\n{truncated}"));
        if trimmed.chars().count() > MAX_PER_ENTRY {
            sections.push_str("\n[…truncated]");
        }
    }

    if sections.is_empty() {
        format!("You are a helpful assistant. Pipeline output:\n\n{summary}")
    } else {
        format!(
            "You are a helpful assistant with full access to all outputs produced by a completed pipeline.\n\
             Each section below is a named result stored in the pipeline context:{sections}"
        )
    }
}

async fn fire_question(
    client: Arc<dyn OllamaClient>,
    model: String,
    summary: String,
    ctx_entries: Vec<(String, serde_json::Value)>,
    conversation: Vec<(String, String)>,
    question: String,
    tx: mpsc::Sender<AnswerChunk>,
) {
    let system = build_system_prompt(&summary, &ctx_entries);
    let mut messages = vec![Message::system(system)];
    for (q, a) in &conversation {
        messages.push(Message::user(q.clone()));
        messages.push(Message::assistant(a.clone()));
    }
    messages.push(Message::user(question));

    let req = ChatRequest {
        model,
        messages,
        stream: Some(true),
        ..Default::default()
    };

    match client.chat_stream(req).await {
        Err(e) => {
            let _ = tx.send(AnswerChunk::Error(e.to_string())).await;
        }
        Ok(mut stream) => {
            while let Some(result) = stream.next().await {
                match result {
                    Err(e) => {
                        let _ = tx.send(AnswerChunk::Error(e.to_string())).await;
                        return;
                    }
                    Ok(chunk) => {
                        if !chunk.message.content.is_empty() {
                            let _ = tx.send(AnswerChunk::Token(chunk.message.content)).await;
                        }
                        if chunk.done {
                            break;
                        }
                    }
                }
            }
            let _ = tx.send(AnswerChunk::Done).await;
        }
    }
}

// ── Top-level render ──────────────────────────────────────────────────────────

fn render(f: &mut Frame, app: &App) {
    let area = f.area();

    if app.debug {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
            .split(area);

        render_phase(f, chunks[0], app);
        render_ctx_panel(f, chunks[1], app);
    } else {
        render_phase(f, area, app);
    }
}

fn render_phase(f: &mut Frame, area: Rect, app: &App) {
    match app.phase {
        Phase::Working => render_working(f, area, app),
        Phase::Reading => render_reading(f, area, app),
        Phase::Answering | Phase::AnswerReady => render_qa(f, area, app),
        Phase::Selecting => render_selecting(f, area, app),
    }
}

// ── Debug context panel ───────────────────────────────────────────────────────

fn render_ctx_panel(f: &mut Frame, area: Rect, app: &App) {
    if app.ctx_detail {
        render_ctx_detail(f, area, app);
    } else {
        render_ctx_list(f, area, app);
    }
}

fn render_ctx_list(f: &mut Frame, area: Rect, app: &App) {
    let inner_w = area.width.saturating_sub(4) as usize;
    let key_w = 18usize.min(inner_w / 3);
    let val_w = inner_w.saturating_sub(key_w + 2); // +2 for selector prefix

    let entry_count = app.ctx_entries.len();
    let hint = "Ctrl+↑↓ nav · Enter expand · R reset";
    let title = format!(" Context  {entry_count} keys ");

    let lines: Vec<Line> = if app.ctx_entries.is_empty() {
        vec![Line::from(Span::styled(
            " Waiting for first step…",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        app.ctx_entries
            .iter()
            .enumerate()
            .map(|(i, (key, val))| {
                let selected = app.ctx_selected == Some(i);
                let selector = if selected { "▶ " } else { "  " };

                let key_str = truncate_str(key, key_w);
                let val_str = format_ctx_value(val, val_w);

                let bg = if selected {
                    Color::DarkGray
                } else {
                    Color::Reset
                };

                Line::from(vec![
                    Span::styled(selector, Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("{key_str:<key_w$}"),
                        Style::default()
                            .fg(Color::Cyan)
                            .bg(bg)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" "),
                    Span::styled(val_str, Style::default().bg(bg)),
                ])
            })
            .collect()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .title_bottom(
            Line::from(hint)
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center),
        )
        .border_style(Style::default().fg(Color::Yellow));

    let para = Paragraph::new(lines)
        .block(block)
        .scroll((app.ctx_scroll, 0));

    f.render_widget(para, area);
}

fn render_ctx_detail(f: &mut Frame, area: Rect, app: &App) {
    let Some(i) = app.ctx_selected else { return };
    let Some((key, val)) = app.ctx_entries.get(i) else {
        return;
    };

    let full = format_full_value(val);
    let title = format!(" {key} ({}/{})", i + 1, app.ctx_entries.len());
    let hint = "Ctrl+↑↓ scroll · Ctrl+Enter back · Click to close";

    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .title_bottom(
            Line::from(hint)
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center),
        )
        .border_style(Style::default().fg(Color::Green));

    let para = Paragraph::new(full.as_str())
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((app.ctx_detail_scroll, 0));

    f.render_widget(para, area);
}

// ── Value formatters ──────────────────────────────────────────────────────────

/// Full pretty-printed value shown in the detail panel.
fn format_full_value(v: &serde_json::Value) -> String {
    match v {
        // Strings are shown without JSON escaping so they're readable.
        serde_json::Value::String(s) => s.clone(),
        _ => serde_json::to_string_pretty(v).unwrap_or_else(|_| format!("{v:?}")),
    }
}

/// Short one-line summary for the list view.
fn format_ctx_value(v: &serde_json::Value, max_width: usize) -> String {
    let s = match v {
        serde_json::Value::String(s) => {
            let flat = s.replace(['\n', '\r', '\t'], " ");
            format!("\"{flat}\"")
        }
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Array(a) => format!("[{} items]", a.len()),
        serde_json::Value::Object(o) => format!("{{{} keys}}", o.len()),
    };
    truncate_str(&s, max_width)
}

fn truncate_str(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    let count = s.chars().count();
    if count > max {
        let t: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{t}…")
    } else {
        s.to_string()
    }
}

// ── Working phase ─────────────────────────────────────────────────────────────

fn render_working(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(3)])
        .split(area);
    render_working_main(f, chunks[0], app);
    render_status_bar(f, chunks[1], app);
}

fn render_working_main(f: &mut Frame, area: Rect, app: &App) {
    if app.completed.is_empty() {
        render_stream_block(f, area, app);
        return;
    }
    let summary_h = (app.completed.len() as u16 + 2).min(area.height / 3);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(summary_h), Constraint::Min(3)])
        .split(area);
    render_completed_block(f, chunks[0], app);
    render_stream_block(f, chunks[1], app);
}

fn render_completed_block(f: &mut Frame, area: Rect, app: &App) {
    let lines: Vec<Line> = app
        .completed
        .iter()
        .map(|s| {
            let tps = if s.elapsed_ms > 0 {
                s.tokens as f64 / (s.elapsed_ms as f64 / 1000.0)
            } else {
                0.0
            };
            let stats = if s.tokens > 0 {
                format!(
                    "  {:>5} tok  {:>6.1}s  {:>5.1} tok/s",
                    s.tokens,
                    s.elapsed_ms as f64 / 1000.0,
                    tps
                )
            } else {
                format!("  {:>6.1}s", s.elapsed_ms as f64 / 1000.0)
            };
            Line::from(vec![
                Span::styled(
                    "✓ ",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("{:<20}", s.name),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::styled(stats, Style::default().fg(Color::DarkGray)),
            ])
        })
        .collect();
    let para =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Completed "));
    f.render_widget(para, area);
}

fn render_stream_block(f: &mut Frame, area: Rect, app: &App) {
    let title = match app.current_step {
        Some(name) => format!(
            " {} — {} ",
            name,
            app.current_model.as_deref().unwrap_or(&app.model)
        ),
        None => {
            if !app.status_message.is_empty() {
                format!(" {} ", app.status_message)
            } else {
                format!(" {} — {} ", app.command, app.model)
            }
        }
    };
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    let scroll_y = compute_scroll_offset(&app.stream_text, inner.width, inner.height);
    let para = Paragraph::new(app.stream_text.as_str())
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((scroll_y, 0));
    f.render_widget(para, area);
}

fn render_status_bar(f: &mut Frame, area: Rect, app: &App) {
    let elapsed = app.workflow_start.elapsed().as_secs_f64();
    let total_tokens: u32 = app.completed.iter().map(|s| s.tokens).sum::<u32>() + app.live_tokens;
    let speed = if let Some(start) = app.step_start {
        let secs = start.elapsed().as_secs_f64();
        if secs > 0.1 {
            app.live_tokens as f64 / secs
        } else {
            0.0
        }
    } else {
        0.0
    };
    let debug_hint = if app.debug {
        "  │  Ctrl+↑↓ nav  Ctrl+Enter expand  Ctrl+R reset"
    } else {
        ""
    };
    let text = format!(
        " Elapsed: {:.1}s  |  Tokens: {}  |  Speed: {:.1} tok/s{debug_hint} ",
        elapsed, total_tokens, speed
    );
    let para = Paragraph::new(text).block(Block::default().borders(Borders::ALL));
    f.render_widget(para, area);
}

// ── Reading phase ─────────────────────────────────────────────────────────────

fn render_reading(f: &mut Frame, area: Rect, app: &App) {
    if let Some(ref err) = app.error_message {
        // Calculate how many lines the error needs so the panel is tall enough.
        let term_w = area.width.saturating_sub(4) as usize; // inside borders + padding
        let err_line_count = if term_w == 0 {
            1
        } else {
            err.chars()
                .collect::<Vec<_>>()
                .chunks(term_w)
                .count()
                .max(1) as u16
        };
        let error_h = (err_line_count + 2).min(area.height / 2); // +2 for borders

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(error_h),
                Constraint::Min(3),
                Constraint::Length(3),
            ])
            .split(area);

        let error_para = Paragraph::new(err.as_str())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" ✗ Pipeline Error ")
                    .border_style(Style::default().fg(Color::Red)),
            )
            .style(Style::default().fg(Color::Red))
            .wrap(Wrap { trim: false });
        f.render_widget(error_para, chunks[0]);

        render_scrollable_text(
            f,
            chunks[1],
            " Output  (↑↓ scroll) ",
            &app.summary_text,
            app.summary_scroll,
        );
        render_input_box(f, chunks[2], app);
    } else {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(3)])
            .split(area);

        render_scrollable_text(
            f,
            chunks[0],
            " Summary  (↑↓ scroll) ",
            &app.summary_text,
            app.summary_scroll,
        );
        render_input_box(f, chunks[1], app);
    }
}

// ── Q&A phase ─────────────────────────────────────────────────────────────────

fn render_qa(f: &mut Frame, area: Rect, app: &App) {
    let summary_h = (area.height / 3).max(6);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(summary_h),
            Constraint::Min(3),
            Constraint::Length(3),
        ])
        .split(area);
    render_scrollable_text(
        f,
        chunks[0],
        " Summary  (↑↓ scroll, PgUp/PgDn) ",
        &app.summary_text,
        app.summary_scroll,
    );
    render_answer_block(f, chunks[1], app);
    render_input_box(f, chunks[2], app);
}

fn render_answer_block(f: &mut Frame, area: Rect, app: &App) {
    let title = match app.phase {
        Phase::Answering => " Answer  (generating…) ",
        Phase::AnswerReady if app.answer_error.is_some() => " Answer  (error) ",
        _ => " Answer  (↑↓ scroll) ",
    };
    let text = if let Some(ref err) = app.answer_error {
        format!("Error: {err}")
    } else {
        app.answer_text.clone()
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(if app.phase == Phase::Answering {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::Green)
        });
    let inner = block.inner(area);
    let scroll_y = if app.answer_auto_scroll {
        compute_scroll_offset(&text, inner.width, inner.height)
    } else {
        app.answer_scroll
    };
    let para = Paragraph::new(text.as_str())
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((scroll_y, 0));
    f.render_widget(para, area);
}

// ── Shared widgets ────────────────────────────────────────────────────────────

fn render_selecting(f: &mut Frame, area: Rect, app: &App) {
    use ratatui::widgets::{List, ListItem};

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(3)])
        .split(area);

    let items: Vec<ListItem> = app
        .select_options
        .iter()
        .enumerate()
        .map(|(i, opt)| {
            let style = if i == app.select_selected {
                Style::default()
                    .bg(Color::Blue)
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(opt.as_str()).style(style)
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" {} ", app.select_prompt)),
    );

    f.render_widget(list, chunks[0]);

    let help = Paragraph::new(" ↑/↓: Navigate • Enter: Select • q/Esc: Cancel ")
        .block(Block::default().borders(Borders::ALL).title(" Controls "))
        .style(Style::default().fg(Color::DarkGray));
    f.render_widget(help, chunks[1]);
}

fn render_scrollable_text(f: &mut Frame, area: Rect, title: &str, text: &str, scroll: u16) {
    let markdown_text = markdown::parse(text);
    let para = Paragraph::new(markdown_text)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    f.render_widget(para, area);
}

fn render_input_box(f: &mut Frame, area: Rect, app: &App) {
    let (content, border_style) = match app.phase {
        Phase::Answering => (
            Span::styled(" Generating answer…", Style::default().fg(Color::DarkGray)),
            Style::default().fg(Color::DarkGray),
        ),
        _ => {
            let prompt = if app.input.is_empty() {
                Span::styled(
                    " Type a question and press Enter  ·  ↑↓ to scroll  ·  q to quit",
                    Style::default().fg(Color::DarkGray),
                )
            } else {
                Span::raw(format!(" > {}▋", app.input))
            };
            (prompt, Style::default().fg(Color::Cyan))
        }
    };
    let para = Paragraph::new(Line::from(vec![content])).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .title(" Ask "),
    );
    f.render_widget(para, area);
}

// ── Scroll helper ─────────────────────────────────────────────────────────────

fn compute_scroll_offset(text: &str, width: u16, height: u16) -> u16 {
    if width == 0 || height == 0 {
        return 0;
    }
    let w = width as usize;
    let h = height as usize;
    let wrapped: usize = text
        .lines()
        .map(|line| {
            if line.is_empty() {
                1
            } else {
                (line.len() + w - 1) / w
            }
        })
        .sum();
    (wrapped.max(1).saturating_sub(h)) as u16
}

// ── Public entry point ────────────────────────────────────────────────────────

pub async fn run(
    mut workflow_rx: mpsc::Receiver<StepEvent>,
    model: String,
    command: String,
    client: Arc<dyn OllamaClient>,
    debug: bool,
) -> anyhow::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(model, command, client, debug);
    let result = run_loop(&mut terminal, &mut app, &mut workflow_rx).await;

    // Always restore the terminal, even on error.
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

async fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    workflow_rx: &mut mpsc::Receiver<StepEvent>,
) -> anyhow::Result<()> {
    loop {
        // Cache terminal size for mouse hit-testing.
        if let Ok((cols, rows)) = crossterm::terminal::size() {
            app.term_cols = cols;
            app.term_rows = rows;
        }

        terminal.draw(|f| render(f, app))?;

        // Non-blocking event poll (keyboard + mouse).
        if event::poll(Duration::from_millis(1))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if app.handle_key(key.code, key.modifiers) {
                        break;
                    }
                }
                Event::Mouse(m) if m.kind == MouseEventKind::Down(MouseButton::Left) => {
                    app.handle_mouse_click(m.column, m.row);
                }
                _ => {}
            }
        }

        // Drain workflow events.
        loop {
            match workflow_rx.try_recv() {
                Ok(evt) => app.handle_workflow_event(evt),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    if app.phase == Phase::Working {
                        app.summary_text = std::mem::take(&mut app.stream_text);
                        app.phase = Phase::Reading;
                    }
                    break;
                }
            }
        }

        // Drain answer chunks.
        if app.answer_rx.is_some() {
            let mut chunks = Vec::new();
            if let Some(rx) = &mut app.answer_rx {
                loop {
                    match rx.try_recv() {
                        Ok(c) => chunks.push(c),
                        Err(_) => break,
                    }
                }
            }
            for chunk in chunks {
                app.handle_answer_chunk(chunk);
            }
        }

        tokio::time::sleep(Duration::from_millis(16)).await;
    }

    Ok(())
}
