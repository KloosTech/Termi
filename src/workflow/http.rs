use crate::workflow::context::WorkflowContext;

/// URL-encodes a string for safe use in query parameters.
///
/// Use this inside your `.url()` closure whenever a context value
/// could contain spaces or special characters:
/// ```ignore
/// .url(|ctx| format!(
///     "https://example.com/search?q={}",
///     url_encode(ctx.get_str("query"))
/// ))
/// ```
pub fn url_encode(s: &str) -> String {
    urlencoding::encode(s).into_owned()
}

/// Controls whether the page is rendered through a headless browser before
/// the HTML is extracted.
#[derive(Debug, Clone, PartialEq)]
pub enum JsRendering {
    /// Plain HTTP fetch via reqwest — fast, no external dependencies.
    None,
    /// Launch a headless Chromium instance via Playwright and wait for the
    /// page to settle before capturing the DOM.
    ///
    /// Requires the `js-render` Cargo feature and Node.js 18+ on `PATH`.
    /// Install browsers once with: `npx playwright@1.59.1 install chromium`
    Headless,
}

pub struct HttpStep {
    pub name: &'static str,
    /// Closure that returns the URL to fetch.
    pub url_fn: Box<dyn Fn(&WorkflowContext) -> String + Send + Sync>,
    /// Context key where the response body (or converted markdown) is stored.
    pub store_as: &'static str,
    /// Optional context key where the HTTP status code (i64) is stored.
    /// When set, non-2xx responses are stored rather than treated as errors.
    pub store_status_as: Option<&'static str>,
    /// When `true`, convert the HTML response body to Markdown via `htmd`.
    pub strip_html: bool,
    /// JS rendering strategy (default: `None`).
    pub js_rendering: JsRendering,
    /// Request timeout in seconds (default: 30).
    pub timeout_secs: u64,
    /// Extra request headers sent with every fetch.
    pub headers: Vec<(String, String)>,
    /// When `Some`, skip this step if the closure returns `true`.
    pub skip_if: Option<Box<dyn Fn(&WorkflowContext) -> bool + Send + Sync>>,
}

// ── Builder ───────────────────────────────────────────────────────────────────

pub struct HttpStepBuilder {
    name: &'static str,
    url_fn: Option<Box<dyn Fn(&WorkflowContext) -> String + Send + Sync>>,
    store_as: Option<&'static str>,
    store_status_as: Option<&'static str>,
    strip_html: bool,
    js_rendering: JsRendering,
    timeout_secs: u64,
    headers: Vec<(String, String)>,
    skip_if: Option<Box<dyn Fn(&WorkflowContext) -> bool + Send + Sync>>,
}

impl HttpStepBuilder {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            url_fn: None,
            store_as: None,
            store_status_as: None,
            strip_html: false,
            js_rendering: JsRendering::None,
            timeout_secs: 30,
            headers: Vec::new(),
            skip_if: None,
        }
    }

    /// Closure that builds the URL from the current context.
    pub fn url<F>(mut self, f: F) -> Self
    where
        F: Fn(&WorkflowContext) -> String + Send + Sync + 'static,
    {
        self.url_fn = Some(Box::new(f));
        self
    }

    /// Context key where the response body (or converted markdown) is stored.
    pub fn store_as(mut self, key: &'static str) -> Self {
        self.store_as = Some(key);
        self
    }

    /// Context key where the HTTP status code is stored as i64.
    /// When set, non-2xx responses are NOT treated as errors — the caller
    /// can inspect the code and decide.
    pub fn store_status_as(mut self, key: &'static str) -> Self {
        self.store_status_as = Some(key);
        self
    }

    /// Convert the HTML response to Markdown before storing it.
    /// Makes LLM prompts significantly shorter and cleaner.
    pub fn strip_html(mut self) -> Self {
        self.strip_html = true;
        self
    }

    /// Fetch the page through a headless Chromium instance so JavaScript
    /// executes before the HTML is captured.
    ///
    /// Requires the `js-render` feature: `cargo build --features js-render`
    pub fn render_js(mut self) -> Self {
        self.js_rendering = JsRendering::Headless;
        self
    }

    /// Override the request timeout (default: 30 s).
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Append a request header.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Skip this step when the closure returns `true`.
    pub fn skip_if<F>(mut self, f: F) -> Self
    where
        F: Fn(&WorkflowContext) -> bool + Send + Sync + 'static,
    {
        self.skip_if = Some(Box::new(f));
        self
    }

    pub fn finish(self) -> HttpStep {
        HttpStep {
            name: self.name,
            url_fn: self.url_fn.unwrap_or_else(|| {
                panic!("HttpStep \"{}\": url() must be called before finish()", self.name)
            }),
            store_as: self.store_as.unwrap_or_else(|| {
                panic!("HttpStep \"{}\": store_as() must be called before finish()", self.name)
            }),
            store_status_as: self.store_status_as,
            strip_html: self.strip_html,
            js_rendering: self.js_rendering,
            timeout_secs: self.timeout_secs,
            headers: self.headers,
            skip_if: self.skip_if,
        }
    }
}
