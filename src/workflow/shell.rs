use crate::workflow::context::WorkflowContext;

pub struct ShellStep {
    pub name: &'static str,
    /// Closure that builds the shell command string from the current context.
    pub command_fn: Box<dyn Fn(&WorkflowContext) -> String + Send + Sync>,
    /// Context key where stdout is stored.
    pub store_stdout_as: std::borrow::Cow<'static, str>,
    /// Optional context key where stderr is stored.
    pub store_stderr_as: Option<std::borrow::Cow<'static, str>>,
    /// Optional context key where the exit code (i64) is stored.
    pub store_exit_code_as: Option<std::borrow::Cow<'static, str>>,
    /// Optional closure that returns the working directory path.
    pub working_dir_fn: Option<Box<dyn Fn(&WorkflowContext) -> String + Send + Sync>>,
    /// Maximum seconds to wait before aborting the command (default: 60).
    pub timeout_secs: u64,
    /// When `Some`, skip this step if the closure returns `true`.
    pub skip_if: Option<Box<dyn Fn(&WorkflowContext) -> bool + Send + Sync>>,
}

// ── Builder ───────────────────────────────────────────────────────────────────

pub struct ShellStepBuilder {
    name: &'static str,
    command_fn: Option<Box<dyn Fn(&WorkflowContext) -> String + Send + Sync>>,
    store_stdout_as: Option<std::borrow::Cow<'static, str>>,
    store_stderr_as: Option<std::borrow::Cow<'static, str>>,
    store_exit_code_as: Option<std::borrow::Cow<'static, str>>,
    working_dir_fn: Option<Box<dyn Fn(&WorkflowContext) -> String + Send + Sync>>,
    timeout_secs: u64,
    skip_if: Option<Box<dyn Fn(&WorkflowContext) -> bool + Send + Sync>>,
}

impl ShellStepBuilder {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            command_fn: None,
            store_stdout_as: None,
            store_stderr_as: None,
            store_exit_code_as: None,
            working_dir_fn: None,
            timeout_secs: 60,
            skip_if: None,
        }
    }

    /// Set the shell command. The closure receives the current context and
    /// returns the command string, which is passed to `sh -c`.
    pub fn command<F>(mut self, f: F) -> Self
    where
        F: Fn(&WorkflowContext) -> String + Send + Sync + 'static,
    {
        self.command_fn = Some(Box::new(f));
        self
    }

    /// Context key where captured stdout is stored as a string.
    pub fn store_stdout_as(mut self, key: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        self.store_stdout_as = Some(key.into());
        self
    }

    /// Context key where captured stderr is stored as a string (optional).
    pub fn store_stderr_as(mut self, key: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        self.store_stderr_as = Some(key.into());
        self
    }

    /// Context key where the exit code is stored as an i64 (optional).
    pub fn store_exit_code_as(mut self, key: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        self.store_exit_code_as = Some(key.into());
        self
    }

    /// Set the working directory for the command.
    pub fn working_dir<F>(mut self, f: F) -> Self
    where
        F: Fn(&WorkflowContext) -> String + Send + Sync + 'static,
    {
        self.working_dir_fn = Some(Box::new(f));
        self
    }

    /// Abort the command if it runs longer than `secs` seconds (default: 60).
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
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

    pub fn finish(self) -> ShellStep {
        ShellStep {
            name: self.name,
            command_fn: self.command_fn.unwrap_or_else(|| {
                panic!(
                    "ShellStep \"{}\": command() must be called before finish()",
                    self.name
                )
            }),
            store_stdout_as: self.store_stdout_as.unwrap_or_else(|| {
                panic!(
                    "ShellStep \"{}\": store_stdout_as() must be called before finish()",
                    self.name
                )
            }),
            store_stderr_as: self.store_stderr_as,
            store_exit_code_as: self.store_exit_code_as,
            working_dir_fn: self.working_dir_fn,
            timeout_secs: self.timeout_secs,
            skip_if: self.skip_if,
        }
    }
}
