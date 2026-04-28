use std::collections::HashMap;

use serde::Serialize;
use serde_json::Value;

/// Shared state passed through every step of a workflow.
///
/// Steps read inputs and write outputs here. It is the only communication
/// channel between steps — no global state, no side channels.
#[derive(Debug, Default, Clone)]
pub struct WorkflowContext {
    values: HashMap<String, Value>,
}

impl WorkflowContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: &str, value: impl Serialize) {
        self.values.insert(
            key.to_string(),
            serde_json::to_value(value).expect("WorkflowContext::set: value must be serializable"),
        );
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }

    /// Returns the string value at `key`, or `""` if absent or not a string.
    pub fn get_str(&self, key: &str) -> &str {
        self.values
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or("")
    }

    /// Returns the array at `key`, or an empty slice if absent or not an array.
    pub fn get_array(&self, key: &str) -> &[Value] {
        self.values
            .get(key)
            .and_then(|v| v.as_array())
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn contains(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get_str() {
        let mut ctx = WorkflowContext::new();
        ctx.set("greeting", "hello");
        assert_eq!(ctx.get_str("greeting"), "hello");
        assert_eq!(ctx.get_str("missing"), "");
    }

    #[test]
    fn test_set_and_get_array() {
        let mut ctx = WorkflowContext::new();
        ctx.set("files", vec!["a.rs", "b.rs"]);
        let arr = ctx.get_array("files");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0].as_str().unwrap(), "a.rs");
    }

    #[test]
    fn test_get_raw_value() {
        let mut ctx = WorkflowContext::new();
        ctx.set("count", 42u32);
        assert_eq!(ctx.get("count").unwrap().as_u64().unwrap(), 42);
    }
}
