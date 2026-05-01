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

    /// Builder-style initialisation: sets `key` to `value` and returns `self`.
    pub fn with(mut self, key: &str, value: impl Serialize) -> Self {
        self.set(key, value);
        self
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

    /// Returns the boolean value at `key`, or `false` if absent or not a boolean.
    pub fn get_bool(&self, key: &str) -> bool {
        self.values
            .get(key)
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    /// Returns the integer value at `key`, or `None` if absent or not an integer.
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.values.get(key).and_then(|v| v.as_i64())
    }

    /// Returns the float value at `key`, or `None` if absent or not a number.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.values.get(key).and_then(|v| v.as_f64())
    }

    /// Returns the JSON object at `key`, or `None` if absent or not an object.
    pub fn get_object(&self, key: &str) -> Option<&serde_json::Map<String, Value>> {
        self.values.get(key).and_then(|v| v.as_object())
    }

    pub fn contains(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    /// Returns an iterator over all keys in the context.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.values.keys().map(|k| k.as_str())
    }

    /// Removes and returns the value at `key`, or `None` if absent.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.values.remove(key)
    }

    /// Returns all key-value pairs sorted by key, suitable for the debug panel.
    pub fn snapshot(&self) -> Vec<(String, Value)> {
        let mut entries: Vec<_> = self
            .values
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    }

    /// Merges all entries from `other` into `self`, overwriting on key conflict.
    pub(crate) fn extend(&mut self, other: &WorkflowContext) {
        for (k, v) in &other.values {
            self.values.insert(k.clone(), v.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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

    #[test]
    fn test_with_builder_pattern() {
        let ctx = WorkflowContext::new()
            .with("name", "alice")
            .with("age", 30u32);
        assert_eq!(ctx.get_str("name"), "alice");
        assert_eq!(ctx.get_i64("age"), Some(30));
    }

    #[test]
    fn test_get_bool() {
        let mut ctx = WorkflowContext::new();
        ctx.set("flag", true);
        assert!(ctx.get_bool("flag"));
        assert!(!ctx.get_bool("missing"));
    }

    #[test]
    fn test_get_i64_and_f64() {
        let mut ctx = WorkflowContext::new();
        ctx.set("count", 42i64);
        ctx.set("ratio", 3.14f64);
        assert_eq!(ctx.get_i64("count"), Some(42));
        assert!(ctx.get_f64("ratio").unwrap() - 3.14 < 0.001);
        assert_eq!(ctx.get_i64("missing"), None);
        assert_eq!(ctx.get_f64("missing"), None);
    }

    #[test]
    fn test_get_object() {
        let mut ctx = WorkflowContext::new();
        ctx.set("user", json!({"name": "bob", "age": 25}));
        let obj = ctx.get_object("user").unwrap();
        assert_eq!(obj["name"].as_str().unwrap(), "bob");
        assert!(ctx.get_object("missing").is_none());
    }

    #[test]
    fn test_keys() {
        let mut ctx = WorkflowContext::new();
        ctx.set("a", "1");
        ctx.set("b", "2");
        let mut keys: Vec<&str> = ctx.keys().collect();
        keys.sort();
        assert_eq!(keys, vec!["a", "b"]);
    }

    #[test]
    fn test_remove() {
        let mut ctx = WorkflowContext::new();
        ctx.set("x", "hello");
        assert!(ctx.contains("x"));
        let val = ctx.remove("x");
        assert!(val.is_some());
        assert!(!ctx.contains("x"));
        assert!(ctx.remove("missing").is_none());
    }

    #[test]
    fn test_extend_merges_contexts() {
        let mut base = WorkflowContext::new();
        base.set("a", "base_a");
        let mut other = WorkflowContext::new();
        other.set("b", "other_b");
        other.set("a", "overwritten");
        base.extend(&other);
        assert_eq!(base.get_str("a"), "overwritten");
        assert_eq!(base.get_str("b"), "other_b");
    }
}
