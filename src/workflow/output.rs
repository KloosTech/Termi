use serde_json::Value;

use crate::error::TermiError;

/// Defines what type of output a step is expected to produce and how to
/// validate and coerce the raw LLM response text.
#[derive(Debug, Clone)]
pub enum OutputFormat {
    /// Plain text — stored as `Value::String`. No validation.
    Text,

    /// Any valid JSON. The response is parsed and stored as-is.
    Json,

    /// JSON that must conform to the given JSON schema.
    /// Currently validates top-level `type` and, for arrays, `items.type`.
    JsonSchema(Value),
}

impl OutputFormat {
    /// Validate and parse `raw` according to the format.
    /// Returns the `Value` to store in the context on success.
    pub fn parse_and_validate(&self, raw: &str) -> Result<Value, TermiError> {
        match self {
            OutputFormat::Text => Ok(Value::String(raw.to_string())),

            OutputFormat::Json => serde_json::from_str(raw).map_err(|e| {
                TermiError::Pipeline(format!("expected JSON output, got parse error: {e}\nRaw: {raw}"))
            }),

            OutputFormat::JsonSchema(schema) => {
                let value: Value = serde_json::from_str(raw).map_err(|e| {
                    TermiError::Pipeline(format!(
                        "expected JSON output, got parse error: {e}\nRaw: {raw}"
                    ))
                })?;
                validate_schema(&value, schema)?;
                Ok(value)
            }
        }
    }

    /// The Ollama `format` field to send in the request, if any.
    pub fn ollama_format(&self) -> Option<Value> {
        match self {
            OutputFormat::Text => None,
            OutputFormat::Json | OutputFormat::JsonSchema(_) => {
                Some(Value::String("json".to_string()))
            }
        }
    }
}

/// Lightweight schema validation — checks `type` and, for object/array schemas,
/// one level of structural constraints. Full JSON Schema support is out of scope
/// for this PoC; this catches the most common mistakes from LLM outputs.
fn validate_schema(value: &Value, schema: &Value) -> Result<(), TermiError> {
    let Some(expected_type) = schema.get("type").and_then(|t| t.as_str()) else {
        return Ok(()); // no type constraint — accept anything
    };

    let actual_type = json_type_name(value);
    if actual_type != expected_type {
        return Err(TermiError::Pipeline(format!(
            "schema validation failed: expected type \"{expected_type}\", got \"{actual_type}\""
        )));
    }

    // For arrays: validate each item against items schema if present
    if expected_type == "array" {
        if let (Some(items_schema), Some(arr)) =
            (schema.get("items"), value.as_array())
        {
            for (i, item) in arr.iter().enumerate() {
                validate_schema(item, items_schema).map_err(|e| {
                    TermiError::Pipeline(format!("schema validation failed at index {i}: {e}"))
                })?;
            }
        }
    }

    // For objects: validate required properties if present
    if expected_type == "object" {
        if let (Some(required), Some(obj)) =
            (schema.get("required").and_then(|r| r.as_array()), value.as_object())
        {
            for req_key in required {
                if let Some(key) = req_key.as_str() {
                    if !obj.contains_key(key) {
                        return Err(TermiError::Pipeline(format!(
                            "schema validation failed: required property \"{key}\" is missing"
                        )));
                    }
                }
            }
        }
    }

    Ok(())
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) => {
            if n.is_f64() { "number" } else { "integer" }
        }
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn text_format_stores_raw_string() {
        let v = OutputFormat::Text.parse_and_validate("hello world").unwrap();
        assert_eq!(v, Value::String("hello world".to_string()));
    }

    #[test]
    fn json_format_parses_object() {
        let v = OutputFormat::Json.parse_and_validate(r#"{"a":1}"#).unwrap();
        assert_eq!(v["a"], json!(1));
    }

    #[test]
    fn json_format_rejects_invalid_json() {
        let err = OutputFormat::Json.parse_and_validate("not json").unwrap_err();
        assert!(matches!(err, crate::error::TermiError::Pipeline(_)));
    }

    #[test]
    fn json_schema_validates_array_of_strings() {
        let schema = json!({"type": "array", "items": {"type": "string"}});
        let v = OutputFormat::JsonSchema(schema)
            .parse_and_validate(r#"["a.rs", "b.rs"]"#)
            .unwrap();
        assert_eq!(v.as_array().unwrap().len(), 2);
    }

    #[test]
    fn json_schema_rejects_array_of_numbers_when_strings_required() {
        let schema = json!({"type": "array", "items": {"type": "string"}});
        let err = OutputFormat::JsonSchema(schema)
            .parse_and_validate(r#"[1, 2]"#)
            .unwrap_err();
        assert!(matches!(err, crate::error::TermiError::Pipeline(_)));
    }

    #[test]
    fn json_schema_rejects_wrong_top_level_type() {
        let schema = json!({"type": "array"});
        let err = OutputFormat::JsonSchema(schema)
            .parse_and_validate(r#"{"key": "val"}"#)
            .unwrap_err();
        assert!(matches!(err, crate::error::TermiError::Pipeline(_)));
    }

    #[test]
    fn json_schema_validates_required_object_properties() {
        let schema = json!({"type": "object", "required": ["name"]});
        // missing "name"
        let err = OutputFormat::JsonSchema(schema.clone())
            .parse_and_validate(r#"{"age": 1}"#)
            .unwrap_err();
        assert!(matches!(err, crate::error::TermiError::Pipeline(_)));
        // present
        OutputFormat::JsonSchema(schema)
            .parse_and_validate(r#"{"name": "alice"}"#)
            .unwrap();
    }
}
