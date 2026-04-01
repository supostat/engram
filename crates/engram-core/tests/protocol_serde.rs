use engram_core::{JsonRequest, JsonResponse};

#[test]
fn deserialize_json_request() {
    let json = r#"{"id":"req-1","method":"memory_store","params":{"key":"value"}}"#;
    let request: JsonRequest = serde_json::from_str(json).unwrap();
    assert_eq!(request.id, "req-1");
    assert_eq!(request.method, "memory_store");
    assert_eq!(request.params["key"], "value");
}

#[test]
fn serialize_success_response() {
    let response = JsonResponse::success("req-1".into(), serde_json::json!({"count": 42}));
    let serialized = serde_json::to_string(&response).unwrap();
    assert!(serialized.contains(r#""ok":true"#));
    assert!(serialized.contains(r#""count":42"#));
    assert!(serialized.contains(r#""id":"req-1""#));
    assert!(serialized.contains(r#""error":null"#));
}

#[test]
fn serialize_error_response() {
    let response = JsonResponse::error("req-2".into(), 6001, "config not found".into());
    let serialized = serde_json::to_string(&response).unwrap();
    assert!(serialized.contains(r#""ok":false"#));
    assert!(serialized.contains(r#""code":6001"#));
    assert!(serialized.contains(r#""config not found"#));
    assert!(serialized.contains(r#""data":null"#));
}

#[test]
fn success_response_has_no_error() {
    let response = JsonResponse::success("id".into(), serde_json::json!(null));
    assert!(response.ok);
    assert!(response.data.is_some());
    assert!(response.error.is_none());
}

#[test]
fn error_response_has_no_data() {
    let response = JsonResponse::error("id".into(), 6002, "msg".into());
    assert!(!response.ok);
    assert!(response.data.is_none());
    assert!(response.error.is_some());
}
