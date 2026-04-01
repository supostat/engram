use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct JsonRequest {
    pub id: String,
    pub method: String,
    pub params: serde_json::Value,
}

#[derive(Serialize)]
pub struct JsonResponse {
    pub id: String,
    pub ok: bool,
    pub data: Option<serde_json::Value>,
    pub error: Option<ErrorPayload>,
}

#[derive(Serialize)]
pub struct ErrorPayload {
    pub code: u32,
    pub message: String,
}

impl JsonResponse {
    pub fn success(id: String, data: serde_json::Value) -> Self {
        Self {
            id,
            ok: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(id: String, code: u32, message: String) -> Self {
        Self {
            id,
            ok: false,
            data: None,
            error: Some(ErrorPayload { code, message }),
        }
    }
}
