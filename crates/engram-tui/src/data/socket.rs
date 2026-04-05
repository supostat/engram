use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

use serde_json::Value;

pub struct SocketClient {
    stream: UnixStream,
}

impl SocketClient {
    pub fn connect(socket_path: &str) -> io::Result<Self> {
        let stream = UnixStream::connect(socket_path)?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(Duration::from_secs(2)))?;
        Ok(Self { stream })
    }

    pub fn call(&mut self, method: &str, params: Value) -> io::Result<Value> {
        let request_id = generate_request_id();
        let request = serde_json::json!({
            "id": request_id,
            "method": method,
            "params": params,
        });
        let mut line = serde_json::to_string(&request).map_err(io::Error::other)?;
        line.push('\n');
        self.stream.write_all(line.as_bytes())?;
        self.stream.flush()?;

        let mut reader = BufReader::new(&self.stream);
        let mut response_line = String::new();
        reader.read_line(&mut response_line)?;
        let response: Value =
            serde_json::from_str(&response_line).map_err(io::Error::other)?;

        if response["ok"].as_bool() == Some(true) {
            Ok(response["data"].clone())
        } else {
            let message = response["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            Err(io::Error::other(message.to_string()))
        }
    }
}

fn generate_request_id() -> String {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_nanos();
    format!("{nanos:032x}")
}
