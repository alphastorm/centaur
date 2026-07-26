use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde_json::Value;

use crate::{HarnessServerError, Result};

pub(crate) const MAX_PHYSICAL_FRAME_BYTES: usize = 1_048_576;
pub(crate) const MAX_REASSEMBLED_FRAME_BYTES: usize = 67_108_864;
const MAX_CHUNKS: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReadyFrame {
    pub(crate) supported_protocol_versions: Vec<u64>,
    pub(crate) max_frame_bytes: usize,
    pub(crate) max_reassembled_frame_bytes: usize,
}

#[derive(Debug)]
struct PendingChunks {
    chunk_id: String,
    next_index: usize,
    count: usize,
    byte_length: usize,
    bytes: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct Decoder {
    protocol_version: u64,
    max_frame_bytes: usize,
    max_reassembled_frame_bytes: usize,
    pending: Option<PendingChunks>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            protocol_version: 1,
            max_frame_bytes: MAX_PHYSICAL_FRAME_BYTES,
            max_reassembled_frame_bytes: MAX_REASSEMBLED_FRAME_BYTES,
            pending: None,
        }
    }
}

impl Decoder {
    pub(crate) fn set_limits_from_ready(&mut self, ready: &ReadyFrame) {
        self.max_frame_bytes = ready.max_frame_bytes.min(MAX_PHYSICAL_FRAME_BYTES);
        self.max_reassembled_frame_bytes = ready
            .max_reassembled_frame_bytes
            .min(MAX_REASSEMBLED_FRAME_BYTES);
    }

    pub(crate) fn enable_v2(&mut self) {
        self.protocol_version = 2;
    }

    pub(crate) fn decode_line(&mut self, line: &[u8]) -> Result<Option<Value>> {
        if line.len() > self.max_frame_bytes {
            return Err(protocol_error(format!(
                "OMP physical frame is {} bytes; limit is {}",
                line.len(),
                self.max_frame_bytes
            )));
        }
        let value: Value = serde_json::from_slice(line).map_err(|error| {
            protocol_error(format!("OMP emitted malformed UTF-8/JSON frame: {error}"))
        })?;
        let object = value
            .as_object()
            .ok_or_else(|| protocol_error("OMP frame must be a JSON object"))?;
        let frame_type = object
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| protocol_error("OMP frame is missing string type"))?;

        if frame_type == "rpc_chunk" {
            if self.protocol_version != 2 {
                return Err(protocol_error(
                    "OMP emitted rpc_chunk before v2 negotiation",
                ));
            }
            return self.push_chunk(object);
        }
        if self.pending.is_some() {
            return Err(protocol_error(
                "OMP interrupted an in-progress rpc_chunk sequence with another frame",
            ));
        }
        Ok(Some(value))
    }

    fn push_chunk(&mut self, object: &serde_json::Map<String, Value>) -> Result<Option<Value>> {
        let chunk_id = required_string(object, "chunkId")?;
        let index = required_usize(object, "index")?;
        let count = required_usize(object, "count")?;
        let byte_length = required_usize(object, "byteLength")?;
        let encoded = required_string(object, "data")?;

        if count == 0 || count > MAX_CHUNKS {
            return Err(protocol_error(format!(
                "OMP rpc_chunk count {count} is outside 1..={MAX_CHUNKS}"
            )));
        }
        if byte_length > self.max_reassembled_frame_bytes {
            return Err(protocol_error(format!(
                "OMP rpc_chunk byteLength {byte_length} exceeds {}",
                self.max_reassembled_frame_bytes
            )));
        }
        let decoded = BASE64_STANDARD
            .decode(encoded)
            .map_err(|error| protocol_error(format!("OMP rpc_chunk base64 is invalid: {error}")))?;

        if self.pending.is_none() {
            if index != 0 {
                return Err(protocol_error(format!(
                    "OMP rpc_chunk sequence starts at index {index}, expected 0"
                )));
            }
            self.pending = Some(PendingChunks {
                chunk_id: chunk_id.to_string(),
                next_index: 0,
                count,
                byte_length,
                bytes: Vec::with_capacity(byte_length.min(1024 * 1024)),
            });
        }

        let pending = self
            .pending
            .as_mut()
            .expect("pending chunk sequence exists");
        if pending.chunk_id != chunk_id
            || pending.count != count
            || pending.byte_length != byte_length
            || pending.next_index != index
        {
            return Err(protocol_error(format!(
                "OMP rpc_chunk sequence mismatch for {chunk_id} at index {index}"
            )));
        }
        if pending.bytes.len().saturating_add(decoded.len()) > pending.byte_length
            || pending.bytes.len().saturating_add(decoded.len()) > self.max_reassembled_frame_bytes
        {
            return Err(protocol_error(
                "OMP rpc_chunk reassembled bytes exceed declared limits",
            ));
        }
        pending.bytes.extend_from_slice(&decoded);
        pending.next_index += 1;

        if pending.next_index < pending.count {
            return Ok(None);
        }
        let completed = self.pending.take().expect("completed sequence exists");
        if completed.bytes.len() != completed.byte_length {
            return Err(protocol_error(format!(
                "OMP rpc_chunk reassembled {} bytes; declared {}",
                completed.bytes.len(),
                completed.byte_length
            )));
        }
        let text = String::from_utf8(completed.bytes).map_err(|error| {
            protocol_error(format!("OMP rpc_chunk is not strict UTF-8: {error}"))
        })?;
        let value: Value = serde_json::from_str(&text).map_err(|error| {
            protocol_error(format!(
                "OMP rpc_chunk does not reassemble to strict JSON: {error}"
            ))
        })?;
        let object = value
            .as_object()
            .ok_or_else(|| protocol_error("OMP reassembled frame must be a JSON object"))?;
        if object.get("type").and_then(Value::as_str).is_none() {
            return Err(protocol_error(
                "OMP reassembled frame is missing string type",
            ));
        }
        if object.get("type").and_then(Value::as_str) == Some("rpc_chunk") {
            return Err(protocol_error("nested OMP rpc_chunk frame is invalid"));
        }
        Ok(Some(value))
    }
}

pub(crate) fn parse_ready(value: &Value) -> Result<ReadyFrame> {
    let object = value
        .as_object()
        .ok_or_else(|| protocol_error("OMP ready frame must be an object"))?;
    if object.get("type").and_then(Value::as_str) != Some("ready") {
        return Err(protocol_error("first OMP frame must be ready"));
    }
    if object.get("protocolVersion").and_then(Value::as_u64) != Some(1) {
        return Err(protocol_error("OMP ready frame must use protocolVersion 1"));
    }
    let versions = object
        .get("supportedProtocolVersions")
        .and_then(Value::as_array)
        .ok_or_else(|| protocol_error("OMP ready frame lacks supportedProtocolVersions"))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| protocol_error("OMP supported protocol version is not an integer"))
        })
        .collect::<Result<Vec<_>>>()?;
    if !versions.contains(&1) {
        return Err(protocol_error(
            "OMP ready frame does not advertise protocol v1",
        ));
    }
    let max_frame_bytes = required_usize(object, "maxFrameBytes")?;
    let max_reassembled_frame_bytes = required_usize(object, "maxReassembledFrameBytes")?;
    if max_frame_bytes == 0 || max_reassembled_frame_bytes == 0 {
        return Err(protocol_error(
            "OMP ready frame advertises a zero frame limit",
        ));
    }
    Ok(ReadyFrame {
        supported_protocol_versions: versions,
        max_frame_bytes,
        max_reassembled_frame_bytes,
    })
}

pub(crate) fn frame_type(value: &Value) -> Result<&str> {
    value
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| protocol_error("OMP frame is missing string type"))
}

pub(crate) fn protocol_error(message: impl Into<String>) -> HarnessServerError {
    HarnessServerError::Protocol(format!("OMP protocol error: {}", message.into()))
}

fn required_string<'a>(object: &'a serde_json::Map<String, Value>, field: &str) -> Result<&'a str> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| protocol_error(format!("OMP frame field {field} must be a string")))
}

fn required_usize(object: &serde_json::Map<String, Value>, field: &str) -> Result<usize> {
    let value = object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| protocol_error(format!("OMP frame field {field} must be an integer")))?;
    usize::try_from(value)
        .map_err(|_| protocol_error(format!("OMP frame field {field} is too large")))
}

#[cfg(test)]
mod tests {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
    use serde_json::json;

    use super::{Decoder, parse_ready};

    #[test]
    fn validates_ready_and_reassembles_v2_chunks() {
        let ready = parse_ready(&json!({
            "type": "ready",
            "protocolVersion": 1,
            "supportedProtocolVersions": [1, 2],
            "maxFrameBytes": 1_048_576,
            "maxReassembledFrameBytes": 67_108_864
        }))
        .unwrap();
        let mut decoder = Decoder::default();
        decoder.set_limits_from_ready(&ready);
        decoder.enable_v2();
        let raw = br#"{"type":"notice","message":"ok"}"#;
        let first = json!({
            "type": "rpc_chunk", "chunkId": "a", "index": 0, "count": 2,
            "byteLength": raw.len(), "data": BASE64_STANDARD.encode(&raw[..10])
        });
        let second = json!({
            "type": "rpc_chunk", "chunkId": "a", "index": 1, "count": 2,
            "byteLength": raw.len(), "data": BASE64_STANDARD.encode(&raw[10..])
        });
        assert!(
            decoder
                .decode_line(first.to_string().as_bytes())
                .unwrap()
                .is_none()
        );
        let value = decoder
            .decode_line(second.to_string().as_bytes())
            .unwrap()
            .unwrap();
        assert_eq!(value["type"], "notice");
    }

    #[test]
    fn rejects_interrupted_chunk_sequence() {
        let mut decoder = Decoder::default();
        decoder.enable_v2();
        let chunk = json!({
            "type": "rpc_chunk", "chunkId": "a", "index": 0, "count": 2,
            "byteLength": 2, "data": BASE64_STANDARD.encode(b"{")
        });
        assert!(
            decoder
                .decode_line(chunk.to_string().as_bytes())
                .unwrap()
                .is_none()
        );
        let error = decoder
            .decode_line(br#"{"type":"notice"}"#)
            .unwrap_err()
            .to_string();
        assert!(error.contains("interrupted"));
    }

    #[test]
    fn rejects_invalid_base64_and_byte_length() {
        let mut decoder = Decoder::default();
        decoder.enable_v2();
        let invalid =
            br#"{"type":"rpc_chunk","chunkId":"a","index":0,"count":1,"byteLength":1,"data":"!"}"#;
        assert!(
            decoder
                .decode_line(invalid)
                .unwrap_err()
                .to_string()
                .contains("base64")
        );
    }
    #[test]
    fn rejects_chunk_identity_order_length_utf8_and_nested_frames() {
        let mut wrong_order = Decoder::default();
        wrong_order.enable_v2();
        let first = json!({
            "type": "rpc_chunk", "chunkId": "a", "index": 0, "count": 2,
            "byteLength": 2, "data": BASE64_STANDARD.encode(b"{")
        });
        wrong_order
            .decode_line(first.to_string().as_bytes())
            .unwrap();
        let second = json!({
            "type": "rpc_chunk", "chunkId": "b", "index": 1, "count": 2,
            "byteLength": 2, "data": BASE64_STANDARD.encode(b"}")
        });
        assert!(
            wrong_order
                .decode_line(second.to_string().as_bytes())
                .unwrap_err()
                .to_string()
                .contains("sequence mismatch")
        );

        for (bytes, expected) in [
            (vec![0xff], "UTF-8"),
            (br#"{"type":"rpc_chunk"}"#.to_vec(), "nested"),
        ] {
            let mut decoder = Decoder::default();
            decoder.enable_v2();
            let chunk = json!({
                "type": "rpc_chunk", "chunkId": "one", "index": 0, "count": 1,
                "byteLength": bytes.len(), "data": BASE64_STANDARD.encode(&bytes)
            });
            assert!(
                decoder
                    .decode_line(chunk.to_string().as_bytes())
                    .unwrap_err()
                    .to_string()
                    .contains(expected)
            );
        }

        let mut wrong_length = Decoder::default();
        wrong_length.enable_v2();
        let chunk = json!({
            "type": "rpc_chunk", "chunkId": "short", "index": 0, "count": 1,
            "byteLength": 2, "data": BASE64_STANDARD.encode(b"{")
        });
        assert!(
            wrong_length
                .decode_line(chunk.to_string().as_bytes())
                .unwrap_err()
                .to_string()
                .contains("declared")
        );
    }

    #[test]
    fn rejects_invalid_ready_and_physical_frame_limit() {
        assert!(
            parse_ready(&json!({
                "type": "ready",
                "protocolVersion": 2,
                "supportedProtocolVersions": [2],
                "maxFrameBytes": 1,
                "maxReassembledFrameBytes": 1
            }))
            .is_err()
        );
        let mut decoder = Decoder::default();
        let oversized = vec![b' '; super::MAX_PHYSICAL_FRAME_BYTES + 1];
        assert!(
            decoder
                .decode_line(&oversized)
                .unwrap_err()
                .to_string()
                .contains("physical frame")
        );
    }
}
