//! Derived-serialization support for the built-in `StateCodec`s. The hand-written byte
//! reader/writer that used to live here is gone — that is the point: per-game byte plumbing was
//! extensive and error-prone; serialization is now derived, and the per-game surface is a single
//! typed `validate_state`.

/// Derived (postcard) state serialization behind a leading layout-version byte. Encoding is the
/// canonical byte form (custom field shims keep it deterministic — e.g. sorted food cells);
/// decoding is STRUCTURAL only — semantic invariants live in each game's `validate_state`.
pub(crate) fn serde_encode<T: serde::Serialize>(version: u8, value: &T) -> Vec<u8> {
    let mut out = vec![version];
    out.extend(postcard::to_stdvec(value).expect("state types always serialize"));
    out
}

pub(crate) fn serde_decode<T: serde::de::DeserializeOwned>(
    version: u8,
    bytes: &[u8],
) -> Result<T, String> {
    match bytes.split_first() {
        Some((&v, rest)) if v == version => {
            let (value, trailing) = postcard::take_from_bytes(rest)
                .map_err(|e| format!("malformed state payload: {e}"))?;
            if !trailing.is_empty() {
                return Err(format!(
                    "{} trailing bytes after state payload",
                    trailing.len()
                ));
            }
            Ok(value)
        }
        Some((&v, _)) => Err(format!(
            "unsupported state layout version {v} (expected {version})"
        )),
        None => Err("empty state payload".into()),
    }
}
