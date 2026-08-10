//! Derived state serialization for built-in games.

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
