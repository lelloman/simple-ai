//! Fixed contract for the Qwen PyTorch embedding provider (no wire changes).
pub const MODEL_ID: &str = "Qwen/Qwen3-Embedding-0.6B";
pub const REVISION: &str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3";
/// Bound CPU tokenization and response memory independently of microbatch size.
pub fn validate(inputs: &[String]) -> Result<(), String> {
    if inputs.is_empty()
        || inputs.len() > 256
        || inputs
            .iter()
            .any(|s| s.trim().is_empty() || s.chars().count() > 32_768)
        || inputs.iter().map(|s| s.len()).sum::<usize>() > 1_048_576
    {
        return Err("input requires 1–256 non-empty texts, at most 32768 characters each and 1 MiB total UTF-8".into());
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_limits() {
        assert!(validate(&["hello".into()]).is_ok());
        for input in [
            vec![],
            vec![" ".into()],
            vec!["a".into(); 257],
            vec!["a".repeat(32769)],
            vec!["a".repeat(32768); 33],
        ] {
            assert!(validate(&input).is_err());
        }
    }
}
