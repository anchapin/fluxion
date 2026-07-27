//! LLM Response Patch Parser
//!
//! Parses TOON-formatted parameter patches from LLM responses,
//! handling markdown codeblock wrapping automatically.

use crate::error::ToonError;

/// Parse a TOON patch from an LLM response string
///
/// Strips markdown codeblock fences (```toon, ```ton, etc.) and
/// extracts the TOON content between them.
///
/// # Arguments
///
/// * `input` - Raw LLM response that may contain TOON in codeblocks
///
/// # Returns
///
/// The extracted TOON string, or an error if parsing fails
pub fn parse_toon_patch(_input: &str) -> Result<String, ToonError> {
    // TODO(#2069): Implement patch parser with codeblock stripping
    // Handles:
    // - ```toon ... ``` (most common)
    // - ```ton ... ```
    // - Plain TOON without codeblocks
    // - Markdown frontmatter
    Err(ToonError::PatchError(
        "patch parser not yet implemented (issue #2069)".to_string()
    ))
}

/// Extract TOON content from a potentially wrapped response
#[allow(dead_code)]
fn strip_codeblock_fences(s: &str) -> &str {
    // TODO(#2069): Implement fence stripping
    s
}
