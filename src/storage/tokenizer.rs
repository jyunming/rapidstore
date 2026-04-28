//! Deterministic, allocation-light tokenizer for BM25.
//!
//! Splits on non-alphanumeric Unicode boundaries, ASCII-lowercases each token,
//! and hashes via FNV-1a. No stopwords, no stemming — those decisions belong
//! to higher layers and would couple the index to a specific language pipeline.

/// Hash a token's bytes via FNV-1a (64-bit). Same constants as `id_pool::fnv1a64`,
/// kept inline here to avoid a cross-module visibility change.
#[inline]
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Tokenize `text` into a vector of FNV-1a token hashes.
///
/// Behaviour:
/// - Splits on any non-alphanumeric `char` (Unicode-aware via `char::is_alphanumeric`).
/// - Lowercases each token via `str::to_lowercase` before hashing, so `Foo` and `foo`
///   collide as expected. Lowercasing here is the whole reason the function allocates;
///   when text is ASCII-only, callers will see one short heap allocation per token.
/// - Empty tokens (consecutive separators) are dropped.
///
/// Determinism: the output is a pure function of the input bytes. Two opens of the
/// same database with the same documents will rebuild identical posting lists.
pub fn tokenize(text: &str) -> Vec<u64> {
    let mut out = Vec::with_capacity(text.len() / 6); // ≈ avg English word length
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        if raw.is_empty() {
            continue;
        }
        // ASCII fast path skips the lowercase allocation when nothing would change.
        if raw.bytes().all(|b| !b.is_ascii_uppercase() && b < 0x80) {
            out.push(fnv1a64(raw.as_bytes()));
        } else {
            let lower = raw.to_lowercase();
            out.push(fnv1a64(lower.as_bytes()));
        }
    }
    out
}

/// Number of distinct tokens in `text`. Useful for callers that need only the doc length.
pub fn token_count(text: &str) -> u32 {
    let mut n: u32 = 0;
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        if !raw.is_empty() {
            n = n.saturating_add(1);
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowercases_ascii() {
        assert_eq!(tokenize("Foo BAR baz"), tokenize("foo bar baz"));
    }

    #[test]
    fn splits_on_non_alphanumeric() {
        // Punctuation, whitespace, and symbols all act as separators.
        let toks = tokenize("hello, world! 123-abc");
        assert_eq!(toks.len(), 4);
    }

    #[test]
    fn drops_empty_tokens() {
        // Consecutive separators must not yield empty entries.
        let toks = tokenize("   ,,,a,,,b...");
        assert_eq!(toks.len(), 2);
    }

    #[test]
    fn unicode_words_kept() {
        // Non-ASCII alphanumeric chars are part of tokens.
        let toks = tokenize("café résumé naïve");
        assert_eq!(toks.len(), 3);
    }

    #[test]
    fn token_count_matches_tokenize_len() {
        for s in ["", "one", "one two", "one,, two!! three"] {
            assert_eq!(token_count(s) as usize, tokenize(s).len(), "input={s:?}");
        }
    }

    #[test]
    fn deterministic_across_calls() {
        let a = tokenize("the quick brown fox");
        let b = tokenize("the quick brown fox");
        assert_eq!(a, b);
    }

    #[test]
    fn case_insensitive_unicode() {
        // Greek capital sigma has two lowercase forms; whatever to_lowercase picks
        // must be the same for both inputs, so the hashes still match.
        let a = tokenize("ΣIGMA");
        let b = tokenize("σigma");
        assert_eq!(a, b);
    }
}
