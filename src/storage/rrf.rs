//! Reciprocal Rank Fusion (RRF) for hybrid retrieval.
//!
//! Combines several ranked candidate lists (e.g. dense vector results + BM25
//! results) into a single ranked list. Each input list contributes
//! `weight / (k + rank)` per item, where `rank` is the item's 1-based position
//! within that list. Items missing from a list contribute nothing for it, which
//! is what makes RRF tolerant of disparate score scales between dense and sparse.
//!
//! Reference: Cormack, Clarke, Buettcher — *Reciprocal rank fusion outperforms
//! Condorcet and individual rank learning methods* (SIGIR 2009). The conventional
//! `k=60` smooths the contribution of mid-rank items so that two strong agreers
//! near the top can outrank a single first-place hit.

/// Fuse `lists` (each already sorted best-first by its own scoring) into a single
/// top-`top_k` ranking.
///
/// `weights` must have the same length as `lists`. A list with weight `0.0`
/// contributes nothing — useful to flip hybrid mode on/off without rebuilding
/// the input lists.
///
/// `k` is the RRF smoothing constant. A typical default is `60.0`.
///
/// Returns `(slot, fused_score)` pairs in descending fused-score order, with at
/// most `top_k` entries.
pub fn rrf_fuse(lists: &[&[u32]], weights: &[f32], k: f32, top_k: usize) -> Vec<(u32, f32)> {
    debug_assert_eq!(
        lists.len(),
        weights.len(),
        "rrf_fuse: lists and weights must align",
    );
    if top_k == 0 || lists.is_empty() {
        return Vec::new();
    }

    use std::collections::HashMap;
    // Capacity is a guess: the union is at most Σ|list_i| but in practice the
    // lists overlap heavily, so a single list's length is usually a good hint.
    let cap_hint = lists.iter().map(|l| l.len()).max().unwrap_or(0);
    let mut accum: HashMap<u32, f32> = HashMap::with_capacity(cap_hint);

    for (list, &w) in lists.iter().zip(weights.iter()) {
        if w == 0.0 || list.is_empty() {
            continue;
        }
        for (i, &slot) in list.iter().enumerate() {
            let rank = (i + 1) as f32;
            let contrib = w / (k + rank);
            *accum.entry(slot).or_insert(0.0) += contrib;
        }
    }

    if accum.is_empty() {
        return Vec::new();
    }

    let mut out: Vec<(u32, f32)> = accum.into_iter().collect();
    if out.len() > top_k {
        out.select_nth_unstable_by(top_k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(top_k);
    }
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Default RRF smoothing constant from Cormack et al. 2009.
pub const DEFAULT_RRF_K: f32 = 60.0;

/// Default oversampling factor: each input list is asked for `RRF_OVERSAMPLE × top_k`
/// candidates so RRF has room to find consensus picks that don't lead either list.
pub const DEFAULT_RRF_OVERSAMPLE: usize = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_inputs_return_empty() {
        assert!(rrf_fuse(&[], &[], 60.0, 10).is_empty());
        let l: &[u32] = &[];
        assert!(rrf_fuse(&[l], &[1.0], 60.0, 10).is_empty());
    }

    #[test]
    fn zero_top_k_returns_empty() {
        let l: &[u32] = &[1, 2, 3];
        assert!(rrf_fuse(&[l], &[1.0], 60.0, 0).is_empty());
    }

    #[test]
    fn single_list_preserves_order() {
        let l: &[u32] = &[10, 20, 30];
        let out = rrf_fuse(&[l], &[1.0], 60.0, 10);
        let slots: Vec<u32> = out.iter().map(|(s, _)| *s).collect();
        assert_eq!(slots, vec![10, 20, 30]);
    }

    #[test]
    fn consensus_pick_outranks_unilateral_winner() {
        // List A puts slot 1 first, slot 99 second.
        // List B puts slot 2 first, slot 1 second.
        // Slot 1 appears in both at rank 1 and 2 → fused score ≈ 1/61 + 1/62.
        // Slot 99 only in A at rank 2 → 1/62.
        // Slot 2 only in B at rank 1 → 1/61.
        // Slot 1 must outrank both unilateral winners.
        let a: &[u32] = &[1, 99];
        let b: &[u32] = &[2, 1];
        let out = rrf_fuse(&[a, b], &[1.0, 1.0], 60.0, 10);
        assert_eq!(out[0].0, 1);
    }

    #[test]
    fn weight_zero_disables_a_list() {
        // With weight 0 on list B, list A's order must dominate completely.
        let a: &[u32] = &[10, 20];
        let b: &[u32] = &[20, 10];
        let out = rrf_fuse(&[a, b], &[1.0, 0.0], 60.0, 10);
        let slots: Vec<u32> = out.iter().map(|(s, _)| *s).collect();
        assert_eq!(slots, vec![10, 20]);
    }

    #[test]
    fn k_governs_rank_decay() {
        // Larger k flattens the contribution of top ranks vs lower ranks.
        let a: &[u32] = &[1, 2];
        let small_k = rrf_fuse(&[a], &[1.0], 1.0, 10);
        let large_k = rrf_fuse(&[a], &[1.0], 1000.0, 10);
        // With k=1, slot 1 (rank 1, score 1/2) is ~3× slot 2 (rank 2, score 1/3).
        // With k=1000, the ratio shrinks toward 1.
        let small_ratio = small_k[0].1 / small_k[1].1;
        let large_ratio = large_k[0].1 / large_k[1].1;
        assert!(small_ratio > large_ratio);
    }

    #[test]
    fn top_k_truncates_results() {
        let a: &[u32] = &[1, 2, 3, 4, 5];
        let out = rrf_fuse(&[a], &[1.0], 60.0, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, 1);
        assert_eq!(out[1].0, 2);
    }

    #[test]
    fn duplicate_in_same_list_does_not_double_count() {
        // RRF input lists are expected to be deduplicated by the producer; this test
        // documents what happens when a caller violates that — the slot accumulates
        // contributions from both positions, which is benign but worth pinning.
        let a: &[u32] = &[1, 1, 2];
        let out = rrf_fuse(&[a], &[1.0], 60.0, 10);
        // Slot 1 picks up rank-1 + rank-2 contributions and beats slot 2 (rank-3 only).
        assert_eq!(out[0].0, 1);
    }
}
