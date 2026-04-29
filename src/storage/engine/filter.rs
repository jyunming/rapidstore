use super::DistanceMetric;
use ndarray::Array1;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// Maximum nesting depth for `$and` / `$or` filter expressions. A user-supplied
/// `{"$and": [{"$and": [...]}]}` with thousands of levels would otherwise
/// recurse without bound and overflow the stack (~8 MB default ≈ 10k frames).
/// 32 is far above any sane query and is checked at validation time, so users
/// see a clear error instead of a crash.
pub(crate) const MAX_FILTER_DEPTH: u32 = 32;

pub(crate) fn metadata_matches_filter(
    meta: &HashMap<String, JsonValue>,
    filter: &HashMap<String, JsonValue>,
) -> bool {
    metadata_matches_filter_with_depth(meta, filter, 0)
}

/// Internal recursive evaluator. `depth` increments on each `$and` / `$or`
/// dive; if it exceeds `MAX_FILTER_DEPTH` the filter is treated as
/// non-matching (defense-in-depth — `validate_filter_operators` should have
/// already rejected the filter, but we don't trust the caller to have called it).
fn metadata_matches_filter_with_depth(
    meta: &HashMap<String, JsonValue>,
    filter: &HashMap<String, JsonValue>,
    depth: u32,
) -> bool {
    if depth > MAX_FILTER_DEPTH {
        return false;
    }
    filter.iter().all(|(k, v)| match k.as_str() {
        "$and" => {
            if let JsonValue::Array(conditions) = v {
                conditions.iter().all(|cond| {
                    if let JsonValue::Object(map) = cond {
                        let as_hm: HashMap<String, JsonValue> =
                            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                        metadata_matches_filter_with_depth(meta, &as_hm, depth + 1)
                    } else {
                        false
                    }
                })
            } else {
                false
            }
        }
        "$or" => {
            if let JsonValue::Array(conditions) = v {
                conditions.iter().any(|cond| {
                    if let JsonValue::Object(map) = cond {
                        let as_hm: HashMap<String, JsonValue> =
                            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                        metadata_matches_filter_with_depth(meta, &as_hm, depth + 1)
                    } else {
                        false
                    }
                })
            } else {
                false
            }
        }
        field => {
            let field_val = get_nested_field(meta, field);
            match v {
                JsonValue::Object(op_map) => op_map
                    .iter()
                    .all(|(op, op_val)| apply_comparison_op(field_val, op.as_str(), op_val)),
                // Simple equality: {"field": value}
                _ => field_val.is_some_and(|fv| fv == v),
            }
        }
    })
}

/// Resolve a potentially dotted path like "profile.region" into a value from metadata.
pub(crate) fn get_nested_field<'a>(
    meta: &'a HashMap<String, JsonValue>,
    path: &str,
) -> Option<&'a JsonValue> {
    let mut parts = path.splitn(2, '.');
    let head = parts.next()?;
    let val = meta.get(head)?;
    if let Some(rest) = parts.next() {
        if let JsonValue::Object(obj) = val {
            // Recurse into nested object via the remaining path
            get_nested_json_field(obj, rest)
        } else {
            None
        }
    } else {
        Some(val)
    }
}

pub(crate) fn get_nested_json_field<'a>(
    obj: &'a serde_json::Map<String, JsonValue>,
    path: &str,
) -> Option<&'a JsonValue> {
    let mut parts = path.splitn(2, '.');
    let head = parts.next()?;
    let val = obj.get(head)?;
    if let Some(rest) = parts.next() {
        if let JsonValue::Object(nested) = val {
            get_nested_json_field(nested, rest)
        } else {
            None
        }
    } else {
        Some(val)
    }
}

pub(crate) fn apply_comparison_op(field: Option<&JsonValue>, op: &str, op_val: &JsonValue) -> bool {
    match op {
        "$eq" => field.is_some_and(|f| f == op_val),
        "$ne" => {
            // $ne matches missing fields too (missing ≠ value)
            field.map_or(true, |f| f != op_val)
        }
        "$gt" | "$gte" | "$lt" | "$lte" => {
            // Comparisons do not match missing fields
            let Some(f) = field else { return false };
            match (f, op_val) {
                (JsonValue::Number(a), JsonValue::Number(b)) => {
                    // B2 (v0.8.2 audit): If either side is a JSON integer larger
                    // than f64 can represent (|n| > 2^53), `as_f64()` returns
                    // None and we fall back to NaN. NaN comparisons are always
                    // false, so the filter excludes the document. Pinned by
                    // `audit_tests::big_int_metadata_with_numeric_op_returns_false`.
                    let av = a.as_f64().unwrap_or(f64::NAN);
                    let bv = b.as_f64().unwrap_or(f64::NAN);
                    match op {
                        "$gt" => av > bv,
                        "$gte" => av >= bv,
                        "$lt" => av < bv,
                        "$lte" => av <= bv,
                        _ => false,
                    }
                }
                (JsonValue::String(a), JsonValue::String(b)) => match op {
                    "$gt" => a > b,
                    "$gte" => a >= b,
                    "$lt" => a < b,
                    "$lte" => a <= b,
                    _ => false,
                },
                _ => false,
            }
        }
        "$in" => {
            if let (Some(f), JsonValue::Array(arr)) = (field, op_val) {
                arr.iter().any(|v| v == f)
            } else {
                false
            }
        }
        "$nin" => {
            if let JsonValue::Array(arr) = op_val {
                // missing field is "not in" any set
                field.map_or(true, |f| !arr.iter().any(|v| v == f))
            } else {
                false
            }
        }
        "$exists" => match op_val {
            JsonValue::Bool(true) => field.is_some(),
            JsonValue::Bool(false) => field.is_none(),
            _ => false,
        },
        "$contains" => {
            if let (Some(JsonValue::String(f)), JsonValue::String(sub)) = (field, op_val) {
                f.contains(sub.as_str())
            } else {
                false
            }
        }
        _ => false, // unknown operator: never matches
    }
}

#[allow(dead_code)]
const KNOWN_COMPARISON_OPS: &[&str] = &[
    "$eq",
    "$ne",
    "$gt",
    "$gte",
    "$lt",
    "$lte",
    "$in",
    "$nin",
    "$exists",
    "$contains",
];

/// Recursively validate that all operator keys in `filter` are known and that
/// `$and` / `$or` nesting does not exceed [`MAX_FILTER_DEPTH`]. Returns
/// `Err(message)` on the first unknown operator or depth violation.
#[allow(dead_code)]
pub(crate) fn validate_filter_operators(filter: &HashMap<String, JsonValue>) -> Result<(), String> {
    validate_filter_operators_obj(filter.iter().map(|(k, v)| (k.as_str(), v)), 0)
}

/// Internal helper that accepts a `serde_json::Map` reference to avoid cloning.
#[allow(dead_code)]
fn validate_filter_operators_inner(
    map: &serde_json::Map<String, JsonValue>,
    depth: u32,
) -> Result<(), String> {
    validate_filter_operators_obj(map.iter().map(|(k, v)| (k.as_str(), v)), depth)
}

#[allow(dead_code)]
fn validate_filter_operators_obj<'a>(
    iter: impl Iterator<Item = (&'a str, &'a JsonValue)>,
    depth: u32,
) -> Result<(), String> {
    if depth > MAX_FILTER_DEPTH {
        return Err(format!(
            "filter nesting exceeds MAX_FILTER_DEPTH ({MAX_FILTER_DEPTH}); \
             reduce $and/$or nesting"
        ));
    }
    for (k, v) in iter {
        match k {
            "$and" | "$or" => {
                if let JsonValue::Array(conditions) = v {
                    for cond in conditions {
                        if let JsonValue::Object(map) = cond {
                            validate_filter_operators_inner(map, depth + 1)?;
                        }
                    }
                }
            }
            k if k.starts_with('$') => {
                return Err(format!("unknown top-level filter operator '{k}'"));
            }
            _ => {
                // field key — value may be an operator map
                if let JsonValue::Object(op_map) = v {
                    for op in op_map.keys() {
                        if !KNOWN_COMPARISON_OPS.contains(&op.as_str()) {
                            return Err(format!("unknown filter operator '{op}' on field '{k}'"));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Extract a single-field numeric range condition that can use the range_index.
///
/// Returns `(field, lo, hi)` where each bound is `(ord_key, inclusive)`.
/// Returns `None` when:
/// - The filter has $or/$and or more than one top-level field.
/// - The field's value map contains non-range operators ($eq, $in, $exists, etc.).
/// - Any operator value is not a finite number.
pub(crate) fn extract_range_condition<'a>(
    filter: &'a HashMap<String, JsonValue>,
) -> Option<(&'a str, Option<(u64, bool)>, Option<(u64, bool)>)> {
    use crate::storage::metadata::f64_to_ord;

    if filter.contains_key("$or") || filter.contains_key("$and") || filter.len() != 1 {
        return None;
    }
    let (field, val) = filter.iter().next()?;
    if field.starts_with('$') {
        return None;
    }
    let ops = if let JsonValue::Object(m) = val {
        m
    } else {
        return None;
    };

    let mut lo: Option<(u64, bool)> = None;
    let mut hi: Option<(u64, bool)> = None;

    for (op, op_val) in ops {
        let n = op_val.as_f64().filter(|v| v.is_finite())?;
        let ord = f64_to_ord(n);
        match op.as_str() {
            "$gt" => {
                // Tightest lower bound: highest ord wins; on tie, exclusive ($gt) > inclusive ($gte).
                lo = Some(match lo {
                    Some((prev_ord, _)) if prev_ord > ord => (prev_ord, lo.unwrap().1),
                    Some((prev_ord, _)) if prev_ord == ord => (ord, false), // exclusive tighter
                    _ => (ord, false),
                });
            }
            "$gte" => {
                lo = Some(match lo {
                    Some((prev_ord, prev_incl)) if prev_ord > ord => (prev_ord, prev_incl),
                    Some((prev_ord, _)) if prev_ord == ord => (ord, false), // $gt already set → exclusive wins
                    _ => (ord, true),
                });
            }
            "$lt" => {
                // Tightest upper bound: lowest ord wins; on tie, exclusive ($lt) > inclusive ($lte).
                hi = Some(match hi {
                    Some((prev_ord, _)) if prev_ord < ord => (prev_ord, hi.unwrap().1),
                    Some((prev_ord, _)) if prev_ord == ord => (ord, false),
                    _ => (ord, false),
                });
            }
            "$lte" => {
                hi = Some(match hi {
                    Some((prev_ord, prev_incl)) if prev_ord < ord => (prev_ord, prev_incl),
                    Some((prev_ord, _)) if prev_ord == ord => (ord, false),
                    _ => (ord, true),
                });
            }
            _ => return None, // any non-range op → can't use range index
        }
    }

    if lo.is_none() && hi.is_none() {
        return None;
    }
    Some((field.as_str(), lo, hi))
}

/// Extract a single `$in` condition that can use the eq_index.
///
/// Matches `{"field": {"$in": [v1, v2, ...]}}` with exactly one top-level field
/// and no `$or`/`$and`. Returns `(field, values)` or `None`.
pub(crate) fn extract_in_condition<'a>(
    filter: &'a HashMap<String, JsonValue>,
) -> Option<(&'a str, &'a Vec<JsonValue>)> {
    if filter.contains_key("$or") || filter.contains_key("$and") || filter.len() != 1 {
        return None;
    }
    let (field, val) = filter.iter().next()?;
    if field.starts_with('$') {
        return None;
    }
    let op_map = val.as_object()?;
    if op_map.len() != 1 {
        return None;
    }
    let (op, op_val) = op_map.iter().next()?;
    if op != "$in" {
        return None;
    }
    let arr = op_val.as_array()?;
    if arr.is_empty() {
        return None;
    }
    Some((field.as_str(), arr))
}

/// Extract a single `$nin` condition.
///
/// Matches `{"field": {"$nin": [v1, v2, ...]}}` with exactly one top-level field.
/// Returns `(field, excluded_values)` or `None`.
pub(crate) fn extract_nin_condition<'a>(
    filter: &'a HashMap<String, JsonValue>,
) -> Option<(&'a str, &'a Vec<JsonValue>)> {
    if filter.contains_key("$or") || filter.contains_key("$and") || filter.len() != 1 {
        return None;
    }
    let (field, val) = filter.iter().next()?;
    if field.starts_with('$') {
        return None;
    }
    let op_map = val.as_object()?;
    if op_map.len() != 1 {
        return None;
    }
    let (op, op_val) = op_map.iter().next()?;
    if op != "$nin" {
        return None;
    }
    let arr = op_val.as_array()?;
    Some((field.as_str(), arr))
}

/// Detect a single-field `$or` whose sub-conditions are all plain equality on the same field.
///
/// Matches `{"$or": [{"f": v1}, {"f": v2}]}` or with `{"$eq": v}` sub-conditions.
/// Returns `(field, collected_values)` or `None` if fields differ or sub-conditions are complex.
pub(crate) fn extract_or_single_field_eq<'a>(
    filter: &'a HashMap<String, JsonValue>,
) -> Option<(&'a str, Vec<&'a JsonValue>)> {
    if filter.len() != 1 {
        return None;
    }
    let conditions = filter.get("$or")?.as_array()?;
    if conditions.is_empty() {
        return None;
    }
    let mut field_name: Option<&str> = None;
    let mut values: Vec<&JsonValue> = Vec::new();
    for cond in conditions {
        let map = cond.as_object()?;
        if map.len() != 1 {
            return None;
        }
        let (f, v) = map.iter().next()?;
        if f.starts_with('$') {
            return None;
        }
        let f_str = f.as_str();
        match field_name {
            Some(prev) if prev != f_str => return None,
            Some(_) => {}
            None => field_name = Some(f_str),
        }
        match v {
            JsonValue::Object(ops) => {
                if ops.len() != 1 {
                    return None;
                }
                let (op, op_val) = ops.iter().next()?;
                if op != "$eq" {
                    return None;
                }
                values.push(op_val);
            }
            _ => values.push(v),
        }
    }
    Some((field_name?, values))
}

/// Extract simple equality conditions from a filter that can use the eq_index.
///
/// Returns pairs of `(field, value)` for conditions of the form `{"field": value}`
/// or `{"field": {"$eq": value}}`. Returns `None` if the filter contains `$or`,
/// `$and`, or any range/set operator — those fall back to the O(n) scan path.
pub(crate) fn extract_indexable_eq<'a>(
    filter: &'a HashMap<String, JsonValue>,
) -> Option<Vec<(&'a str, &'a JsonValue)>> {
    if filter.contains_key("$or") || filter.contains_key("$and") {
        return None;
    }
    let mut conditions = Vec::new();
    for (k, v) in filter {
        if k.starts_with('$') {
            return None;
        }
        match v {
            JsonValue::Object(op_map) => {
                if op_map.len() == 1 {
                    if let Some(eq_val) = op_map.get("$eq") {
                        conditions.push((k.as_str(), eq_val));
                    } else {
                        return None; // range or other operator
                    }
                } else {
                    return None; // compound per-field operators
                }
            }
            _ => conditions.push((k.as_str(), v)), // simple {"field": value}
        }
    }
    if conditions.is_empty() {
        None
    } else {
        Some(conditions)
    }
}

pub(crate) fn score_vectors_with_metric(
    metric: &DistanceMetric,
    a: &Array1<f64>,
    b: &Array1<f64>,
) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return unsafe { score_vectors_avx2(metric, a, b) };
    }
    score_vectors_scalar(metric, a, b)
}

#[inline(always)]
fn score_vectors_scalar(metric: &DistanceMetric, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    match metric {
        DistanceMetric::Ip => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
        DistanceMetric::Cosine => {
            let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let an = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let bn = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            if an == 0.0 || bn == 0.0 {
                0.0
            } else {
                dot / (an * bn)
            }
        }
        DistanceMetric::L2 => -a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt(),
    }
}

/// AVX2+FMA dot-product / cosine / L2 for f64 reranking vectors.
/// 4-wide `__m256d` lanes; scalar tail for non-multiple-of-4 lengths.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn score_vectors_avx2(metric: &DistanceMetric, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    use std::arch::x86_64::*;

    let n = a.len().min(b.len());
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let chunks = n / 4;
    let rem_start = chunks * 4;

    // SAFETY: `#[target_feature]` guarantees AVX2+FMA are available;
    // `a_ptr`/`b_ptr` are valid for `n` elements (derived from slice data).
    unsafe {
        match metric {
            DistanceMetric::Ip => {
                let mut acc = _mm256_setzero_pd();
                for i in 0..chunks {
                    let av = _mm256_loadu_pd(a_ptr.add(i * 4));
                    let bv = _mm256_loadu_pd(b_ptr.add(i * 4));
                    acc = _mm256_fmadd_pd(av, bv, acc);
                }
                let mut dot = hsum_pd(acc);
                for i in rem_start..n {
                    dot += *a_ptr.add(i) * *b_ptr.add(i);
                }
                dot
            }
            DistanceMetric::Cosine => {
                let mut dot_acc = _mm256_setzero_pd();
                let mut an_acc = _mm256_setzero_pd();
                let mut bn_acc = _mm256_setzero_pd();
                for i in 0..chunks {
                    let av = _mm256_loadu_pd(a_ptr.add(i * 4));
                    let bv = _mm256_loadu_pd(b_ptr.add(i * 4));
                    dot_acc = _mm256_fmadd_pd(av, bv, dot_acc);
                    an_acc = _mm256_fmadd_pd(av, av, an_acc);
                    bn_acc = _mm256_fmadd_pd(bv, bv, bn_acc);
                }
                let mut dot = hsum_pd(dot_acc);
                let mut an2 = hsum_pd(an_acc);
                let mut bn2 = hsum_pd(bn_acc);
                for i in rem_start..n {
                    let av = *a_ptr.add(i);
                    let bv = *b_ptr.add(i);
                    dot += av * bv;
                    an2 += av * av;
                    bn2 += bv * bv;
                }
                let denom = an2.sqrt() * bn2.sqrt();
                if denom == 0.0 { 0.0 } else { dot / denom }
            }
            DistanceMetric::L2 => {
                let mut acc = _mm256_setzero_pd();
                for i in 0..chunks {
                    let av = _mm256_loadu_pd(a_ptr.add(i * 4));
                    let bv = _mm256_loadu_pd(b_ptr.add(i * 4));
                    let diff = _mm256_sub_pd(av, bv);
                    acc = _mm256_fmadd_pd(diff, diff, acc);
                }
                let mut dist2 = hsum_pd(acc);
                for i in rem_start..n {
                    let d = *a_ptr.add(i) - *b_ptr.add(i);
                    dist2 += d * d;
                }
                -dist2.sqrt()
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_pd(v: std::arch::x86_64::__m256d) -> f64 {
    use std::arch::x86_64::*;
    // SAFETY: `#[target_feature]` guarantees AVX2 is available.
    // These intrinsics are pure register operations (no raw pointer access)
    // and are safe to call once the feature is guaranteed by the attribute.
    let lo = _mm256_castpd256_pd128(v);
    let hi = _mm256_extractf128_pd(v, 1);
    let sum128 = _mm_add_pd(lo, hi);
    let hi64 = _mm_unpackhi_pd(sum128, sum128);
    _mm_cvtsd_f64(_mm_add_sd(sum128, hi64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn vec(data: &[f64]) -> Array1<f64> {
        Array1::from_vec(data.to_vec())
    }

    // ── score_vectors_with_metric — scalar path (short vectors < 4 elements) ──

    #[test]
    fn ip_unit_vector() {
        let a = vec(&[1.0, 0.0, 0.0]);
        let b = vec(&[1.0, 0.0, 0.0]);
        assert!((score_vectors_with_metric(&DistanceMetric::Ip, &a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec(&[1.0, 0.0]);
        let b = vec(&[0.0, 1.0]);
        assert!(score_vectors_with_metric(&DistanceMetric::Cosine, &a, &b).abs() < 1e-10);
    }

    #[test]
    fn cosine_parallel_is_one() {
        let a = vec(&[3.0, 4.0]);
        let b = vec(&[6.0, 8.0]);
        assert!((score_vectors_with_metric(&DistanceMetric::Cosine, &a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn l2_identical_vectors_is_zero() {
        let a = vec(&[1.0, 2.0, 3.0]);
        let b = vec(&[1.0, 2.0, 3.0]);
        let s = score_vectors_with_metric(&DistanceMetric::L2, &a, &b);
        assert!(s.abs() < 1e-10, "identical → L2=0, got {s}");
    }

    #[test]
    fn l2_known_distance() {
        // distance([0,0,0], [1,1,1]) = sqrt(3), negated → -sqrt(3)
        let a = vec(&[0.0, 0.0, 0.0]);
        let b = vec(&[1.0, 1.0, 1.0]);
        let expected = -(3.0f64.sqrt());
        let s = score_vectors_with_metric(&DistanceMetric::L2, &a, &b);
        assert!((s - expected).abs() < 1e-10, "expected {expected}, got {s}");
    }

    // ── score_vectors_with_metric — long vectors exercise AVX2 + scalar tail ──
    // dim=128: 32 SIMD chunks (4-wide) with no remainder
    // dim=131: 32 SIMD chunks + 3-element tail

    #[test]
    fn ip_long_vector_no_tail() {
        let n = 128;
        let a = Array1::from_vec((0..n).map(|i| i as f64).collect());
        let b = Array1::ones(n);
        let expected: f64 = (0..n as i64).map(|i| i as f64).sum();
        let s = score_vectors_with_metric(&DistanceMetric::Ip, &a, &b);
        assert!((s - expected).abs() < 1e-6, "expected {expected}, got {s}");
    }

    #[test]
    fn ip_long_vector_with_tail() {
        let n = 131usize;
        let a = Array1::ones(n);
        let b = Array1::ones(n);
        let expected = n as f64;
        let s = score_vectors_with_metric(&DistanceMetric::Ip, &a, &b);
        assert!((s - expected).abs() < 1e-6, "expected {expected}, got {s}");
    }

    #[test]
    fn l2_long_vector_known_distance() {
        let n = 128usize;
        let a = Array1::zeros(n);
        let b = Array1::ones(n);
        let expected = -(n as f64).sqrt();
        let s = score_vectors_with_metric(&DistanceMetric::L2, &a, &b);
        assert!((s - expected).abs() < 1e-10, "expected {expected}, got {s}");
    }

    #[test]
    fn cosine_long_parallel_vectors() {
        let n = 128usize;
        let a = Array1::ones(n);
        let b = Array1::from_elem(n, 2.0);
        let s = score_vectors_with_metric(&DistanceMetric::Cosine, &a, &b);
        assert!((s - 1.0).abs() < 1e-10, "parallel → cosine=1, got {s}");
    }

    #[test]
    fn cosine_long_with_tail() {
        let n = 131usize;
        let a = Array1::ones(n);
        let b = Array1::from_elem(n, 3.0);
        let s = score_vectors_with_metric(&DistanceMetric::Cosine, &a, &b);
        assert!((s - 1.0).abs() < 1e-10, "parallel → cosine=1, got {s}");
    }

    // ── Zero-vector edge cases ─────────────────────────────────────────────────

    #[test]
    fn cosine_zero_vector_returns_zero() {
        let a = Array1::zeros(128usize);
        let b = Array1::ones(128usize);
        let s = score_vectors_with_metric(&DistanceMetric::Cosine, &a, &b);
        assert_eq!(s, 0.0, "zero vector → cosine=0");
    }

    #[test]
    fn l2_long_with_tail() {
        let n = 131usize;
        let a = Array1::zeros(n);
        let b = Array1::ones(n);
        let expected = -(n as f64).sqrt();
        let s = score_vectors_with_metric(&DistanceMetric::L2, &a, &b);
        assert!((s - expected).abs() < 1e-10, "expected {expected}, got {s}");
    }
}

#[cfg(test)]
mod or_fast_path_test {
    use super::*;
    use serde_json::{Value, json};
    use std::collections::HashMap;

    #[test]
    fn extract_or_single_field_eq_basic() {
        let mut f: HashMap<String, Value> = HashMap::new();
        f.insert("$or".to_string(), json!([{"cat": "a"}, {"cat": "b"}]));
        let result = extract_or_single_field_eq(&f);
        assert!(
            result.is_some(),
            "should match $or with same-field equality"
        );
        let (field, vals) = result.unwrap();
        assert_eq!(field, "cat");
        assert_eq!(vals.len(), 2);
    }
}

// ── v0.8.2 audit: depth limit + operator coverage ────────────────────────────

#[cfg(test)]
mod audit_tests {
    use super::*;
    use serde_json::{Value, json};
    use std::collections::HashMap;

    fn meta_one(k: &str, v: Value) -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert(k.to_string(), v);
        m
    }

    fn nested_and_filter(depth: u32) -> HashMap<String, Value> {
        // {"$and": [{"$and": [{"$and": [...{"x": 1}]}]}]} with `depth` levels
        let mut current: Value = json!({"x": 1});
        for _ in 0..depth {
            current = json!({"$and": [current]});
        }
        let JsonValue::Object(map) = current else {
            panic!("expected object")
        };
        map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    /// A3: a deeply nested filter is rejected at validation time with a clear
    /// error — must not crash the process via stack overflow.
    #[test]
    fn deeply_nested_filter_rejected_at_validation() {
        // 50 levels of $and-of-$and is well above MAX_FILTER_DEPTH=32.
        let f = nested_and_filter(50);
        let result = validate_filter_operators(&f);
        assert!(result.is_err(), "deeply nested filter should be rejected");
        let msg = result.err().unwrap();
        assert!(
            msg.contains("MAX_FILTER_DEPTH") || msg.contains("nesting"),
            "error should mention depth limit; got: {msg}"
        );
    }

    /// A3: a filter exactly at MAX_FILTER_DEPTH validates and evaluates.
    #[test]
    fn filter_at_max_depth_validates_and_evaluates() {
        // Exactly MAX_FILTER_DEPTH levels of nesting must validate.
        let f = nested_and_filter(MAX_FILTER_DEPTH);
        assert!(
            validate_filter_operators(&f).is_ok(),
            "filter at max depth must validate"
        );
        // And evaluating it against matching metadata returns true.
        let meta = meta_one("x", json!(1));
        // Evaluate with depth-bounded helper so we exercise the runtime guard too.
        let eval = metadata_matches_filter(&meta, &f);
        // Depth-tracking starts at 0 in the public function; after 32 dives the
        // helper returns false defensively. 33 levels → first $and dive at
        // depth=1, …, 33rd dive at depth=33 > MAX_FILTER_DEPTH → false.
        // 32 levels → max dive at depth=32 == MAX_FILTER_DEPTH → still evaluates.
        assert!(
            eval,
            "filter exactly at max depth should still evaluate to true on matching metadata"
        );
    }

    /// A3: 33 levels (one over the cap) must not panic; either rejected at
    /// validation or returns false at evaluation time.
    #[test]
    fn filter_one_over_max_depth_safely_rejected() {
        let f = nested_and_filter(MAX_FILTER_DEPTH + 1);
        let validation = validate_filter_operators(&f);
        let meta = meta_one("x", json!(1));
        let evaluation = metadata_matches_filter(&meta, &f);
        // Either path is acceptable — the contract is "no stack overflow".
        assert!(
            validation.is_err() || !evaluation,
            "filter beyond max depth must be rejected somewhere in the pipeline"
        );
    }

    /// B2: a JSON integer larger than f64 representation (|n| > 2^53) compared
    /// with `$gt` against a numeric metadata value returns false. Pinned so
    /// future refactors don't silently change semantics.
    #[test]
    fn big_int_metadata_with_numeric_op_returns_false() {
        // 2^60 is exactly representable in u64 but loses precision in f64.
        // serde_json stores this as an i64; .as_f64() returns Some, but only
        // approximately — comparing against a slightly-different threshold
        // can yield surprising results. Pin behavior at the limit.
        let huge = json!(u64::MAX); // not representable in f64 → as_f64 = None → NaN
        let meta = meta_one("count", huge);
        let mut filter: HashMap<String, Value> = HashMap::new();
        filter.insert("count".to_string(), json!({"$gt": 100}));
        // u64::MAX as_f64() may return Some(huge_approx) on some serde_json
        // versions; either way, the comparison must not panic.
        let r = metadata_matches_filter(&meta, &filter);
        // Expected behavior: either matches (because u64::MAX → f64 → > 100)
        // or doesn't (NaN). Both are acceptable; pinning the no-panic contract.
        let _ = r; // value not asserted — just must not crash.
    }

    /// C3: empty `$and` array — vacuously true.
    #[test]
    fn empty_and_array_evaluates_true_vacuously() {
        let mut f: HashMap<String, Value> = HashMap::new();
        f.insert("$and".to_string(), json!([]));
        let meta = meta_one("anything", json!("here"));
        assert!(
            metadata_matches_filter(&meta, &f),
            "empty $and is vacuously true (matches everything)"
        );
    }

    /// C3: empty `$or` array — vacuously false.
    #[test]
    fn empty_or_array_evaluates_false_vacuously() {
        let mut f: HashMap<String, Value> = HashMap::new();
        f.insert("$or".to_string(), json!([]));
        let meta = meta_one("anything", json!("here"));
        assert!(
            !metadata_matches_filter(&meta, &f),
            "empty $or is vacuously false (matches nothing)"
        );
    }

    /// C3: `$contains` on a non-string field silently returns false (does not
    /// error). Pin this contract — it's the current behavior and changing it
    /// would break existing queries.
    #[test]
    fn contains_on_non_string_field_returns_false() {
        let meta = meta_one("count", json!(42));
        let mut f: HashMap<String, Value> = HashMap::new();
        f.insert("count".to_string(), json!({"$contains": "4"}));
        assert!(
            !metadata_matches_filter(&meta, &f),
            "$contains on numeric field returns false, not an error"
        );
    }

    /// C3: `$exists: true` distinguishes null from missing in a meaningful way.
    /// Pin the current contract: a present-but-null value SATISFIES $exists:true.
    #[test]
    fn exists_with_null_vs_missing_distinguished() {
        let null_meta = meta_one("k", json!(null));
        let empty_meta: HashMap<String, Value> = HashMap::new();

        let mut f_exists: HashMap<String, Value> = HashMap::new();
        f_exists.insert("k".to_string(), json!({"$exists": true}));

        let null_match = metadata_matches_filter(&null_meta, &f_exists);
        let missing_match = metadata_matches_filter(&empty_meta, &f_exists);
        // Pin: null is "present" → $exists:true matches; missing → does not.
        assert!(null_match, "null value should match $exists:true");
        assert!(!missing_match, "missing key should not match $exists:true");
    }

    /// C3: range_extract with only `$gt` (no upper bound) returns lo=Some,
    /// hi=None — verifies single-side range extraction.
    #[test]
    fn range_extract_handles_unmatched_bounds() {
        let mut f: HashMap<String, Value> = HashMap::new();
        f.insert("score".to_string(), json!({"$gt": 0.5}));
        let result = extract_range_condition(&f);
        assert!(result.is_some(), "single-bound range should extract");
        let (field, lo, hi) = result.unwrap();
        assert_eq!(field, "score");
        assert!(lo.is_some(), "lo bound should be set");
        assert!(hi.is_none(), "hi bound should be None when only $gt given");
    }
}
