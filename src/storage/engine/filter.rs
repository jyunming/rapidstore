use super::DistanceMetric;
use ndarray::Array1;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
pub(crate) fn metadata_matches_filter(
    meta: &HashMap<String, JsonValue>,
    filter: &HashMap<String, JsonValue>,
) -> bool {
    filter.iter().all(|(k, v)| match k.as_str() {
        "$and" => {
            if let JsonValue::Array(conditions) = v {
                conditions.iter().all(|cond| {
                    if let JsonValue::Object(map) = cond {
                        let as_hm: HashMap<String, JsonValue> =
                            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                        metadata_matches_filter(meta, &as_hm)
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
                        metadata_matches_filter(meta, &as_hm)
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

/// Recursively validate that all operator keys in `filter` are known.
/// Returns `Err(message)` on the first unknown operator encountered.
pub(crate) fn validate_filter_operators(filter: &HashMap<String, JsonValue>) -> Result<(), String> {
    validate_filter_operators_obj(filter.iter().map(|(k, v)| (k.as_str(), v)))
}

/// Internal helper that accepts a `serde_json::Map` reference to avoid cloning.
fn validate_filter_operators_inner(map: &serde_json::Map<String, JsonValue>) -> Result<(), String> {
    validate_filter_operators_obj(map.iter().map(|(k, v)| (k.as_str(), v)))
}

fn validate_filter_operators_obj<'a>(
    iter: impl Iterator<Item = (&'a str, &'a JsonValue)>,
) -> Result<(), String> {
    for (k, v) in iter {
        match k {
            "$and" | "$or" => {
                if let JsonValue::Array(conditions) = v {
                    for cond in conditions {
                        if let JsonValue::Object(map) = cond {
                            validate_filter_operators_inner(map)?;
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_pd(v: std::arch::x86_64::__m256d) -> f64 {
    use std::arch::x86_64::*;
    let lo = _mm256_castpd256_pd128(v);
    let hi = _mm256_extractf128_pd(v, 1);
    let sum128 = _mm_add_pd(lo, hi);
    let hi64 = _mm_unpackhi_pd(sum128, sum128);
    _mm_cvtsd_f64(_mm_add_sd(sum128, hi64))
}
