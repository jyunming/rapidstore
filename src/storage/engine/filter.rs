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

pub(crate) fn score_vectors_with_metric(
    metric: &DistanceMetric,
    a: &Array1<f64>,
    b: &Array1<f64>,
) -> f64 {
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
