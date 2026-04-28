"""Cross-framework filter translators.

Each integration (LangChain, LlamaIndex, …) has its own filter syntax. TQDB
speaks one MongoDB-style dialect (``{field: {"$op": value}}``). These functions
translate from a framework's filter shape into the TQDB dialect.

Both translators accept ``None`` and pass it through unchanged; both raise
``ValueError`` on shapes they can't represent.
"""

from __future__ import annotations

from typing import Any, Optional


# ── LangChain → TQDB ───────────────────────────────────────────────────


def langchain_filter_to_mongo(filt: Any) -> Optional[dict]:
    """Translate a LangChain `filter` argument to TQDB MongoDB-style.

    LangChain doesn't standardise filter shapes — vendors vary widely. Two
    common patterns are accepted here:

    1. **Plain dict (``{"field": value}``)**: passed through unchanged. TQDB
       reads it as an implicit ``$eq`` per field.
    2. **MongoDB-style (``{"field": {"$op": value}}``)**: passed through
       unchanged.

    Any other shape (e.g. LangChain's ``StructuredFilter`` Pydantic model)
    raises ``ValueError`` so the caller can adapt.
    """
    if filt is None:
        return None
    if isinstance(filt, dict):
        return filt
    raise ValueError(
        f"unsupported LangChain filter type {type(filt).__name__}; "
        "pass a dict in MongoDB-style ({'field': {'$op': value}}) "
        "or simple equality ({'field': value})"
    )


# ── LlamaIndex → TQDB ──────────────────────────────────────────────────

_LLAMA_OP_MAP = {
    "==": "$eq",
    "!=": "$ne",
    ">": "$gt",
    ">=": "$gte",
    "<": "$lt",
    "<=": "$lte",
    "in": "$in",
    "nin": "$nin",
    "not in": "$nin",
    "contains": "$contains",
}


def _llama_filter_clause_to_mongo(f) -> dict:
    """Convert a single LlamaIndex MetadataFilter to a TQDB clause."""
    op = getattr(f, "operator", None)
    op_str = getattr(op, "value", op)  # FilterOperator enum or raw str
    op_str = str(op_str) if op_str is not None else "=="
    if op_str not in _LLAMA_OP_MAP:
        raise ValueError(
            f"unsupported LlamaIndex filter operator {op_str!r}; "
            f"supported: {sorted(_LLAMA_OP_MAP)}"
        )
    tqdb_op = _LLAMA_OP_MAP[op_str]
    field = getattr(f, "key", None) or getattr(f, "field", None)
    if not field:
        raise ValueError("LlamaIndex MetadataFilter is missing `key`/`field`")
    value = getattr(f, "value", None)
    return {field: {tqdb_op: value}}


def llama_index_filters_to_mongo(filters) -> Optional[dict]:
    """Translate LlamaIndex MetadataFilters into TQDB MongoDB-style.

    ``filters`` is either ``None`` or a ``MetadataFilters`` instance with
    ``.filters`` (list of ``MetadataFilter``) and ``.condition`` ("and"/"or").
    Nested ``MetadataFilters`` are supported recursively.
    """
    if filters is None:
        return None
    cond = getattr(filters, "condition", None)
    cond_str = str(getattr(cond, "value", cond) or "and").lower()
    children = getattr(filters, "filters", None)
    if children is None:
        # Bare MetadataFilter (some LlamaIndex paths pass one without a wrapper).
        return _llama_filter_clause_to_mongo(filters)

    clauses = []
    for child in children:
        if hasattr(child, "filters"):
            clauses.append(llama_index_filters_to_mongo(child))
        else:
            clauses.append(_llama_filter_clause_to_mongo(child))

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    if cond_str == "or":
        return {"$or": clauses}
    return {"$and": clauses}
