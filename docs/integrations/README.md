# Integrations

Native wrappers around `tqdb.Database` for popular RAG and vector-store frameworks.
Each integration's third-party dep is **optional** — install only the extras you
actually use.

| Framework | Module | Doc | Install |
|---|---|---|---|
| LangChain v2 | `tqdb.vectorstore.TurboQuantVectorStore` | [langchain.md](langchain.md) | `pip install 'tqdb[langchain]'` |
| LlamaIndex | `tqdb.llama_index.TurboQuantVectorStore` | [llama_index.md](llama_index.md) | `pip install 'tqdb[llamaindex]'` |

The integration modules import their third-party libraries lazily, so users on
plain `pip install tqdb` don't pay the import cost until they touch the relevant
module.

## Related guides

These aren't strictly "integrations" but cover related v0.8 surfaces and live
one directory up:

- **Multi-vector / ColBERT** — late-interaction retrieval with MaxSim scoring.
  See [`../MULTI_VECTOR.md`](../MULTI_VECTOR.md).
- **Async API** — `AsyncDatabase`, the asyncio facade. See
  [`../PYTHON_API.md` — Async API](../PYTHON_API.md#async-api).
- **Migration toolkit** — one-command import from existing Chroma / LanceDB
  collections. See [`../MIGRATION.md`](../MIGRATION.md).

## Version compatibility

The supported version range for each third-party library lives in
[`../COMPATIBILITY_MATRIX.md`](../COMPATIBILITY_MATRIX.md) (Integration Version
Matrix section). The same versions are pinned in
[`../../pyproject.toml`](../../pyproject.toml) under `[project.optional-dependencies]`.

## Reporting integration bugs

The filter translator at
[`python/tqdb/_filter_translator.py`](https://github.com/jyunming/TurboQuantDB/blob/main/python/tqdb/_filter_translator.py)
is the most common source of integration friction (each framework has its own
filter shape). When you hit an unsupported filter, the translator raises
`ValueError` naming the unsupported operator/key — include that in the bug
report and link the framework's filter snippet that failed.
