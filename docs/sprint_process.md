# TurboQuantDB Sprint Process

GitHub Project: https://github.com/users/jyunming/projects/3

---

## Schedule (1 milestone per week)

| Sprint | Phase | Due |
|--------|-------|-----|
| Sprint 1 | v0.3 · Quality Foundation | Apr 12, 2026 |
| Sprint 2 | v0.4 · Index + Data Safety | Apr 19, 2026 |
| Sprint 3 | v0.5 · Ecosystem + Search Quality | Apr 26, 2026 |
| Sprint 4 | v0.6 · Performance + Structure | May 3, 2026 |
| Sprint 5 | v0.7–0.9 · Feature Expansion | May 10, 2026 |
| Sprint 6 | v1.0 · Stable Release | May 17, 2026 |

---

## Project field IDs (for CLI use)

```
Project ID:       PVT_kwHOBCfwF84BTy45
Status field:     PVTSSF_lAHOBCfwF84BTy45zhA_VKQ
Phase field:      PVTSSF_lAHOBCfwF84BTy45zhA_VMk
Due Date field:   PVTF_lAHOBCfwF84BTy45zhA_Vro
Sprint field:     PVTIF_lAHOBCfwF84BTy45zhA_X-U
```

Status option IDs:
```
Todo:        f75ad846
In Progress: 47fc9ee4
Done:        98236657
```

Sprint iteration IDs:
```
Sprint 1 (v0.3): 7116b8f1
Sprint 2 (v0.4): 0b780266
Sprint 3 (v0.5): 8a151f63
Sprint 4 (v0.6): fd4e787f
Sprint 5 (v0.7-0.9): bdb0ded6
Sprint 6 (v1.0):  598bdf16
```

---

## Weekly sprint workflow

### Monday — start sprint

1. Open the project board, filter `Sprint = Sprint N`
2. Pick the first item, move to **In Progress**

Via CLI:
```bash
GITHUB_TOKEN="" gh api graphql -f query='
  mutation {
    updateProjectV2ItemFieldValue(input: {
      projectId: "PVT_kwHOBCfwF84BTy45"
      itemId: "<item-id>"
      fieldId: "PVTSSF_lAHOBCfwF84BTy45zhA_VKQ"
      value: { singleSelectOptionId: "47fc9ee4" }
    }) { projectV2Item { id } }
  }'
```

### During the week

Items flow: **Todo → In Progress → Done**

Each item = one branch + one PR. When the PR merges, move item to Done.
Automate via: Project Settings → Workflows → "Item closed → set Status to Done".

### Convert a draft item to a real issue

```bash
# Create the issue on the repo
GITHUB_TOKEN="" gh issue create \
  --title "Wire QJL-Hamming HNSW navigation" \
  --body "..." \
  --repo jyunming/turboquantDB \
  --label "v0.3"

# Link it to the project
GITHUB_TOKEN="" gh project item-add 3 --owner jyunming --url <issue-url>
```

### Sunday — close sprint

1. All Done → archive the iteration in the project UI
2. Incomplete items → reassign to next sprint iteration
3. Tag the release:
```bash
git tag v0.3.0
GITHUB_TOKEN="" git push origin v0.3.0
```

---

## Roadmap v2 summary

### v0.3 · Quality Foundation ✅ Released as v0.3.0 (2026-04-07)
- ✅ RwLock poison-safe helpers
- ✅ Sprint smoke test
- ✅ Wire QJL-Hamming HNSW navigation (recall 0.164 → 0.831)
- ✅ Python type stubs (.pyi) — shipped in wheel
- ✅ WAL + segment CRC32 checksums (v5 WAL format)
- ✅ Database.open(path) — parameterless reopen from manifest.json
- ✅ delete_batch(where_filter=...) — filter-based bulk delete
- ✅ Automated perf CI with history (TQDB_TRACK=1 opt-in)
- 🔲 .unwrap() audit pass 1 (write-path + Python-reachable only) — deferred to v0.4

### v0.4 · Index + Data Safety
- 🔲 Delta index (brute-force overlay; search queries both; rebuild merges)
- 🔲 Parallel batch ingest via Rayon
- 🔲 Compaction crash recovery
- 🔲 .unwrap() audit pass 1 (deferred from v0.3)

### v0.5 · Ecosystem + Search Quality
- ✅ ChromaDB drop-in shim (tqdb.chroma_compat) — delivered early in v0.3
- ✅ LanceDB drop-in shim (tqdb.lancedb_compat) — delivered early in v0.3
- 🔲 True incremental HNSW (re-evaluate after v0.4)
- 🔲 LangChain v2 — full ABC: add_texts, similarity_search, similarity_search_with_score, from_texts
- 🔲 LlamaIndex integration
- 🔲 Filtered search pushdown into Rust
- 🔲 Async Python API (asyncio/FastAPI)

### v0.6 · Performance + Structure
- ✅ engine.rs decomposition — delivered early in v0.3 (PR #13)
- ✅ Cloud storage backend for server (S3/GCS) — delivered early in v0.3
- 🔲 SIMD distance kernels (AVX2, 4–8× brute-force throughput)
- 🔲 Defensive .unwrap() cleanup (deferred from v0.3)

### v0.7–0.9 · Feature Expansion
- ✅ Prometheus /metrics endpoint — delivered early in v0.3
- ✅ Server restore endpoint — delivered early in v0.3
- 🔲 Hybrid search (BM25 + dense vector fusion)
- 🔲 Multi-vector documents (ColBERT-style late interaction)
- 🔲 Metadata field indexing (fast pre-filter)
- 🔲 Minimum server auth — API-key gating on all endpoints (v0.9)

### v1.0 · Stable Release
- 🔲 API freeze
- 🔲 ≥80% test coverage
- 🔲 Automated perf CI with full benchmark history
- 🔲 Complete docs + 0.x → 1.0 migration guide
