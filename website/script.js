document.addEventListener('DOMContentLoaded', () => {

    // ── Copy: pip install cmd ─────────────────────────────────────────────────
    window.copyInstallCmd = function() {
        const cmd = 'pip install tqdb';
        navigator.clipboard.writeText(cmd).then(() => {
            const btn = document.querySelector('.copy-btn[onclick="copyInstallCmd()"]');
            const orig = btn.innerHTML;
            btn.innerHTML = '<span style="font-size:14px;font-weight:bold;color:var(--accent-primary)">Copied!</span>';
            setTimeout(() => { btn.innerHTML = orig; }, 2000);
        });
    };

    // ── Scroll-triggered entrance animations ─────────────────────────────────
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.15, rootMargin: '0px 0px -60px 0px' });

    document.querySelectorAll('.fade-in-up, .fade-in-left, .fade-in-right')
        .forEach(el => observer.observe(el));

    // ── Version badge — GitHub releases API ──────────────────────────────────
    const navPill = document.getElementById('nav-version-pill');
    fetch('https://api.github.com/repos/jyunming/TurboQuantDB/releases/latest', {
        headers: { Accept: 'application/vnd.github+json' }
    })
    .then(r => r.ok ? r.json() : Promise.reject())
    .then(data => {
        if (navPill && data.tag_name) navPill.textContent = data.tag_name;
    })
    .catch(() => { /* keep static fallback */ });

    // ──────────────────────────────────────────────────────────────────────────
    // BENCHMARK DATA — paper bench numbers, n=100k brute fast_mode=True
    // Pulled from benchmarks/perf_history.json (latest v0.8.3 entry)
    // ──────────────────────────────────────────────────────────────────────────
    // ri4 = rerank with residual_int4 (v0.8.3 default for compression-first users).
    // Numbers from bench_p12c_residual_rust_v3.log + bench_p12a_residual_alldims.log.
    const benchData = {
        dbpedia1536: {
            name: 'DBpedia-1536',
            queries: '1,000',
            n: '100,000',
            rows: [
                { config: 'b=2 rerank=F',         r1: 0.837, r4: 0.993, p50: 5.62, disk: '46.8 MB',  paper: 0.895, dpaper: -0.058 },
                { config: 'b=2 rerank=ri4 (new)', r1: 0.975, r4: 0.998, p50: 5.95, disk: '124 MB',   paper: null, dpaper: null, ri4: true },
                { config: 'b=2 rerank=int8',      r1: 0.997, r4: 1.000, p50: 5.77, disk: '193.7 MB', paper: null, dpaper: null },
                { config: 'b=4 rerank=F',         r1: 0.958, r4: 1.000, p50: 8.28, disk: '83.4 MB',  paper: 0.970, dpaper: -0.012 },
                { config: 'b=4 rerank=ri4 (new)', r1: 0.995, r4: 1.000, p50: 8.36, disk: '181.1 MB', paper: null, dpaper: null, ri4: true, best: true },
                { config: 'b=4 rerank=int8',      r1: 0.997, r4: 1.000, p50: 8.03, disk: '230.3 MB', paper: null, dpaper: null },
            ]
        },
        dbpedia3072: {
            name: 'DBpedia-3072',
            queries: '1,000',
            n: '100,000',
            rows: [
                { config: 'b=2 rerank=F',         r1: 0.896, r4: 1.000, p50: 15.6, disk: '110.6 MB', paper: 0.905, dpaper: -0.009 },
                { config: 'b=2 rerank=ri4 (new)', r1: 0.980, r4: 0.999, p50: 16.4, disk: '258 MB',   paper: null, dpaper: null, ri4: true },
                { config: 'b=2 rerank=int8',      r1: 0.997, r4: 1.000, p50: 16.6, disk: '403.9 MB', paper: null, dpaper: null },
                { config: 'b=4 rerank=F',         r1: 0.963, r4: 1.000, p50: 21.0, disk: '183.8 MB', paper: 0.975, dpaper: -0.012 },
                { config: 'b=4 rerank=ri4 (new)', r1: 0.995, r4: 1.000, p50: 21.8, disk: '331 MB',   paper: null, dpaper: null, ri4: true, best: true },
                { config: 'b=4 rerank=int8',      r1: 0.997, r4: 1.000, p50: 22.3, disk: '477.1 MB', paper: null, dpaper: null },
            ]
        },
        glove200: {
            name: 'GloVe-200',
            queries: '10,000',
            n: '100,000',
            rows: [
                { config: 'b=2 rerank=F',         r1: 0.511, r4: 0.799, p50: 1.08, disk: '6.1 MB',  paper: 0.550, dpaper: -0.039 },
                { config: 'b=2 rerank=ri4 (new)', r1: 0.920, r4: 0.998, p50: 1.22, disk: '17.5 MB', paper: null, dpaper: null, ri4: true },
                { config: 'b=2 rerank=int8',      r1: 0.987, r4: 1.000, p50: 1.27, disk: '25.5 MB', paper: null, dpaper: null },
                { config: 'b=4 rerank=F',         r1: 0.819, r4: 0.991, p50: 1.14, disk: '10.8 MB', paper: 0.860, dpaper: -0.041 },
                { config: 'b=4 rerank=ri4 (new)', r1: 0.975, r4: 1.000, p50: 1.25, disk: '21.6 MB', paper: null, dpaper: null, ri4: true, best: true },
                { config: 'b=4 rerank=int8',      r1: 0.987, r4: 1.000, p50: 1.30, disk: '30.3 MB', paper: null, dpaper: null },
            ]
        }
    };

    function renderBench(key) {
        const data = benchData[key];
        if (!data) return;
        const tbody = document.getElementById('bench-tbody');
        tbody.innerHTML = data.rows.map(row => {
            const cls = [row.best && 'row-best', row.ri4 && 'row-ri4'].filter(Boolean).join(' ');
            return `<tr class="${cls}">
                <td class="config-cell">${row.config}</td>
                <td class="num" data-label="R@1">${row.r1.toFixed(3)}</td>
                <td class="num" data-label="R@4">${row.r4.toFixed(3)}</td>
                <td class="num" data-label="p50">${row.p50.toFixed(2)} ms</td>
                <td class="num" data-label="disk">${row.disk}</td>
            </tr>`;
        }).join('');
    }

    // Initial render
    renderBench('dbpedia1536');

    // Tabs
    document.querySelectorAll('.bench-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.bench-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            renderBench(tab.dataset.bench);
        });
    });

    // ──────────────────────────────────────────────────────────────────────────
    // CODE GALLERY — six tabs, each with code + explainer
    // ──────────────────────────────────────────────────────────────────────────
    const codeSamples = {
        quickstart: {
            title: 'examples/quickstart.py',
            code: `import numpy as np
from sentence_transformers import SentenceTransformer
from tqdb import Database

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dim = model.get_sentence_embedding_dimension()  # 384

db = Database.open("./my_db", dimension=dim,
                   bits=4, metric="ip", rerank=True)

docs = [
    ("rust",   "Rust uses ownership for memory safety."),
    ("python", "Python prioritizes readability."),
    ("vector", "A vector DB stores embeddings for k-NN."),
]
ids   = [d[0] for d in docs]
texts = [d[1] for d in docs]
db.insert_batch(
    ids,
    model.encode(texts, normalize_embeddings=True).astype("f4"),
    documents=texts,
)

q = model.encode("How to avoid memory bugs?",
                 normalize_embeddings=True).astype("f4")
for r in db.search(q, top_k=2):
    print(f"  [{r['score']:.3f}] {r['id']} — {r['document']}")`,
            explainer: "<p><strong>End-to-end semantic search in 20 lines.</strong> Real embeddings (sentence-transformers), real query, real result ranking. The output looks like:</p><pre style='background:#0f0f14;padding:10px;border-radius:4px;font-size:0.82rem;color:var(--accent-primary);margin-top:0.75rem'>[0.687] rust   — Rust uses ownership for memory safety.\n[0.298] vector — A vector DB stores embeddings for k-NN.</pre>",
            bullets: [
                ['01', '<strong>No <code>train()</code> step.</strong> Codebook is closed-form; quantize on insert.'],
                ['02', '<strong>Auto-discovers dimension</strong> from your embedder via <code>get_sentence_embedding_dimension()</code>.'],
                ['03', '<strong>L2-normalize</strong> embeddings (<code>normalize_embeddings=True</code>) so IP scores ≡ cosine.'],
                ['04', '<strong>Persistent</strong> — close + reopen the DB; codes survive on disk via WAL + segments.']
            ]
        },
        rag: {
            title: 'examples/rag.py',
            code: `from tqdb.rag import TurboQuantRetriever
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def embed(texts):
    return model.encode(texts, normalize_embeddings=True).astype("f4")

retriever = TurboQuantRetriever(
    db_path="./rag_db",
    dimension=model.get_sentence_embedding_dimension(),
    bits=4,
    rerank_precision="f16",        # higher precision, more disk
    embedding_function=embed,
)
retriever.add_texts(
    texts=["TurboQuant is a 2-stage quantizer.",
           "RAG grounds LLMs in retrieved passages."],
    ids=["1", "2"],
    metadatas=[{"year": 2025}, {"year": 2025}],
)

docs = retriever.similarity_search(
    query="What is RAG?",
    k=3,
)
context = "\\n".join(f"- {d.page_content}" for d in docs)
prompt = f"Use the context to answer.\\n{context}\\n\\nQuestion: ..."
# feed prompt to your LLM of choice`,
            explainer: "<p><strong>LangChain-style retriever wrapper.</strong> Pass an embedding function once; everything else flows through standard LangChain idioms (<code>add_texts</code>, <code>similarity_search</code>, <code>as_retriever()</code>).</p>",
            bullets: [
                ['01', '<strong>Native LangChain v2 / LlamaIndex</strong> via <code>tqdb[langchain]</code> / <code>tqdb[llamaindex]</code>.'],
                ['02', '<strong>Metadata filters</strong> — pass <code>filter={"year": {"$gte": 2024}}</code> to narrow before scoring.'],
                ['03', '<strong>Hybrid retrieval</strong> — pass <code>hybrid={"text": "your query"}</code> for BM25+dense fusion via RRF.'],
                ['04', '<strong>Async too</strong> — <code>tqdb.aio.AsyncDatabase</code> for FastAPI / Starlette services.']
            ]
        },
        hybrid: {
            title: 'examples/hybrid.py',
            code: `from tqdb import Database
import numpy as np

db = Database.open("./db", dimension=384,
                   bits=4, metric="ip", rerank=True)

# BM25 index builds incrementally from the document field — no train()
db.insert_batch(
    ids=["a", "b", "c"],
    vectors=embeddings,                 # (N, 384) float32
    documents=[
        "Error: WAL replay failed at frame 1024",
        "How does the Lloyd-Max codebook work?",
        "TurboQuant paper arXiv:2504.19874",
    ],
)

# Pure dense
results = db.search(query_vec, top_k=10)

# Hybrid: dense + BM25 fused via Reciprocal Rank Fusion
results = db.search(
    query_vec,
    top_k=10,
    hybrid={
        "text": "WAL replay error frame 1024",  # exact-match recovery
        "weight": 0.3,                          # sparse weight in fusion
        "rrf_k": 60,                            # standard RRF constant
    },
)`,
            explainer: "<p><strong>Recover keyword matches that pure-dense misses.</strong> Paper IDs, function names, error messages, code symbols — semantic embeddings dilute these signals; BM25 catches them. RRF fusion blends both.</p>",
            bullets: [
                ['01', '<strong>BM25 is incremental</strong> — index updates on every insert. No <code>build_text_index()</code> step.'],
                ['02', '<strong>Pure dense unchanged</strong> when <code>hybrid=</code> is omitted (zero overhead).'],
                ['03', '<strong>RRF over rank, not score</strong> — robust to score-magnitude differences between BM25 and dense.'],
                ['04', '<strong>Per-row hybrid in batch</strong> — <code>hybrid={"texts": [str, ...]}</code> for <code>db.query(...)</code>.']
            ]
        },
        async: {
            title: 'examples/async_api.py',
            code: `import asyncio
from tqdb.aio import AsyncDatabase

async def main():
    db = await AsyncDatabase.open(
        "./db", dimension=1536, bits=4,
        rerank=True, rerank_precision="residual_int4",
    )

    # All long-running ops are awaitable
    await db.insert("doc-1", vec, document="...")
    await db.insert_batch(ids, vecs, documents=docs)

    # Concurrent queries run in true parallel — Rust releases the GIL
    queries = [embed(q) for q in questions]
    results = await asyncio.gather(*[
        db.search(q, top_k=5) for q in queries
    ])

    await db.close()

asyncio.run(main())`,
            explainer: "<p><strong>Real parallelism for FastAPI / Starlette services.</strong> The Rust extension releases the GIL inside every long-running call, so concurrent <code>await db.search(...)</code> calls run in genuine parallel — not just interleaved.</p>",
            bullets: [
                ['01', '<strong>Pass <code>executor=</code></strong> to share a thread pool across multiple databases.'],
                ['02', '<strong>Awaitable wrappers</strong> for: insert, insert_batch, upsert, search, query, create_index, checkpoint, close.'],
                ['03', '<strong>No double-locking</strong> — internal <code>RwLock</code> permits concurrent reads.'],
                ['04', '<strong>Throughput scales with cores</strong> — measured 10× speedup on 8-core boxes for parallel <code>search()</code>.']
            ]
        },
        migrate: {
            title: 'examples/migrate_from_chroma.py',
            code: `# CLI — one-liner
$ pip install 'tqdb[migrate-chroma]'
$ python -m tqdb.migrate chroma /old/chroma_db /new/tqdb_db

# Or programmatic — verify the destination immediately
from tqdb import Database
from tqdb.migrate import migrate_chroma

n = migrate_chroma(
    src_path="./chroma_db",
    dst_path="./tqdb_db",
    bits=4,
    batch_size=1000,        # tune for your RAM
)
print(f"Migrated {n} vectors")

# Spot-check the destination
db = Database.open("./tqdb_db")
print(db.stats())                       # vector_count, disk_bytes, ...

for record in db.get_many(db.list_ids(limit=3)):
    print(record["id"], record["document"][:60])`,
            explainer: "<p><strong>Drop-in migration from Chroma or LanceDB.</strong> IDs, vectors, metadata, and document text preserved. Pick the bits parameter based on your recall vs disk preference; the Config Advisor scores both for your dataset.</p>",
            bullets: [
                ['01', '<strong>Streaming batches</strong> — <code>batch_size</code> controls RAM peak (default 1000; try 100 for low-RAM machines).'],
                ['02', '<strong>Per-collection</strong> — pass <code>--collection foo</code> to migrate only one Chroma collection.'],
                ['03', '<strong>LanceDB symmetric</strong> — <code>migrate_lancedb(src, dst, table_name="docs")</code>.'],
                ['04', '<strong>Re-runnable</strong> — append-only writes; no destructive replace of the destination.']
            ]
        },
        server: {
            title: 'tqdb-server (curl)',
            code: `# Launch the bundled Axum server (no extra install)
$ tqdb-server                  # listens on 127.0.0.1:8080

$ AUTH='Authorization: ApiKey dev-key'

# 1. Create a 3-dim collection (production: use 384/768/1536)
$ curl -X POST http://127.0.0.1:8080/v1/tenants/dev/databases/main/collections \\
    -H "$AUTH" -H 'Content-Type: application/json' \\
    -d '{"name":"docs","dimension":3,"bits":4}'

# 2. Insert vectors
$ curl -X POST .../docs/add \\
    -H "$AUTH" -H 'Content-Type: application/json' \\
    -d '{"ids":["a","b"],"embeddings":[[0.1,0.2,0.3],[0.4,0.5,0.6]],
         "metadatas":[{"src":"faq"},{"src":"blog"}]}'

# 3. Top-5 nearest neighbours
$ curl -X POST .../docs/query \\
    -H "$AUTH" -H 'Content-Type: application/json' \\
    -d '{"query_embeddings":[[0.1,0.2,0.3]],"n_results":5}'

# Snapshot to S3 / restore async
$ curl -X POST .../docs/snapshot -d '{"snapshot_name":"v1"}'
$ curl -X GET  /v1/jobs/<job_id>      # poll to "succeeded"`,
            explainer: "<p><strong>Optional Axum HTTP server bundled in the wheel.</strong> No extra install on Linux x86-64, macOS, or Windows. Multi-tenant, RBAC, async background jobs, snapshot/restore, Prometheus metrics.</p>",
            bullets: [
                ['01', '<strong>Bootstraps a dev API key</strong> on first launch (<code>dev-key</code>, tenant <code>dev</code>) — replace before production.'],
                ['02', '<strong>RBAC</strong> via API keys with scopes (read / write / admin) at tenant/database/collection level.'],
                ['03', '<strong>Async jobs</strong> — compaction, index build, snapshots, restores. Restart-safe with up to 3 retries.'],
                ['04', '<strong>Prometheus</strong> at <code>/metrics</code>; quotas at <code>/v1/tenants/{t}/databases/{d}/quota_usage</code>.']
            ]
        }
    };

    function renderCode(key) {
        const sample = codeSamples[key];
        if (!sample) return;
        document.getElementById('code-title').textContent = sample.title;
        document.getElementById('active-code').textContent = sample.code;
    }

    renderCode('quickstart');
    document.querySelectorAll('.code-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.code-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            renderCode(tab.dataset.code);
        });
    });

    // Copy code button (rebuilt for new layout)
    const copyCodeBtn = document.getElementById('copy-code-btn');
    if (copyCodeBtn) {
        copyCodeBtn.addEventListener('click', () => {
            const code = document.getElementById('active-code');
            navigator.clipboard.writeText(code.innerText).then(() => {
                const orig = copyCodeBtn.innerText;
                copyCodeBtn.innerText = 'Copied!';
                setTimeout(() => { copyCodeBtn.innerText = orig; }, 1500);
            });
        });
    }

    // ──────────────────────────────────────────────────────────────────────────
    // ANIMATED COUNTERS — bench callouts
    // ──────────────────────────────────────────────────────────────────────────
    const animateCounter = (el) => {
        const target = parseFloat(el.dataset.target);
        const dur = 1400;
        const start = performance.now();
        const startVal = 0;
        const tick = (now) => {
            const t = Math.min(1, (now - start) / dur);
            // easeOutExpo
            const eased = t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
            const val = startVal + (target - startVal) * eased;
            el.textContent = (target % 1 === 0) ? Math.round(val) : val.toFixed(1);
            if (t < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    };

    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach(e => {
            if (e.isIntersecting) {
                animateCounter(e.target);
                counterObserver.unobserve(e.target);
            }
        });
    }, { threshold: 0.4 });

    document.querySelectorAll('.bench-callout-counter').forEach(el => counterObserver.observe(el));

});
