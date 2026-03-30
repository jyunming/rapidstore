import argparse
import gc
import json
import math
import os
import random
import shutil
import statistics
import string
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
import lancedb
import numpy as np
import psutil
import turboquantdb
from sentence_transformers import SentenceTransformer

MB = 1024 * 1024


@dataclass
class Scenario:
    mode: str
    total_mb: int
    file_count: int

    @property
    def name(self) -> str:
        return f"{self.mode}_{self.total_mb}mb_{self.file_count}files"


class MemorySampler:
    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = interval_sec
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = None
        self.peak_rss = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def start(self) -> None:
        self.peak_rss = self._proc.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak_rss



def resolve_embedding_model(model_name: str, hf_home: Path) -> str:
    if model_name != "BAAI/bge-large-en-v1.5":
        return model_name

    candidates = [
        hf_home / "transformers" / "models--BAAI--bge-large-en-v1.5" / "snapshots",
        hf_home / "hub" / "models--BAAI--bge-large-en-v1.5" / "snapshots",
    ]
    for base in candidates:
        if not base.exists():
            continue
        snaps = [p for p in base.iterdir() if p.is_dir()]
        snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in snaps:
            cfg = snap / "config.json"
            if cfg.exists():
                return str(snap)
    return model_name
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def random_word(rng: random.Random, min_len: int = 3, max_len: int = 12) -> str:
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def synth_paragraph(rng: random.Random, idx: int) -> str:
    topics = [
        "retrieval augmented generation",
        "vector indexing",
        "quantization pipeline",
        "metadata filtering",
        "storage compaction",
        "approximate nearest neighbors",
    ]
    topic = topics[idx % len(topics)]
    words = " ".join(random_word(rng) for _ in range(80))
    return (
        f"Document {idx}. This section discusses {topic} and practical tradeoffs. "
        f"It includes synthetic yet readable content for benchmarking. {words}.\n"
    )


def generate_file(path: Path, target_bytes: int, rng: random.Random) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    idx = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        while written < target_bytes:
            p = synth_paragraph(rng, idx)
            if idx % 11 == 0:
                p += "MIXED_BLOCK " + " ".join(random_word(rng) for _ in range(32)) + "\n"
            f.write(p)
            written += len(p.encode("utf-8"))
            idx += 1


def build_dataset_files(base_dir: Path, scenario: Scenario, seed: int) -> List[Path]:
    ds_dir = base_dir / scenario.name
    ensure_clean_dir(ds_dir)
    total_bytes = scenario.total_mb * MB
    per_file = total_bytes // scenario.file_count
    rem = total_bytes % scenario.file_count
    rng = random.Random(seed + scenario.total_mb * 1000 + scenario.file_count)
    files: List[Path] = []
    for i in range(scenario.file_count):
        size = per_file + (1 if i < rem else 0)
        fp = ds_dir / f"part_{i:04d}.txt"
        generate_file(fp, size, rng)
        files.append(fp)
    return files


def make_chunks_from_files(
    files: List[Path],
    chunk_chars: int,
    sample_stride_bytes: int,
) -> Tuple[List[str], List[str], List[dict]]:
    ids: List[str] = []
    texts: List[str] = []
    metas: List[dict] = []
    idx = 0
    for fp in files:
        raw = fp.read_bytes()
        if not raw:
            continue
        offsets = [0] if len(raw) <= sample_stride_bytes else list(range(0, len(raw), sample_stride_bytes))
        for off in offsets:
            end = min(len(raw), off + chunk_chars * 4)
            chunk = raw[off:end].decode("utf-8", errors="ignore").strip()
            if not chunk:
                continue
            if len(chunk) > chunk_chars:
                chunk = chunk[:chunk_chars]
            ids.append(f"doc_{idx:08d}")
            texts.append(chunk)
            metas.append({"source": fp.name, "offset": int(off)})
            idx += 1
    return ids, texts, metas


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embs, dtype=np.float32)


def build_queries(rng: random.Random, texts: List[str], embeddings: np.ndarray, n_queries: int) -> Tuple[List[str], np.ndarray]:
    n = len(texts)
    if n == 0:
        return [], np.zeros((0, embeddings.shape[1]), dtype=np.float32)
    take = min(n_queries, n)
    idxs = rng.sample(range(n), take)
    q_texts = [texts[i] for i in idxs]
    q_embs = embeddings[idxs].copy()
    while len(q_texts) < n_queries:
        j = rng.randrange(n)
        q_texts.append(texts[j])
        q_embs = np.vstack([q_embs, embeddings[j : j + 1]])
    return q_texts, q_embs


def summarize_times(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    vals_ms = [v * 1000.0 for v in values]
    vals_ms_sorted = sorted(vals_ms)
    p50 = statistics.median(vals_ms_sorted)
    p95 = vals_ms_sorted[min(len(vals_ms_sorted) - 1, math.floor(0.95 * (len(vals_ms_sorted) - 1)))]
    return {
        "avg_ms": float(sum(vals_ms) / len(vals_ms)),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
    }


def bench_turboquant(db_dir: Path, ids: List[str], vectors: np.ndarray, texts: List[str], metas: List[dict], query_vecs: np.ndarray, top_k: int, batch_size: int, bits: int) -> Dict[str, object]:
    ensure_clean_dir(db_dir)
    proc = psutil.Process(os.getpid())
    d = int(vectors.shape[1])
    try:
        db = turboquantdb.Database.open(str(db_dir), d, bits, 42, None, "ip")
    except TypeError:
        db = turboquantdb.Database.open(str(db_dir), d, bits, 42)

    vectors64 = np.asarray(vectors, dtype=np.float64)
    queries64 = np.asarray(query_vecs, dtype=np.float64)

    baseline_write = proc.memory_info().rss
    ms_write = MemorySampler()
    ms_write.start()
    t0 = time.perf_counter()
    for s in range(0, len(ids), batch_size):
        e = min(len(ids), s + batch_size)
        batch_ids = ids[s:e]
        batch_vecs = np.ascontiguousarray(vectors64[s:e], dtype=np.float64)
        batch_metas = metas[s:e]
        batch_docs = texts[s:e]
        if hasattr(db, "insert_batch"):
            db.insert_batch(batch_ids, batch_vecs, batch_metas, batch_docs, "insert")
        elif hasattr(db, "insert_many"):
            db.insert_many(
                batch_ids,
                [row for row in batch_vecs],
                batch_metas,
                batch_docs,
                "insert",
            )
        else:
            for i in range(s, e):
                db.insert(ids[i], vectors64[i], metas[i] if i < len(metas) else None)
    db.flush()
    index_build_s = 0.0
    if hasattr(db, "create_index"):
        t_index0 = time.perf_counter()
        db.create_index(32, 128)
        index_build_s = time.perf_counter() - t_index0
    write_s = time.perf_counter() - t0
    peak_write = ms_write.stop()

    baseline_read = proc.memory_info().rss
    ms_read = MemorySampler()
    ms_read.start()
    q_lat: List[float] = []
    for q in queries64:
        t1 = time.perf_counter()
        try:
            _ = db.search(q, top_k, None, True, 128)
        except TypeError:
            _ = db.search(q, top_k)
        q_lat.append(time.perf_counter() - t1)
    read_total_s = sum(q_lat)
    peak_read = ms_read.stop()

    stats = db.stats() if hasattr(db, "stats") else {}
    db.flush()
    db.close()
    gc.collect()

    return {
        "write_time_s": float(write_s),
        "index_build_time_s": float(index_build_s),
        "read_total_time_s": float(read_total_s),
        "read_latency": summarize_times(q_lat),
        "db_size_bytes": int(dir_size_bytes(db_dir)),
        "write_peak_rss_mb": float((peak_write - baseline_write) / MB),
        "read_peak_rss_mb": float((peak_read - baseline_read) / MB),
        "has_index": bool(stats.get("has_index", False)),
        "index_nodes": int(stats.get("index_nodes", 0) or 0),
    }

def bench_chromadb(db_dir: Path, ids: List[str], vectors: np.ndarray, texts: List[str], metas: List[dict], query_vecs: np.ndarray, top_k: int, batch_size: int) -> Dict[str, object]:
    ensure_clean_dir(db_dir)
    proc = psutil.Process(os.getpid())

    client = chromadb.PersistentClient(path=str(db_dir))
    col = client.get_or_create_collection(name="bench", metadata={"hnsw:space": "cosine"})

    baseline_write = proc.memory_info().rss
    ms_write = MemorySampler()
    ms_write.start()
    t0 = time.perf_counter()
    for s in range(0, len(ids), batch_size):
        e = min(len(ids), s + batch_size)
        col.add(ids=ids[s:e], embeddings=[v.tolist() for v in vectors[s:e]], metadatas=metas[s:e], documents=texts[s:e])
    write_s = time.perf_counter() - t0
    peak_write = ms_write.stop()

    baseline_read = proc.memory_info().rss
    ms_read = MemorySampler()
    ms_read.start()
    q_lat: List[float] = []
    for q in query_vecs:
        t1 = time.perf_counter()
        _ = col.query(query_embeddings=[q.tolist()], n_results=top_k)
        q_lat.append(time.perf_counter() - t1)
    read_total_s = sum(q_lat)
    peak_read = ms_read.stop()

    del col
    del client
    gc.collect()

    return {
        "write_time_s": float(write_s),
        "read_total_time_s": float(read_total_s),
        "read_latency": summarize_times(q_lat),
        "db_size_bytes": int(dir_size_bytes(db_dir)),
        "write_peak_rss_mb": float((peak_write - baseline_write) / MB),
        "read_peak_rss_mb": float((peak_read - baseline_read) / MB),
    }


def bench_lancedb(db_dir: Path, ids: List[str], vectors: np.ndarray, texts: List[str], metas: List[dict], query_vecs: np.ndarray, top_k: int, batch_size: int) -> Dict[str, object]:
    ensure_clean_dir(db_dir)
    proc = psutil.Process(os.getpid())

    db = lancedb.connect(str(db_dir))
    table = None

    baseline_write = proc.memory_info().rss
    ms_write = MemorySampler()
    ms_write.start()
    t0 = time.perf_counter()
    for s in range(0, len(ids), batch_size):
        e = min(len(ids), s + batch_size)
        rows = [
            {
                "id": ids[i],
                "vector": vectors[i].tolist(),
                "text": texts[i],
                "source": metas[i].get("source", ""),
                "offset": int(metas[i].get("offset", 0)),
            }
            for i in range(s, e)
        ]
        if table is None:
            table = db.create_table("bench", data=rows, mode="overwrite")
        else:
            table.add(rows)
    write_s = time.perf_counter() - t0
    peak_write = ms_write.stop()

    baseline_read = proc.memory_info().rss
    ms_read = MemorySampler()
    ms_read.start()
    q_lat: List[float] = []
    for q in query_vecs:
        t1 = time.perf_counter()
        _ = table.search(q.tolist()).metric("cosine").limit(top_k).to_list()
        q_lat.append(time.perf_counter() - t1)
    read_total_s = sum(q_lat)
    peak_read = ms_read.stop()

    del table
    del db
    gc.collect()

    return {
        "write_time_s": float(write_s),
        "read_total_time_s": float(read_total_s),
        "read_latency": summarize_times(q_lat),
        "db_size_bytes": int(dir_size_bytes(db_dir)),
        "write_peak_rss_mb": float((peak_write - baseline_write) / MB),
        "read_peak_rss_mb": float((peak_read - baseline_read) / MB),
    }


def build_scenarios() -> List[Scenario]:
    sizes = [1, 10, 20, 50]
    return [
        Scenario("single", mb, 1) for mb in sizes
    ] + [
        Scenario("multi", n, n) for n in sizes
    ]


def load_existing_runs(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("runs", [])


def save_runs(path: Path, runs: List[Dict[str, object]]) -> None:
    path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")


def upsert_run(runs: List[Dict[str, object]], new_run: Dict[str, object]) -> List[Dict[str, object]]:
    out = [r for r in runs if r.get("scenario") != new_run.get("scenario")]
    out.append(new_run)
    ordered = {sc.name: i for i, sc in enumerate(build_scenarios())}
    out.sort(key=lambda r: ordered.get(r.get("scenario", ""), 999))
    return out


def run(args: argparse.Namespace) -> Dict[str, object]:
    hf_home = Path(args.hf_home).resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    random.seed(args.seed)
    np.random.seed(args.seed)

    base = Path(args.work_dir).resolve()
    datasets_root = base / "datasets"
    vector_root = base / "vectorstores"
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "vectorstore_compare_results.json"
    runs = load_existing_runs(result_path)

    model_ref = resolve_embedding_model(args.embedding_model, hf_home)
    model = SentenceTransformer(model_ref, cache_folder=str(hf_home), local_files_only=True)

    scenarios = build_scenarios()
    if args.scenario:
        scenarios = [s for s in scenarios if s.name == args.scenario]
        if not scenarios:
            raise ValueError(f"Unknown scenario: {args.scenario}")

    for sc in scenarios:
        if args.skip_done and any(r.get("scenario") == sc.name for r in runs):
            print(f"Skipping completed scenario: {sc.name}")
            continue

        print(f"\n=== Scenario: {sc.name} ===", flush=True)
        files = build_dataset_files(datasets_root, sc, args.seed)
        data_size_bytes = sum(p.stat().st_size for p in files)
        ids, texts, metas = make_chunks_from_files(files, args.chunk_chars, args.sample_stride_bytes)
        print(f"files={len(files)} data_size_mb={data_size_bytes/MB:.2f} chunks={len(texts)} model={args.embedding_model}", flush=True)

        t_embed0 = time.perf_counter()
        vectors = encode_texts(model, texts, args.embed_batch_size)
        embed_s = time.perf_counter() - t_embed0
        dim = int(vectors.shape[1]) if len(vectors) else 0
        print(f"embedding_time_s={embed_s:.2f} dim={dim}", flush=True)

        _, q_vecs = build_queries(random.Random(args.seed + sc.total_mb), texts, vectors, args.query_count)
        if len(q_vecs) != args.query_count:
            raise RuntimeError("failed to build required query set")

        run_item = {
            "scenario": sc.name,
            "mode": sc.mode,
            "total_mb": sc.total_mb,
            "file_count": sc.file_count,
            "source_data_bytes": int(data_size_bytes),
            "chunk_count": len(texts),
            "embedding_model": args.embedding_model,
            "embedding_dim": dim,
            "embedding_time_s": float(embed_s),
            "query_count": args.query_count,
            "top_k": args.top_k,
            "batch_size": args.write_batch_size,
            "chunk_chars": args.chunk_chars,
            "sample_stride_bytes": args.sample_stride_bytes,
            "started_at": now_iso(),
            "stores": {},
        }

        store_paths = {
            "turboquantdb": vector_root / sc.name / "turboquantdb",
            "chromadb": vector_root / sc.name / "chromadb",
            "lancedb": vector_root / sc.name / "lancedb",
        }

        print("running turboquantdb...", flush=True)
        run_item["stores"]["turboquantdb"] = bench_turboquant(
            store_paths["turboquantdb"], ids, vectors, texts, metas, q_vecs, args.top_k, args.write_batch_size, args.turbo_bits
        )

        print("running chromadb...", flush=True)
        run_item["stores"]["chromadb"] = bench_chromadb(
            store_paths["chromadb"], ids, vectors, texts, metas, q_vecs, args.top_k, args.write_batch_size
        )

        print("running lancedb...", flush=True)
        run_item["stores"]["lancedb"] = bench_lancedb(
            store_paths["lancedb"], ids, vectors, texts, metas, q_vecs, args.top_k, args.write_batch_size
        )

        run_item["ended_at"] = now_iso()
        runs = upsert_run(runs, run_item)
        save_runs(result_path, runs)
        print(f"completed scenario: {sc.name}", flush=True)
        gc.collect()

    return {"runs": runs}


def write_csv(report: Dict[str, object], csv_path: Path) -> None:
    header = "scenario,store,mode,total_mb,file_count,source_data_mb,chunk_count,embedding_time_s,write_time_s,read_total_time_s,read_avg_ms,read_p50_ms,read_p95_ms,db_size_mb,write_peak_rss_mb,read_peak_rss_mb"
    lines = [header]
    for run in report["runs"]:
        for store, metrics in run["stores"].items():
            lines.append(
                ",".join(
                    [
                        run["scenario"], store, run["mode"], str(run["total_mb"]), str(run["file_count"]),
                        f"{run['source_data_bytes'] / MB:.4f}", str(run["chunk_count"]), f"{run['embedding_time_s']:.6f}",
                        f"{metrics['write_time_s']:.6f}", f"{metrics['read_total_time_s']:.6f}",
                        f"{metrics['read_latency']['avg_ms']:.6f}", f"{metrics['read_latency']['p50_ms']:.6f}",
                        f"{metrics['read_latency']['p95_ms']:.6f}", f"{metrics['db_size_bytes'] / MB:.6f}",
                        f"{metrics['write_peak_rss_mb']:.6f}", f"{metrics['read_peak_rss_mb']:.6f}",
                    ]
                )
            )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def print_human_summary(report: Dict[str, object]) -> None:
    runs = report.get("runs", [])
    if not runs:
        return

    def avg_metric(store: str, key: str, subkey: str = "") -> float:
        vals = []
        for run in runs:
            m = run["stores"][store]
            v = m[key] if not subkey else m[key][subkey]
            vals.append(float(v))
        return sum(vals) / max(1, len(vals))

    stores = ["turboquantdb", "lancedb", "chromadb"]
    ingest = {s: avg_metric(s, "write_time_s") for s in stores}
    ingest_vps = {s: (sum(r["chunk_count"] for r in runs) / ingest[s]) if ingest[s] > 0 else 0.0 for s in stores}
    disk = {s: avg_metric(s, "db_size_bytes") for s in stores}
    search = {s: avg_metric(s, "read_latency", "p50_ms") for s in stores}
    ram = {s: avg_metric(s, "read_peak_rss_mb") for s in stores}

    def win_max(d: Dict[str, float]) -> str:
        return max(d.items(), key=lambda kv: kv[1])[0]

    def win_min(d: Dict[str, float]) -> str:
        return min(d.items(), key=lambda kv: kv[1])[0]

    print("\nSummary (avg across scenarios):")
    print(f"Ingest speed  : tqdb={ingest_vps['turboquantdb']:.0f} v/s, lancedb={ingest_vps['lancedb']:.0f} v/s, chromadb={ingest_vps['chromadb']:.0f} v/s | winner={win_max(ingest_vps)}")
    print(f"Disk footprint: tqdb={disk['turboquantdb']/MB:.2f} MB, lancedb={disk['lancedb']/MB:.2f} MB, chromadb={disk['chromadb']/MB:.2f} MB | winner={win_min(disk)}")
    print(f"Search p50    : tqdb={search['turboquantdb']:.2f} ms, lancedb={search['lancedb']:.2f} ms, chromadb={search['chromadb']:.2f} ms | winner={win_min(search)}")
    print(f"RAM usage     : tqdb={ram['turboquantdb']:.2f} MB, lancedb={ram['lancedb']:.2f} MB, chromadb={ram['chromadb']:.2f} MB | winner={win_min(ram)}")

def main() -> None:
    p = argparse.ArgumentParser(description="Compare TurboQuantDB vs ChromaDB vs LanceDB")
    p.add_argument("--work-dir", default="benchmarks/store_compare_work")
    p.add_argument("--output-dir", default="benchmarks/artifacts")
    p.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--embed-batch-size", type=int, default=32)
    p.add_argument("--query-count", type=int, default=100)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--write-batch-size", type=int, default=256)
    p.add_argument("--chunk-chars", type=int, default=2048)
    p.add_argument("--sample-stride-bytes", type=int, default=131072)
    p.add_argument("--turbo-bits", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-home", default=".hf_home")
    p.add_argument("--scenario", default="", help="Run one scenario by name, e.g. single_100mb_1files")
    p.add_argument("--skip-done", action="store_true", help="Skip scenarios already present in vectorstore_compare_results.json")
    args = p.parse_args()

    report = run(args)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path(args.output_dir).resolve()
    out_json = out_dir / f"store_compare_{ts}.json"
    out_csv = out_dir / f"store_compare_{ts}.csv"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(report, out_csv)
    print(f"wrote: {out_json}")
    print(f"wrote: {out_csv}")
    print_human_summary(report)


if __name__ == "__main__":
    main()











