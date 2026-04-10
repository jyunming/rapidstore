"""
Fair SRHT vs. Exact Quantizer Comparison — Equal Codes, Power-of-Two Dimensions
=================================================================================
The only way to truly isolate rotation matrix quality (SRHT vs. QR) from code
count is to run both modes at dimensions d that are already powers of two.

When d = 2^k:
  - SRHT: next_power_of_two(d) = d → n = d codes, no padding
  - Exact: n = d codes (always)
  Both modes store the same number of MSE codes and QJL bits, use the same
  codebook, and produce the same disk footprint.  The ONLY difference is the
  rotation matrix applied before quantization:
    SRHT  — Walsh-Hadamard transform × random +-1 diagonal (O(d log d))
    Exact — Haar-random orthogonal via modified Gram-Schmidt QR (O(d²))

This script runs both modes on synthetic Gaussian vectors at d ∈ {128, 256,
512, 1024} across multiple bit widths, and reports recall@k, MRR, ingest speed,
and disk.  Because the code counts are identical, any recall difference is
attributable solely to the rotation matrix choice.

Usage:
    python benchmarks/compare_quantizers_fair.py
    python benchmarks/compare_quantizers_fair.py --dims 256 512
    python benchmarks/compare_quantizers_fair.py --bits 2 4 8
    python benchmarks/compare_quantizers_fair.py --save-json benchmarks/_fair_compare.json
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bench_core import compute_recalls, compute_mrr, disk_size_mb

import tqdb

# Power-of-two dimensions — SRHT never pads here so n=d for both modes.
DIMS_DEFAULT = [128, 256, 512, 1024]
BITS_DEFAULT = [2, 4, 8]
N_CORPUS     = 100_000
N_QUERIES    = 1_000
TOP_K        = 64
SEED         = 42


def assert_power_of_two(d: int) -> None:
    if d & (d - 1) != 0:
        raise ValueError(
            f"d={d} is not a power of two — SRHT would pad to {d.bit_length().__rshift__(0)} "
            f"and the comparison would be unfair. Use a power-of-two dimension."
        )


def make_dataset(d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic Gaussian corpus + queries.  Ground-truth top-1 = exact inner product
    on L2-normalised corpus (equivalent to cosine similarity).
    """
    corpus  = rng.standard_normal((N_CORPUS,  d)).astype("f4")
    queries = rng.standard_normal((N_QUERIES, d)).astype("f4")

    # Normalise corpus for ground-truth computation (inner product ≡ cosine).
    normed = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12)
    scores = queries @ normed.T          # (Q, N)
    true_top1 = np.argsort(-scores, axis=1)[:, 0].astype("int32")  # best match per query

    return corpus, queries, true_top1


def run_one(qt: str, corpus: np.ndarray, queries: np.ndarray,
            true_top1: np.ndarray, bits: int) -> dict:
    N, D = corpus.shape
    assert_power_of_two(D)

    ids = [str(i) for i in range(N)]
    with tempfile.TemporaryDirectory() as tmp:
        db = tqdb.Database.open(
            tmp, dimension=D, bits=bits, metric="ip",
            rerank=False, quantizer_type=qt,
        )
        t0 = time.perf_counter()
        batch = 5_000
        for s in range(0, N, batch):
            db.insert_batch(ids[s:s+batch], corpus[s:s+batch])
        db.flush()
        ingest_s = time.perf_counter() - t0

        dm = disk_size_mb(tmp)

        latencies: list[float] = []
        all_results: list[list[str]] = []
        for q in queries:
            t0 = time.perf_counter()
            hits = db.search(q, top_k=TOP_K)
            latencies.append(time.perf_counter() - t0)
            all_results.append([r["id"] for r in hits])

    recalls = compute_recalls(all_results, true_top1)
    mrr     = compute_mrr(all_results, true_top1)

    return {
        "d":          D,
        "bits":       bits,
        "qt":         qt,
        "r1":         recalls.get(1, 0.0),
        "r4":         recalls.get(4, 0.0),
        "r8":         recalls.get(8, 0.0),
        "mrr":        mrr,
        "disk_mb":    dm,
        "ingest_vps": N / ingest_s,
        "p50_ms":     float(np.median(latencies)) * 1000,
    }


def print_dim_results(d: int, bits_list: list[int],
                      srht_rows: list[dict], dense_rows: list[dict]) -> None:
    W = 84
    print(f"\n{'=' * W}")
    print(f"  d={d}  (n=d={d} for both modes — no SRHT padding)")
    print(f"{'=' * W}")
    print(f"  {'Bits':>4}  {'Mode':>6}  {'R@1':>7}  {'R@4':>7}  {'R@8':>7}  {'MRR':>7}"
          f"  {'Disk MB':>8}  {'Ingest vps':>11}")
    print(f"  {'-'*76}")

    srht_by_b  = {r["bits"]: r for r in srht_rows}
    exact_by_b = {r["bits"]: r for r in dense_rows}

    for b in sorted(bits_list):
        for qt, tbl in [("SRHT", srht_by_b), ("Exact", exact_by_b)]:
            p = tbl.get(b)
            if p is None:
                continue
            print(f"  {b:>4}  {qt:>6}  {p['r1']:>7.3f}  {p['r4']:>7.3f}  {p['r8']:>7.3f}"
                  f"  {p['mrr']:>7.4f}  {p['disk_mb']:>8.2f}  {p['ingest_vps']:>11,.0f}")

    # Direct delta (disk is identical so no interpolation needed)
    print(f"\n  Direct recall delta (Exact - SRHT, same disk, same n={d} codes):")
    print(f"  {'Bits':>4}  {'SRHT R@1':>9}  {'Exact R@1':>10}  {'D R@1':>8}"
          f"  {'SRHT MRR':>9}  {'Exact MRR':>10}  {'D MRR':>8}"
          f"  {'Speed ratio':>12}")
    print(f"  {'-'*90}")
    for b in sorted(bits_list):
        sp = srht_by_b.get(b)
        ep = exact_by_b.get(b)
        if sp is None or ep is None:
            continue
        dr1  = ep["r1"]  - sp["r1"]
        dmrr = ep["mrr"] - sp["mrr"]
        speed_ratio = sp["ingest_vps"] / max(ep["ingest_vps"], 1e-9)
        r1_arrow   = f"+{dr1:.3f}"  if dr1  > 0.002 else (f"{dr1:.3f}"  if dr1  < -0.002 else "~0.000")
        mrr_arrow  = f"+{dmrr:.4f}" if dmrr > 0.001 else (f"{dmrr:.4f}" if dmrr < -0.001 else "~0.000")
        print(f"  {b:>4}  {sp['r1']:>9.3f}  {ep['r1']:>10.3f}  {r1_arrow:>8}"
              f"  {sp['mrr']:>9.4f}  {ep['mrr']:>10.4f}  {mrr_arrow:>8}"
              f"  {speed_ratio:>10.1f}×")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fair SRHT vs exact comparison at equal code counts (power-of-two d)")
    ap.add_argument("--dims", nargs="+", type=int, default=DIMS_DEFAULT,
                    help=f"Dimensions to test — must be powers of two (default: {DIMS_DEFAULT})")
    ap.add_argument("--bits", nargs="+", type=int, default=BITS_DEFAULT,
                    help=f"Bit widths (default: {BITS_DEFAULT})")
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    for d in args.dims:
        assert_power_of_two(d)

    print("=" * 84)
    print("Fair SRHT vs. Exact Comparison — Power-of-Two Dimensions (n = d for both modes)")
    print("=" * 84)
    print(f"  dims: {args.dims}   bits: {args.bits}")
    print(f"  corpus: {N_CORPUS:,} synthetic Gaussian vectors per dimension")
    print(f"  queries: {N_QUERIES:,}  top_k={TOP_K}")
    print()
    print("  Since d is a power of two, SRHT uses n=d (no padding) and exact uses n=d.")
    print("  Both modes produce identical code counts, disk usage, and codebooks.")
    print("  The ONLY difference is the rotation matrix: SRHT (Walsh-Hadamard × +-1 diag)")
    print("  vs. Exact (Haar-uniform QR via modified Gram-Schmidt).")
    print("  Any recall difference is attributable solely to the rotation quality.")

    rng = np.random.default_rng(SEED)
    all_results: dict = {}

    for d in args.dims:
        print(f"\n-- d={d} ----------------------------------------------------------", flush=True)
        corpus, queries, true_top1 = make_dataset(d, rng)

        srht_rows:  list[dict] = []
        dense_rows: list[dict] = []

        for bits in args.bits:
            for qt, rows in [("srht", srht_rows), ("dense", dense_rows)]:
                print(f"  {qt:>5} b={bits} ...", end="", flush=True)
                res = run_one(qt, corpus, queries, true_top1, bits)
                rows.append(res)
                print(f"  R@1={res['r1']:.3f}  disk={res['disk_mb']:.2f} MB"
                      f"  ingest={res['ingest_vps']:,.0f} vps", flush=True)

        print_dim_results(d, args.bits, srht_rows, dense_rows)
        all_results[str(d)] = {"srht": srht_rows, "dense": dense_rows}

    print(f"\n{'=' * 84}")
    print("Summary")
    print(f"{'=' * 84}")
    print("""
  Positive D = exact better, Negative D = SRHT better, ~0 = indistinguishable.
  Speed ratio = SRHT ingest vps / Dense ingest vps.

  Since code counts are identical, this comparison directly measures whether
  the Walsh-Hadamard rotation (SRHT) is as effective as Haar-uniform QR as a
  mixing operator prior to scalar quantization.
""")

    if args.save_json:
        import json

        def _ser(rows):
            return [{k: float(v) if isinstance(v, (float, np.floating)) else v
                     for k, v in r.items()} for r in rows]

        out = {d: {"srht": _ser(v["srht"]), "dense": _ser(v["dense"])}
               for d, v in all_results.items()}
        Path(args.save_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  Results saved to {args.save_json}", flush=True)


if __name__ == "__main__":
    main()
