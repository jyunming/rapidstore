"""
Post-benchmark verification script.

Checks that _full_config_results.json has exactly 32 configs × 4 datasets
(128 entries total). Identifies any missing or errored entries and re-runs
just those configs.

Usage:
    python benchmarks/_verify_and_fill.py          # check + fill gaps
    python benchmarks/_verify_and_fill.py --check  # check only, no re-run
"""
import argparse
import json
import sys
from pathlib import Path

# Add benchmarks dir to path so we can import full_config_bench
sys.path.insert(0, str(Path(__file__).parent))

import full_config_bench as fcb
from bench_core import load_glove, load_dbpedia, load_arxiv768

EXPECTED_DATASETS = {
    "glove-200":    ("glove",       load_glove),
    "arxiv-768":    ("arxiv768",    load_arxiv768),
    "dbpedia-1536": ("dbpedia1536", lambda: load_dbpedia(1536)),
    "dbpedia-3072": ("dbpedia3072", lambda: load_dbpedia(3072)),
}
EXPECTED_CONFIGS = 32


def load_results() -> dict:
    if not fcb.RESULTS_PATH.exists():
        return {}
    return json.loads(fcb.RESULTS_PATH.read_text(encoding="utf-8"))


def config_key(bits, rerank, ann, fast_mode, qtype):
    return fcb.config_label(bits, rerank, ann, fast_mode, qtype)


def expected_labels() -> list[str]:
    return [config_key(*c) for c in fcb.FULL_CONFIGS]


def check(results: dict) -> dict[str, list[str]]:
    """Returns {dataset_label: [missing_config_labels]}."""
    expected = expected_labels()
    gaps: dict[str, list[str]] = {}
    for ds_label in EXPECTED_DATASETS:
        present = {
            r["label"]
            for r in results.get(ds_label, [])
            if "error" not in r
        }
        missing = [lbl for lbl in expected if lbl not in present]
        errored = [
            r["label"]
            for r in results.get(ds_label, [])
            if "error" in r
        ]
        all_gaps = list(dict.fromkeys(missing + errored))  # preserve order, dedup
        if all_gaps:
            gaps[ds_label] = all_gaps
    return gaps


def print_report(results: dict, gaps: dict[str, list[str]]) -> None:
    print("\n=== Benchmark completeness check ===\n")
    for ds_label in EXPECTED_DATASETS:
        n = len(results.get(ds_label, []))
        errors = sum(1 for r in results.get(ds_label, []) if "error" in r)
        g = len(gaps.get(ds_label, []))
        status = "OK" if g == 0 else f"MISSING {g}"
        err_str = f"  ({errors} errored)" if errors else ""
        print(f"  {ds_label:20s}  {n:2d}/32 configs  [{status}]{err_str}")
    total = sum(len(v) for v in results.values())
    total_gaps = sum(len(v) for v in gaps.values())
    print(f"\n  Total: {total}/128 entries, {total_gaps} gap(s)\n")


def fill_gaps(gaps: dict[str, list[str]], results: dict) -> None:
    """Re-run missing configs for each dataset."""
    # Build label → config tuple map
    label_to_cfg = {config_key(*c): c for c in fcb.FULL_CONFIGS}

    for ds_label, missing_labels in gaps.items():
        loader_key, loader_fn = EXPECTED_DATASETS[ds_label]
        print(f"\n{'='*72}")
        print(f"  Filling {len(missing_labels)} gap(s) for {ds_label}")
        print(f"{'='*72}")

        vecs, qvecs, true_top1 = loader_fn()

        existing = {r["label"]: r for r in results.get(ds_label, [])}
        new_results = []
        for i, lbl in enumerate(missing_labels, 1):
            cfg = label_to_cfg[lbl]
            bits, rerank, ann, fast_mode, qtype = cfg
            print(f"\n  [{i}/{len(missing_labels)}] {lbl}", flush=True)
            try:
                r = fcb.run_one_config(
                    vecs, qvecs, true_top1, bits, rerank, ann, fast_mode, qtype
                )
                new_results.append(r)
                r1 = r["recall"].get("1", 0)
                p50 = r.get("p50_ms", 0)
                print(f"    R@1={r1:.3f}  p50={p50:.1f}ms", flush=True)
            except Exception as exc:
                print(f"    ERROR: {exc}", flush=True)
                new_results.append({
                    "label": lbl,
                    "bits": bits,
                    "rerank": rerank,
                    "ann": ann,
                    "fast_mode": fast_mode,
                    "quantizer_type": qtype,
                    "error": str(exc),
                })

        # Merge: existing good entries + newly run entries
        merged = list(existing.values()) + new_results
        order = {config_key(*c): i for i, c in enumerate(fcb.FULL_CONFIGS)}
        merged.sort(key=lambda r: order.get(r["label"], 999))
        results[ds_label] = merged

        fcb.RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n  Saved — {ds_label} now has {len(merged)} entries.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Verify and fill benchmark gaps")
    parser.add_argument("--check", action="store_true", help="Check only, do not re-run")
    args = parser.parse_args()

    results = load_results()
    gaps = check(results)
    print_report(results, gaps)

    if not gaps:
        print("All 128 configs present and error-free. Nothing to do.\n")
        return

    if args.check:
        print(f"--check mode: {sum(len(v) for v in gaps.values())} gap(s) found. Exiting.\n")
        sys.exit(1)

    print("Filling gaps...\n")
    fill_gaps(gaps, results)

    # Final check
    results = load_results()
    gaps_after = check(results)
    print_report(results, gaps_after)
    if gaps_after:
        print(f"WARNING: {sum(len(v) for v in gaps_after.values())} gap(s) remain after fill.\n")
        sys.exit(1)
    else:
        print("All gaps filled successfully.\n")


if __name__ == "__main__":
    main()
