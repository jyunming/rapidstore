"""
Build website/data/advisor_data.json from local benchmark results.

Merges two sources:
  - benchmarks/_dim_sweep_results/dim_sweep_results.json  (bits=2,4 across 9 datasets)
  - benchmarks/_bit_sweep_results/bit_sweep_results.json  (bits=1,3 across 5 low-dim datasets)

Output format mirrors what advisor.js expects:
  {
    "dims": {"dataset-name": d, ...},
    "configs": {
      "dataset-name": [
        {"ds": ..., "d": ..., "bits": ..., "rerank": ..., "ann": ...,
         "fast": ..., "qtype": ..., "rk": {"1": .., "2": .., ..., "32": ..},
         "mrr": .., "p50": .., "disk": .., "compr": .., "vps": .., "src": "dim"|"bit"},
        ...
      ],
      ...
    }
  }

Usage:
    python benchmarks/build_advisor_data.py
    python benchmarks/build_advisor_data.py --dry-run   # print stats only, no write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
DIM_SWEEP_PATH  = REPO_DIR / "benchmarks" / "_dim_sweep_results" / "dim_sweep_results.json"
BIT_SWEEP_PATH  = REPO_DIR / "benchmarks" / "_bit_sweep_results" / "bit_sweep_results.json"
ADVISOR_OUT     = REPO_DIR / "website" / "data" / "advisor_data.json"

# k-values exposed in the advisor UI
RK_KEYS = ["1", "2", "4", "8", "16", "32"]


def _compress_ratio(n: int, d: int, disk_mb: float) -> float:
    """float32 footprint divided by actual on-disk size."""
    float32_mb = n * d * 4 / (1024 ** 2)
    return round(float32_mb / disk_mb, 2) if disk_mb > 0 else 0.0


def _to_advisor_entry(ds_name: str, entry: dict, src: str) -> dict:
    """Convert one dim/bit-sweep entry to advisor_data format."""
    recall_raw = entry.get("recall", {})
    rk = {k: round(float(recall_raw[k]), 4) for k in RK_KEYS if k in recall_raw}

    n    = entry["n"]
    d    = entry["dim"]
    disk = entry["disk_mb"]
    qtype_raw = entry.get("quantizer_type")

    return {
        "ds":     ds_name,
        "d":      d,
        "bits":   entry["bits"],
        "rerank": entry["rerank"],
        "ann":    entry["ann"],
        "fast":   entry["fast_mode"],
        "qtype":  "srht" if qtype_raw == "srht" else "dense",
        "rk":     rk,
        "mrr":    round(float(entry.get("mrr", 0)), 4),
        "p50":    round(float(entry.get("p50_ms", 0)), 3),
        "disk":   round(disk, 2),
        "compr":  _compress_ratio(n, d, disk),
        "vps":    int(entry.get("throughput_vps", 0)),
        "src":    src,
    }


def load_source(path: Path, src_tag: str) -> dict[str, list[dict]]:
    """Load a sweep results file and convert all entries to advisor format."""
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for ds_name, entries in raw.items():
        converted = []
        for e in entries:
            if "error" in e:
                continue
            converted.append(_to_advisor_entry(ds_name, e, src_tag))
        if converted:
            out[ds_name] = converted
    return out


def merge(dim_data: dict, bit_data: dict) -> dict[str, list[dict]]:
    """Merge dim and bit sweep configs per dataset (union by unique key)."""
    all_ds = sorted(set(list(dim_data) + list(bit_data)))
    merged: dict[str, list[dict]] = {}
    for ds in all_ds:
        seen: set[tuple] = set()
        configs: list[dict] = []
        for entry in list(dim_data.get(ds, [])) + list(bit_data.get(ds, [])):
            key = (entry["bits"], entry["rerank"], entry["ann"],
                   entry["fast"], entry["qtype"])
            if key not in seen:
                seen.add(key)
                configs.append(entry)
        merged[ds] = configs
    return merged


def build(dry_run: bool = False) -> None:
    print("Loading dim_sweep results...")
    dim_data = load_source(DIM_SWEEP_PATH, "dim")
    print(f"  {sum(len(v) for v in dim_data.values())} entries across {len(dim_data)} datasets")

    print("Loading bit_sweep results...")
    bit_data = load_source(BIT_SWEEP_PATH, "bit")
    print(f"  {sum(len(v) for v in bit_data.values())} entries across {len(bit_data)} datasets")

    merged = merge(dim_data, bit_data)
    total = sum(len(v) for v in merged.values())
    print(f"Merged: {total} entries across {len(merged)} datasets")

    dims = {ds: entries[0]["d"] for ds, entries in merged.items() if entries}

    result = {"dims": dims, "configs": merged}

    if dry_run:
        for ds, entries in sorted(merged.items()):
            bits_vals = sorted({e["bits"] for e in entries})
            srcs = sorted({e["src"] for e in entries})
            print(f"  {ds:20s}  d={dims[ds]:4d}  {len(entries):3d} configs  bits={bits_vals}  src={srcs}")
        return

    ADVISOR_OUT.parent.mkdir(parents=True, exist_ok=True)
    ADVISOR_OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {ADVISOR_OUT}  ({ADVISOR_OUT.stat().st_size // 1024} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary only; do not write advisor_data.json")
    args = parser.parse_args()
    build(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
