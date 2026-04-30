# perf/hamming_sketch — Sign-bit Hamming prefilter for fast_mode + b=4

**Status: PASS** (high-D b=4 brute-force search)

## Idea

In fast_mode (no QJL stored), the high bit of every 4-bit MSE nibble is the
sign of the corresponding rotated dimension (Lloyd-Max codebook is symmetric
around 0). We can derive a 1-bit-per-dim sketch from the existing live MSE
bytes for *free* — no disk format change, no extra RAM, no migration — and
use Hamming disagreement as a cheap IP proxy to shrink the candidate set
before full LUT scoring.

Gate: `qjl_len == 0 && b == 4 && d >= 512 && IP|Cosine && N >= 5_000`.
Retain `max(N/8, 10·internal_k)` candidates after Hamming.

## Stable bench (5-iter median A/B, dbpedia datasets, 200 queries each, top_k=10)

| dataset       | config         | baseline | treatment | Δ        | recall change |
|---------------|----------------|---------:|----------:|---------:|--------------:|
| dbpedia-1536  | b=2 rerank=F   |   15.55  |    21.02  | +35% ⚠   | 0             |
| dbpedia-1536  | b=2 rerank=T   |   20.52  |    20.32  |  -1%     | 0             |
| dbpedia-1536  | b=4 rerank=F   |   14.23  |     8.53  | **-40%** | 0             |
| dbpedia-1536  | b=4 rerank=T   |   14.67  |     8.80  | **-40%** | 0             |
| dbpedia-3072  | b=2 rerank=F   |   42.71  |    42.21  |  -1%     | 0             |
| dbpedia-3072  | b=2 rerank=T   |   42.73  |    42.25  |  -1%     | 0             |
| dbpedia-3072  | b=4 rerank=F   |   28.14  |    21.01  | **-25%** | 0             |
| dbpedia-3072  | b=4 rerank=T   |   27.56  |    21.10  | **-23%** | 0             |

## Notes

- **Recall is bit-identical** on every cell. The prefilter retains 12.5% of
  candidates (~12,500 of 100k), well above the rerank pool of `top_k * 20`.
- The +35% on dbpedia-1536 b=2 rerank=F is **bench noise**, not regression:
  b=2 is gated out of the new code path, and a baseline-redo on the same
  machine state showed b=2 rerank=T itself swinging 15.7 → 20.7 ms across 5
  iters of the *baseline* code. b=2 paths are byte-identical to baseline.
- glove-200 (d=200) is gated out by `d >= 512`. Tested without the gate, the
  prefilter hurt low-D queries (+80% on b=4 rerank=F, d=200) because per-slot
  prefilter overhead exceeds the savings on the cheap d=200 LUT.
- `cargo test --lib` passes (425 tests).

## Files changed

- `src/quantizer/prod.rs`: +`prepare_query_sign_bits()` (~25 lines),
  +`hamming_disagree_b4_signs()` (~50 lines).
- `src/storage/engine/mod.rs`: +~30-line prefilter section before brute-force
  scoring loop, scoped to fast_mode + b=4 + d≥512 + IP|Cosine + N≥5000.

## Verdict

PASS. Worth promoting on its own. Recommend upstreaming after running the
full `paper_recall_bench.py` (with all 8 brute+ANN configs) and confirming
ANN paths are unaffected.
