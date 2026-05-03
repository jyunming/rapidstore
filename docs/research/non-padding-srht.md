# Non-padding alternatives to SRHT — research note

**Status:** investigation, not committed. Authored 2026-05-04 during the v0.9 quantizer
sweep. Purpose: capture the design space for closing SRHT's pow2-padding disk tax at
non-pow2 dimensions (1536, 3072 — the OpenAI/Cohere production embedding shapes).

## Why this matters

At d=1536 and d=3072, SRHT pads the input to the next power of two (2048, 4096) before
applying the Walsh–Hadamard transform. The bench shows the practical impact:

| dim | code data only | SRHT padded codes | padding overhead |
|---:|---:|---:|---:|
| 1536 | 38.4 MB (b=2) / 76.8 MB (b=4) | 51.2 MB / 102.4 MB | +33% |
| 3072 | 76.8 MB / 153.6 MB | 102.4 MB / 204.8 MB | +33% |

SRHT wins on every other axis (recall, ingest, p50, RAM) but pays this 33% disk tax
at non-pow2 dims. If we can apply a structured pseudo-random orthogonal transform
without padding, SRHT becomes Pareto-better than dense Haar QR everywhere.

## Candidate approaches

Ranked from most viable to least.

### 1. DCT-II with random sign vector (DCT-RS) — most promising

Apply `y = DCT-II(x ⊙ s)` where `s ∈ {±1}^d` is a random sign vector. DCT-II is a
real orthogonal transform with an O(d log d) fast algorithm, defined for any length
d (no padding). The random signs provide the JL-quality randomness; the DCT provides
the structured mixing.

**Pros**
- No padding at any d
- O(d log d) cost, comparable to SRHT
- Real orthogonal — preserves inner products exactly modulo numerical precision
- Well-studied in compressive sensing; concentration bounds match SRHT for `d × d → m`
  random projections (Tropp 2011, Krahmer-Ward 2011)
- Length-d sign vector is the only persistent state — disk overhead identical to SRHT

**Cons**
- Requires an FFT crate. Candidates: `rustfft` (mature, MIT licence), `realfft` (DCT
  on top of `rustfft`)
- DCT primitive is more expensive per-element than the Hadamard butterfly (real
  multiplies vs sign flips), so per-vector wall-time is ~1.5–2× SRHT despite same
  asymptotic order
- Concentration is asymptotically equivalent to SRHT but the per-element variance has
  a slightly different constant — may need a calibration pass to match recall

**Implementation cost:** ~1 week. Add a new `quantizer_type="dct"` variant. Reuse the
codebook + bit-packing path; only the rotation kernel changes.

**Expected impact**
- Disk: -25 to -50 MB at d=3072 b=4 (closes the 51 MB padding gap)
- Ingest: ~80% of SRHT speed (still 1.5–2× faster than dense at d=3072)
- Recall: equal to SRHT within 0.2 pp (theoretical concentration guarantees match)

### 2. Block Hadamard with cross-block mixing permutation

Decompose d into powers of two greedily (e.g. d=3072 = 2048+1024). Apply
`H_2048` to the first block and `H_1024` to the second. To restore mixing across
blocks, apply a random permutation π to the input *before* the block transforms and
another permutation σ *after*. Final transform: `σ ∘ blockH ∘ π ∘ diag(s) ∘ x`.

**Pros**
- Reuses the existing Hadamard butterfly code (`src/linalg/hadamard.rs`)
- Hadamard cost (sign-flip butterfly) is faster per-element than DCT
- O(d log d) overall — sum of blocks
- No new dependency

**Cons**
- Per-block mixing is exact JL; cross-block mixing comes only from the permutations,
  which is weaker than full Hadamard
- Concentration constant is worse than full Hadamard or DCT-RS; theoretical bound
  loses a `log(num_blocks)` factor (Krahmer-Ward extension result)
- For "awkward" d like 3072, the decomposition `2048 + 1024` has imbalanced blocks
  → the smaller block dominates the per-coordinate variance
- More implementation surface: permutation generation, block boundary management

**Implementation cost:** ~2–3 days. Less disruptive than DCT but with a weaker
guarantee.

**Expected impact**
- Disk: same as DCT (no padding)
- Ingest: ~95% of SRHT speed (same butterfly, just smaller per block)
- Recall: -0.5 to -1.5 pp vs full SRHT, maybe more at d where decomposition is unbalanced

### 3. Truncated SRHT

Pad to next pow2 `n`, apply full SRHT to length-n input, but only quantize and store
the first `d` output coordinates. The remaining `n − d` coordinates are dropped.

**Why this almost works:** all output coordinates of `H · D · (x, 0)` are linear
combinations of the d non-zero input entries; no information is "in" the padding
positions. So truncating outputs is a lossy projection but information-theoretically
not catastrophic.

**Why it's lossy:** the kept d coordinates aren't a uniform subset of an orthogonal
basis — they're the first d rows of `H · D`, which are a *biased* subset. Variance
of the remaining d-dim projection is no longer 1/d uniformly across coordinates,
so the codebook's quantization noise is non-uniform, hurting recall.

**Pros**
- No new code beyond a slice on the output side
- Removes the disk tax exactly (codes are length d, not n)

**Cons**
- Bias hurts recall — needs empirical measurement, but theoretical bound suggests
  ~5–10 pp R@1 drop at b=4 (rough estimate from JL-with-row-sampling literature)
- Probably not viable in practice; included for completeness

**Implementation cost:** half a day.

### 4. Subsampled circulant rotation

Use a length-d random circulant matrix as the rotation. Circulant matrices have an
O(d log d) FFT-based apply. Combined with a random diagonal sign matrix, gives a
restricted-isometry-property–satisfying transform without padding (Romberg 2009).

**Pros**
- No padding
- O(d log d) cost
- Same FFT crate as DCT-RS (often subsumed)

**Cons**
- The full state is a length-d vector (the seed of the circulant) — same as DCT-RS,
  no advantage
- Not as well-tested in JL-projection settings as DCT-RS or block Hadamard
- Higher implementation risk for marginal benefit over DCT-RS

**Verdict:** dominated by DCT-RS. Skip.

### 5. Generalized Hadamard at length 4k

Hadamard matrices exist at all sizes that are multiples of 4 with the conjectured
"Hadamard property" — not just powers of 2. The Paley construction provides
`H_p` for any prime `p ≡ 3 (mod 4)`, plus tensor products (`H_{ab} = H_a ⊗ H_b`).

For d=1536 = 2^9 × 3 = 512 × 3 → could use `H_512 ⊗ H_3`? But `H_3` doesn't exist
(Hadamard's theorem requires multiples of 4). Doesn't easily apply to 1536 or 3072.

For d=3072 = 1024 × 3, same issue.

**Verdict:** elegant in theory, doesn't fit the dimensions of interest. Skip.

## Recommendation

Pursue **DCT-RS (option 1)** as a third `quantizer_type` value (e.g. `"dct"` or
`"srht_nopad"`). It is:

- Well-studied with strong concentration guarantees
- O(d log d) without padding
- A clean drop-in replacement for SRHT at non-pow2 dimensions

Implementation steps (rough):
1. Add `rustfft` (or `realfft`) to `Cargo.toml`
2. New variant `MseQuantizer::new_dct(d, b, seed)` mirroring `new_srht` but using a
   length-d sign vector and DCT-II for the rotation kernel
3. Update auto-default: at d ≥ 1024, prefer DCT-RS over SRHT when d is not a power
   of two; keep SRHT when d is a power of two (no padding tax there)
4. Bench at d=1536, 3072 to verify recall parity
5. Document in `QUANTIZER_MODES.md`

**Skip block Hadamard** unless adding an FFT dependency is unacceptable. The recall
penalty is real and not easily bounded.

**Skip truncated SRHT** — likely too lossy.

## Open questions

- Is `rustfft` audit-clean and small-binary-impact? It pulls in `num-complex` and
  has historically required SIMD-feature-flag care on Windows ARM64.
- Does FAISS's IVF-PQ implementation use a similar trick? Worth checking — they
  handle non-pow2 dim and report similar recall to padded SRHT.
- For very high d (≥8192) the DCT primitive's per-element constant matters more.
  Worth measuring crossover where dense Haar QR with bf16 rotation is still
  competitive on disk despite the d² state cost.

## Decision deferred

This is a v0.10+ candidate. v0.9 ships SRHT default at d ≥ 1024 and bf16 rotation
storage; non-padding SRHT is a follow-up if the padding tax actually bothers users
in practice.
