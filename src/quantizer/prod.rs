use nalgebra::DMatrix;
use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

use super::CodeIndex;
use super::mse::MseQuantizer;
use super::qjl::QjlQuantizer;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProdQuantizer {
    pub d: usize,
    pub n: usize,
    pub b: usize,
    pub mse_quantizer: MseQuantizer,
    pub qjl_quantizer: QjlQuantizer,
    /// When true, skip the QJL residual quantization step. Ingest is ~30% faster;
    /// recall is marginally lower. Has no effect on dequantization (gamma=0 → QJL=0).
    #[serde(default)]
    pub fast_mode: bool,
}

pub struct PreparedIpQuery {
    // Lookup table flattened as [dim0 c0..cK, dim1 c0..cK, ...]
    mse_lut: Vec<f32>,
    mse_lut_width: usize,
    /// i16-quantized version of `mse_lut` with a single global scale.
    /// Halves the L1/L2 footprint of the LUT (96 KB -> 48 KB at d=1536, lut_w=16).
    /// Per the P6 numpy validation, ranking is bit-identical to the f32 LUT.
    mse_lut_i16: Vec<i16>,
    mse_lut_i16_scale: f32,
    sq: Vec<f32>,
    qjl_scale: f32,
}

pub struct PreparedIpQueryLite {
    y: Vec<f32>,
    sq: Vec<f32>,
    qjl_scale: f32,
}

impl ProdQuantizer {
    pub fn new(d: usize, b: usize, seed: u64) -> Self {
        assert!(b >= 2, "ProdQuantizer requires at least b=2");

        let mse_quantizer = MseQuantizer::new(d, b - 1, seed);
        let qjl_quantizer = QjlQuantizer::new(d, seed ^ 0xdeadbeef);
        let n = d.next_power_of_two();

        Self {
            d,
            n,
            b,
            mse_quantizer,
            qjl_quantizer,
            fast_mode: false,
        }
    }

    /// Dense mode: Haar-uniform QR rotation (MSE) + dense N(0,1) Gaussian projection (QJL),
    /// as specified in arXiv:2504.19874. No padding: n=d.
    /// O(d²) quantize/score time and O(d²) storage vs O(d log d) / O(d) for SRHT.
    pub fn new_dense(d: usize, b: usize, seed: u64) -> Self {
        assert!(b >= 2, "ProdQuantizer requires at least b=2");
        let mse_quantizer = MseQuantizer::new_dense(d, b - 1, seed);
        let qjl_quantizer = QjlQuantizer::new_dense(d, seed ^ 0xdeadbeef);
        Self {
            d,
            n: d, // no padding in exact mode
            b,
            mse_quantizer,
            qjl_quantizer,
            fast_mode: false,
        }
    }

    /// Dense fast-path: Haar-uniform QR rotation + all b bits to MSE (no QJL).
    /// Best recall of all modes; O(d²) ingest cost.
    pub fn new_dense_fast(d: usize, b: usize, seed: u64) -> Self {
        // fast_mode assigns all bits to MSE and stores no QJL residual, so b=1 is valid.
        assert!(b >= 1, "ProdQuantizer requires at least b=1 in fast mode");
        let mse_quantizer = MseQuantizer::new_dense(d, b, seed);
        // fast_mode never uses the QJL projection — use a no-op placeholder (empty
        // struct, zero disk/RAM) instead of the dense D×D Gaussian matrix.
        let qjl_quantizer = QjlQuantizer::new_noop(d);
        Self {
            d,
            n: d,
            b,
            mse_quantizer,
            qjl_quantizer,
            fast_mode: true,
        }
    }

    /// Fast-path: skips the QJL residual quantization step. Ingest is ~30% faster;
    /// approximate scores omit the residual correction (slightly lower recall).
    pub fn new_srht(d: usize, b: usize, seed: u64) -> Self {
        // fast_mode: QJL not stored, so all b bits go to MSE codebook (not b-1).
        let mse_quantizer = MseQuantizer::new(d, b, seed);
        let qjl_quantizer = QjlQuantizer::new(d, seed ^ 0xdeadbeef);
        let n = d.next_power_of_two();

        Self {
            d,
            n,
            b,
            mse_quantizer,
            qjl_quantizer,
            fast_mode: true,
        }
    }

    /// Bits used per MSE code index. Equals `mse_quantizer.b`:
    /// `b-1` in prod mode (QJL takes 1 bit) or `b` in fast_mode (all bits → MSE).
    pub fn mse_bits_per_idx(&self) -> usize {
        self.mse_quantizer.b
    }

    fn qjl_query_scale(&self) -> f32 {
        if self.qjl_quantizer.projection_matrix.is_some() {
            // Exact mode uses sq = S q / sqrt(d), so the correction term needs
            // sqrt(pi/2) / sqrt(d) to match gamma * sqrt(pi/2) / d * S^T z.
            (PI / 2.0).sqrt() / (self.d as f32).sqrt()
        } else {
            // SRHT output is unit-variance per dim (normalized Hadamard), so scale
            // matches qjl.rs dequant scale_base = sqrt(pi/(2n)) = sqrt(pi/2)/sqrt(n).
            (PI / 2.0).sqrt() / (self.n as f32).sqrt()
        }
    }

    pub fn prepare_ip_query(&self, query: &Array1<f64>) -> PreparedIpQuery {
        assert_eq!(query.len(), self.d);

        let query_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
        let mut y = vec![0.0f32; self.n];
        self.mse_quantizer.apply_rotation(&query_f32, &mut y);

        // Precompute MSE lookup scores for each dimension/code.
        let centroids = &self.mse_quantizer.centroids;
        let lut_w = centroids.len();
        let mut mse_lut = vec![0.0f32; self.n * lut_w];
        for i in 0..self.n {
            let yi = y[i];
            let row_off = i * lut_w;
            for (k, &c) in centroids.iter().enumerate() {
                mse_lut[row_off + k] = c * yi;
            }
        }

        // fast_mode: all gammas are 0 so sq is never used in scoring — skip projection.
        let mut sq = vec![0.0f32; self.n];
        if !self.fast_mode {
            self.qjl_quantizer.apply_projection(&query_f32, &mut sq);
        }

        // Build i16 mirror with a single global max-abs scale. Used only by the b=4 fast-mode
        // SIMD path; cost is one pass over n*lut_w f32 entries — at d=1536/lut_w=16 that's
        // ~25k mul+round+cast ops = ~50us, marginal vs the ~1ms prepare_ip_query baseline.
        let max_abs = mse_lut.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let mse_lut_i16_scale = if max_abs > 0.0 { max_abs / 32767.0 } else { 1.0 };
        let inv_scale = if max_abs > 0.0 { 32767.0 / max_abs } else { 1.0 };
        let mse_lut_i16: Vec<i16> = mse_lut
            .iter()
            .map(|&v| (v * inv_scale).round().clamp(-32768.0, 32767.0) as i16)
            .collect();

        PreparedIpQuery {
            mse_lut,
            mse_lut_width: lut_w,
            mse_lut_i16,
            mse_lut_i16_scale,
            sq,
            qjl_scale: self.qjl_query_scale(),
        }
    }

    pub fn prepare_ip_query_lite(&self, query: &Array1<f64>) -> PreparedIpQueryLite {
        assert_eq!(query.len(), self.d);

        let query_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
        let mut y = vec![0.0f32; self.n];
        self.mse_quantizer.apply_rotation(&query_f32, &mut y);

        // fast_mode: all gammas are 0 so sq is never used in scoring — skip projection.
        let mut sq = vec![0.0f32; self.n];
        if !self.fast_mode {
            self.qjl_quantizer.apply_projection(&query_f32, &mut sq);
        }

        PreparedIpQueryLite {
            y,
            sq,
            qjl_scale: self.qjl_query_scale(),
        }
    }

    /// Hamming similarity between two QJL bit codes: (matching bits) / (total bits).
    /// Fraction of matching bits between two QJL bit codes.  Returns a value in
    /// [0, 1] where 1 = identical codes and 0.5 = random (expected value for
    /// independent bits).  Distinct from the free function [`hamming_score`]
    /// which returns a centred value in [−1, 1].  Used as a proxy for inner-product
    /// proximity during HNSW construction when raw vectors are unavailable.
    pub fn hamming_proximity(&self, from_qjl: &[u8], to_qjl: &[u8]) -> f64 {
        let n_bytes = from_qjl.len().min(to_qjl.len());
        if n_bytes == 0 {
            return 0.5;
        }
        let matching_bits: u32 = (0..n_bytes)
            .map(|i| (!(from_qjl[i] ^ to_qjl[i])).count_ones())
            .sum();
        matching_bits as f64 / (n_bytes as f64 * 8.0)
    }

    /// Build a `PreparedIpQueryLite` directly from stored MSE codes without any SRHT.
    ///
    /// Each `y[i]` is set to `centroids[idx[i]]` — the quantized approximation of the
    /// SRHT-rotated vector. This skips the 3-SRHT round-trip
    /// (`dequantize` → d-dim → `apply_rotation` → n-dim) and replaces it with an O(n)
    /// centroid lookup. Used in HNSW construction when no raw vectors are stored.
    pub fn prepare_ip_query_from_codes(&self, idx: &[CodeIndex]) -> PreparedIpQueryLite {
        let centroids = &self.mse_quantizer.centroids;
        let y: Vec<f32> = (0..self.n).map(|i| centroids[idx[i] as usize]).collect();
        PreparedIpQueryLite {
            y,
            sq: vec![0.0f32; self.n],
            qjl_scale: self.qjl_query_scale(),
        }
    }

    pub fn score_ip_encoded_lite(
        &self,
        prep: &PreparedIpQueryLite,
        idx: &[CodeIndex],
        qjl: &[u8],
        gamma: f64,
    ) -> f64 {
        // A5 (v0.8.2 audit): see comment on score_ip_encoded.
        assert_eq!(
            idx.len(),
            self.n,
            "score_ip_encoded_lite: idx.len() must equal quantizer.n ({})",
            self.n
        );
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { self.score_ip_encoded_lite_simd(prep, idx, qjl, gamma) };
            }
        }
        self.score_ip_encoded_lite_scalar(prep, idx, qjl, gamma)
    }

    fn score_ip_encoded_lite_scalar(
        &self,
        prep: &PreparedIpQueryLite,
        idx: &[CodeIndex],
        qjl: &[u8],
        gamma: f64,
    ) -> f64 {
        let mut mse_score = 0.0f32;
        let centroids = &self.mse_quantizer.centroids;
        let prep_y = &prep.y;

        for i in 0..self.n {
            let c = unsafe { *centroids.get_unchecked(idx[i] as usize) };
            mse_score += c * unsafe { *prep_y.get_unchecked(i) };
        }

        // fast_mode stores gamma=0 — skip QJL entirely (saves ~50% of scoring work).
        if gamma == 0.0 {
            return mse_score as f64;
        }

        let mut qjl_score = 0.0f32;
        let sq = &prep.sq;
        let qjl_len = qjl.len();

        for byte_idx in 0..qjl_len {
            let byte = unsafe { *qjl.get_unchecked(byte_idx) };
            let base_i = byte_idx << 3;

            for bit_idx in 0..8 {
                let i = base_i + bit_idx;
                if i >= self.n {
                    break;
                }
                let bit_set = ((byte >> bit_idx) & 1u8) == 1u8;
                let s = unsafe { *sq.get_unchecked(i) };
                qjl_score += if bit_set { s } else { -s };
            }
        }

        (mse_score + (gamma as f32) * prep.qjl_scale * qjl_score) as f64
    }

    /// AVX2+FMA SIMD path for score_ip_encoded_lite (used in HNSW construction).
    ///
    /// MSE part: manual 8-wide gather from the codebook (16–256 entries, L1-resident)
    /// followed by FMA against the sequential prep_y array.
    ///
    /// QJL part: bitwise sign expansion via integer SIMD + multiply-accumulate.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn score_ip_encoded_lite_simd(
        &self,
        prep: &PreparedIpQueryLite,
        idx: &[CodeIndex],
        qjl: &[u8],
        gamma: f64,
    ) -> f64 {
        use std::arch::x86_64::*;

        let centroids = self.mse_quantizer.centroids.as_ptr();
        let prep_y = prep.y.as_ptr();
        let idx_ptr = idx.as_ptr();

        let mut mse_acc = _mm256_setzero_ps();
        let mut i = 0usize;

        // Main 8-wide FMA loop — centroids fits in L1 (16–256 entries), so the
        // manual gather has no cache-miss penalty.
        while i + 7 < self.n {
            let mut cents = [0.0f32; 8];
            for j in 0..8 {
                cents[j] = *centroids.add(*idx_ptr.add(i + j) as usize);
            }
            let c_vec = _mm256_loadu_ps(cents.as_ptr());
            let y_vec = _mm256_loadu_ps(prep_y.add(i));
            mse_acc = _mm256_fmadd_ps(c_vec, y_vec, mse_acc);
            i += 8;
        }

        // Horizontal reduction of mse_acc
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), mse_acc);
        let mut mse_sum: f32 =
            tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

        // Scalar tail for MSE
        while i < self.n {
            mse_sum += *centroids.add(*idx_ptr.add(i) as usize) * *prep_y.add(i);
            i += 1;
        }

        // fast_mode stores gamma=0 — skip QJL entirely (saves ~50% of scoring work).
        if gamma == 0.0 {
            return mse_sum as f64;
        }

        // QJL part: for each byte, expand 8 bits → ±1.0 sign vector via integer SIMD,
        // then multiply-accumulate against the sequential sq array.
        let sq_ptr = prep.sq.as_ptr();
        let qjl_ptr = qjl.as_ptr();
        let qjl_len = qjl.len();

        let bit_masks = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
        let zero_si = _mm256_setzero_si256();
        let pos_one = _mm256_set1_ps(1.0f32);
        let neg_one = _mm256_set1_ps(-1.0f32);

        let mut qjl_acc = _mm256_setzero_ps();
        let mut qjl_tail = 0.0f32;
        let mut b = 0usize;

        while b < qjl_len {
            let byte = *qjl_ptr.add(b) as i32;
            let base_i = b << 3;

            if base_i + 7 < self.n {
                // Expand byte bits to ±1.0 sign floats using integer SIMD.
                let byte_broadcast = _mm256_set1_epi32(byte);
                let masked = _mm256_and_si256(byte_broadcast, bit_masks);
                let is_zero = _mm256_cmpeq_epi32(masked, zero_si);
                // bit=1 → pos_one, bit=0 → neg_one
                let signs = _mm256_blendv_ps(pos_one, neg_one, _mm256_castsi256_ps(is_zero));
                let s_vec = _mm256_loadu_ps(sq_ptr.add(base_i));
                qjl_acc = _mm256_fmadd_ps(signs, s_vec, qjl_acc);
            } else {
                for bit in 0..8 {
                    let qi = base_i + bit;
                    if qi >= self.n {
                        break;
                    }
                    let s = *sq_ptr.add(qi);
                    qjl_tail += if (byte & (1 << bit)) != 0 { s } else { -s };
                }
            }
            b += 1;
        }

        _mm256_storeu_ps(tmp.as_mut_ptr(), qjl_acc);
        let qjl_sum =
            tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7] + qjl_tail;

        (mse_sum + (gamma as f32) * prep.qjl_scale * qjl_sum) as f64
    }

    pub fn score_ip_encoded(
        &self,
        prep: &PreparedIpQuery,
        idx: &[CodeIndex],
        qjl: &[u8],
        gamma: f64,
    ) -> f64 {
        // A5 (v0.8.2 audit): the inner SIMD/scalar loops use `idx.get_unchecked(i)`
        // for i in 0..self.n; a too-short `idx` would be undefined behavior.
        // Asserting at the safe wrapper turns UB into a clear panic.
        assert_eq!(
            idx.len(),
            self.n,
            "score_ip_encoded: idx.len() must equal quantizer.n ({})",
            self.n
        );
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { self.score_ip_encoded_simd(prep, idx, qjl, gamma) };
            }
        }
        self.score_ip_encoded_scalar(prep, idx, qjl, gamma)
    }

    fn score_ip_encoded_scalar(
        &self,
        prep: &PreparedIpQuery,
        idx: &[CodeIndex],
        qjl: &[u8],
        gamma: f64,
    ) -> f64 {
        let mut mse_score = 0.0f32;
        let lut_width = prep.mse_lut_width;
        let lut = &prep.mse_lut;

        let mut row_off = 0;
        for i in 0..self.n {
            let val = unsafe { *lut.get_unchecked(row_off + idx[i] as usize) };
            mse_score += val;
            row_off += lut_width;
        }

        let mut qjl_score = 0.0f32;
        let sq = &prep.sq;

        let qjl_len = qjl.len();
        for byte_idx in 0..qjl_len {
            let byte = unsafe { *qjl.get_unchecked(byte_idx) };
            let base_i = byte_idx << 3;

            for bit_idx in 0..8 {
                let i = base_i + bit_idx;
                if i >= self.n {
                    break;
                }
                let bit_set = ((byte >> bit_idx) & 1u8) == 1u8;
                let s = unsafe { *sq.get_unchecked(i) };
                qjl_score += if bit_set { s } else { -s };
            }
        }

        (mse_score + (gamma as f32) * prep.qjl_scale * qjl_score) as f64
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn score_ip_encoded_simd(
        &self,
        prep: &PreparedIpQuery,
        idx: &[CodeIndex],
        qjl: &[u8],
        gamma: f64,
    ) -> f64 {
        use std::arch::x86_64::*;

        let mut mse_acc = _mm256_setzero_ps();
        let lut = prep.mse_lut.as_ptr();
        let lut_width = prep.mse_lut_width;

        let mut i = 0;
        while i + 7 < self.n {
            let mut vals = [0.0f32; 8];
            for j in 0..8 {
                let row = i + j;
                unsafe {
                    vals[j] = *lut.add(row * lut_width + idx[row] as usize);
                }
            }
            let v = _mm256_loadu_ps(vals.as_ptr());
            mse_acc = _mm256_add_ps(mse_acc, v);
            i += 8;
        }

        let mut mse_sum = 0.0f32;
        let mut res = [0.0f32; 8];
        _mm256_storeu_ps(res.as_mut_ptr(), mse_acc);
        for val in res {
            mse_sum += val;
        }
        while i < self.n {
            unsafe {
                mse_sum += *lut.add(i * lut_width + idx[i] as usize);
            }
            i += 1;
        }

        let mut qjl_acc = _mm256_setzero_ps();
        let sq = prep.sq.as_ptr();
        let qjl_ptr = qjl.as_ptr();
        let qjl_len = qjl.len();

        // Constants hoisted out of the loop for integer SIMD sign expansion.
        let bit_masks = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
        let zero_si = _mm256_setzero_si256();
        let pos_one = _mm256_set1_ps(1.0f32);
        let neg_one = _mm256_set1_ps(-1.0f32);

        let mut b = 0;
        while b < qjl_len {
            let byte = unsafe { *qjl_ptr.add(b) as i32 };
            let base_i = b << 3;
            if base_i + 7 < self.n {
                let s_vec = _mm256_loadu_ps(unsafe { sq.add(base_i) });
                // Expand 8 packed sign bits → [±1.0; 8] using integer SIMD blend.
                // This replaces a scalar 8-iteration loop that wrote to a stack array.
                let byte_broadcast = _mm256_set1_epi32(byte);
                let masked = _mm256_and_si256(byte_broadcast, bit_masks);
                let is_zero = _mm256_cmpeq_epi32(masked, zero_si);
                let signs = _mm256_blendv_ps(pos_one, neg_one, _mm256_castsi256_ps(is_zero));
                qjl_acc = _mm256_fmadd_ps(signs, s_vec, qjl_acc);
            } else {
                let mut qs = 0.0f32;
                for bit in 0..8 {
                    let idx = base_i + bit;
                    if idx >= self.n {
                        break;
                    }
                    let s = unsafe { *sq.add(idx) };
                    qs += if (byte & (1 << bit)) != 0 { s } else { -s };
                }
                mse_sum += (gamma as f32) * prep.qjl_scale * qs;
            }
            b += 1;
        }

        _mm256_storeu_ps(res.as_mut_ptr(), qjl_acc);
        let mut qjl_sum = 0.0f32;
        for val in res {
            qjl_sum += val;
        }

        (mse_sum + (gamma as f32) * prep.qjl_scale * qjl_sum) as f64
    }

    /// Fused nibble-unpack + LUT-gather scoring for b=4 (production fast path).
    ///
    /// Reads the raw packed MSE bytes directly and extracts nibbles inline,
    /// avoiding the intermediate `idx_buf: Vec<u16>` materialization (~4-8KB
    /// of L1 traffic per slot at d=2048). Falls back to the unpack+score path
    /// for b != 4.
    pub fn score_ip_encoded_packed(
        &self,
        prep: &PreparedIpQuery,
        packed_mse: &[u8],
        qjl: &[u8],
        gamma: f64,
    ) -> f64 {
        if self.mse_quantizer.b == 4 {
            debug_assert_eq!(
                packed_mse.len(),
                self.n / 2,
                "score_ip_encoded_packed: packed_mse.len() must equal n/2 ({})",
                self.n / 2
            );
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // P7: in fast_mode the QJL term is zero, so the i16 LUT path can
                    // do the whole MSE accumulate in i32 with half the L1/L2 footprint.
                    if self.fast_mode && qjl.is_empty() {
                        return unsafe {
                            self.score_ip_encoded_packed_b4_simd_i16(prep, packed_mse)
                        };
                    }
                    return unsafe {
                        self.score_ip_encoded_packed_b4_simd(prep, packed_mse, qjl, gamma)
                    };
                }
            }
        }
        // b != 4 (or no AVX2): fall back to two-step path with a temporary buffer.
        let mut idx = vec![0 as CodeIndex; self.n];
        self.unpack_mse_indices(packed_mse, &mut idx);
        self.score_ip_encoded(prep, &idx, qjl, gamma)
    }

    /// AVX2/FMA fused path: 16 indices per outer iteration (= 8 packed bytes).
    /// Nibbles are kept in registers / a tiny stack buffer instead of being
    /// materialized into a heap-backed Vec.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn score_ip_encoded_packed_b4_simd(
        &self,
        prep: &PreparedIpQuery,
        packed_mse: &[u8],
        qjl: &[u8],
        gamma: f64,
    ) -> f64 {
        use std::arch::x86_64::*;

        let mut mse_acc = _mm256_setzero_ps();
        let lut = prep.mse_lut.as_ptr();
        let lut_width = prep.mse_lut_width;
        let n = self.n;
        let packed_ptr = packed_mse.as_ptr();

        let mut i = 0usize;
        let mut byte_i = 0usize;

        // Process 16 indices = 8 packed bytes per iteration.
        while i + 16 <= n {
            let p64 = unsafe { std::ptr::read_unaligned(packed_ptr.add(byte_i) as *const u64) };
            // Extract 16 nibbles via bit math.
            let mut nibs = [0u8; 16];
            let mut bits = p64;
            for k in 0..8 {
                nibs[2 * k] = (bits & 0x0f) as u8;
                nibs[2 * k + 1] = ((bits >> 4) & 0x0f) as u8;
                bits >>= 8;
            }

            // First 8 lanes
            let mut vals = [0.0f32; 8];
            for j in 0..8 {
                unsafe {
                    vals[j] = *lut.add((i + j) * lut_width + nibs[j] as usize);
                }
            }
            mse_acc = _mm256_add_ps(mse_acc, _mm256_loadu_ps(vals.as_ptr()));

            // Second 8 lanes
            for j in 0..8 {
                unsafe {
                    vals[j] = *lut.add((i + 8 + j) * lut_width + nibs[8 + j] as usize);
                }
            }
            mse_acc = _mm256_add_ps(mse_acc, _mm256_loadu_ps(vals.as_ptr()));

            i += 16;
            byte_i += 8;
        }

        // Reduce mse_acc.
        let mut res = [0.0f32; 8];
        _mm256_storeu_ps(res.as_mut_ptr(), mse_acc);
        let mut mse_sum = 0.0f32;
        for v in res {
            mse_sum += v;
        }

        // Scalar tail for any remaining indices (n not multiple of 16).
        while i < n {
            let b = unsafe { *packed_ptr.add(byte_i) };
            let nib = if i % 2 == 0 { b & 0x0f } else { b >> 4 };
            unsafe {
                mse_sum += *lut.add(i * lut_width + nib as usize);
            }
            i += 1;
            if i % 2 == 0 {
                byte_i += 1;
            }
        }

        // QJL portion: identical to score_ip_encoded_simd.
        let mut qjl_acc = _mm256_setzero_ps();
        let sq = prep.sq.as_ptr();
        let qjl_ptr = qjl.as_ptr();
        let qjl_len = qjl.len();
        let bit_masks = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
        let zero_si = _mm256_setzero_si256();
        let pos_one = _mm256_set1_ps(1.0f32);
        let neg_one = _mm256_set1_ps(-1.0f32);

        let mut b = 0;
        while b < qjl_len {
            let byte = unsafe { *qjl_ptr.add(b) as i32 };
            let base_i = b << 3;
            if base_i + 7 < n {
                let s_vec = _mm256_loadu_ps(unsafe { sq.add(base_i) });
                let byte_broadcast = _mm256_set1_epi32(byte);
                let masked = _mm256_and_si256(byte_broadcast, bit_masks);
                let is_zero = _mm256_cmpeq_epi32(masked, zero_si);
                let signs = _mm256_blendv_ps(pos_one, neg_one, _mm256_castsi256_ps(is_zero));
                qjl_acc = _mm256_fmadd_ps(signs, s_vec, qjl_acc);
            } else {
                let mut qs = 0.0f32;
                for bit in 0..8 {
                    let idx = base_i + bit;
                    if idx >= n {
                        break;
                    }
                    let s = unsafe { *sq.add(idx) };
                    qs += if (byte & (1 << bit)) != 0 { s } else { -s };
                }
                mse_sum += (gamma as f32) * prep.qjl_scale * qs;
            }
            b += 1;
        }

        _mm256_storeu_ps(res.as_mut_ptr(), qjl_acc);
        let mut qjl_sum = 0.0f32;
        for v in res {
            qjl_sum += v;
        }

        (mse_sum + (gamma as f32) * prep.qjl_scale * qjl_sum) as f64
    }

    /// P7: i16 LUT variant of the b=4 fast-mode scorer.
    /// Halves the LUT footprint (96 KB -> 48 KB at d=1536, lut_w=16) and uses an
    /// i32 accumulator (1-cycle latency vs f32 add's 4 cycles). Single global scale
    /// applied at the very end.
    /// Bit-identical R@1/R@10 to the f32 variant per P6 numpy validation.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn score_ip_encoded_packed_b4_simd_i16(
        &self,
        prep: &PreparedIpQuery,
        packed_mse: &[u8],
    ) -> f64 {
        use std::arch::x86_64::*;

        let mut acc_i32 = _mm256_setzero_si256();
        let lut = prep.mse_lut_i16.as_ptr();
        let lut_width = prep.mse_lut_width;
        let n = self.n;
        let packed_ptr = packed_mse.as_ptr();

        let mut i = 0usize;
        let mut byte_i = 0usize;

        // Process 16 indices = 8 packed bytes per iteration.
        while i + 16 <= n {
            let p64 = unsafe { std::ptr::read_unaligned(packed_ptr.add(byte_i) as *const u64) };
            let mut nibs = [0u8; 16];
            let mut bits = p64;
            for k in 0..8 {
                nibs[2 * k] = (bits & 0x0f) as u8;
                nibs[2 * k + 1] = ((bits >> 4) & 0x0f) as u8;
                bits >>= 8;
            }

            // First 8 lanes
            let mut vals = [0i16; 8];
            for j in 0..8 {
                unsafe {
                    vals[j] = *lut.add((i + j) * lut_width + nibs[j] as usize);
                }
            }
            let v128 = _mm_loadu_si128(vals.as_ptr() as *const __m128i);
            let v256 = _mm256_cvtepi16_epi32(v128);
            acc_i32 = _mm256_add_epi32(acc_i32, v256);

            // Second 8 lanes
            for j in 0..8 {
                unsafe {
                    vals[j] = *lut.add((i + 8 + j) * lut_width + nibs[8 + j] as usize);
                }
            }
            let v128 = _mm_loadu_si128(vals.as_ptr() as *const __m128i);
            let v256 = _mm256_cvtepi16_epi32(v128);
            acc_i32 = _mm256_add_epi32(acc_i32, v256);

            i += 16;
            byte_i += 8;
        }

        // Reduce acc_i32 horizontally.
        let mut tmp = [0i32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc_i32);
        let mut sum_i64: i64 = 0;
        for v in tmp {
            sum_i64 += v as i64;
        }

        // Scalar tail (n not multiple of 16).
        while i < n {
            let b = unsafe { *packed_ptr.add(byte_i) };
            let nib = if i % 2 == 0 { b & 0x0f } else { b >> 4 };
            sum_i64 += unsafe { *lut.add(i * lut_width + nib as usize) } as i64;
            i += 1;
            if i % 2 == 0 {
                byte_i += 1;
            }
        }

        // Convert i32 sum to f32 by multiplying by global scale.
        (sum_i64 as f32 * prep.mse_lut_i16_scale) as f64
    }

    pub fn quantize(&self, x: &[f32]) -> (Vec<CodeIndex>, Vec<u8>, f64) {
        let idx = self.mse_quantizer.quantize(x);

        if self.fast_mode {
            // Skip QJL residual: zero bits, gamma=0. ~30% faster ingest.
            let qjl = vec![0u8; self.n.div_ceil(8)];
            return (idx, qjl, 0.0);
        }

        let x_tilde_mse = self.mse_quantizer.dequantize(&idx);
        let mut residual = Array1::zeros(self.d);
        let mut gamma_sq = 0.0f64;
        for i in 0..self.d {
            let rv = x[i] as f64 - x_tilde_mse[i];
            residual[i] = rv;
            gamma_sq += rv * rv;
        }
        let gamma = gamma_sq.sqrt();
        let qjl = self.qjl_quantizer.quantize(&residual);

        (idx, qjl, gamma)
    }

    pub fn quantize_batch(&self, xs: &[&[f32]]) -> Vec<(Vec<CodeIndex>, Vec<u8>, f64)> {
        // Dense mode: batch all rotations via a single GEMM (Y = R × X) instead of
        // B separate matrix-vector multiplies. Break-even at B ≥ 64.
        const DENSE_BATCH_MIN: usize = 64;
        if self.mse_quantizer.rotation_matrix.is_some() && xs.len() >= DENSE_BATCH_MIN {
            return self.quantize_batch_dense_gemm(xs);
        }
        // Below this threshold the Rayon thread-park/unpark overhead exceeds the
        // quantization work on all supported platforms (especially Windows).
        const PAR_THRESHOLD: usize = 512;
        if xs.len() <= PAR_THRESHOLD {
            xs.iter().map(|x| self.quantize(x)).collect()
        } else {
            xs.par_iter().map(|x| self.quantize(x)).collect()
        }
    }

    /// Dense-mode batch quantization: rotate all B vectors at once via GEMM,
    /// then per-vector centroid lookup + optional QJL residual encoding.
    fn quantize_batch_dense_gemm(&self, xs: &[&[f32]]) -> Vec<(Vec<CodeIndex>, Vec<u8>, f64)> {
        let d = self.d;
        let b = xs.len();
        // Build input matrix X (d×B): column col = vector xs[col].
        let mut x_mat = DMatrix::<f32>::zeros(d, b);
        for (col, x) in xs.iter().enumerate() {
            for (row, &val) in x.iter().enumerate() {
                x_mat[(row, col)] = val;
            }
        }
        // GEMM: Y = R × X via matrixmultiply's SGEMM kernel.
        // Interpret the rotation matrix slice directly without DMatrix allocation.
        let mat = self.mse_quantizer.rotation_matrix.as_ref().unwrap();
        let mut y_mat = DMatrix::<f32>::zeros(d, b);
        // SAFETY: mat, x_mat, y_mat are valid f32 slices; dimensions match.
        unsafe {
            matrixmultiply::sgemm(
                d,
                d,
                b,
                1.0,
                mat.as_ptr(),
                d as isize,
                1, // R: row-major (d×d)
                x_mat.as_ptr(),
                1,
                d as isize, // X: col-major (d×b)
                0.0,
                y_mat.as_mut_ptr(),
                1,
                d as isize, // Y: col-major (d×b)
            );
        }

        // B1: parallelize the per-vector centroid lookup + (optional) QJL residual.
        // Each column is independent; using par_iter scales linearly with cores
        // for the dominant cost in dense ingest at high d.
        // Gate by TOTAL work (b * d). At small d the per-vector work is so cheap
        // that Rayon park/unpark overhead exceeds the saving (measured: d=200 was
        // 12% SLOWER under always-on par). Threshold tuned so d=1536+ at b≥1000
        // parallelizes (positive bench), d=200 stays sequential.
        const PAR_WORK_THRESHOLD: usize = 5_000_000;
        let parallel = b.saturating_mul(d) >= PAR_WORK_THRESHOLD;
        // Snapshot the columns we need before entering Rayon — DMatrix isn't Sync.
        let cols: Vec<Vec<f32>> = (0..b)
            .map(|col| y_mat.column(col).iter().copied().collect())
            .collect();
        let process = |col: usize, y_col: &[f32]| -> (Vec<CodeIndex>, Vec<u8>, f64) {
            let idx: Vec<CodeIndex> = y_col
                .iter()
                .map(|&val| self.mse_quantizer.nearest_centroid_index(val))
                .collect();
            if self.fast_mode {
                return (idx, vec![0u8; self.n.div_ceil(8)], 0.0f64);
            }
            let x_mse = self.mse_quantizer.dequantize(&idx);
            let mut residual = Array1::zeros(d);
            let mut gamma_sq = 0.0f64;
            for i in 0..d {
                let rv = xs[col][i] as f64 - x_mse[i];
                residual[i] = rv;
                gamma_sq += rv * rv;
            }
            let gamma = gamma_sq.sqrt();
            let qjl = self.qjl_quantizer.quantize(&residual);
            (idx, qjl, gamma)
        };
        if parallel {
            cols.par_iter()
                .enumerate()
                .map(|(c, y)| process(c, y))
                .collect()
        } else {
            cols.iter()
                .enumerate()
                .map(|(c, y)| process(c, y))
                .collect()
        }
    }

    pub fn dequantize(&self, idx: &[CodeIndex], qjl: &[u8], gamma: f64) -> Array1<f64> {
        let x_mse = self.mse_quantizer.dequantize(idx);
        if self.fast_mode {
            // fast_mode: no QJL stored — MSE reconstruction is the full estimate.
            return x_mse;
        }
        let x_qjl = self.qjl_quantizer.dequantize(qjl, gamma);
        x_mse + x_qjl
    }

    /// Rotate a d-dimensional query into the n-dimensional SRHT space used
    /// by dequantize().  Inner products and L2 distances are preserved by the
    /// orthogonal SRHT, so scoring rotate_query_for_reranking(q) against
    /// dequantize(codes) gives the same result as scoring q against the
    /// original vector — without needing live_vectors.bin.
    pub fn rotate_query_for_reranking(&self, query: &Array1<f64>) -> Array1<f64> {
        let x_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
        let mut y = vec![0.0f32; self.n];
        self.mse_quantizer.apply_rotation(&x_f32, &mut y);
        Array1::from_iter(y.iter().map(|&v| v as f64))
    }

    pub fn dequantize_single_no_parallel(
        &self,
        idx: &[CodeIndex],
        qjl: &[u8],
        gamma: f64,
    ) -> Array1<f64> {
        self.dequantize(idx, qjl, gamma)
    }

    pub fn dequantize_batch(&self, encoded: &[(Vec<CodeIndex>, Vec<u8>, f64)]) -> Vec<Array1<f64>> {
        encoded
            .par_iter()
            .map(|(idx, qjl, gamma)| self.dequantize(idx, qjl, *gamma))
            .collect()
    }

    pub fn pack_mse_indices(&self, indices: &[CodeIndex]) -> Vec<u8> {
        let bits_per_idx = self.mse_quantizer.b;
        if bits_per_idx == 8 {
            return indices.iter().map(|v| *v as u8).collect();
        }

        let total_bits = self.n * bits_per_idx;
        let mut packed = vec![0u8; total_bits.div_ceil(8)];

        for i in 0..self.n {
            let val = indices[i] as u32;
            let bit_start = i * bits_per_idx;
            for b in 0..bits_per_idx {
                if (val >> b) & 1 == 1 {
                    let bit_pos = bit_start + b;
                    packed[bit_pos / 8] |= 1 << (bit_pos % 8);
                }
            }
        }
        packed
    }

    pub fn unpack_mse_indices(&self, packed: &[u8], out: &mut [CodeIndex]) {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            return unsafe { self.unpack_mse_indices_avx2(packed, out) };
        }
        self.unpack_mse_indices_scalar(packed, out);
    }

    /// SIMD nibble unpacker for b=4 (the common production case).
    ///
    /// Processes 16 packed bytes per iteration → 32 u16 indices using:
    ///   - `_mm_and_si128` to extract lo/hi nibbles
    ///   - `_mm_unpacklo/hi_epi8` to interleave them in index order
    ///   - `_mm256_cvtepu8_epi16` to zero-extend u8 → u16 for storage
    ///
    /// b=2 and b=8 fall through to the scalar path (uncommon in practice).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn unpack_mse_indices_avx2(&self, packed: &[u8], out: &mut [CodeIndex]) {
        use std::arch::x86_64::*;

        let bits_per_idx = self.mse_quantizer.b;

        if bits_per_idx == 4 {
            // Each byte holds 2 nibbles: lo=index[2i], hi=index[2i+1].
            // Process 16 packed bytes → 32 u16 output values per iteration.
            // Bound by out.len() so we never write past the caller's buffer.
            let n = out.len().min(self.n);
            let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);
            let n_full = (n / 32) * 32; // full 32-index chunks
            let mut out_i = 0usize;
            let mut byte_i = 0usize;

            while out_i < n_full {
                let v = _mm_loadu_si128(packed.as_ptr().add(byte_i) as *const __m128i);
                // lo nibbles: bits[3:0] of each byte → index[2i]
                let lo = _mm_and_si128(v, mask_lo);
                // hi nibbles: bits[7:4] → index[2i+1] (shift right 4 within each u16 lane)
                let hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask_lo);
                // Interleave: [lo0,hi0,lo1,hi1,...] within each 8-byte half
                let interleaved_lo = _mm_unpacklo_epi8(lo, hi);
                let interleaved_hi = _mm_unpackhi_epi8(lo, hi);
                // Zero-extend u8 → u16 and store
                let out16_lo = _mm256_cvtepu8_epi16(interleaved_lo);
                let out16_hi = _mm256_cvtepu8_epi16(interleaved_hi);
                _mm256_storeu_si256(out.as_mut_ptr().add(out_i) as *mut __m256i, out16_lo);
                _mm256_storeu_si256(out.as_mut_ptr().add(out_i + 16) as *mut __m256i, out16_hi);
                out_i += 32;
                byte_i += 16;
            }
            // Scalar tail for any remainder.
            if out_i < n {
                self.unpack_mse_indices_scalar(&packed[byte_i..], &mut out[out_i..]);
            }
            return;
        }

        // b=2 and b=8: scalar path (uncommon configs).
        self.unpack_mse_indices_scalar(packed, out);
    }

    fn unpack_mse_indices_scalar(&self, packed: &[u8], out: &mut [CodeIndex]) {
        let bits_per_idx = self.mse_quantizer.b;
        if bits_per_idx == 8 {
            for (dst, src) in out.iter_mut().zip(packed.iter()) {
                *dst = *src as CodeIndex;
            }
            return;
        }
        let mask = (1u32 << bits_per_idx) - 1;
        let mut bit_pos = 0usize;
        for i in 0..out.len().min(self.n) {
            let byte_idx = bit_pos >> 3;
            let bit_off = bit_pos & 7;
            let mut val = (packed[byte_idx] as u32) >> bit_off;
            let mut bits = 8 - bit_off;
            let mut byte_offset = 1usize;
            while bits < bits_per_idx {
                val |= (packed[byte_idx + byte_offset] as u32) << bits;
                bits += 8;
                byte_offset += 1;
            }
            out[i] = (val & mask) as CodeIndex;
            bit_pos += bits_per_idx;
        }
    }

    /// Compute the query's QJL residual bits for use as a fast navigation fingerprint
    /// during HNSW graph traversal (see [`hamming_score`]).
    ///
    /// Encodes the query through the full `quantize()` pipeline and returns the
    /// bit-packed QJL codes of the MSE residual — the same representation stored
    /// per-vector in `live_codes.bin`. Comparing these bits against stored QJL codes
    /// with Hamming distance gives a fast (~64× cheaper than LUT) proxy for inner
    /// product proximity, enabling fast HNSW navigation followed by accurate LUT
    /// re-scoring of the final candidate set.
    #[allow(dead_code)]
    pub(crate) fn prepare_navigation_bits(&self, query: &Array1<f64>) -> Vec<u8> {
        let query_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
        let (_, qjl_bits, _) = self.quantize(&query_f32);
        qjl_bits
    }

    /// Compute query sign-bit sketch from the rotated query for fast Hamming
    /// prefiltering when fast_mode + b=4 (no QJL stored). Output length = n/8 bytes,
    /// where bit i of byte i/8 is 1 iff the i-th rotated dimension is non-negative.
    ///
    /// This matches the doc-side sketch derivable from the high bit of each MSE
    /// nibble (Lloyd-Max codebook is symmetric around 0, so idx >= 1<<(b-1) ↔
    /// centroid > 0).
    pub(crate) fn prepare_query_sign_bits(&self, query: &Array1<f64>) -> Vec<u8> {
        let query_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
        let mut y = vec![0.0f32; self.n];
        self.mse_quantizer.apply_rotation(&query_f32, &mut y);
        let mut bits = vec![0u8; self.n.div_ceil(8)];
        for i in 0..self.n {
            if y[i] >= 0.0 {
                bits[i >> 3] |= 1u8 << (i & 7);
            }
        }
        bits
    }
}

/// Hamming disagreement between a query sign sketch (`query_signs`) and the
/// document's implicit sign sketch derivable from b=4 packed MSE nibbles.
///
/// For each MSE byte (= two 4-bit indices), bit 3 is the sign of the low-nibble
/// index and bit 7 is the sign of the high-nibble index (since the Lloyd-Max
/// codebook is sorted by centroid value with negatives in the lower half). We
/// extract those 8 sign bits per 4 MSE bytes and XOR-popcount against the query.
///
/// Lower return value = closer in IP space. Caller is responsible for ensuring
/// `query_signs.len() == mse_bytes.len() / 4` (i.e. b=4 stride).
#[inline]
pub fn hamming_disagree_b4_signs(query_signs: &[u8], mse_bytes: &[u8]) -> u32 {
    debug_assert_eq!(query_signs.len() * 4, mse_bytes.len());
    let mut disagree = 0u32;
    let mut i = 0usize;
    let mut q_i = 0usize;
    // Process 8 mse bytes (16 nibbles, 16 sign bits = 2 sketch bytes) per iter.
    while i + 8 <= mse_bytes.len() {
        let chunk = u64::from_le_bytes(mse_bytes[i..i + 8].try_into().unwrap());
        // Mask = 0x88 per byte: bit 3 (low-nibble sign) and bit 7 (high-nibble sign).
        // Compress those 16 bits via shift/or into the low 16 bits of `packed`.
        // For each byte k (0..8) at bit offset 8k, we want output bits 2k and 2k+1.
        // Per byte: output bit 2k   = chunk bit (8k + 3)
        //           output bit 2k+1 = chunk bit (8k + 7)
        // Right-shift bit (8k+3) to position 2k => shift right by (8k+3) - 2k = 6k+3.
        // Right-shift bit (8k+7) to position 2k+1 => shift right by (8k+7) - (2k+1) = 6k+6.
        // Doing this per-bit is 16 shifts; cheaper to use scalar byte loop:
        let mut packed: u16 = 0;
        for k in 0..8 {
            let byte = (chunk >> (8 * k)) as u8;
            packed |= ((byte as u16 >> 3) & 0x01) << (2 * k);
            packed |= ((byte as u16 >> 7) & 0x01) << (2 * k + 1);
        }
        let q = u16::from_le_bytes([query_signs[q_i], query_signs[q_i + 1]]);
        disagree += (packed ^ q).count_ones();
        i += 8;
        q_i += 2;
    }
    // Tail: 4 mse bytes → 8 sign bits → 1 sketch byte.
    while i + 4 <= mse_bytes.len() {
        let m0 = mse_bytes[i];
        let m1 = mse_bytes[i + 1];
        let m2 = mse_bytes[i + 2];
        let m3 = mse_bytes[i + 3];
        let packed: u8 = ((m0 >> 3) & 0x01)
            | ((m0 >> 6) & 0x02)
            | ((m1 >> 1) & 0x04)
            | ((m1 >> 4) & 0x08)
            | ((m2 << 1) & 0x10)
            | ((m2 >> 2) & 0x20)
            | ((m3 << 3) & 0x40)
            | (m3 & 0x80);
        disagree += (packed ^ query_signs[q_i]).count_ones();
        i += 4;
        q_i += 1;
    }
    disagree
}

/// Normalized Hamming similarity between two bit-packed byte slices.
///
/// Returns `(agreements - disagreements) / n_bits` ∈ [-1.0, 1.0].  A score of
/// 1.0 means identical bit patterns; −1.0 means all bits flipped.  Processing
/// is done in 64-bit words (8 bytes at a time) to exploit native `POPCNT`.
///
/// The shorter slice determines the number of bits compared; any extra bytes in
/// the longer slice are ignored.
#[allow(dead_code)]
pub(crate) fn hamming_score(a: &[u8], b: &[u8]) -> f64 {
    let n_bytes = a.len().min(b.len());
    if n_bytes == 0 {
        return 0.0;
    }

    let mut disagreements = 0u32;
    let mut i = 0usize;

    // Process 8 bytes at a time as u64 for native POPCNT throughput.
    while i + 8 <= n_bytes {
        let wa = u64::from_le_bytes(a[i..i + 8].try_into().unwrap());
        let wb = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        disagreements += (wa ^ wb).count_ones();
        i += 8;
    }

    // Scalar tail for remaining bytes.
    while i < n_bytes {
        disagreements += (a[i] ^ b[i]).count_ones();
        i += 1;
    }

    let n_bits = (n_bytes * 8) as f64;
    1.0 - 2.0 * disagreements as f64 / n_bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_pq(d: usize) -> ProdQuantizer {
        ProdQuantizer::new(d, 4, 42)
    }

    fn make_dense_pq(d: usize) -> ProdQuantizer {
        ProdQuantizer::new_dense(d, 4, 42)
    }

    #[test]
    fn new_creates_correct_config() {
        let d = 64;
        let pq = make_pq(d);
        assert_eq!(pq.d, d);
        assert_eq!(pq.n, d.next_power_of_two());
        assert_eq!(pq.b, 4);
        assert!(!pq.fast_mode);
    }

    #[test]
    fn new_srht_sets_fast_mode() {
        let pq = ProdQuantizer::new_srht(64, 4, 42);
        assert!(pq.fast_mode);
    }

    #[test]
    fn quantize_returns_correct_output_sizes() {
        let d = 64;
        let pq = make_pq(d);
        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.01).collect();
        let (idx, qjl, gamma) = pq.quantize(&x);
        assert_eq!(idx.len(), pq.n);
        assert_eq!(qjl.len(), pq.n.div_ceil(8));
        assert!(gamma >= 0.0);
    }

    #[test]
    fn fast_mode_returns_zero_gamma() {
        let d = 64;
        let pq = ProdQuantizer::new_srht(d, 4, 42);
        let x: Vec<f32> = vec![1.0f32; d];
        let (_idx, _qjl, gamma) = pq.quantize(&x);
        assert_eq!(gamma, 0.0, "fast_mode should always produce gamma=0");
    }

    #[test]
    fn fast_mode_qjl_all_zero_bits() {
        let d = 64;
        let pq = ProdQuantizer::new_srht(d, 4, 42);
        let x: Vec<f32> = vec![1.0f32; d];
        let (_idx, qjl, _gamma) = pq.quantize(&x);
        assert!(
            qjl.iter().all(|&b| b == 0),
            "fast_mode qjl bits should all be 0"
        );
    }

    #[test]
    fn dequantize_returns_correct_dimension() {
        let d = 64;
        let pq = make_pq(d);
        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.01).collect();
        let (idx, qjl, gamma) = pq.quantize(&x);
        let recon = pq.dequantize(&idx, &qjl, gamma);
        assert_eq!(recon.len(), d);
    }

    #[test]
    fn quantize_dequantize_reasonable_reconstruction() {
        let d = 64;
        let pq = make_pq(d);
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1 - 3.2).sin()).collect();
        let (idx, qjl, gamma) = pq.quantize(&x);
        let recon = pq.dequantize(&idx, &qjl, gamma);
        let mse: f64 = x
            .iter()
            .zip(recon.iter())
            .map(|(&a, &b)| (a as f64 - b).powi(2))
            .sum::<f64>()
            / d as f64;
        assert!(mse < 2.0, "MSE too high: {}", mse);
    }

    #[test]
    fn pack_unpack_mse_indices_roundtrip() {
        let d = 64;
        let pq = make_pq(d);
        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.01).collect();
        let (idx, _, _) = pq.quantize(&x);
        let packed = pq.pack_mse_indices(&idx);
        let mut unpacked = vec![0u16; pq.n];
        pq.unpack_mse_indices(&packed, &mut unpacked);
        assert_eq!(idx, unpacked, "pack/unpack roundtrip should be lossless");
    }

    #[test]
    fn pack_mse_8bit_path_roundtrip() {
        // b=9 → bits_per_idx = 8 → uses the fast 1-byte-per-index path
        let d = 64;
        let pq = ProdQuantizer::new(d, 9, 42);
        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.005).collect();
        let (idx, _, _) = pq.quantize(&x);
        let packed = pq.pack_mse_indices(&idx);
        assert_eq!(packed.len(), pq.n, "8-bit path: 1 byte per index");
        let mut unpacked = vec![0u16; pq.n];
        pq.unpack_mse_indices(&packed, &mut unpacked);
        assert_eq!(idx, unpacked);
    }

    #[test]
    fn pack_mse_low_bits_roundtrip() {
        // b=2 → bits_per_idx=1
        let d = 32;
        let pq = ProdQuantizer::new(d, 2, 42);
        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.01).collect();
        let (idx, _, _) = pq.quantize(&x);
        let packed = pq.pack_mse_indices(&idx);
        let mut unpacked = vec![0u16; pq.n];
        pq.unpack_mse_indices(&packed, &mut unpacked);
        assert_eq!(idx, unpacked);
    }

    #[test]
    fn quantize_batch_matches_single() {
        let d = 32;
        let pq = make_pq(d);
        let x1: Vec<f32> = (0..d).map(|i| i as f32 * 0.01).collect();
        let x2: Vec<f32> = (0..d).map(|i| -(i as f32) * 0.01).collect();
        let batch = pq.quantize_batch(&[&x1, &x2]);
        let single1 = pq.quantize(&x1);
        let single2 = pq.quantize(&x2);
        assert_eq!(batch[0], single1);
        assert_eq!(batch[1], single2);
    }

    #[test]
    fn dequantize_batch_matches_single() {
        let d = 32;
        let pq = make_pq(d);
        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.01).collect();
        let (idx, qjl, gamma) = pq.quantize(&x);
        let single = pq.dequantize(&idx, &qjl, gamma);
        let batch = pq.dequantize_batch(&[(idx, qjl, gamma)]);
        for i in 0..d {
            let diff = (single[i] - batch[0][i]).abs();
            assert!(
                diff < 1e-4,
                "mismatch at dim {}: {} vs {}",
                i,
                single[i],
                batch[0][i]
            );
        }
    }

    #[test]
    fn prepare_ip_query_and_score_positive() {
        let d = 32;
        let pq = make_pq(d);
        let x: Vec<f32> = vec![1.0f32; d];
        let (idx, qjl, gamma) = pq.quantize(&x);
        let query = Array1::from_iter(x.iter().map(|&v| v as f64));
        let prep = pq.prepare_ip_query(&query);
        let score = pq.score_ip_encoded(&prep, &idx, &qjl, gamma);
        assert!(score > 0.0, "self IP score should be positive: {}", score);
    }

    #[test]
    fn prepare_ip_query_lite_and_score_positive() {
        let d = 32;
        let pq = make_pq(d);
        let x: Vec<f32> = vec![1.0f32; d];
        let (idx, qjl, gamma) = pq.quantize(&x);
        let query = Array1::from_iter(x.iter().map(|&v| v as f64));
        let prep_lite = pq.prepare_ip_query_lite(&query);
        let score = pq.score_ip_encoded_lite(&prep_lite, &idx, &qjl, gamma);
        assert!(
            score > 0.0,
            "lite self IP score should be positive: {}",
            score
        );
    }

    #[test]
    fn prepare_ip_query_from_codes_scores_finite() {
        let d = 32;
        let pq = make_pq(d);
        let x: Vec<f32> = vec![1.0f32; d];
        let (idx, qjl, gamma) = pq.quantize(&x);
        let prep = pq.prepare_ip_query_from_codes(&idx);
        let score = pq.score_ip_encoded_lite(&prep, &idx, &qjl, gamma);
        assert!(score.is_finite(), "score from codes should be finite");
    }

    #[test]
    fn score_ip_with_zero_gamma_skips_qjl() {
        let d = 32;
        let pq = make_pq(d);
        let x: Vec<f32> = vec![0.5f32; d];
        let (idx, qjl, _gamma) = pq.quantize(&x);
        let query = Array1::from_iter(x.iter().map(|&v| v as f64));

        let prep = pq.prepare_ip_query(&query);
        let score_full = pq.score_ip_encoded(&prep, &idx, &qjl, 0.0);
        assert!(score_full.is_finite());

        let prep_lite = pq.prepare_ip_query_lite(&query);
        let score_lite = pq.score_ip_encoded_lite(&prep_lite, &idx, &qjl, 0.0);
        assert!(score_lite.is_finite());
    }

    #[test]
    fn rotate_query_for_reranking_returns_n_dims() {
        let d = 64;
        let pq = make_pq(d);
        let query = Array1::from_iter((0..d).map(|i| i as f64 * 0.01));
        let rotated = pq.rotate_query_for_reranking(&query);
        assert_eq!(rotated.len(), pq.n);
    }

    #[test]
    fn dequantize_single_no_parallel_matches_dequantize() {
        let d = 32;
        let pq = make_pq(d);
        let x: Vec<f32> = vec![0.5f32; d];
        let (idx, qjl, gamma) = pq.quantize(&x);
        let a = pq.dequantize(&idx, &qjl, gamma);
        let b = pq.dequantize_single_no_parallel(&idx, &qjl, gamma);
        for i in 0..d {
            let diff = (a[i] - b[i]).abs();
            assert!(diff < 1e-5, "mismatch at dim {}", i);
        }
    }

    #[test]
    fn score_ip_query_vs_query_lite_close() {
        // Scores from full LUT vs lite centroid lookup should be close (same input)
        let d = 32;
        let pq = make_pq(d);
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.13).sin()).collect();
        let (idx, qjl, gamma) = pq.quantize(&x);
        let query = Array1::from_iter(x.iter().map(|&v| v as f64));

        let prep = pq.prepare_ip_query(&query);
        let score_full = pq.score_ip_encoded(&prep, &idx, &qjl, gamma);

        let prep_lite = pq.prepare_ip_query_lite(&query);
        let score_lite = pq.score_ip_encoded_lite(&prep_lite, &idx, &qjl, gamma);

        // They use different paths but both estimate the same inner product
        let diff = (score_full - score_lite).abs();
        assert!(
            diff < 10.0,
            "full vs lite score should be reasonably close: {} vs {}",
            score_full,
            score_lite
        );
    }

    #[test]
    fn exact_mode_scores_match_dequantized_inner_product() {
        let d = 32;
        let pq = make_dense_pq(d);
        let x: Vec<f32> = (0..d)
            .map(|i| ((i as f32 * 0.17).sin() + 0.25).cos())
            .collect();
        let query = Array1::from_iter(
            (0..d).map(|i| ((i as f64 * 0.11).cos() - 0.2) * (1.0 + i as f64 * 0.001)),
        );

        let (idx, qjl, gamma) = pq.quantize(&x);
        let dequantized = pq.dequantize(&idx, &qjl, gamma);
        let direct: f64 = query
            .iter()
            .zip(dequantized.iter())
            .map(|(q, x_hat)| q * x_hat)
            .sum();

        let prep = pq.prepare_ip_query(&query);
        let score = pq.score_ip_encoded(&prep, &idx, &qjl, gamma);
        let full_diff = (score - direct).abs();
        assert!(
            full_diff < 1e-4,
            "exact-mode full score should match query·dequantize(code): score={score}, direct={direct}, diff={full_diff}"
        );

        let prep_lite = pq.prepare_ip_query_lite(&query);
        let score_lite = pq.score_ip_encoded_lite(&prep_lite, &idx, &qjl, gamma);
        let lite_diff = (score_lite - direct).abs();
        assert!(
            lite_diff < 1e-4,
            "exact-mode lite score should match query·dequantize(code): score={score_lite}, direct={direct}, diff={lite_diff}"
        );
    }

    /// Directly invoke the scalar scoring paths (avoids AVX2 dispatch on x86_64).
    #[test]
    fn score_ip_encoded_scalar_gives_finite_result() {
        let d = 32;
        let pq = make_pq(d);
        let x: Vec<f32> = vec![1.0f32; d];
        let (idx, qjl, gamma) = pq.quantize(&x);
        let query = Array1::from_iter(x.iter().map(|&v| v as f64));
        let prep = pq.prepare_ip_query(&query);
        let score = pq.score_ip_encoded_scalar(&prep, &idx, &qjl, gamma);
        assert!(
            score.is_finite(),
            "scalar score_ip_encoded should be finite: {}",
            score
        );
        assert!(score > 0.0, "scalar self-score should be positive");
    }

    #[test]
    fn score_ip_encoded_lite_scalar_gives_finite_result() {
        let d = 32;
        let pq = make_pq(d);
        let x: Vec<f32> = vec![1.0f32; d];
        let (idx, qjl, gamma) = pq.quantize(&x);
        let query = Array1::from_iter(x.iter().map(|&v| v as f64));
        let prep_lite = pq.prepare_ip_query_lite(&query);
        let score = pq.score_ip_encoded_lite_scalar(&prep_lite, &idx, &qjl, gamma);
        assert!(
            score.is_finite(),
            "scalar score_ip_encoded_lite should be finite: {}",
            score
        );
        assert!(score > 0.0, "scalar lite self-score should be positive");
    }

    #[test]
    fn score_ip_encoded_scalar_zero_gamma_path() {
        let d = 32;
        let pq = make_pq(d);
        let x: Vec<f32> = vec![0.5f32; d];
        let (idx, qjl, _gamma) = pq.quantize(&x);
        let query = Array1::from_iter(x.iter().map(|&v| v as f64));
        let prep = pq.prepare_ip_query(&query);
        let score = pq.score_ip_encoded_scalar(&prep, &idx, &qjl, 0.0);
        assert!(score.is_finite());
    }

    #[test]
    fn score_ip_encoded_lite_scalar_zero_gamma_path() {
        let d = 32;
        let pq = make_pq(d);
        let x: Vec<f32> = vec![0.5f32; d];
        let (idx, qjl, _gamma) = pq.quantize(&x);
        let query = Array1::from_iter(x.iter().map(|&v| v as f64));
        let prep_lite = pq.prepare_ip_query_lite(&query);
        let score = pq.score_ip_encoded_lite_scalar(&prep_lite, &idx, &qjl, 0.0);
        assert!(score.is_finite());
    }

    // ── SIMD scalar tail coverage (lines 239-242, 278-284) ───────────────────
    // With d=4, n=4: the SIMD main loop `while i+7 < 4` never runs,
    // so the MSE scalar tail and QJL scalar tail are both exercised.

    #[test]
    fn small_dim_simd_scalar_tail_covered_with_d4() {
        // d=4 → n=4; SIMD main loop `while i+7 < n` never executes →
        // scalar tail runs for all 4 dims (lines 239-242)
        // QJL: qjl_len=1, base_i=0, 0+7=7 >= 4 → scalar QJL tail (lines 278-284)
        let d = 4;
        let pq = ProdQuantizer::new(d, 4, 42);
        let x: Vec<f32> = vec![0.7f32, 0.1, 0.5, 0.3];
        let (idx, qjl, gamma) = pq.quantize(&x);
        let query = Array1::from_iter(x.iter().map(|&v| v as f64));
        let prep_lite = pq.prepare_ip_query_lite(&query);
        let score = pq.score_ip_encoded_lite(&prep_lite, &idx, &qjl, gamma);
        assert!(
            score.is_finite(),
            "d=4 scalar-tail score must be finite: {score}"
        );
        // Also test with non-zero gamma to exercise the full QJL scalar tail
        let score_g = pq.score_ip_encoded_lite(&prep_lite, &idx, &qjl, 1.0);
        assert!(
            score_g.is_finite(),
            "d=4 scalar-tail with gamma=1 must be finite: {score_g}"
        );
    }

    // ── score_ip_encoded with d=4: MSE+QJL scalar tails (lines 390-394, 414-423) ──
    // score_ip_encoded_lite was already tested above; this exercises score_ip_encoded
    // (the full SIMD path) with d=4 so n=4 and the scalar-tail branches are taken.

    #[test]
    fn score_ip_encoded_d4_exercises_simd_scalar_tails() {
        let d = 4;
        let pq = ProdQuantizer::new(d, 4, 42);
        let x: Vec<f32> = vec![0.7f32, 0.1, 0.5, 0.3];
        let (idx, qjl, gamma) = pq.quantize(&x);
        let query = Array1::from_iter(x.iter().map(|&v| v as f64));
        // prepare_ip_query (full version, not _lite) is required for score_ip_encoded
        let prep = pq.prepare_ip_query(&query);
        // With d=4 (n=4): SIMD main loop `while i+7 < 4` never runs → MSE scalar tail
        // (lines 390-394); QJL base_i=0, 0+7=7>=4 → QJL scalar tail (lines 414-423)
        let score = pq.score_ip_encoded(&prep, &idx, &qjl, gamma);
        assert!(
            score.is_finite(),
            "d=4 score_ip_encoded must be finite: {score}"
        );
        let score_g = pq.score_ip_encoded(&prep, &idx, &qjl, 1.0);
        assert!(
            score_g.is_finite(),
            "d=4 score_ip_encoded with gamma=1 must be finite: {score_g}"
        );
    }

    #[test]
    fn encode_decode_roundtrip_preserves_order() {
        // v1 is parallel to query → high IP; v2 is orthogonal → IP ≈ 0.
        // Quantized scores must preserve this ordering.
        let d = 64;
        let pq = ProdQuantizer::new(d, 4, 42);

        let scale = 1.0_f32 / (d as f32).sqrt();
        let query: Vec<f32> = vec![scale; d];
        // v1 in the same direction, twice the magnitude → IP(q, v1) = 2.0
        let v1: Vec<f32> = vec![scale * 2.0; d];
        // v2 orthogonal to query: first half positive, second half negative → IP(q, v2) = 0.0
        let v2: Vec<f32> = (0..d)
            .map(|i| if i < d / 2 { scale } else { -scale })
            .collect();

        let query_arr = Array1::from_iter(query.iter().map(|&v| v as f64));
        let (idx1, qjl1, g1) = pq.quantize(&v1);
        let (idx2, qjl2, g2) = pq.quantize(&v2);

        let prep = pq.prepare_ip_query(&query_arr);
        let score1 = pq.score_ip_encoded(&prep, &idx1, &qjl1, g1);
        let score2 = pq.score_ip_encoded(&prep, &idx2, &qjl2, g2);

        assert!(
            score1 > score2,
            "parallel vector (score {:.4}) should outscore orthogonal vector (score {:.4})",
            score1,
            score2
        );
    }

    #[test]
    fn fast_mode_scores_correlate_with_full_mode() {
        // Both modes share the same MSE codebook; QJL correction is the only difference.
        // Rank order should agree on at least 60% of pairs (Kendall's tau ≥ 0.6).
        let d = 64;
        let pq_full = ProdQuantizer::new(d, 4, 42);
        let pq_fast = ProdQuantizer::new_srht(d, 4, 42);

        let scale = 1.0_f32 / (d as f32).sqrt();
        let query = Array1::from_iter((0..d).map(|i| (i as f64 * 0.1 + 1.0) * scale as f64));

        let n = 12usize;
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..d)
                    .map(|j| ((i * d + j) as f32 * 0.07 + 0.3).sin() * scale)
                    .collect()
            })
            .collect();

        let prep_full = pq_full.prepare_ip_query(&query);
        let prep_fast = pq_fast.prepare_ip_query(&query);

        let scores_full: Vec<f64> = vecs
            .iter()
            .map(|v| {
                let (idx, qjl, gamma) = pq_full.quantize(v);
                pq_full.score_ip_encoded(&prep_full, &idx, &qjl, gamma)
            })
            .collect();

        let scores_fast: Vec<f64> = vecs
            .iter()
            .map(|v| {
                let (idx, qjl, gamma) = pq_fast.quantize(v);
                pq_fast.score_ip_encoded(&prep_fast, &idx, &qjl, gamma)
            })
            .collect();

        let mut concordant = 0usize;
        let mut total = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                total += 1;
                let full_order = scores_full[i] > scores_full[j];
                let fast_order = scores_fast[i] > scores_fast[j];
                if full_order == fast_order {
                    concordant += 1;
                }
            }
        }

        let tau = concordant as f64 / total as f64;
        assert!(
            tau >= 0.6,
            "rank correlation (Kendall's tau) too low: {:.2} ({}/{} concordant pairs)",
            tau,
            concordant,
            total
        );
    }

    // ── QJL-Hamming navigation tests ──────────────────────────────────────────

    #[test]
    fn prepare_navigation_bits_returns_correct_length() {
        let d = 64;
        let pq = make_pq(d);
        let query = Array1::from_iter((0..d).map(|i| i as f64 * 0.01));
        let bits = pq.prepare_navigation_bits(&query);
        assert_eq!(bits.len(), pq.n.div_ceil(8));
    }

    #[test]
    fn hamming_score_identical_bits_returns_one() {
        let a = vec![0b10110011u8, 0b01001100u8, 0xFF, 0x00];
        assert!(
            (hamming_score(&a, &a) - 1.0).abs() < 1e-9,
            "identical bits → score 1.0"
        );
    }

    #[test]
    fn hamming_score_all_flipped_returns_minus_one() {
        let a = vec![0xAAu8; 8];
        let b: Vec<u8> = a.iter().map(|x| !x).collect();
        let s = hamming_score(&a, &b);
        assert!(
            (s - (-1.0)).abs() < 1e-9,
            "all bits flipped → score −1.0, got {}",
            s
        );
    }

    #[test]
    fn hamming_score_is_symmetric() {
        let a = vec![0b10101010u8, 0b11001100u8, 0x42, 0x99];
        let b = vec![0b01010101u8, 0b00110011u8, 0x24, 0x66];
        assert!(
            (hamming_score(&a, &b) - hamming_score(&b, &a)).abs() < 1e-12,
            "hamming_score must be symmetric"
        );
    }

    #[test]
    fn hamming_score_in_range() {
        let a = vec![0x5Au8; 32];
        let b = vec![0x3Cu8; 32];
        let s = hamming_score(&a, &b);
        assert!(
            (-1.0..=1.0).contains(&s),
            "hamming_score must be in [-1, 1], got {}",
            s
        );
    }

    #[test]
    fn navigation_bits_similar_vectors_closer_than_orthogonal() {
        // Query direction: uniform unit vector.
        // v_parallel ≈ query direction → high Hamming agreement.
        // v_orthogonal ⊥ query → ~random bit agreement (Hamming ≈ 0.0).
        let d = 128;
        let pq = make_pq(d);
        let scale = 1.0_f64 / (d as f64).sqrt();
        let query = Array1::from_iter(vec![scale; d]);
        let v_parallel: Vec<f32> = vec![scale as f32 * 0.9; d];
        let v_orthogonal: Vec<f32> = (0..d)
            .map(|i| {
                if i < d / 2 {
                    scale as f32
                } else {
                    -scale as f32
                }
            })
            .collect();

        let q_bits = pq.prepare_navigation_bits(&query);
        let (_, qjl_par, _) = pq.quantize(&v_parallel);
        let (_, qjl_orth, _) = pq.quantize(&v_orthogonal);

        let s_par = hamming_score(&q_bits, &qjl_par);
        let s_orth = hamming_score(&q_bits, &qjl_orth);

        assert!(
            s_par > s_orth,
            "similar vector (score {:.3}) should beat orthogonal vector (score {:.3}) on Hamming",
            s_par,
            s_orth
        );
    }

    // ── v0.8.2 audit: dimension assertions + numeric edges ────────────────────

    /// A5: passing `idx` shorter than `quantizer.n` panics with a clear message
    /// (rather than the previous undefined-behavior get_unchecked read).
    #[test]
    #[should_panic(expected = "idx.len() must equal quantizer.n")]
    fn score_ip_encoded_panics_on_short_idx() {
        let pq = make_pq(64);
        let q: Array1<f64> = Array1::from_elem(64, 0.1);
        let prep = pq.prepare_ip_query(&q);
        let short_idx = vec![0 as CodeIndex; pq.n - 1];
        let qjl_len = (pq.n + 7) / 8;
        let qjl = vec![0u8; qjl_len];
        let _ = pq.score_ip_encoded(&prep, &short_idx, &qjl, 0.0);
    }

    /// A5: same for the lite variant.
    #[test]
    #[should_panic(expected = "idx.len() must equal quantizer.n")]
    fn score_ip_encoded_lite_panics_on_short_idx() {
        let pq = make_pq(64);
        let q: Array1<f64> = Array1::from_elem(64, 0.1);
        let prep = pq.prepare_ip_query_lite(&q);
        let short_idx = vec![0 as CodeIndex; pq.n - 1];
        let qjl_len = (pq.n + 7) / 8;
        let qjl = vec![0u8; qjl_len];
        let _ = pq.score_ip_encoded_lite(&prep, &short_idx, &qjl, 0.0);
    }

    /// C4: zero vector quantize/dequantize is a no-panic edge case. The result
    /// is approximately zero (within quantization noise from the rotation step).
    #[test]
    fn quantize_zero_vector_does_not_panic() {
        let d = 64;
        let pq = make_pq(d);
        let zero: Vec<f32> = vec![0.0; d];
        let (idx, qjl, gamma) = pq.quantize(&zero);
        assert_eq!(idx.len(), pq.n);
        assert_eq!(qjl.len(), (pq.n + 7) / 8);
        // Dequantize and verify result is finite (NaN/Inf would fail downstream).
        let recon = pq.dequantize(&idx, &qjl, gamma);
        for v in recon.iter() {
            assert!(
                v.is_finite(),
                "dequantized zero vector should be finite, got {v}"
            );
        }
    }

    /// C4: a unit basis vector quantizes without producing NaN, and the score
    /// against itself via `score_ip_encoded` is finite and matches the
    /// `dequantize_then_score` path within rounding.
    #[test]
    fn dequantize_then_score_matches_score_ip_encoded() {
        let d = 64;
        let pq = make_pq(d);
        let mut v: Vec<f32> = vec![0.0; d];
        v[0] = 1.0;
        let (idx, qjl, gamma) = pq.quantize(&v);
        let q: Array1<f64> = Array1::from_iter(v.iter().map(|&x| x as f64));

        let prep = pq.prepare_ip_query(&q);
        let direct_score = pq.score_ip_encoded(&prep, &idx, &qjl, gamma);

        let recon = pq.dequantize(&idx, &qjl, gamma);
        let dot_score: f64 = q.iter().zip(recon.iter()).map(|(a, b)| a * b).sum();

        assert!(direct_score.is_finite(), "direct score must be finite");
        assert!(dot_score.is_finite(), "dequant+dot score must be finite");
        // The two paths use slightly different math (LUT rounding vs direct
        // dequantize), but should agree within ~1% of |query|·|recon|.
        let rel = ((direct_score - dot_score).abs() + 1e-6) / (dot_score.abs() + 1e-6);
        assert!(
            rel < 0.10,
            "direct ({direct_score:.4}) and dequant+dot ({dot_score:.4}) should agree within 10%; got rel={rel:.3}"
        );
    }
}
