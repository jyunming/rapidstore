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

    /// Fast-path: skips the QJL residual quantization step. Ingest is ~30% faster;
    /// approximate scores omit the residual correction (slightly lower recall).
    pub fn new_srht(d: usize, b: usize, seed: u64) -> Self {
        let mse_quantizer = MseQuantizer::new(d, b - 1, seed);
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

        let mut sq = vec![0.0f32; self.n];
        self.qjl_quantizer.apply_projection(&query_f32, &mut sq);

        PreparedIpQuery {
            mse_lut,
            mse_lut_width: lut_w,
            sq,
            qjl_scale: (PI / 2.0).sqrt() / self.n as f32,
        }
    }

    pub fn prepare_ip_query_lite(&self, query: &Array1<f64>) -> PreparedIpQueryLite {
        assert_eq!(query.len(), self.d);

        let query_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
        let mut y = vec![0.0f32; self.n];
        self.mse_quantizer.apply_rotation(&query_f32, &mut y);

        let mut sq = vec![0.0f32; self.n];
        self.qjl_quantizer.apply_projection(&query_f32, &mut sq);

        PreparedIpQueryLite {
            y,
            sq,
            qjl_scale: (PI / 2.0).sqrt() / self.n as f32,
        }
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
            qjl_scale: (PI / 2.0).sqrt() / self.n as f32,
        }
    }

    pub fn score_ip_encoded_lite(
        &self,
        prep: &PreparedIpQueryLite,
        idx: &[CodeIndex],
        qjl: &[u8],
        gamma: f64,
    ) -> f64 {
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

        let mut b = 0;
        while b < qjl_len {
            let byte = unsafe { *qjl_ptr.add(b) as i32 };
            let base_i = b << 3;
            if base_i + 7 < self.n {
                let s_vec = _mm256_loadu_ps(unsafe { sq.add(base_i) });
                let mut signs = [0.0f32; 8];
                for bit in 0..8 {
                    signs[bit] = if (byte & (1 << bit)) != 0 { 1.0 } else { -1.0 };
                }
                let sign_vec = _mm256_loadu_ps(signs.as_ptr());
                qjl_acc = _mm256_add_ps(qjl_acc, _mm256_mul_ps(s_vec, sign_vec));
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
        xs.par_iter().map(|x| self.quantize(x)).collect()
    }

    pub fn dequantize(&self, idx: &[CodeIndex], qjl: &[u8], gamma: f64) -> Array1<f64> {
        let x_mse = self.mse_quantizer.dequantize(idx);
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
        let bits_per_idx = self.b - 1;
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
        let bits_per_idx = self.b - 1;
        if bits_per_idx == 8 {
            for (dst, src) in out.iter_mut().zip(packed.iter()) {
                *dst = *src as CodeIndex;
            }
            return;
        }
        let mask = (1u32 << bits_per_idx) - 1;
        let mut bit_pos = 0usize;
        for i in 0..self.n {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_pq(d: usize) -> ProdQuantizer {
        ProdQuantizer::new(d, 4, 42)
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
}
