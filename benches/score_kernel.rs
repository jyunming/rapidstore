// Microbench harness for the brute-force scoring hot path.
// Run with: cargo bench --bench score_kernel
// Or a single group: cargo bench --bench score_kernel score_packed
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use tqdb::quantizer::prod::{hamming_disagree_b4_signs, ProdQuantizer};

const DIMS: &[usize] = &[200, 1536, 3072];
const N_SLOTS: usize = 4096; // microbench scope: in-cache, isolates kernel arithmetic

fn gen_vec(d: usize, rng: &mut StdRng) -> Vec<f32> {
    let dist = StandardNormal;
    (0..d).map(|_| Distribution::<f64>::sample(&dist, rng) as f32).collect()
}

fn bench_score_packed_b4(c: &mut Criterion) {
    let mut group = c.benchmark_group("score_packed_b4");
    for &d in DIMS {
        let mut rng = StdRng::seed_from_u64(42);
        let q = ProdQuantizer::new_dense_fast(d, 4, 7);

        let v = gen_vec(d, &mut rng);
        let (idx, qjl, gamma) = q.quantize(&v);
        let packed = q.pack_mse_indices(&idx);

        let qf32 = gen_vec(d, &mut rng);
        let qarr: Array1<f64> = Array1::from(qf32.iter().map(|&x| x as f64).collect::<Vec<_>>());
        let prep = q.prepare_ip_query(&qarr);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, _| {
            b.iter(|| {
                let s = q.score_ip_encoded_packed(
                    black_box(&prep),
                    black_box(&packed),
                    black_box(&qjl),
                    black_box(gamma),
                );
                black_box(s)
            });
        });
    }
    group.finish();
}

fn bench_prepare_ip_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("prepare_ip_query");
    for &d in DIMS {
        let mut rng = StdRng::seed_from_u64(42);
        let q = ProdQuantizer::new_dense_fast(d, 4, 7);
        let qf32 = gen_vec(d, &mut rng);
        let qarr: Array1<f64> = Array1::from(qf32.iter().map(|&x| x as f64).collect::<Vec<_>>());

        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, _| {
            b.iter(|| {
                let p = q.prepare_ip_query(black_box(&qarr));
                black_box(p)
            });
        });
    }
    group.finish();
}

fn bench_hamming_disagree(c: &mut Criterion) {
    let mut group = c.benchmark_group("hamming_disagree_b4");
    for &d in DIMS {
        let mut rng = StdRng::seed_from_u64(42);
        let q = ProdQuantizer::new_dense_fast(d, 4, 7);

        // Per-slot packed MSE bytes, plus a query "sign byte" sketch (one nibble per dim sign-bit
        // packed as the high bit of each nibble — same shape the prefilter uses).
        let v = gen_vec(d, &mut rng);
        let (idx, _, _) = q.quantize(&v);
        let mse_packed = q.pack_mse_indices(&idx);
        // The query-side input to hamming_disagree_b4_signs is a 1-bit-per-dim sign vector.
        // Sketch length is d/8 bytes (8 dims per byte).
        let sketch_len = d.next_power_of_two() / 8;
        let mut q_signs = vec![0u8; sketch_len];
        for byte in q_signs.iter_mut() {
            *byte = (rng.next_u32() & 0xff) as u8;
        }

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, _| {
            b.iter(|| {
                let h = hamming_disagree_b4_signs(black_box(&q_signs), black_box(&mse_packed));
                black_box(h)
            });
        });
    }
    group.finish();
}

fn bench_per_corpus_scan(c: &mut Criterion) {
    // Sequential scan microbench: how long to score N_SLOTS slots end-to-end?
    // Compares apples-to-apples to the d=1536 hot path used in production.
    let mut group = c.benchmark_group("scan_4k_slots");
    for &d in DIMS {
        let mut rng = StdRng::seed_from_u64(42);
        let q = ProdQuantizer::new_dense_fast(d, 4, 7);

        let mut packed_per_slot: Vec<Vec<u8>> = Vec::with_capacity(N_SLOTS);
        let mut qjl_per_slot: Vec<Vec<u8>> = Vec::with_capacity(N_SLOTS);
        let mut gamma_per_slot: Vec<f64> = Vec::with_capacity(N_SLOTS);
        for _ in 0..N_SLOTS {
            let v = gen_vec(d, &mut rng);
            let (idx, qjl, gamma) = q.quantize(&v);
            packed_per_slot.push(q.pack_mse_indices(&idx));
            qjl_per_slot.push(qjl);
            gamma_per_slot.push(gamma);
        }

        let qf32 = gen_vec(d, &mut rng);
        let qarr: Array1<f64> = Array1::from(qf32.iter().map(|&x| x as f64).collect::<Vec<_>>());
        let prep = q.prepare_ip_query(&qarr);

        group.throughput(Throughput::Elements(N_SLOTS as u64));
        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, _| {
            b.iter(|| {
                let mut acc = 0.0f64;
                for s in 0..N_SLOTS {
                    acc += q.score_ip_encoded_packed(
                        &prep,
                        &packed_per_slot[s],
                        &qjl_per_slot[s],
                        gamma_per_slot[s],
                    );
                }
                black_box(acc)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_score_packed_b4,
    bench_prepare_ip_query,
    bench_hamming_disagree,
    bench_per_corpus_scan,
);
criterion_main!(benches);
