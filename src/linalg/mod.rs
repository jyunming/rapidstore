//! Linear algebra primitives used by the TurboQuant quantizer.
//!
//! | Module | Responsibility |
//! |--------|---------------|
//! | [`hadamard`] | In-place O(d log d) Fast Walsh-Hadamard Transform (FWHT) and SRHT |
//! | [`matmul`] | GEMM / SGEMM wrappers via the `matrixmultiply` crate |
//! | [`rotation`] | Seeded random rotation / projection matrix generators |

// Linear algebra utilities
pub mod hadamard;
pub mod matmul;
pub mod rotation;
