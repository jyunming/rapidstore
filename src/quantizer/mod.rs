//! Quantizer implementations for TurboQuant two-stage vector quantization.
//!
//! ## Pipeline
//!
//! 1. [`mse::MseQuantizer`] — Applies a seeded SRHT rotation, then encodes each
//!    dimension using a Lloyd-Max scalar codebook (b-1 bits per dimension).
//! 2. [`qjl::QjlQuantizer`] — Projects the MSE residual via a dense Gaussian
//!    matrix and stores the sign bit (1 bit per dimension, bit-packed).
//! 3. [`prod::ProdQuantizer`] — Orchestrates both stages, manages lookup tables
//!    for fast asymmetric inner-product scoring, and exposes batch encode/decode.
//!
//! Quantization is **data-oblivious**: all random matrices are derived from a
//! single integer seed — no training, no fitting, no warm-up phase.

// Quantizer implementation
pub mod codebook;
pub mod mse;
pub mod prod;
pub mod qjl;

pub type CodeIndex = u16;
