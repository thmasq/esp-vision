pub mod alloc;
pub mod matrix;
pub mod storage;

use alloc::AlignedVec;
use nalgebra::{Dyn, Matrix};
use storage::EspAlignedStorage;

/// A dynamically sized, 16-byte aligned matrix specifically designed for `esp-dsp` SIMD acceleration.
pub type AlignedDMat<T> = Matrix<T, Dyn, Dyn, EspAlignedStorage<T, Dyn, Dyn>>;

/// Extension trait to provide constructor methods.
pub trait AlignedDMatExt<T> {
    fn zeros(nrows: usize, ncols: usize) -> Self;
    fn from_slice(nrows: usize, ncols: usize, slice: &[T]) -> Self;
}

/// Extension trait exposing hardware accelerated math routines (Floating Point).
pub trait EspMatrixMath {
    /// Multiplies `self` by `rhs` utilizing ESP32-S3 hardware acceleration when possible.
    fn esp_mul(&self, rhs: &Self) -> Self;

    /// Multiplies a sub-region of `self` by a sub-region of `rhs` using ESP32-S3 SIMD.
    /// `a_stride` and `b_stride` are the total columns of the parent buffers.
    /// Returns a new, contiguous `AlignedDMat` of size (m x k).
    fn esp_mul_ex(
        &self,
        a_stride: usize,
        rhs: &Self,
        b_stride: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Self;
}

/// Extension trait exposing hardware accelerated fixed-point math routines (Q-format).
pub trait EspFixedMatrixMath {
    /// Multiplies `self` by `rhs` using ESP32-S3 vector instructions.
    /// The `shift` parameter controls the bit-shift applied after multiplication
    /// to maintain the chosen Q-format (e.g., Q15 typically uses a shift of 15).
    fn esp_mul_fixed(&self, rhs: &Self, shift: i32) -> Self;
}

impl<T: nalgebra::Scalar + Copy> AlignedDMatExt<T> for AlignedDMat<T> {
    fn zeros(nrows: usize, ncols: usize) -> Self {
        let storage =
            EspAlignedStorage::new(Dyn(nrows), Dyn(ncols), AlignedVec::zeros(nrows * ncols));
        Self::from_data(storage)
    }

    fn from_slice(nrows: usize, ncols: usize, slice: &[T]) -> Self {
        let storage = EspAlignedStorage::new(Dyn(nrows), Dyn(ncols), AlignedVec::from_slice(slice));
        Self::from_data(storage)
    }
}

// Implement the accelerated operations exclusively for f32 matrices
impl EspMatrixMath for AlignedDMat<f32> {
    fn esp_mul(&self, rhs: &Self) -> Self {
        let mut result = Self::zeros(self.nrows(), rhs.ncols());
        crate::dsp::matrix::dspm_mult_f32::esp_gemm(self, rhs, &mut result);
        result
    }

    fn esp_mul_ex(
        &self,
        a_stride: usize,
        rhs: &Self,
        b_stride: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Self {
        // The output matrix is a contiguous block of size m * k
        let mut result = Self::zeros(m, k);

        crate::dsp::matrix::dspm_mult_ex_f32::esp_gemm_ex(
            self.as_slice(),
            a_stride,
            rhs.as_slice(),
            b_stride,
            result.as_mut_slice(),
            k, // Since result is contiguous, stride is exactly k
            m,
            n,
            k,
        );

        result
    }
}

// Implement the accelerated vector operations exclusively for i16 matrices
impl EspFixedMatrixMath for AlignedDMat<i16> {
    fn esp_mul_fixed(&self, rhs: &Self, shift: i32) -> Self {
        let mut result = Self::zeros(self.nrows(), rhs.ncols());
        crate::dsp::matrix::dspm_mult_s16::esp_gemm_s16(
            self.as_slice(),
            rhs.as_slice(),
            result.as_mut_slice(),
            self.nrows(),
            self.ncols(),
            rhs.ncols(),
            shift,
        );
        result
    }
}
