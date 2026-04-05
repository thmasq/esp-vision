pub mod alloc;
pub mod matrix;
pub mod storage;

use alloc::AlignedVec;
use nalgebra::base::storage::{RawStorage, RawStorageMut};
use nalgebra::{Dyn, Matrix};
use storage::EspAlignedStorage;

const fn round_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

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

impl<T: nalgebra::Scalar + Copy + Default> AlignedDMatExt<T> for AlignedDMat<T> {
    fn zeros(nrows: usize, ncols: usize) -> Self {
        // Pad dimensions for f32 (4) or i16 (8) hardware bounds
        let align = if core::mem::size_of::<T>() == 2 { 8 } else { 4 };

        let padded_rows = round_up(nrows, align);
        let padded_cols = round_up(ncols, align);

        // Allocate the larger padded vector, initialized to pure zeros
        let data = AlignedVec::zeros(padded_rows * padded_cols);

        let storage = EspAlignedStorage::new(Dyn(nrows), Dyn(ncols), data, padded_rows);
        Self::from_data(storage)
    }

    fn from_slice(nrows: usize, ncols: usize, slice: &[T]) -> Self {
        let mut mat = Self::zeros(nrows, ncols);

        for c in 0..ncols {
            for r in 0..nrows {
                mat[(r, c)] = slice[c * nrows + r];
            }
        }
        mat
    }
}

// Implement the accelerated operations exclusively for f32 matrices
impl EspMatrixMath for AlignedDMat<f32> {
    fn esp_mul(&self, rhs: &Self) -> Self {
        let m = self.nrows();
        let n = self.ncols();
        let k = rhs.ncols();

        let a_stride = self.data.physical_stride;
        let b_stride = rhs.data.physical_stride;

        // Hardware sees the padded boundaries
        let m_pad = round_up(m, 4);
        let n_pad = round_up(n, 4);
        let k_pad = round_up(k, 4);

        let mut result = Self::zeros(m, k);
        let c_stride = result.data.physical_stride;

        unsafe {
            // Using `esp_gemm_ex` uniformly handles the non-contiguous storage gaps
            crate::dsp::matrix::dspm_mult_ex_f32::esp_gemm_ex(
                self.data.as_slice_unchecked(),
                a_stride,
                rhs.data.as_slice_unchecked(),
                b_stride,
                result.data.as_mut_slice_unchecked(),
                c_stride,
                m_pad,
                n_pad,
                k_pad,
            );
        }

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
        let mut result = Self::zeros(m, k);
        let c_stride = result.data.physical_stride;

        let m_pad = round_up(m, 4);
        let n_pad = round_up(n, 4);
        let k_pad = round_up(k, 4);

        unsafe {
            crate::dsp::matrix::dspm_mult_ex_f32::esp_gemm_ex(
                self.data.as_slice_unchecked(),
                a_stride,
                rhs.data.as_slice_unchecked(),
                b_stride,
                result.data.as_mut_slice_unchecked(),
                c_stride,
                m_pad,
                n_pad,
                k_pad,
            );
        }

        result
    }
}

// Implement the accelerated vector operations exclusively for i16 matrices
impl EspFixedMatrixMath for AlignedDMat<i16> {
    fn esp_mul_fixed(&self, rhs: &Self, shift: i32) -> Self {
        let m = self.nrows();
        let n = self.ncols();
        let k = rhs.ncols();

        // Q15 fixed-point multiplies require multiples of 8 for SIMD unrolling
        let m_pad = round_up(m, 8);
        let n_pad = round_up(n, 8);
        let k_pad = round_up(k, 8);

        let mut result = Self::zeros(m, k);

        unsafe {
            // Because our arrays are zero-padded physically, we can pass the padded
            // dimensions directly into the standard block multiplier. Any row/col
            // in the padding area correctly calculates out to 0, keeping standard data intact.
            crate::dsp::matrix::dspm_mult_s16::esp_gemm_s16(
                self.data.as_slice_unchecked(),
                rhs.data.as_slice_unchecked(),
                result.data.as_mut_slice_unchecked(),
                m_pad,
                n_pad,
                k_pad,
                shift,
            );
        }

        result
    }
}
