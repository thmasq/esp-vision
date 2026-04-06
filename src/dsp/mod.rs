pub mod alloc;
pub mod matrix;
pub mod storage;
pub mod vector;

use alloc::AlignedVec;
use nalgebra::ComplexField;
use nalgebra::base::storage::{RawStorage, RawStorageMut};
use nalgebra::{Dim, Dyn, Matrix, U1};
use storage::EspAlignedStorage;

const fn round_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

/// A dynamically sized, 16-byte aligned matrix specifically designed for `esp-dsp` SIMD acceleration.
pub type AlignedDMat<T> = Matrix<T, Dyn, Dyn, EspAlignedStorage<T, Dyn, Dyn>>;

/// A dynamically sized, 16-byte aligned column vector specifically designed for `esp-dsp` SIMD acceleration.
pub type AlignedDVec<T> = Matrix<T, Dyn, U1, EspAlignedStorage<T, Dyn, U1>>;

/// Extension trait to provide constructor methods for matrices.
pub trait AlignedDMatExt<T> {
    fn zeros(nrows: usize, ncols: usize) -> Self;
    fn from_slice(nrows: usize, ncols: usize, slice: &[T]) -> Self;
}

/// Extension trait to provide constructor methods for column vectors.
pub trait AlignedDVecExt<T> {
    fn zeros(nrows: usize) -> Self;
    fn from_slice(nrows: usize, slice: &[T]) -> Self;
}

/// Extension trait exposing hardware accelerated matrix math routines (Floating Point).
pub trait EspMatrixMath {
    /// Multiplies `self` by `rhs` utilizing ESP32-S3 hardware acceleration.
    /// Allocates and returns a new `AlignedDMat`.
    fn esp_mul(&self, rhs: &Self) -> Self;

    /// Multiplies `self` by `rhs`, writing the result into a pre-allocated `out` matrix.
    /// Use this in hot loops to avoid heap allocation overhead.
    fn esp_mul_to(&self, rhs: &Self, out: &mut Self);

    /// Multiplies a sub-region of `self` by a sub-region of `rhs` using ESP32-S3 SIMD.
    /// `a_stride` and `b_stride` are the total columns of the parent buffers.
    /// Allocates and returns a new `AlignedDMat` of size (m x k).
    fn esp_mul_ex(
        &self,
        a_stride: usize,
        rhs: &Self,
        b_stride: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Self;

    /// Multiplies a sub-region of `self` by a sub-region of `rhs` using ESP32-S3 SIMD,
    /// writing directly to a pre-allocated `out` matrix.
    fn esp_mul_ex_to(
        &self,
        a_stride: usize,
        rhs: &Self,
        b_stride: usize,
        out: &mut Self,
        m: usize,
        n: usize,
        k: usize,
    );
}

/// Extension trait to add hardware-accelerated dot products to nalgebra vectors.
pub trait EspDotProd {
    /// Computes the dot product using ESP32-S3 PIE SIMD instructions.
    fn esp_dot(&self, other: &Self) -> f32;
}

/// Extension trait exposing hardware accelerated element-wise vector routines (Floating Point).
pub trait EspVectorMath {
    /// Element-wise addition using ESP32-S3 hardware acceleration.
    /// Allocates and returns a new `AlignedDVec`.
    fn esp_add(&self, rhs: &Self) -> Self;

    /// Element-wise addition, writing directly to a pre-allocated `out` vector.
    fn esp_add_to(&self, rhs: &Self, out: &mut Self);

    /// Element-wise subtraction using ESP32-S3 hardware acceleration.
    /// Allocates and returns a new `AlignedDVec`.
    fn esp_sub(&self, rhs: &Self) -> Self;

    /// Element-wise subtraction, writing directly to a pre-allocated `out` vector.
    fn esp_sub_to(&self, rhs: &Self, out: &mut Self);

    /// Element-wise multiplication using ESP32-S3 hardware acceleration.
    /// Allocates and returns a new `AlignedDVec`.
    fn esp_mul_elem(&self, rhs: &Self) -> Self;

    /// Element-wise multiplication, writing directly to a pre-allocated `out` vector.
    fn esp_mul_elem_to(&self, rhs: &Self, out: &mut Self);
}

/// Extension trait exposing hardware accelerated fixed-point math routines (Q-format).
pub trait EspFixedMatrixMath {
    /// Multiplies `self` by `rhs` using ESP32-S3 vector instructions.
    /// Allocates and returns a new `AlignedDMat`.
    /// The `shift` parameter controls the bit-shift applied after multiplication.
    fn esp_mul_fixed(&self, rhs: &Self, shift: i32) -> Self;

    /// Multiplies `self` by `rhs` using ESP32-S3 vector instructions,
    /// writing directly to a pre-allocated `out` matrix.
    fn esp_mul_fixed_to(&self, rhs: &Self, shift: i32, out: &mut Self);
}

/// Extension trait for in-place Golub-Kahan Bidiagonalization SVD.
pub trait EspBidiagonalization {
    /// Bidiagonalizes the matrix in-place using ESP32-S3 SIMD acceleration.
    fn esp_bidiagonalize(&mut self);
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

impl<T: nalgebra::Scalar + Copy + Default> AlignedDVecExt<T> for AlignedDVec<T> {
    fn zeros(nrows: usize) -> Self {
        let align = if core::mem::size_of::<T>() == 2 { 8 } else { 4 };
        let padded_rows = round_up(nrows, align);

        let data = AlignedVec::zeros(padded_rows);
        let storage = EspAlignedStorage::new(Dyn(nrows), U1, data, padded_rows);
        Self::from_data(storage)
    }

    fn from_slice(nrows: usize, slice: &[T]) -> Self {
        let mut vec = Self::zeros(nrows);
        for r in 0..nrows {
            vec[(r, 0)] = slice[r];
        }
        vec
    }
}

impl<R: Dim> EspDotProd for Matrix<f32, R, U1, EspAlignedStorage<f32, R, U1>> {
    #[inline(always)]
    fn esp_dot(&self, other: &Self) -> f32 {
        assert_eq!(
            self.nrows(),
            other.nrows(),
            "Vectors must have the same mathematical length"
        );

        assert_eq!(
            self.data.physical_stride, other.data.physical_stride,
            "Vectors must have the same padded physical stride"
        );

        unsafe {
            let a_ptr = self.data.ptr();
            let b_ptr = other.data.ptr();

            core::hint::assert_unchecked(a_ptr as usize % 16 == 0);
            core::hint::assert_unchecked(b_ptr as usize % 16 == 0);

            let padded_len = self.data.physical_stride;

            crate::dsp::vector::dsps_dotprod_f32::dsps_dotprod_f32_aes3_core(
                a_ptr, b_ptr, padded_len,
            )
        }
    }
}

impl EspVectorMath for AlignedDVec<f32> {
    fn esp_add(&self, rhs: &Self) -> Self {
        let mut result = Self::zeros(self.nrows());
        self.esp_add_to(rhs, &mut result);
        result
    }

    fn esp_add_to(&self, rhs: &Self, out: &mut Self) {
        assert_eq!(
            self.nrows(),
            rhs.nrows(),
            "Vectors must have the same length"
        );
        assert_eq!(
            self.nrows(),
            out.nrows(),
            "Output vector must have the same length"
        );
        assert_eq!(
            self.data.physical_stride, rhs.data.physical_stride,
            "Vectors must have the same padded physical stride"
        );

        unsafe {
            let a_ptr = self.data.ptr();
            let b_ptr = rhs.data.ptr();
            let out_ptr = out.data.ptr_mut();

            core::hint::assert_unchecked(a_ptr as usize % 16 == 0);
            core::hint::assert_unchecked(b_ptr as usize % 16 == 0);
            core::hint::assert_unchecked(out_ptr as usize % 16 == 0);

            let padded_len = self.data.physical_stride;

            crate::dsp::vector::dsps_add_f32::dsps_add_f32_aes3_core(
                a_ptr, b_ptr, out_ptr, padded_len,
            )
        }
    }

    fn esp_sub(&self, rhs: &Self) -> Self {
        let mut result = Self::zeros(self.nrows());
        self.esp_sub_to(rhs, &mut result);
        result
    }

    fn esp_sub_to(&self, rhs: &Self, out: &mut Self) {
        assert_eq!(
            self.nrows(),
            rhs.nrows(),
            "Vectors must have the same length"
        );
        assert_eq!(
            self.nrows(),
            out.nrows(),
            "Output vector must have the same length"
        );
        assert_eq!(
            self.data.physical_stride, rhs.data.physical_stride,
            "Vectors must have the same padded physical stride"
        );

        unsafe {
            let a_ptr = self.data.ptr();
            let b_ptr = rhs.data.ptr();
            let out_ptr = out.data.ptr_mut();

            core::hint::assert_unchecked(a_ptr as usize % 16 == 0);
            core::hint::assert_unchecked(b_ptr as usize % 16 == 0);
            core::hint::assert_unchecked(out_ptr as usize % 16 == 0);

            let padded_len = self.data.physical_stride;

            crate::dsp::vector::dsps_sub_f32::dsps_sub_f32_aes3_core(
                a_ptr, b_ptr, out_ptr, padded_len,
            )
        }
    }

    fn esp_mul_elem(&self, rhs: &Self) -> Self {
        let mut result = Self::zeros(self.nrows());
        self.esp_mul_elem_to(rhs, &mut result);
        result
    }

    fn esp_mul_elem_to(&self, rhs: &Self, out: &mut Self) {
        assert_eq!(
            self.nrows(),
            rhs.nrows(),
            "Vectors must have the same length"
        );
        assert_eq!(
            self.nrows(),
            out.nrows(),
            "Output vector must have the same length"
        );
        assert_eq!(
            self.data.physical_stride, rhs.data.physical_stride,
            "Vectors must have the same padded physical stride"
        );

        unsafe {
            let a_ptr = self.data.ptr();
            let b_ptr = rhs.data.ptr();
            let out_ptr = out.data.ptr_mut();

            core::hint::assert_unchecked(a_ptr as usize % 16 == 0);
            core::hint::assert_unchecked(b_ptr as usize % 16 == 0);
            core::hint::assert_unchecked(out_ptr as usize % 16 == 0);

            let padded_len = self.data.physical_stride;

            crate::dsp::vector::dsps_mul_f32::dsps_mul_f32_aes3_core(
                a_ptr, b_ptr, out_ptr, padded_len,
            )
        }
    }
}

// Implement the accelerated matrix operations exclusively for f32 matrices
impl EspMatrixMath for AlignedDMat<f32> {
    fn esp_mul(&self, rhs: &Self) -> Self {
        let mut result = Self::zeros(self.nrows(), rhs.ncols());
        self.esp_mul_to(rhs, &mut result);
        result
    }

    fn esp_mul_to(&self, rhs: &Self, out: &mut Self) {
        let m = self.nrows();
        let n = self.ncols();
        let k = rhs.ncols();

        // Safety checks
        assert_eq!(
            n,
            rhs.nrows(),
            "Matrix dimensions mismatch for multiplication!"
        );
        assert_eq!(
            out.nrows(),
            m,
            "Output matrix has incorrect number of rows!"
        );
        assert_eq!(
            out.ncols(),
            k,
            "Output matrix has incorrect number of columns!"
        );

        let a_stride = self.data.physical_stride;
        let b_stride = rhs.data.physical_stride;
        let c_stride = out.data.physical_stride;

        // Hardware sees the padded boundaries
        let m_pad = round_up(m, 4);
        let n_pad = round_up(n, 4);
        let k_pad = round_up(k, 4);

        unsafe {
            // Using `esp_gemm_ex` uniformly handles the non-contiguous storage gaps
            crate::dsp::matrix::dspm_mult_ex_f32::esp_gemm_ex(
                self.data.as_slice_unchecked(),
                a_stride,
                rhs.data.as_slice_unchecked(),
                b_stride,
                out.data.as_mut_slice_unchecked(),
                c_stride,
                m_pad,
                n_pad,
                k_pad,
            );
        }
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
        self.esp_mul_ex_to(a_stride, rhs, b_stride, &mut result, m, n, k);
        result
    }

    fn esp_mul_ex_to(
        &self,
        a_stride: usize,
        rhs: &Self,
        b_stride: usize,
        out: &mut Self,
        m: usize,
        n: usize,
        k: usize,
    ) {
        assert_eq!(
            out.nrows(),
            m,
            "Output matrix has incorrect number of rows!"
        );
        assert_eq!(
            out.ncols(),
            k,
            "Output matrix has incorrect number of columns!"
        );

        let c_stride = out.data.physical_stride;
        let m_pad = round_up(m, 4);
        let n_pad = round_up(n, 4);
        let k_pad = round_up(k, 4);

        unsafe {
            crate::dsp::matrix::dspm_mult_ex_f32::esp_gemm_ex(
                self.data.as_slice_unchecked(),
                a_stride,
                rhs.data.as_slice_unchecked(),
                b_stride,
                out.data.as_mut_slice_unchecked(),
                c_stride,
                m_pad,
                n_pad,
                k_pad,
            );
        }
    }
}

// Implement the accelerated matrix operations exclusively for i16 matrices
impl EspFixedMatrixMath for AlignedDMat<i16> {
    fn esp_mul_fixed(&self, rhs: &Self, shift: i32) -> Self {
        let mut result = Self::zeros(self.nrows(), rhs.ncols());
        self.esp_mul_fixed_to(rhs, shift, &mut result);
        result
    }

    fn esp_mul_fixed_to(&self, rhs: &Self, shift: i32, out: &mut Self) {
        let m = self.nrows();
        let n = self.ncols();
        let k = rhs.ncols();

        assert_eq!(
            n,
            rhs.nrows(),
            "Matrix dimensions mismatch for multiplication!"
        );
        assert_eq!(
            out.nrows(),
            m,
            "Output matrix has incorrect number of rows!"
        );
        assert_eq!(
            out.ncols(),
            k,
            "Output matrix has incorrect number of columns!"
        );

        // Q15 fixed-point multiplies require multiples of 8 for SIMD unrolling
        let m_pad = round_up(m, 8);
        let n_pad = round_up(n, 8);
        let k_pad = round_up(k, 8);

        unsafe {
            // Because our arrays are zero-padded physically, we can pass the padded
            // dimensions directly into the standard block multiplier. Any row/col
            // in the padding area correctly calculates out to 0, keeping standard data intact.
            crate::dsp::matrix::dspm_mult_s16::esp_gemm_s16(
                self.data.as_slice_unchecked(),
                rhs.data.as_slice_unchecked(),
                out.data.as_mut_slice_unchecked(),
                m_pad,
                n_pad,
                k_pad,
                shift,
            );
        }
    }
}

impl EspBidiagonalization for AlignedDMat<f32> {
    fn esp_bidiagonalize(&mut self) {
        let m = self.nrows();
        let n = self.ncols();
        let limit = m.min(n);

        // Use the storage's physical stride for padded rows
        let padded_m = self.data.physical_stride;
        let padded_n = round_up(n, 4);
        let a_stride = self.data.physical_stride;

        let mut u = AlignedDVec::<f32>::zeros(m);
        let mut v = AlignedDVec::<f32>::zeros(n);
        let mut w = AlignedDVec::<f32>::zeros(n); // Temp vector for A^T * u
        let mut z = AlignedDVec::<f32>::zeros(m); // Temp vector for A * v

        for k in 0..limit {
            // 1. ELIMINATE COLUMN (Apply Householder Left)
            for i in 0..padded_m {
                u[(i, 0)] = self[(i, k)];
            }
            let diag_val = householder_vec(&mut u, k, m);
            self[(k, k)] = diag_val;

            unsafe {
                crate::dsp::matrix::dspm_bidiag_f32::dsps_gemv_t_f32_aes3(
                    self.data.ptr(),
                    u.data.ptr(),
                    w.data.ptr_mut(),
                    padded_m,
                    padded_n,
                    a_stride,
                );
            }

            // Mask out the vectors affecting already solved columns
            for j in 0..=k {
                w[(j, 0)] = 0.0;
            }

            unsafe {
                crate::dsp::matrix::dspm_bidiag_f32::dsps_ger_f32_aes3(
                    self.data.ptr_mut(),
                    u.data.ptr(),
                    w.data.ptr(),
                    2.0,
                    padded_m,
                    padded_n,
                    a_stride,
                );
            }
            // Zero out values below the diagonal for cleanliness
            for i in (k + 1)..m {
                self[(i, k)] = 0.0;
            }

            // 2. ELIMINATE ROW (Apply Householder Right)
            if k < n - 2 {
                for j in 0..padded_n {
                    v[(j, 0)] = self[(k, j)];
                }

                let superdiag_val = householder_vec(&mut v, k + 1, n);

                unsafe {
                    crate::dsp::matrix::dspm_bidiag_f32::dsps_gemv_f32_aes3(
                        self.data.ptr(),
                        v.data.ptr(),
                        z.data.ptr_mut(),
                        padded_m,
                        padded_n,
                        a_stride,
                    );
                }

                // Mask out the vectors affecting already solved rows
                for i in 0..=k {
                    z[(i, 0)] = 0.0;
                }

                unsafe {
                    crate::dsp::matrix::dspm_bidiag_f32::dsps_ger_f32_aes3(
                        self.data.ptr_mut(),
                        z.data.ptr(),
                        v.data.ptr(),
                        2.0,
                        padded_m,
                        padded_n,
                        a_stride,
                    );
                }

                self[(k, k + 1)] = superdiag_val;
                for j in (k + 2)..n {
                    self[(k, j)] = 0.0;
                }
            }
        }
    }
}

/// Helper to generate the Householder reflection vector and return the scalar root.
fn householder_vec(x: &mut AlignedDVec<f32>, start: usize, end: usize) -> f32 {
    let mut norm_sq = 0.0;
    for i in (start + 1)..end {
        norm_sq += x[(i, 0)] * x[(i, 0)];
    }

    let x_start = x[(start, 0)];
    let mut alpha = (x_start * x_start + norm_sq).sqrt();
    if x_start > 0.0 {
        alpha = -alpha;
    }

    let r = (0.5 * (alpha * alpha - x_start * alpha)).sqrt();
    x[(start, 0)] -= alpha;

    let inv_2r = 1.0 / (2.0 * r);
    for i in start..end {
        x[(i, 0)] *= inv_2r;
    }

    // Crucial: Mask out bounds to keep 16-byte SIMD alignments clean
    for i in 0..start {
        x[(i, 0)] = 0.0;
    }
    for i in end..x.nrows() {
        x[(i, 0)] = 0.0;
    }

    alpha
}
