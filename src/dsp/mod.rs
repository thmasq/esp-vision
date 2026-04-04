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

/// Extension trait exposing hardware accelerated math routines.
pub trait EspMatrixMath {
    /// Multiplies `self` by `rhs` utilizing ESP32-S3 hardware acceleration when possible.
    fn esp_mul(&self, rhs: &Self) -> Self;
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
        crate::dsp::matrix::esp_gemm(self, rhs, &mut result);
        result
    }
}
