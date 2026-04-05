use nalgebra::base::allocator::Allocator;
use nalgebra::base::default_allocator::DefaultAllocator;
use nalgebra::base::dimension::{Dim, U1};
use nalgebra::base::storage::{Owned, RawStorage, RawStorageMut, Storage};
use nalgebra::{Dyn, OMatrix, Scalar};

use super::alloc::AlignedVec;

#[derive(Debug, Clone)]
pub struct EspAlignedStorage<T, R: Dim, C: Dim> {
    data: AlignedVec<T>,
    nrows: R,
    ncols: C,
    pub physical_stride: usize,
}

impl<T, R: Dim, C: Dim> EspAlignedStorage<T, R, C> {
    pub fn new(nrows: R, ncols: C, data: AlignedVec<T>, physical_stride: usize) -> Self {
        Self {
            data,
            nrows,
            ncols,
            physical_stride,
        }
    }
}

unsafe impl<T, R: Dim, C: Dim> RawStorage<T, R, C> for EspAlignedStorage<T, R, C> {
    type RStride = U1;
    type CStride = Dyn;

    #[inline]
    fn ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    #[inline]
    fn shape(&self) -> (R, C) {
        (self.nrows.clone(), self.ncols.clone())
    }

    #[inline]
    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (U1, Dyn(self.physical_stride))
    }

    #[inline]
    fn is_contiguous(&self) -> bool {
        self.nrows.value() == self.physical_stride
    }

    #[inline]
    unsafe fn as_slice_unchecked(&self) -> &[T] {
        &self.data
    }
}

unsafe impl<T, R: Dim, C: Dim> RawStorageMut<T, R, C> for EspAlignedStorage<T, R, C> {
    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline]
    unsafe fn as_mut_slice_unchecked(&mut self) -> &mut [T] {
        &mut self.data
    }
}

unsafe impl<T: Scalar, R: Dim, C: Dim> Storage<T, R, C> for EspAlignedStorage<T, R, C> {
    #[inline]
    fn into_owned(self) -> Owned<T, R, C>
    where
        DefaultAllocator: Allocator<R, C>,
    {
        self.clone_owned()
    }

    #[inline]
    fn clone_owned(&self) -> Owned<T, R, C>
    where
        DefaultAllocator: Allocator<R, C>,
    {
        let iter = (0..self.ncols.value()).flat_map(|c| {
            let start = c * self.physical_stride;
            let end = start + self.nrows.value();
            self.data[start..end].iter().cloned()
        });

        OMatrix::<T, R, C>::from_iterator_generic(self.nrows.clone(), self.ncols.clone(), iter).data
    }

    #[inline]
    fn forget_elements(self) {
        core::mem::forget(self.data);
    }
}
