extern crate alloc;

use alloc::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use core::mem::{align_of, size_of};
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull};

const ALIGNMENT: usize = 16;

#[derive(Debug)]
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
}

impl<T> AlignedVec<T> {
    fn layout(capacity: usize) -> Layout {
        let align = ALIGNMENT.max(align_of::<T>());
        let size = capacity * size_of::<T>();
        Layout::from_size_align(size, align).expect("Invalid layout for AlignedVec")
    }

    pub fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
        }
    }

    pub fn zeros(len: usize) -> Self {
        if len == 0 {
            return Self::new();
        }

        let layout = Self::layout(len);
        let ptr = unsafe { alloc_zeroed(layout) as *mut T };

        let ptr = NonNull::new(ptr).unwrap_or_else(|| handle_alloc_error(layout));

        Self {
            ptr,
            len,
            capacity: len,
        }
    }

    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Copy,
    {
        let vec = Self::zeros(slice.len());
        unsafe {
            ptr::copy_nonoverlapping(slice.as_ptr(), vec.ptr.as_ptr(), slice.len());
        }
        vec
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.capacity != 0 {
            let layout = Self::layout(self.capacity);
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Clone> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        let mut new_vec = Self::zeros(self.len);
        for (i, item) in self.iter().enumerate() {
            new_vec[i] = item.clone();
        }
        new_vec
    }
}
