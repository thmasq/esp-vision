extern crate alloc;
use alloc::alloc::{Layout, alloc_zeroed, dealloc};
use core::slice;

/// A SIMD-friendly, 16-byte aligned grayscale image buffer.
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    data: *mut u8,
    layout: Layout,
}

unsafe impl Send for Image {}
unsafe impl Sync for Image {}

impl Image {
    /// Allocates a new 16-byte aligned image buffer in PSRAM
    pub fn new(width: usize, height: usize, stride: usize) -> Self {
        let size = height * stride;
        let layout = Layout::from_size_align(size, 16).expect("Invalid layout");

        let data = unsafe { alloc_zeroed(layout) };
        if data.is_null() {
            alloc::alloc::handle_alloc_error(layout);
        }

        Self {
            width,
            height,
            stride,
            data,
            layout,
        }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.data, self.height * self.stride) }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.data, self.height * self.stride) }
    }

    #[inline(always)]
    pub fn row(&self, y: usize) -> &[u8] {
        let start = y * self.stride;
        &self.as_slice()[start..start + self.width]
    }

    #[inline(always)]
    pub fn row_mut(&mut self, y: usize) -> &mut [u8] {
        let start = y * self.stride;
        let width = self.width;

        &mut self.as_mut_slice()[start..start + width]
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.data, self.layout);
        }
    }
}
