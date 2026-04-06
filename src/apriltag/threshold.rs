extern crate alloc;
use crate::apriltag::image::Image;
use crate::dsp::image::dspi_min_max_u8::dspi_min_max_u8;
use alloc::vec::Vec;

const TILE_SIZE: usize = 4;
const MIN_CONTRAST: u8 = 30;

/// Performs adaptive thresholding on the input image, writing the binarized
/// result (0 for black, 255 for white) to the output image.
pub fn process(input: &Image, output: &mut Image) {
    debug_assert_eq!(input.width, output.width);
    debug_assert_eq!(input.height, output.height);

    let tiles_x = input.width / TILE_SIZE;
    let tiles_y = input.height / TILE_SIZE;

    // Allocate tile extrema buffers. For 640x480, this is 160x120 = 19,200 bytes each.
    // This is small enough that it might fit into internal SRAM with a custom allocator,
    // avoiding PSRAM latency during the Phase 2 neighborhood check.
    let mut tile_mins = Vec::with_capacity(tiles_x * tiles_y);
    let mut tile_maxs = Vec::with_capacity(tiles_x * tiles_y);

    // =========================================================================
    // PHASE 1: Localized Min/Max Extraction (SIMD Accelerated)
    // =========================================================================
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let mut tile_buf = [0u8; 16];
            let x_start = tx * TILE_SIZE;

            for i in 0..TILE_SIZE {
                let y = ty * TILE_SIZE + i;
                let row_slice = &input.row(y)[x_start..x_start + TILE_SIZE];
                tile_buf[i * 4..(i + 1) * 4].copy_from_slice(row_slice);
            }

            let (min, max) = dspi_min_max_u8(&tile_buf);

            tile_mins.push(min);
            tile_maxs.push(max);
        }
    }

    // =========================================================================
    // PHASE 2: 3x3 Neighborhood Smoothing & Binarization
    // =========================================================================
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let min_tx = tx.saturating_sub(1);
            let max_tx = (tx + 1).min(tiles_x - 1);
            let min_ty = ty.saturating_sub(1);
            let max_ty = (ty + 1).min(tiles_y - 1);

            let mut local_min = 255;
            let mut local_max = 0;

            for ny in min_ty..=max_ty {
                for nx in min_tx..=max_tx {
                    let idx = ny * tiles_x + nx;
                    local_min = local_min.min(tile_mins[idx]);
                    local_max = local_max.max(tile_maxs[idx]);
                }
            }

            let contrast = local_max.saturating_sub(local_min);
            let thresh = local_min as u16 + (contrast as u16 >> 1);

            let x_start = tx * TILE_SIZE;

            for i in 0..TILE_SIZE {
                let y = ty * TILE_SIZE + i;
                let in_row = &input.row(y)[x_start..x_start + TILE_SIZE];
                let out_row = &mut output.row_mut(y)[x_start..x_start + TILE_SIZE];

                for j in 0..TILE_SIZE {
                    if contrast < MIN_CONTRAST {
                        out_row[j] = 0;
                    } else {
                        out_row[j] = if (in_row[j] as u16) < thresh { 255 } else { 0 };
                    }
                }
            }
        }
    }
}
