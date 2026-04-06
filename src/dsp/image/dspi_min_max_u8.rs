#[inline(always)]
pub fn dspi_min_max_u8(pixels: &[u8]) -> (u8, u8) {
    if pixels.is_empty() {
        return (0, 0);
    }

    pixels
        .iter()
        .fold((pixels[0], pixels[0]), |(current_min, current_max), &p| {
            (current_min.min(p), current_max.max(p))
        })
}
