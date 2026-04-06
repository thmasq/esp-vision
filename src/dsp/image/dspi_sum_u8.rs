use core::arch::asm;

/// Accumulates an array of u8 into a single u32 sum using the 40-bit PIE accumulator.
///
/// # Requirements:
/// - `pixels` must point to a 16-byte aligned memory address.
#[inline(always)]
pub unsafe fn dspi_sum_u8(pixels: &[u8]) -> u32 {
    let mut p_runner = pixels.as_ptr() as usize;
    let len_count = pixels.len() / 16;
    let len_rem = pixels.len() % 16;

    let mut global_sum: u32 = 0;

    if len_count > 0 {
        let loop_count = len_count;

        // Pass constants via Rust to prevent inline asm literal pool dumps
        let shift_zero: u32 = 0;
        let ones: u32 = 0x01010101;

        unsafe {
            core::hint::assert_unchecked(p_runner % 16 == 0);
            asm!(
                // 1. Clear the 40-bit accumulator
                "EE.ZERO.ACCX",

                // 2. Load the dummy vector of 1s into q1
                "EE.MOVI.32.Q q1, {ones}, 0",
                "EE.MOVI.32.Q q1, {ones}, 1",
                "EE.MOVI.32.Q q1, {ones}, 2",
                "EE.MOVI.32.Q q1, {ones}, 3",

                "1:",
                    // Load 16 pixels into q0
                    "EE.VLD.128.IP q0, {p_runner}, 16",

                    // Multiply q0 by q1, horizontal sum, and add to ACCX
                    "EE.VMULAS.U8.ACCX q0, q1",

                    "addi {loop_count}, {loop_count}, -1",
                    "bnez {loop_count}, 1b",

                // 3. Shift ACCX by 0 and store into sum
                // (The ESP32-S3 TRM requires a trailing 0 for this macro)
                "EE.SRS.ACCX {sum}, {shift}, 0",

                p_runner = inout(reg) p_runner,
                loop_count = inout(reg) loop_count => _,
                ones = in(reg) ones,
                shift = in(reg) shift_zero,
                sum = out(reg) global_sum,
                options(nostack)
            );
        }
    }

    // Process remainder standardly
    if len_rem > 0 {
        let p_rem = unsafe { core::slice::from_raw_parts(p_runner as *const u8, len_rem) };
        for &val in p_rem {
            global_sum += val as u32;
        }
    }

    global_sum
}
