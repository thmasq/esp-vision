use core::arch::asm;

/// Vectorized single-precision float dot product using ESP32-S3 SIMD extension.
///
/// # Requirements:
/// - Pointers `ptr_a` and `ptr_b` must be 16-byte aligned.
/// - `padded_len` MUST be a multiple of 4.
#[inline(always)]
pub unsafe fn dsps_dotprod_f32_aes3_core(
    ptr_a: *const f32,
    ptr_b: *const f32,
    padded_len: usize,
) -> f32 {
    let a_runner = ptr_a as usize;
    let b_runner = ptr_b as usize;
    let mut loop_count = padded_len / 4;
    let mut result_sum: f32 = 0.0;

    if loop_count > 0 {
        loop_count -= 1;

        unsafe {
            asm!(
                // --- Loop Peeling (First Iteration) ---
                // Load 4 floats (16 bytes) from A and B, auto-increment pointers by 16
                "EE.LDF.128.IP f7, f6, f5, f4, {a_runner}, 16",
                "EE.LDF.128.IP f11, f10, f9, f8, {b_runner}, 16",

                // Multiply to initialize the accumulators (f0-f3).
                // Doing this on the first cycle avoids having to manually zero f0-f3 beforehand.
                "mul.s f0, f4, f8",
                "mul.s f1, f5, f9",
                "mul.s f2, f6, f10",
                "mul.s f3, f7, f11",

                // If the vector was exactly 4 elements long, skip the loop
                "beqz {loop_count}, 2f",

                // --- Main SIMD Loop ---
                "1:",
                    // Load the next 4 floats from A and B
                    "EE.LDF.128.IP f7, f6, f5, f4, {a_runner}, 16",
                    "EE.LDF.128.IP f11, f10, f9, f8, {b_runner}, 16",

                    // Multiply and accumulate into existing sums
                    "madd.s f0, f4, f8",
                    "madd.s f1, f5, f9",
                    "madd.s f2, f6, f10",
                    "madd.s f3, f7, f11",

                    "addi {loop_count}, {loop_count}, -1",
                    "bnez {loop_count}, 1b",
                "2:",

                // --- Reduction (Horizontal Sum) ---
                // Fold the 4 accumulators down into the single `result` register
                "add.s f0, f0, f1",
                "add.s f2, f2, f3",
                "add.s {result}, f0, f2",

                // Bindings
                a_runner = inout(reg) a_runner=> _,
                b_runner = inout(reg) b_runner=> _,
                loop_count = inout(reg) loop_count => _,
                result = out(freg) result_sum,

                // Clobbers: Warn the compiler that these FPU registers were trashed
                out("f0") _, out("f1") _, out("f2") _, out("f3") _,
                out("f4") _, out("f5") _, out("f6") _, out("f7") _,
                out("f8") _, out("f9") _, out("f10") _, out("f11") _,

                options(nostack)
            );
        }
    }

    result_sum
}

/// Hardware accelerated dot product of two single-precision arrays.
pub fn esp_dotprod_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Slices must have the same length");

    unsafe {
        // Assert 16-byte alignment to the compiler optimizer.
        // The PIE `EE.LDF.128.IP` instruction requires strict 16-byte boundaries.
        core::hint::assert_unchecked(a.as_ptr() as usize % 16 == 0);
        core::hint::assert_unchecked(b.as_ptr() as usize % 16 == 0);

        dsps_dotprod_f32_aes3_core(a.as_ptr(), b.as_ptr(), a.len())
    }
}
