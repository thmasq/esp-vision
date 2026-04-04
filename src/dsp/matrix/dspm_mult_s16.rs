use core::arch::asm;

#[repr(C, align(16))]
struct AlignedArray([i16; 8]);

/// Vectorized int16 matrix multiplication using the ESP32-S3 qacc engine.
/// Requirements:
/// - k must be a multiple of 8.
/// - A, B, C pointers must be 16-byte aligned.
#[inline(never)]
pub unsafe fn dspm_mult_s16_aes3_core(
    ptr_a: *const i16,
    ptr_b: *const i16,
    ptr_c: *mut i16,
    m: usize,
    n: usize,
    k: usize,
    shift: i32,
) {
    let mut round_data = AlignedArray([0; 8]);
    let round_val = if shift < 0 {
        (16383 >> -shift) as i16
    } else {
        (32767 >> shift) as i16
    };
    for i in 0..8 {
        round_data.0[i] = round_val;
    }

    let mut ptr_c_runner = ptr_c as usize;
    let b_stride_bytes = k * 2; // 2 bytes per i16
    let mut ptr_a_row = ptr_a as usize;

    // Unrolled loop counters
    let n_half = n / 2;
    let n_odd = n % 2;

    // FIX: Pass the absolute shift directly to the hardware instruction
    let qacc_shift = shift;

    // Outer Loop (M): Iterate over the rows of A
    for _ in 0..m {
        let mut ptr_b_col_block = ptr_b as usize;

        // Middle Loop (K): Iterate over the columns of B in chunks of 8
        for _ in 0..(k / 8) {
            let ptr_a_runner = ptr_a_row;
            let ptr_b_runner = ptr_b_col_block;
            let ptr_round = round_data.0.as_ptr() as usize;

            // Inner Loop (N): The Hardware Dot-Product Pipeline
            unsafe {
                asm!(
                    // Load 128-bits of rounding offset data into the 'qacc' accumulator
                    "ee.ldqa.u16.128.ip {ptr_round}, 0",

                    // Pre-load: Broadcast 1 int16 from A into q1, load 8 int16s from B into q0
                    "ee.vldbc.16.ip q1, {ptr_a_runner}, 2",
                    "ee.vld.128.xp q0, {ptr_b_runner}, {b_stride_bytes}",

                    // If N is odd, process one initial iteration to align the unrolled loop
                    "beqz {n_odd}, 1f",
                    "ee.vmulas.s16.qacc.ldbc.incp q1, {ptr_a_runner}, q0, q1",
                    "ee.vld.128.xp q0, {ptr_b_runner}, {b_stride_bytes}",
                    "1:",

                    // Hardware inner loop over remaining N blocks (Unrolled by 2)
                    "loopnez {n_half}, 2f",
                        "ee.vld.128.xp q2, {ptr_b_runner}, {b_stride_bytes}",
                        "ee.vmulas.s16.qacc.ldbc.incp q1, {ptr_a_runner}, q0, q1",

                        "ee.vld.128.xp q0, {ptr_b_runner}, {b_stride_bytes}",
                        "ee.vmulas.s16.qacc.ldbc.incp q1, {ptr_a_runner}, q2, q1",
                    "2:",

                    // End of row: Shift, round, and compress qacc down to q0, then store to C
                    "ee.srcmb.s16.qacc q0, {qacc_shift}, 1",
                    "ee.vst.128.ip q0, {ptr_c_runner}, 16",

                    // Routing variables
                    ptr_round = inout(reg) ptr_round => _,
                    ptr_a_runner = inout(reg) ptr_a_runner => _,
                    ptr_b_runner = inout(reg) ptr_b_runner => _,
                    ptr_c_runner = inout(reg) ptr_c_runner,

                    n_odd = in(reg) n_odd,
                    n_half = in(reg) n_half,
                    b_stride_bytes = in(reg) b_stride_bytes,
                    qacc_shift = in(reg) qacc_shift,

                    options(nostack)
                );
            }

            // Advance B block pointer by 8 i16 columns (16 bytes)
            ptr_b_col_block += 16;
        }
        // Advance A row pointer to the next row (N * 2 bytes)
        ptr_a_row += n * 2;
    }
}

/// Safely multiply two Q15 fixed-point matrices.
pub fn esp_gemm_s16(a: &[i16], b: &[i16], c: &mut [i16], m: usize, n: usize, k: usize, shift: i32) {
    // The vector pipeline operates on 8 elements simultaneously
    assert!(
        k % 8 == 0,
        "k must be a multiple of 8 for SIMD S16 multiplication."
    );

    let ptr_a = a.as_ptr();
    let ptr_b = b.as_ptr();
    let ptr_c = c.as_mut_ptr();

    if ptr_a as usize % 16 == 0 && ptr_b as usize % 16 == 0 && ptr_c as usize % 16 == 0 {
        unsafe {
            dspm_mult_s16_aes3_core(ptr_a, ptr_b, ptr_c, m, n, k, shift);
        }
    } else {
        // Software fallback for non-aligned data
        for i in 0..m {
            for j in 0..k {
                let mut acc: i32 = 0;
                for s in 0..n {
                    acc += (a[i * n + s] as i32) * (b[s * k + j] as i32);
                }
                c[i * k + j] = (acc >> shift) as i16;
            }
        }
    }
}
