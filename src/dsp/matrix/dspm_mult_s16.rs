use core::arch::asm;

#[repr(C, align(16))]
struct AlignedArray([i16; 8]);

/// Vectorized int16 matrix multiplication using the ESP32-S3 qacc engine.
/// Requirements:
/// - m must be a multiple of 8.
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
    let mut ptr_b_col = ptr_b as usize;

    let a_stride_bytes = m * 2;
    let b_col_bytes = n * 2;

    let n_half = n / 2;
    let n_odd = n % 2;
    let qacc_shift = shift;

    // Iterate over the columns of B and C
    for _ in 0..k {
        let mut ptr_a_row_block = ptr_a as usize;

        // Iterate down the columns of A and C in chunks of 8
        for _ in 0..(m / 8) {
            let ptr_a_runner = ptr_a_row_block;
            let ptr_b_runner = ptr_b_col;
            let ptr_round = round_data.0.as_ptr() as usize;

            // Hardware Dot-Product Pipeline
            unsafe {
                asm!(
                    // Load rounding offset data into the 'qacc' accumulator
                    "ee.ldqa.u16.128.ip {ptr_round}, 0",

                    // Pre-load: Broadcast 1 int16 from B into q1, load 8 int16s from A into q0
                    "ee.vldbc.16.ip q1, {ptr_b_runner}, 2",
                    "ee.vld.128.xp q0, {ptr_a_runner}, {a_stride_bytes}",

                    // If N is odd, process one initial iteration to align the unrolled loop
                    "beqz {n_odd}, 1f",
                    "ee.vmulas.s16.qacc.ldbc.incp q1, {ptr_b_runner}, q0, q1",
                    "ee.vld.128.xp q0, {ptr_a_runner}, {a_stride_bytes}",
                    "1:",

                    // Hardware inner loop over remaining N blocks (Unrolled by 2)
                    "loopnez {n_half}, 2f",
                        "ee.vld.128.xp q2, {ptr_a_runner}, {a_stride_bytes}",
                        "ee.vmulas.s16.qacc.ldbc.incp q1, {ptr_b_runner}, q0, q1",

                        "ee.vld.128.xp q0, {ptr_a_runner}, {a_stride_bytes}",
                        "ee.vmulas.s16.qacc.ldbc.incp q1, {ptr_b_runner}, q2, q1",
                    "2:",

                    // End of chunk: Shift, round, and compress qacc down to q0, store to C
                    "ee.srcmb.s16.qacc q0, {qacc_shift}, 1",
                    "ee.vst.128.ip q0, {ptr_c_runner}, 16",

                    // Routing variables
                    ptr_round = inout(reg) ptr_round => _,
                    ptr_a_runner = inout(reg) ptr_a_runner => _,
                    ptr_b_runner = inout(reg) ptr_b_runner => _,
                    ptr_c_runner = inout(reg) ptr_c_runner,

                    n_odd = in(reg) n_odd,
                    n_half = in(reg) n_half,
                    a_stride_bytes = in(reg) a_stride_bytes,
                    qacc_shift = in(reg) qacc_shift,

                    options(nostack)
                );
            }

            // Advance A row block pointer by 8 i16 elements down the column (16 bytes)
            ptr_a_row_block += 16;
        }
        // Advance B pointer to the next column
        ptr_b_col += b_col_bytes;
    }
}

/// Safely multiply two Q15 fixed-point matrices.
pub fn esp_gemm_s16(a: &[i16], b: &[i16], c: &mut [i16], m: usize, n: usize, k: usize, shift: i32) {
    assert!(
        m % 8 == 0,
        "m must be a multiple of 8 for SIMD S16 column-major multiplication."
    );

    let ptr_a = a.as_ptr();
    let ptr_b = b.as_ptr();
    let ptr_c = c.as_mut_ptr();

    if ptr_a as usize % 16 == 0 && ptr_b as usize % 16 == 0 && ptr_c as usize % 16 == 0 {
        unsafe {
            dspm_mult_s16_aes3_core(ptr_a, ptr_b, ptr_c, m, n, k, shift);
        }
    } else {
        let round_val = if shift < 0 {
            16383 >> -shift
        } else {
            32767 >> shift
        };

        for j in 0..k {
            for i in 0..m {
                let mut acc: i32 = round_val;
                for s in 0..n {
                    let a_val = a[s * m + i] as i32;
                    let b_val = b[j * n + s] as i32;
                    acc += a_val * b_val;
                }
                c[j * m + i] = (acc >> shift) as i16;
            }
        }
    }
}
