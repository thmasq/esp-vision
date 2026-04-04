use crate::dsp::AlignedDMat;
use core::arch::asm;

/// Requirements: m, n, k must be multiples of 4. A, B, C must be 16-byte aligned.
#[inline(never)]
unsafe fn dspm_mult_f32_aes3_core(
    ptr_a: *const f32,
    ptr_b: *const f32,
    ptr_c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) {
    let mut _b_shift: usize;
    let mut _k_step: usize;
    let mut _n_count: usize;
    let mut _x_idx: usize;
    let mut _y_ptr: usize;
    let mut _c_ptr_runner: usize;
    let mut _c_col_start: usize = ptr_c as usize;
    let mut _y_idx: usize;
    let mut _a_runner: usize;

    unsafe {
        asm!(
            "movi.n {_b_shift}, 0",
            "slli {_k_step}, {k}, 2",
            "srli {_n_count}, {n}, 2",
            "addi.n {_n_count}, {_n_count}, -1",
            "movi.n {_x_idx}, 0",

            "2:", // .loop_x_aes3
                "movi.n {_y_idx}, 0",
                "mov {_a_runner}, {ptr_a}",
                "mov {_c_ptr_runner}, {_c_col_start}",

                "3:", // .loop_y_aes3
                    "add {_y_ptr}, {ptr_b}, {_b_shift}",

                    // Load A row into f8-f11, Load B col into f4-f7
                    "EE.LDF.128.IP f11, f10, f9, f8, {_a_runner}, 16",
                    "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {_k_step}",
                    "mul.s f0, f4, f8",
                    "mul.s f1, f5, f8",
                    "mul.s f2, f6, f8",
                    "mul.s f3, f7, f8",

                    "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {_k_step}",
                    "madd.s f0, f4, f9",
                    "madd.s f1, f5, f9",
                    "madd.s f2, f6, f9",
                    "madd.s f3, f7, f9",

                    "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {_k_step}",
                    "madd.s f0, f4, f10",
                    "madd.s f1, f5, f10",
                    "madd.s f2, f6, f10",
                    "madd.s f3, f7, f10",

                    "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {_k_step}",
                    "madd.s f0, f4, f11",
                    "madd.s f1, f5, f11",
                    "madd.s f2, f6, f11",
                    "madd.s f3, f7, f11",

                    // Hardware inner loop over N/4 remaining blocks
                    "loopnez {_n_count}, 4f",
                        "EE.LDF.128.IP f11, f10, f9, f8, {_a_runner}, 16",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {_k_step}",
                        "madd.s f0, f4, f8", "madd.s f1, f5, f8", "madd.s f2, f6, f8", "madd.s f3, f7, f8",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {_k_step}",
                        "madd.s f0, f4, f9", "madd.s f1, f5, f9", "madd.s f2, f6, f9", "madd.s f3, f7, f9",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {_k_step}",
                        "madd.s f0, f4, f10", "madd.s f1, f5, f10", "madd.s f2, f6, f10", "madd.s f3, f7, f10",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {_k_step}",
                        "madd.s f0, f4, f11", "madd.s f1, f5, f11", "madd.s f2, f6, f11", "madd.s f3, f7, f11",
                    "4:", // .loop_end_m_aes3

                    // Store 4 output floats back to Matrix C
                    "EE.STF.128.XP f3, f2, f1, f0, {_c_ptr_runner}, {_k_step}",

                    "addi {_y_idx}, {_y_idx}, 1",

                    // MANUAL BRANCH RELAXATION:
                    // blt {_y_idx}, {m}, 3b -> (Invert condition, use unconditional jump)
                    "bge {_y_idx}, {m}, 5f", // If y_idx >= m, skip jump and exit inner loop
                    "j 3b",                  // Else, jump back to 3: (Range up to 256KB)
                    "5:",

                "addi {_c_col_start}, {_c_col_start}, 16",
                "addi {_b_shift}, {_b_shift}, 16",
                "addi {_x_idx}, {_x_idx}, 16",

                // MANUAL BRANCH RELAXATION:
                "bge {_x_idx}, {_k_step}, 6f",
                "j 2b",
                "6:",

            _b_shift = out(reg) _b_shift,
            _k_step = out(reg) _k_step,
            _n_count = out(reg) _n_count,
            _x_idx = out(reg) _x_idx,
            _y_ptr = out(reg) _y_ptr,
            _c_ptr_runner = out(reg) _c_ptr_runner,
            _c_col_start = inout(reg) _c_col_start,
            _y_idx = out(reg) _y_idx,
            _a_runner = out(reg) _a_runner,

            ptr_a = in(reg) ptr_a,
            ptr_b = in(reg) ptr_b,
            m = in(reg) m,
            n = in(reg) n,
            k = in(reg) k,
            options(nostack)
        );
    }
}

/// Safely multiply two aligned matrices using ESP32-S3 SIMD acceleration.
pub fn esp_gemm(a: &AlignedDMat<f32>, b: &AlignedDMat<f32>, c: &mut AlignedDMat<f32>) {
    let m = a.nrows();
    let n = a.ncols();
    let k = b.ncols();

    assert_eq!(n, b.nrows(), "Matrix dimension mismatch for multiplication");
    assert_eq!(c.nrows(), m, "Output matrix rows mismatch");
    assert_eq!(c.ncols(), k, "Output matrix cols mismatch");

    if m % 4 == 0 && n % 4 == 0 && k % 4 == 0 {
        unsafe {
            let ptr_a = a.as_slice().as_ptr();
            let ptr_b = b.as_slice().as_ptr();
            let ptr_c = c.as_mut_slice().as_mut_ptr();

            core::hint::assert_unchecked(ptr_a as usize % 16 == 0);
            core::hint::assert_unchecked(ptr_b as usize % 16 == 0);
            core::hint::assert_unchecked(ptr_c as usize % 16 == 0);

            dspm_mult_f32_aes3_core(ptr_b, ptr_a, ptr_c, k, n, m);
        }
    } else {
        c.gemm(1.0, a, b, 0.0);
    }
}
