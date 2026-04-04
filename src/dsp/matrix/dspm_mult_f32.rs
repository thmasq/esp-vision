use crate::dsp::AlignedDMat;
use core::arch::asm;

/// Natively Column-Major hardware matrix multiplication.
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
    let mut _c_runner: usize = ptr_c as usize;
    let mut _b_runner: usize = ptr_b as usize;
    let mut _k_loops = k;
    let mut _a_runner: usize;
    let mut _y_ptr: usize;
    let mut _x_ptr: usize;

    let m_step = m * 4;
    let n_bytes = n * 4;
    let n_count = (n / 4) - 1;
    let ptr_a_end = (ptr_a as usize) + m * 4;

    unsafe {
        asm!(
            "2:", // .loop_j (Iterate over columns of C and B)
                "mov {_a_runner}, {ptr_a}",

                "3:", // .loop_i (Iterate down the rows of A and C in blocks of 4)
                    "mov {_y_ptr}, {_b_runner}",
                    "mov {_x_ptr}, {_a_runner}",

                    // Load 4 elements of B's column into f8-f11 (contiguous)
                    // Load 4 elements of A's column into f4-f7 (contiguous)
                    "EE.LDF.128.IP f11, f10, f9, f8, {_y_ptr}, 16",
                    "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {m_step}",

                    // Multiply broadcast B over A
                    "mul.s f0, f4, f8",
                    "mul.s f1, f5, f8",
                    "mul.s f2, f6, f8",
                    "mul.s f3, f7, f8",

                    "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {m_step}",
                    "madd.s f0, f4, f9",
                    "madd.s f1, f5, f9",
                    "madd.s f2, f6, f9",
                    "madd.s f3, f7, f9",

                    "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {m_step}",
                    "madd.s f0, f4, f10",
                    "madd.s f1, f5, f10",
                    "madd.s f2, f6, f10",
                    "madd.s f3, f7, f10",

                    "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {m_step}",
                    "madd.s f0, f4, f11",
                    "madd.s f1, f5, f11",
                    "madd.s f2, f6, f11",
                    "madd.s f3, f7, f11",

                    // Hardware inner loop over N/4 remaining blocks
                    "loopnez {n_count}, 4f",
                        "EE.LDF.128.IP f11, f10, f9, f8, {_y_ptr}, 16",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {m_step}",
                        "madd.s f0, f4, f8", "madd.s f1, f5, f8", "madd.s f2, f6, f8", "madd.s f3, f7, f8",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {m_step}",
                        "madd.s f0, f4, f9", "madd.s f1, f5, f9", "madd.s f2, f6, f9", "madd.s f3, f7, f9",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {m_step}",
                        "madd.s f0, f4, f10", "madd.s f1, f5, f10", "madd.s f2, f6, f10", "madd.s f3, f7, f10",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {m_step}",
                        "madd.s f0, f4, f11", "madd.s f1, f5, f11", "madd.s f2, f6, f11", "madd.s f3, f7, f11",
                    "4:", // .loop_end_inner

                    // Store 4 output floats contiguously to Matrix C
                    // Because C is column-major, this natively glides down the column.
                    "EE.STF.128.IP f3, f2, f1, f0, {_c_runner}, 16",

                    "addi {_a_runner}, {_a_runner}, 16",
                    "bge {_a_runner}, {ptr_a_end}, 5f",
                    "j 3b",
                    "5:",

                // Advance B to the next column
                "add {_b_runner}, {_b_runner}, {n_bytes}",
                "addi {_k_loops}, {_k_loops}, -1",
                "beqz {_k_loops}, 6f",
                "j 2b",
                "6:",

            _a_runner = out(reg) _a_runner,
            _y_ptr = out(reg) _y_ptr,
            _x_ptr = out(reg) _x_ptr,
            _c_runner = inout(reg) _c_runner,
            _b_runner = inout(reg) _b_runner,
            _k_loops = inout(reg) _k_loops,

            ptr_a = in(reg) ptr_a,
            ptr_a_end = in(reg) ptr_a_end,
            m_step = in(reg) m_step,
            n_bytes = in(reg) n_bytes,
            n_count = in(reg) n_count,
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

            dspm_mult_f32_aes3_core(ptr_a, ptr_b, ptr_c, m, n, k);
        }
    } else {
        c.gemm(1.0, a, b, 0.0);
    }
}
