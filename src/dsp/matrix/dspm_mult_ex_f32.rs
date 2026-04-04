use core::arch::asm;

/// Extended floating-point matrix multiplication with paddings (strides).
/// Requirements:
/// - m, n, k must be multiples of 4.
/// - a_padding, b_padding, c_padding must be multiples of 4.
/// - A, B, C pointers must be 16-byte aligned.
#[inline(never)]
pub unsafe fn dspm_mult_ex_f32_aes3_core(
    ptr_a: *const f32,
    ptr_b: *const f32,
    ptr_c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    a_padding: usize,
    b_padding: usize,
    c_padding: usize,
) {
    let c_col_start: usize = ptr_c as usize;
    let k_loops = k / 4;
    let b_step_bytes = (k + b_padding) * 4;
    let c_step_bytes = (k + c_padding) * 4;
    let a_pad_bytes = a_padding * 4;

    let mut n_count = n / 4;
    if n_count > 0 {
        n_count -= 1;
    }

    unsafe {
        asm!(
            "2:", // .loop_x_mult_ex
                "mov {_y_idx}, {m}",
                "mov {_a_runner}, {ptr_a}",
                "mov {_c_ptr_runner}, {_c_col_start}",

                "3:", // .loop_y_mult_ex
                    "mov {_y_ptr}, {ptr_b}",

                    "EE.LDF.128.IP f11, f10, f9, f8, {_a_runner}, 16",
                    "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {b_step_bytes}",
                    "mul.s f0, f4, f8",
                    "mul.s f1, f5, f8",
                    "mul.s f2, f6, f8",
                    "mul.s f3, f7, f8",

                    "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {b_step_bytes}",
                    "madd.s f0, f4, f9",
                    "madd.s f1, f5, f9",
                    "madd.s f2, f6, f9",
                    "madd.s f3, f7, f9",

                    "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {b_step_bytes}",
                    "madd.s f0, f4, f10",
                    "madd.s f1, f5, f10",
                    "madd.s f2, f6, f10",
                    "madd.s f3, f7, f10",

                    "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {b_step_bytes}",
                    "madd.s f0, f4, f11",
                    "madd.s f1, f5, f11",
                    "madd.s f2, f6, f11",
                    "madd.s f3, f7, f11",

                    "loopnez {n_count}, 4f",
                        "EE.LDF.128.IP f11, f10, f9, f8, {_a_runner}, 16",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {b_step_bytes}",
                        "madd.s f0, f4, f8", "madd.s f1, f5, f8", "madd.s f2, f6, f8", "madd.s f3, f7, f8",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {b_step_bytes}",
                        "madd.s f0, f4, f9", "madd.s f1, f5, f9", "madd.s f2, f6, f9", "madd.s f3, f7, f9",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {b_step_bytes}",
                        "madd.s f0, f4, f10", "madd.s f1, f5, f10", "madd.s f2, f6, f10", "madd.s f3, f7, f10",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_y_ptr}, {b_step_bytes}",
                        "madd.s f0, f4, f11", "madd.s f1, f5, f11", "madd.s f2, f6, f11", "madd.s f3, f7, f11",
                    "4:", // .loop_end_m_aes3

                    "EE.STF.128.XP f3, f2, f1, f0, {_c_ptr_runner}, {c_step_bytes}",
                    "add {_a_runner}, {_a_runner}, {a_pad_bytes}",

                    "addi {_y_idx}, {_y_idx}, -1",
                    "beqz {_y_idx}, 5f",
                    "j 3b",
                    "5:",

                "addi {_c_col_start}, {_c_col_start}, 16",
                "addi {ptr_b}, {ptr_b}, 16",

                "addi {k_loops}, {k_loops}, -1",
                "beqz {k_loops}, 6f",
                "j 2b",
                "6:",

            _c_col_start = inout(reg) c_col_start => _,
            ptr_b = inout(reg) ptr_b => _,
            k_loops = inout(reg) k_loops => _,

            _y_idx = out(reg) _,
            _c_ptr_runner = out(reg) _,
            _y_ptr = out(reg) _,
            _a_runner = out(reg) _,

            ptr_a = in(reg) ptr_a,
            m = in(reg) m,
            b_step_bytes = in(reg) b_step_bytes,
            c_step_bytes = in(reg) c_step_bytes,
            a_pad_bytes = in(reg) a_pad_bytes,
            n_count = in(reg) n_count,

            options(nostack)
        );
    }
}

/// Safely multiply two extended/padded aligned matrices.
/// `a_stride`, `b_stride`, and `c_stride` correspond to the total number of elements in a row.
pub fn esp_gemm_ex(
    a: &[f32],
    a_stride: usize,
    b: &[f32],
    b_stride: usize,
    c: &mut [f32],
    c_stride: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    assert!(
        m % 4 == 0 && n % 4 == 0 && k % 4 == 0,
        "Dimensions must be multiples of 4"
    );
    assert!(
        a_stride % 4 == 0 && b_stride % 4 == 0 && c_stride % 4 == 0,
        "Strides must be multiples of 4"
    );

    let a_padding = a_stride - n;
    let b_padding = b_stride - k;
    let c_padding = c_stride - k;

    let ptr_a = a.as_ptr();
    let ptr_b = b.as_ptr();
    let ptr_c = c.as_mut_ptr();

    if ptr_a as usize % 16 == 0 && ptr_b as usize % 16 == 0 && ptr_c as usize % 16 == 0 {
        unsafe {
            dspm_mult_ex_f32_aes3_core(
                ptr_a, ptr_b, ptr_c, m, n, k, a_padding, b_padding, c_padding,
            );
        }
    } else {
        panic!("Extended S3 Matrix multiplication requires 16-byte aligned pointers.");
    }
}
