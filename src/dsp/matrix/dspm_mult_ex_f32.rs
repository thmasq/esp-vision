use core::arch::asm;

/// Extended floating-point matrix multiplication with paddings (strides).
/// Requirements:
/// - m, n, k must be multiples of 4.
/// - a_stride, b_stride, c_stride must be multiples of 4 and >= their respective column lengths.
/// - A, B, C pointers must be 16-byte aligned.
#[inline(never)]
pub unsafe fn dspm_mult_ex_f32_aes3_core(
    ptr_a: *const f32,
    ptr_b: *const f32,
    ptr_c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    a_stride: usize,
    b_stride: usize,
    c_stride: usize,
) {
    let a_stride_bytes = a_stride * 4;
    let b_stride_bytes = b_stride * 4;
    let c_pad_bytes = (c_stride - m) * 4;

    let mut _k_loops = k;
    let mut _b_col_start = ptr_b as usize;
    let mut _c_runner = ptr_c as usize;
    let mut _a_runner: usize;
    let mut _y_ptr: usize;
    let mut _x_ptr: usize;

    let ptr_a_end = (ptr_a as usize) + m * 4;

    let mut n_count = n / 4;
    if n_count > 0 {
        n_count -= 1;
    }

    unsafe {
        asm!(
            "2:", // .loop_k (Iterate over columns of C and B)
                "mov {_a_runner}, {ptr_a}",

                "3:", // .loop_m (Iterate down chunks of rows for A and C)
                    "mov {_y_ptr}, {_b_col_start}",
                    "mov {_x_ptr}, {_a_runner}",

                    // Load B column chunk (4 elements, contiguous)
                    "EE.LDF.128.IP f11, f10, f9, f8, {_y_ptr}, 16",
                    // Load A row chunk (stride between columns)
                    "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {a_stride_bytes}",

                    "mul.s f0, f4, f8",
                    "mul.s f1, f5, f8",
                    "mul.s f2, f6, f8",
                    "mul.s f3, f7, f8",

                    "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {a_stride_bytes}",
                    "madd.s f0, f4, f9",
                    "madd.s f1, f5, f9",
                    "madd.s f2, f6, f9",
                    "madd.s f3, f7, f9",

                    "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {a_stride_bytes}",
                    "madd.s f0, f4, f10",
                    "madd.s f1, f5, f10",
                    "madd.s f2, f6, f10",
                    "madd.s f3, f7, f10",

                    "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {a_stride_bytes}",
                    "madd.s f0, f4, f11",
                    "madd.s f1, f5, f11",
                    "madd.s f2, f6, f11",
                    "madd.s f3, f7, f11",

                    "loopnez {n_count}, 4f",
                        "EE.LDF.128.IP f11, f10, f9, f8, {_y_ptr}, 16",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {a_stride_bytes}",
                        "madd.s f0, f4, f8", "madd.s f1, f5, f8", "madd.s f2, f6, f8", "madd.s f3, f7, f8",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {a_stride_bytes}",
                        "madd.s f0, f4, f9", "madd.s f1, f5, f9", "madd.s f2, f6, f9", "madd.s f3, f7, f9",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {a_stride_bytes}",
                        "madd.s f0, f4, f10", "madd.s f1, f5, f10", "madd.s f2, f6, f10", "madd.s f3, f7, f10",

                        "EE.LDF.128.XP f7, f6, f5, f4, {_x_ptr}, {a_stride_bytes}",
                        "madd.s f0, f4, f11", "madd.s f1, f5, f11", "madd.s f2, f6, f11", "madd.s f3, f7, f11",
                    "4:", // .loop_end_n

                    // Store naturally contiguous C output and advance ptr
                    "EE.STF.128.IP f3, f2, f1, f0, {_c_runner}, 16",

                    // Move to next 4 rows of A's current columns
                    "addi {_a_runner}, {_a_runner}, 16",
                    "bge {_a_runner}, {ptr_a_end}, 5f",
                    "j 3b",
                    "5:", // .loop_m_end

                // Jump to the start of the next B and C columns
                "add {_b_col_start}, {_b_col_start}, {b_stride_bytes}",
                "add {_c_runner}, {_c_runner}, {c_pad_bytes}",
                "addi {_k_loops}, {_k_loops}, -1",
                "beqz {_k_loops}, 6f",
                "j 2b",
                "6:",

            _b_col_start = inout(reg) _b_col_start,
            _k_loops = inout(reg) _k_loops,
            _c_runner = inout(reg) _c_runner,

            _a_runner = out(reg) _a_runner,
            _y_ptr = out(reg) _y_ptr,
            _x_ptr = out(reg) _x_ptr,

            ptr_a = in(reg) ptr_a,
            ptr_a_end = in(reg) ptr_a_end,
            a_stride_bytes = in(reg) a_stride_bytes,
            b_stride_bytes = in(reg) b_stride_bytes,
            c_pad_bytes = in(reg) c_pad_bytes,
            n_count = in(reg) n_count,

            options(nostack)
        );
    }
}

/// Safely multiply two extended/padded aligned matrices.
/// In column-major layout, `stride` (`lda`, `ldb`, `ldc`) refers to the distance
/// in elements between the start of one column and the next.
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
    assert!(
        a_stride >= m && b_stride >= n && c_stride >= m,
        "Stride cannot be smaller than the matrix column length"
    );

    let ptr_a = a.as_ptr();
    let ptr_b = b.as_ptr();
    let ptr_c = c.as_mut_ptr();

    if ptr_a as usize % 16 == 0 && ptr_b as usize % 16 == 0 && ptr_c as usize % 16 == 0 {
        unsafe {
            dspm_mult_ex_f32_aes3_core(ptr_a, ptr_b, ptr_c, m, n, k, a_stride, b_stride, c_stride);
        }
    } else {
        panic!("Extended S3 Matrix multiplication requires 16-byte aligned pointers.");
    }
}
