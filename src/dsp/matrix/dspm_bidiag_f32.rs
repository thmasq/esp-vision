use core::arch::asm;

/// Computes y = y + A * x
/// m and n must be multiples of 4.
#[inline(always)]
pub unsafe fn dsps_gemv_f32_aes3(
    ptr_a: *const f32,
    ptr_x: *const f32,
    ptr_y: *mut f32,
    m: usize,
    n: usize,
    a_stride: usize,
) {
    let a_stride_bytes = a_stride * 4;
    let n_count = n;
    let m_div_4 = m / 4;

    let ptr_a_col = ptr_a as usize;
    let ptr_x_runner = ptr_x as usize;

    // 1. Zero out Y
    let y_zero = ptr_y as usize;
    let m_zero = m_div_4;
    unsafe {
        asm!(
            "sub.s f0, f0, f0", "sub.s f1, f1, f1",
            "sub.s f2, f2, f2", "sub.s f3, f3, f3",
            "1:",
                "EE.STF.128.IP f3, f2, f1, f0, {y_zero}, 16",
                "addi {m_zero}, {m_zero}, -1",
                "bnez {m_zero}, 1b",
            y_zero = inout(reg) y_zero => _,
            m_zero = inout(reg) m_zero => _,
            out("f0") _, out("f1") _, out("f2") _, out("f3") _,
            options(nostack)
        );
    }

    // 2. Accumulate columns
    unsafe {
        asm!(
            "1:", // .loop_cols
                "lsi f8, {ptr_x_runner}, 0", // f8 = x[j]
                "mov {a_read}, {ptr_a_col}",
                "mov {y_read}, {ptr_y}",
                "mov {m_count}, {m_div_4}",

                "2:", // .loop_rows
                    "mov {y_write}, {y_read}",
                    "EE.LDF.128.IP f3, f2, f1, f0, {y_read}, 16",
                    "EE.LDF.128.IP f7, f6, f5, f4, {a_read}, 16",

                    "madd.s f0, f4, f8",
                    "madd.s f1, f5, f8",
                    "madd.s f2, f6, f8",
                    "madd.s f3, f7, f8",

                    "EE.STF.128.IP f3, f2, f1, f0, {y_write}, 16",

                    "addi {m_count}, {m_count}, -1",
                    "bnez {m_count}, 2b",

                "add {ptr_a_col}, {ptr_a_col}, {a_stride_bytes}",
                "addi {ptr_x_runner}, {ptr_x_runner}, 4",
                "addi {n_count}, {n_count}, -1",
                "bnez {n_count}, 1b",

            // Named arguments
            ptr_a_col = inout(reg) ptr_a_col => _,
            ptr_x_runner = inout(reg) ptr_x_runner => _,
            ptr_y = in(reg) ptr_y,
            a_stride_bytes = in(reg) a_stride_bytes,
            m_div_4 = in(reg) m_div_4,
            n_count = inout(reg) n_count => _,

            // Scratch registers bound to names
            a_read = out(reg) _,
            y_read = out(reg) _,
            y_write = out(reg) _,
            m_count = out(reg) _,

            // Explicit registers (must be at the very end)
            out("f0") _, out("f1") _, out("f2") _, out("f3") _,
            out("f4") _, out("f5") _, out("f6") _, out("f7") _, out("f8") _,
            options(nostack)
        );
    }
}

/// Computes y = A^T * x (Dot product of A's columns with x)
#[inline(always)]
pub unsafe fn dsps_gemv_t_f32_aes3(
    ptr_a: *const f32,
    ptr_x: *const f32,
    ptr_y: *mut f32,
    m: usize,
    n: usize,
    a_stride: usize,
) {
    let a_stride_bytes = a_stride * 4;
    let n_count = n;
    let m_div_4 = m / 4;

    let ptr_a_col = ptr_a as usize;
    let ptr_y_runner = ptr_y as usize;

    unsafe {
        asm!(
            "1:", // .loop_cols
                "mov {a_read}, {ptr_a_col}",
                "mov {x_read}, {ptr_x}",
                "mov {m_count}, {m_div_4}",
                "sub.s f8, f8, f8", // Accumulator for the column

                "2:", // .loop_rows
                    "EE.LDF.128.IP f3, f2, f1, f0, {a_read}, 16",
                    "EE.LDF.128.IP f7, f6, f5, f4, {x_read}, 16",

                    "madd.s f8, f0, f4",
                    "madd.s f8, f1, f5",
                    "madd.s f8, f2, f6",
                    "madd.s f8, f3, f7",

                    "addi {m_count}, {m_count}, -1",
                    "bnez {m_count}, 2b",

                "ssi f8, {ptr_y_runner}, 0", // Store dot product in y

                "add {ptr_a_col}, {ptr_a_col}, {a_stride_bytes}",
                "addi {ptr_y_runner}, {ptr_y_runner}, 4",
                "addi {n_count}, {n_count}, -1",
                "bnez {n_count}, 1b",

            ptr_a_col = inout(reg) ptr_a_col => _,
            ptr_y_runner = inout(reg) ptr_y_runner => _,
            ptr_x = in(reg) ptr_x,
            a_stride_bytes = in(reg) a_stride_bytes,
            m_div_4 = in(reg) m_div_4,
            n_count = inout(reg) n_count => _,

            a_read = out(reg) _,
            x_read = out(reg) _,
            m_count = out(reg) _,

            out("f0") _, out("f1") _, out("f2") _, out("f3") _,
            out("f4") _, out("f5") _, out("f6") _, out("f7") _, out("f8") _,
            options(nostack)
        );
    }
}

/// Rank-1 Update: A = A - alpha * u * v^T
#[inline(always)]
pub unsafe fn dsps_ger_f32_aes3(
    ptr_a: *mut f32,
    ptr_u: *const f32,
    ptr_v: *const f32,
    alpha: f32,
    m: usize,
    n: usize,
    a_stride: usize,
) {
    let alpha_ptr = &alpha as *const f32;
    let a_stride_bytes = a_stride * 4;
    let n_count = n;
    let m_div_4 = m / 4;

    let ptr_a_col = ptr_a as usize;
    let ptr_v_runner = ptr_v as usize;

    unsafe {
        asm!(
            "1:", // .loop_cols
                "lsi f8, {ptr_v_runner}, 0",
                "lsi f9, {alpha_ptr}, 0",
                "mul.s f8, f8, f9", // f8 = alpha * v[j]

                "mov {u_runner}, {ptr_u}",
                "mov {a_read}, {ptr_a_col}",
                "mov {m_count}, {m_div_4}",

                "2:", // .loop_rows
                    "mov {a_write}, {a_read}",
                    "EE.LDF.128.IP f3, f2, f1, f0, {a_read}, 16",
                    "EE.LDF.128.IP f7, f6, f5, f4, {u_runner}, 16",

                    "msub.s f0, f4, f8",
                    "msub.s f1, f5, f8",
                    "msub.s f2, f6, f8",
                    "msub.s f3, f7, f8",

                    "EE.STF.128.IP f3, f2, f1, f0, {a_write}, 16",

                    "addi {m_count}, {m_count}, -1",
                    "bnez {m_count}, 2b",

                "add {ptr_a_col}, {ptr_a_col}, {a_stride_bytes}",
                "addi {ptr_v_runner}, {ptr_v_runner}, 4",
                "addi {n_count}, {n_count}, -1",
                "bnez {n_count}, 1b",

            ptr_a_col = inout(reg) ptr_a_col => _,
            ptr_v_runner = inout(reg) ptr_v_runner => _,
            ptr_u = in(reg) ptr_u,
            alpha_ptr = in(reg) alpha_ptr,
            a_stride_bytes = in(reg) a_stride_bytes,
            m_div_4 = in(reg) m_div_4,
            n_count = inout(reg) n_count => _,

            u_runner = out(reg) _,
            a_read = out(reg) _,
            a_write = out(reg) _,
            m_count = out(reg) _,

            out("f0") _, out("f1") _, out("f2") _, out("f3") _, out("f4") _,
            out("f5") _, out("f6") _, out("f7") _, out("f8") _, out("f9") _,
            options(nostack)
        );
    }
}
