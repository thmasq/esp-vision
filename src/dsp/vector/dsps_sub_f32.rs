use core::arch::asm;

/// Vectorized single-precision float subtraction using ESP32-S3 SIMD extension.
#[inline(always)]
pub unsafe fn dsps_sub_f32_aes3_core(
    ptr_a: *const f32,
    ptr_b: *const f32,
    ptr_out: *mut f32,
    len: usize,
) {
    let mut a_runner = ptr_a as usize;
    let mut b_runner = ptr_b as usize;
    let mut out_runner = ptr_out as usize;

    let len_count = len / 4;
    let len_rem = len % 4;

    if len_count > 0 {
        let loop_count = len_count;

        unsafe {
            asm!(
                "1:",
                    "EE.LDF.128.IP f7, f6, f5, f4, {a_runner}, 16",
                    "EE.LDF.128.IP f11, f10, f9, f8, {b_runner}, 16",

                    // Subtraction using Xtensa FPU: dest = source1 - source2
                    "sub.s f0, f4, f8",
                    "sub.s f1, f5, f9",
                    "sub.s f2, f6, f10",
                    "sub.s f3, f7, f11",

                    "EE.STF.128.IP f3, f2, f1, f0, {out_runner}, 16",

                    "addi {loop_count}, {loop_count}, -1",
                    "bnez {loop_count}, 1b",

                a_runner = inout(reg) a_runner,
                b_runner = inout(reg) b_runner,
                out_runner = inout(reg) out_runner,
                loop_count = inout(reg) loop_count => _,

                out("f0") _, out("f1") _, out("f2") _, out("f3") _,
                out("f4") _, out("f5") _, out("f6") _, out("f7") _,
                out("f8") _, out("f9") _, out("f10") _, out("f11") _,

                options(nostack)
            );
        }
    }

    if len_rem > 0 {
        let a_rem = unsafe { core::slice::from_raw_parts(a_runner as *const f32, len_rem) };
        let b_rem = unsafe { core::slice::from_raw_parts(b_runner as *const f32, len_rem) };
        let out_rem = unsafe { core::slice::from_raw_parts_mut(out_runner as *mut f32, len_rem) };
        for i in 0..len_rem {
            out_rem[i] = a_rem[i] - b_rem[i];
        }
    }
}

/// Hardware accelerated subtraction of two single-precision arrays.
pub fn esp_sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "Slices must have the same length");
    assert_eq!(a.len(), out.len(), "Output slice must have the same length");

    unsafe {
        core::hint::assert_unchecked(a.as_ptr() as usize % 16 == 0);
        core::hint::assert_unchecked(b.as_ptr() as usize % 16 == 0);
        core::hint::assert_unchecked(out.as_ptr() as usize % 16 == 0);

        dsps_sub_f32_aes3_core(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len())
    }
}
