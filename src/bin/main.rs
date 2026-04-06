#![no_std]
#![no_main]
#![deny(
    clippy::mem_forget,
    reason = "mem::forget is generally not safe to do with esp_hal types, especially those \
    holding buffers for the duration of a data transfer."
)]
#![deny(clippy::large_stack_frames)]

use embassy_executor::Spawner;
use embassy_time::{Duration, Instant, Timer};
use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::timer::timg::TimerGroup;

use esp_vision::dsp::{
    AlignedDMat, AlignedDMatExt, AlignedDVec, AlignedDVecExt, EspDotProd, EspFixedMatrixMath,
    EspImageMath, EspMatrixMath, EspVectorMath,
};
use log::{error, info};

extern crate alloc;

esp_bootloader_esp_idf::esp_app_desc!();

fn test_matrix_math() {
    info!("--- FUNCTIONAL TEST (FLOAT) ---");
    info!("Initializing 4x4 test matrices...");

    let mut mat_a = AlignedDMat::<f32>::zeros(4, 4);
    let mut mat_b = AlignedDMat::<f32>::zeros(4, 4);

    for c in 0..4 {
        for r in 0..4 {
            let i = c * 4 + r;
            mat_a[(r, c)] = (i + 1) as f32;
            mat_b[(r, c)] = (16 - i) as f32;
        }
    }

    let result_hardware = mat_a.esp_mul(&mat_b);

    let mut result_software = AlignedDMat::<f32>::zeros(4, 4);
    result_software.gemm(1.0, &mat_a, &mat_b, 0.0);

    let mut success = true;
    for c in 0..4 {
        for r in 0..4 {
            let diff = (result_hardware[(r, c)] - result_software[(r, c)]).abs();
            if diff > 0.0001 {
                error!(
                    "MISMATCH at (row: {}, col: {}): Hardware = {}, Software = {}",
                    r,
                    c,
                    result_hardware[(r, c)],
                    result_software[(r, c)]
                );
                success = false;
            }
        }
    }

    if success {
        info!("SUCCESS: Assembly SIMD matches standard nalgebra math");
    } else {
        error!("FAILED: Assembly logic has a math or memory alignment error.");
    }
}

fn test_matrix_math_ex() {
    info!("--- FUNCTIONAL TEST (EXTENDED STRIDES) ---");
    info!("Extracting 4x4 sub-matrices from 8x8 parents...");

    let a_stride = 8;
    let b_stride = 8;
    let m = 4;
    let n = 4;
    let k = 4;

    let mut mat_a = AlignedDMat::<f32>::zeros(8, 8);
    let mut mat_b = AlignedDMat::<f32>::zeros(8, 8);

    for c in 0..8 {
        for r in 0..8 {
            let i = c * 8 + r;
            mat_a[(r, c)] = (i + 1) as f32;
            mat_b[(r, c)] = (64 - i) as f32;
        }
    }

    let result_hardware = mat_a.esp_mul_ex(a_stride, &mat_b, b_stride, m, n, k);

    let mut result_software = AlignedDMat::<f32>::zeros(m, k);

    // Manual math leveraging standard (row, col) indexing
    for i in 0..m {
        for j in 0..k {
            let mut sum = 0.0;
            for s in 0..n {
                sum += mat_a[(i, s)] * mat_b[(s, j)];
            }
            result_software[(i, j)] = sum;
        }
    }

    let mut success = true;
    for j in 0..k {
        for i in 0..m {
            let diff = (result_hardware[(i, j)] - result_software[(i, j)]).abs();
            if diff > 0.0001 {
                error!(
                    "MISMATCH EX at (row: {}, col: {}): Hardware = {}, Software = {}",
                    i,
                    j,
                    result_hardware[(i, j)],
                    result_software[(i, j)]
                );
                success = false;
            }
        }
    }

    if success {
        info!("SUCCESS: Assembly SIMD EX matches row-major baseline");
    } else {
        error!("FAILED: Assembly logic EX has a math or memory alignment error.");
    }
}

fn test_matrix_math_fixed() {
    info!("--- FUNCTIONAL TEST (Q15 FIXED-POINT) ---");
    info!("Initializing 8x8 fixed-point test matrices...");

    let m = 8;
    let n = 8;
    let k = 8;

    let mut mat_a = AlignedDMat::<i16>::zeros(m, n);
    let mut mat_b = AlignedDMat::<i16>::zeros(n, k);

    for c in 0..n {
        for r in 0..m {
            let i = c * m + r;
            mat_a[(r, c)] = (i as i16 * 10) % 1000;
        }
    }

    for c in 0..k {
        for r in 0..n {
            let i = c * n + r;
            mat_b[(r, c)] = (2000 - i as i16 * 5) % 1000;
        }
    }

    let shift = 15;
    let result_hardware = mat_a.esp_mul_fixed(&mat_b, shift);

    let mut result_software = AlignedDMat::<i16>::zeros(m, k);

    for i in 0..m {
        for j in 0..k {
            let mut sum: i32 = 0;
            for s in 0..n {
                sum += (mat_a[(i, s)] as i32) * (mat_b[(s, j)] as i32);
            }
            let round_offset = if shift > 0 { 32767 >> shift } else { 0 };
            result_software[(i, j)] = ((sum + round_offset) >> shift) as i16;
        }
    }

    let mut success = true;
    for j in 0..k {
        for i in 0..m {
            let diff = (result_hardware[(i, j)] as i32 - result_software[(i, j)] as i32).abs();
            if diff > 1 {
                error!(
                    "MISMATCH FIXED at (row: {}, col: {}): Hardware = {}, Software = {}",
                    i,
                    j,
                    result_hardware[(i, j)],
                    result_software[(i, j)]
                );
                success = false;
            }
        }
    }

    if success {
        info!("SUCCESS: Assembly Vector Math matches fixed-point baseline");
    } else {
        error!("FAILED: Fixed-point assembly logic has an error.");
    }
}

fn test_vector_math() {
    info!("--- FUNCTIONAL TEST (VECTOR MATH) ---");

    let size = 15;
    info!("Initializing vectors of size {}...", size);

    let mut vec_a = AlignedDVec::<f32>::zeros(size);
    let mut vec_b = AlignedDVec::<f32>::zeros(size);

    for i in 0..size {
        vec_a[(i, 0)] = (i + 1) as f32;
        vec_b[(i, 0)] = (size - i) as f32;
    }

    let mut success = true;

    // 1. Dot Product
    let dot_hw = vec_a.esp_dot(&vec_b);
    let dot_sw = vec_a.dot(&vec_b);
    if (dot_hw - dot_sw).abs() > 0.0001 {
        error!("DOT FAILED: HW = {}, SW = {}", dot_hw, dot_sw);
        success = false;
    }

    // 2. Addition
    let add_hw = vec_a.esp_add(&vec_b);
    for i in 0..size {
        let sw_val = vec_a[(i, 0)] + vec_b[(i, 0)];
        if (add_hw[(i, 0)] - sw_val).abs() > 0.0001 {
            error!(
                "ADD FAILED at {}: HW = {}, SW = {}",
                i,
                add_hw[(i, 0)],
                sw_val
            );
            success = false;
        }
    }

    // 3. Subtraction
    let sub_hw = vec_a.esp_sub(&vec_b);
    for i in 0..size {
        let sw_val = vec_a[(i, 0)] - vec_b[(i, 0)];
        if (sub_hw[(i, 0)] - sw_val).abs() > 0.0001 {
            error!(
                "SUB FAILED at {}: HW = {}, SW = {}",
                i,
                sub_hw[(i, 0)],
                sw_val
            );
            success = false;
        }
    }

    // 4. Multiplication
    let mul_hw = vec_a.esp_mul_elem(&vec_b);
    for i in 0..size {
        let sw_val = vec_a[(i, 0)] * vec_b[(i, 0)];
        if (mul_hw[(i, 0)] - sw_val).abs() > 0.0001 {
            error!(
                "MUL FAILED at {}: HW = {}, SW = {}",
                i,
                mul_hw[(i, 0)],
                sw_val
            );
            success = false;
        }
    }

    if success {
        info!("SUCCESS: Assembly SIMD Vector math matches standard software math");
    } else {
        error!("FAILED: One or more vector operations produced incorrect results.");
    }
}

fn benchmark_matrix_math() {
    info!("--- BENCHMARK TEST (FLOAT) ---");
    let size = 64;

    info!("Allocating {}x{} matrices for benchmarking...", size, size);

    let mut mat_a = AlignedDMat::<f32>::zeros(size, size);
    let mut mat_b = AlignedDMat::<f32>::zeros(size, size);

    for c in 0..size {
        for r in 0..size {
            let i = c * size + r;
            mat_a[(r, c)] = i as f32 % 7.0;
            mat_b[(r, c)] = (size * size - i) as f32 % 11.0;
        }
    }

    info!("Executing Hardware Accelerated Multiplication (Xtensa SIMD)...");
    let start_hw = Instant::now();
    let result_hw = mat_a.esp_mul(&mat_b);
    let duration_hw = start_hw.elapsed();
    info!(
        "-> Hardware Time: {} ms ({} microseconds)",
        duration_hw.as_millis(),
        duration_hw.as_micros()
    );

    info!("Executing ANSI Software Multiplication (nalgebra default)...");
    let mut result_sw = AlignedDMat::<f32>::zeros(size, size);
    let start_sw = Instant::now();
    result_sw.gemm(1.0, &mat_a, &mat_b, 0.0);
    let duration_sw = start_sw.elapsed();
    info!(
        "-> Software Time: {} ms ({} microseconds)",
        duration_sw.as_millis(),
        duration_sw.as_micros()
    );

    let mut diff_accum = 0.0;
    for c in 0..size {
        for r in 0..size {
            diff_accum += (result_hw[(r, c)] - result_sw[(r, c)]).abs();
        }
    }

    info!(
        "Benchmark Result Consistency Check (Cumulative Error): {}",
        diff_accum
    );

    let perf_gain = duration_sw.as_micros() as f32 / duration_hw.as_micros() as f32;
    info!("===============================================");
    info!(">>> PERFORMANCE GAIN: {:.2}x FASTER <<<", perf_gain);
    info!("===============================================");
}

fn benchmark_matrix_math_ex() {
    info!("--- BENCHMARK TEST (EXTENDED STRIDES) ---");
    let m = 32;
    let n = 32;
    let k = 32;
    let a_stride = 64;
    let b_stride = 64;

    info!("Allocating 64x64 matrices for benchmarking 32x32 sub-regions...");

    let mut mat_a = AlignedDMat::<f32>::zeros(a_stride, a_stride);
    let mut mat_b = AlignedDMat::<f32>::zeros(b_stride, b_stride);

    for c in 0..a_stride {
        for r in 0..a_stride {
            let i = c * a_stride + r;
            mat_a[(r, c)] = i as f32 % 7.0;
        }
    }

    for c in 0..b_stride {
        for r in 0..b_stride {
            let i = c * b_stride + r;
            mat_b[(r, c)] = (b_stride * b_stride - i) as f32 % 11.0;
        }
    }

    info!("Executing Hardware Accelerated Multiplication (Xtensa SIMD EX)...");
    let start_hw = Instant::now();
    let result_hw = mat_a.esp_mul_ex(a_stride, &mat_b, b_stride, m, n, k);
    let duration_hw = start_hw.elapsed();
    info!(
        "-> Hardware EX Time: {} ms ({} microseconds)",
        duration_hw.as_millis(),
        duration_hw.as_micros()
    );

    info!("Executing ANSI Software Multiplication (Manual Nested Loops)...");
    let mut result_sw = AlignedDMat::<f32>::zeros(m, k);

    let start_sw = Instant::now();
    for i in 0..m {
        for j in 0..k {
            let mut sum = 0.0;
            for s in 0..n {
                sum += mat_a[(i, s)] * mat_b[(s, j)];
            }
            result_sw[(i, j)] = sum;
        }
    }
    let duration_sw = start_sw.elapsed();
    info!(
        "-> Software EX Time: {} ms ({} microseconds)",
        duration_sw.as_millis(),
        duration_sw.as_micros()
    );

    let mut diff_accum = 0.0;
    for j in 0..k {
        for i in 0..m {
            diff_accum += (result_hw[(i, j)] - result_sw[(i, j)]).abs();
        }
    }

    info!(
        "Benchmark EX Result Consistency Check (Cumulative Error): {}",
        diff_accum
    );

    let perf_gain = duration_sw.as_micros() as f32 / duration_hw.as_micros() as f32;
    info!("===============================================");
    info!(">>> EX PERFORMANCE GAIN: {:.2}x FASTER <<<", perf_gain);
    info!("===============================================");
}

fn benchmark_matrix_math_fixed() {
    info!("--- BENCHMARK TEST (Q15 FIXED-POINT) ---");
    let size = 64;

    info!(
        "Allocating {}x{} fixed-point matrices for benchmarking...",
        size, size
    );

    let mut mat_a = AlignedDMat::<i16>::zeros(size, size);
    let mut mat_b = AlignedDMat::<i16>::zeros(size, size);

    for c in 0..size {
        for r in 0..size {
            let i = c * size + r;
            mat_a[(r, c)] = (i as i16 * 17) % 2000;
            mat_b[(r, c)] = (3000 - i as i16 * 13) % 2000;
        }
    }

    info!("Executing Vector Accelerated Multiplication (Xtensa qacc)...");
    let start_hw = Instant::now();
    let result_hw = mat_a.esp_mul_fixed(&mat_b, 15);
    let duration_hw = start_hw.elapsed();
    info!(
        "-> Hardware FIXED Time: {} ms ({} microseconds)",
        duration_hw.as_millis(),
        duration_hw.as_micros()
    );

    info!("Executing ANSI Software Fixed-Point Multiplication...");
    let mut result_sw = AlignedDMat::<i16>::zeros(size, size);

    let start_sw = Instant::now();
    let round_offset = 32767 >> 15;

    for i in 0..size {
        for j in 0..size {
            let mut sum: i32 = 0;
            for s in 0..size {
                sum += (mat_a[(i, s)] as i32) * (mat_b[(s, j)] as i32);
            }
            result_sw[(i, j)] = ((sum + round_offset) >> 15) as i16;
        }
    }
    let duration_sw = start_sw.elapsed();
    info!(
        "-> Software FIXED Time: {} ms ({} microseconds)",
        duration_sw.as_millis(),
        duration_sw.as_micros()
    );

    let mut diff_accum: i32 = 0;
    for j in 0..size {
        for i in 0..size {
            diff_accum += (result_hw[(i, j)] as i32 - result_sw[(i, j)] as i32).abs();
        }
    }

    info!(
        "Benchmark FIXED Result Consistency Check (Cumulative Rounding Drift): {}",
        diff_accum
    );

    let perf_gain = duration_sw.as_micros() as f32 / duration_hw.as_micros() as f32;
    info!("===============================================");
    info!(">>> FIXED PERFORMANCE GAIN: {:.2}x FASTER <<<", perf_gain);
    info!("===============================================");
}

fn benchmark_matrix_math_small_hot_loops() {
    info!("--- BENCHMARK TEST (SMALL MATRICES HOT LOOP) ---");
    let sizes = [4, 6, 8, 10];
    let iterations = 5000;

    for &size in &sizes {
        info!(
            "Allocating and benchmarking {}x{} matrix over {} iterations...",
            size, size, iterations
        );

        let mut mat_a = AlignedDMat::<f32>::zeros(size, size);
        let mut mat_b = AlignedDMat::<f32>::zeros(size, size);

        let mut result_hw = AlignedDMat::<f32>::zeros(size, size);
        let mut result_sw = AlignedDMat::<f32>::zeros(size, size);

        for c in 0..size {
            for r in 0..size {
                let i = c * size + r;
                mat_a[(r, c)] = i as f32 % 7.0;
                mat_b[(r, c)] = (size * size - i) as f32 % 11.0;
            }
        }

        let start_hw = Instant::now();
        for _ in 0..iterations {
            mat_a.esp_mul_to(&mat_b, &mut result_hw);
        }
        let duration_hw = start_hw.elapsed();
        info!(
            "-> Hardware {}x{} Time: {} ms ({} microseconds)",
            size,
            size,
            duration_hw.as_millis(),
            duration_hw.as_micros()
        );

        let start_sw = Instant::now();
        for _ in 0..iterations {
            result_sw.gemm(1.0, &mat_a, &mat_b, 0.0);
        }
        let duration_sw = start_sw.elapsed();
        info!(
            "-> Software {}x{} Time: {} ms ({} microseconds)",
            size,
            size,
            duration_sw.as_millis(),
            duration_sw.as_micros()
        );

        let perf_gain = duration_sw.as_micros() as f32 / duration_hw.as_micros() as f32;
        info!("===============================================");
        info!(
            ">>> {}x{} HOT LOOP GAIN: {:.2}x FASTER <<<",
            size, size, perf_gain
        );
        info!("===============================================");
    }
}

fn benchmark_vector_math() {
    info!("--- BENCHMARK TEST (VECTOR MATH) ---");
    let size = 1024;
    let iterations = 10000;

    info!(
        "Allocating {}x1 vectors for benchmarking over {} iterations...",
        size, iterations
    );

    let mut vec_a = AlignedDVec::<f32>::zeros(size);
    let mut vec_b = AlignedDVec::<f32>::zeros(size);
    let mut vec_out = AlignedDVec::<f32>::zeros(size);

    for i in 0..size {
        vec_a[(i, 0)] = i as f32 % 7.0;
        vec_b[(i, 0)] = (size - i) as f32 % 11.0;
    }

    // --- DOT PRODUCT ---
    let start_hw_dot = Instant::now();
    for _ in 0..iterations {
        // Prevent the compiler from optimizing away the loop!
        core::hint::black_box(vec_a.esp_dot(&vec_b));
    }
    let dur_hw_dot = start_hw_dot.elapsed();

    let start_sw_dot = Instant::now();
    for _ in 0..iterations {
        // Prevent the compiler from optimizing away the loop!
        core::hint::black_box(vec_a.dot(&vec_b));
    }
    let dur_sw_dot = start_sw_dot.elapsed();

    info!(
        "Dot Product: HW {}us vs SW {}us ({:.2}x FASTER)",
        dur_hw_dot.as_micros(),
        dur_sw_dot.as_micros(),
        dur_sw_dot.as_micros() as f32 / dur_hw_dot.as_micros() as f32
    );

    // --- ADDITION ---
    let start_hw_add = Instant::now();
    for _ in 0..iterations {
        vec_a.esp_add_to(&vec_b, &mut vec_out);
    }
    let dur_hw_add = start_hw_add.elapsed();

    let start_sw_add = Instant::now();
    for _ in 0..iterations {
        for i in 0..size {
            vec_out[(i, 0)] = vec_a[(i, 0)] + vec_b[(i, 0)];
        }
        core::hint::black_box(&vec_out); // Just to be safe
    }
    let dur_sw_add = start_sw_add.elapsed();

    info!(
        "Addition:    HW {}us vs SW {}us ({:.2}x FASTER)",
        dur_hw_add.as_micros(),
        dur_sw_add.as_micros(),
        dur_sw_add.as_micros() as f32 / dur_hw_add.as_micros() as f32
    );

    // --- SUBTRACTION ---
    let start_hw_sub = Instant::now();
    for _ in 0..iterations {
        vec_a.esp_sub_to(&vec_b, &mut vec_out);
    }
    let dur_hw_sub = start_hw_sub.elapsed();

    let start_sw_sub = Instant::now();
    for _ in 0..iterations {
        for i in 0..size {
            vec_out[(i, 0)] = vec_a[(i, 0)] - vec_b[(i, 0)];
        }
        core::hint::black_box(&vec_out);
    }
    let dur_sw_sub = start_sw_sub.elapsed();

    info!(
        "Subtraction: HW {}us vs SW {}us ({:.2}x FASTER)",
        dur_hw_sub.as_micros(),
        dur_sw_sub.as_micros(),
        dur_sw_sub.as_micros() as f32 / dur_hw_sub.as_micros() as f32
    );

    // --- MULTIPLICATION ---
    let start_hw_mul = Instant::now();
    for _ in 0..iterations {
        vec_a.esp_mul_elem_to(&vec_b, &mut vec_out);
    }
    let dur_hw_mul = start_hw_mul.elapsed();

    let start_sw_mul = Instant::now();
    for _ in 0..iterations {
        for i in 0..size {
            vec_out[(i, 0)] = vec_a[(i, 0)] * vec_b[(i, 0)];
        }
        core::hint::black_box(&vec_out);
    }
    let dur_sw_mul = start_sw_mul.elapsed();

    info!(
        "Multiply:    HW {}us vs SW {}us ({:.2}x FASTER)",
        dur_hw_mul.as_micros(),
        dur_sw_mul.as_micros(),
        dur_sw_mul.as_micros() as f32 / dur_hw_mul.as_micros() as f32
    );

    info!("===============================================");
}

fn test_image_math() {
    info!("--- FUNCTIONAL TEST (IMAGE MATH SIMD) ---");
    let size = 64; // A multiple of 16 to test clean SIMD lanes

    // Allocate raw vecs and slice them to guarantee 16-byte alignment
    let mut raw_a = alloc::vec![0u8; size + 16];
    let offset_a = raw_a.as_ptr().align_offset(16);
    let slice_a = &mut raw_a[offset_a..offset_a + size];

    let mut raw_b = alloc::vec![0u8; size + 16];
    let offset_b = raw_b.as_ptr().align_offset(16);
    let slice_b = &mut raw_b[offset_b..offset_b + size];

    for i in 0..size {
        slice_a[i] = (i * 3) as u8;
        slice_b[i] = (255 - i) as u8;
    }

    let mut success = true;

    // 1. Min/Max Test
    let (min_hw, max_hw) = slice_a.esp_min_max();
    let min_sw = *slice_a.iter().min().unwrap();
    let max_sw = *slice_a.iter().max().unwrap();

    if min_hw != min_sw || max_hw != max_sw {
        error!(
            "MIN/MAX FAILED: HW=({}, {}), SW=({}, {})",
            min_hw, max_hw, min_sw, max_sw
        );
        success = false;
    }

    // 2. Sum Test
    let sum_hw = slice_a.esp_sum();
    // Use an iterator mapping to u32 so the software implementation doesn't overflow
    let sum_sw: u32 = slice_a.iter().map(|&x| x as u32).sum();

    if sum_hw != sum_sw {
        error!("SUM FAILED: HW={}, SW={}", sum_hw, sum_sw);
        success = false;
    }

    if success {
        info!("SUCCESS: Assembly SIMD Image math matches standard software math");
    } else {
        error!("FAILED: Image math SIMD operations produced incorrect results.");
    }
}

fn benchmark_image_math() {
    info!("--- BENCHMARK TEST (IMAGE MATH SIMD) ---");
    // Simulate a standard QQVGA image buffer size (160x120 = 19,200 pixels)
    let size = 19_200;
    let iterations = 1000;

    info!(
        "Allocating {} byte image buffers for benchmarking over {} iterations...",
        size, iterations
    );

    // Guaranteed 16-byte aligned buffers
    let mut raw_a = alloc::vec![0u8; size + 16];
    let offset_a = raw_a.as_ptr().align_offset(16);
    let slice_a = &mut raw_a[offset_a..offset_a + size];

    let mut raw_b = alloc::vec![0u8; size + 16];
    let offset_b = raw_b.as_ptr().align_offset(16);
    let slice_b = &mut raw_b[offset_b..offset_b + size];

    // Pre-fill with arbitrary data
    for i in 0..size {
        slice_a[i] = (i % 256) as u8;
        slice_b[i] = ((size - i) % 256) as u8;
    }

    // --- MIN / MAX BENCHMARK ---
    let start_hw_minmax = Instant::now();
    for _ in 0..iterations {
        core::hint::black_box(slice_a.esp_min_max());
    }
    let dur_hw_minmax = start_hw_minmax.elapsed();

    let start_sw_minmax = Instant::now();
    for _ in 0..iterations {
        let min = *slice_a.iter().min().unwrap();
        let max = *slice_a.iter().max().unwrap();
        core::hint::black_box((min, max));
    }
    let dur_sw_minmax = start_sw_minmax.elapsed();

    info!(
        "Min/Max: HW {}us vs SW {}us ({:.2}x FASTER)",
        dur_hw_minmax.as_micros(),
        dur_sw_minmax.as_micros(),
        dur_sw_minmax.as_micros() as f32 / dur_hw_minmax.as_micros() as f32
    );

    // --- SUM BENCHMARK ---
    let start_hw_sum = Instant::now();
    for _ in 0..iterations {
        core::hint::black_box(slice_a.esp_sum());
    }
    let dur_hw_sum = start_hw_sum.elapsed();

    let start_sw_sum = Instant::now();
    for _ in 0..iterations {
        let sum: u32 = slice_a.iter().map(|&x| x as u32).sum();
        core::hint::black_box(sum);
    }
    let dur_sw_sum = start_sw_sum.elapsed();

    info!(
        "Sum:     HW {}us vs SW {}us ({:.2}x FASTER)",
        dur_hw_sum.as_micros(),
        dur_sw_sum.as_micros(),
        dur_sw_sum.as_micros() as f32 / dur_hw_sum.as_micros() as f32
    );

    info!("===============================================");
}

#[allow(
    clippy::large_stack_frames,
    reason = "it's not unusual to allocate larger buffers etc. in main"
)]
#[esp_rtos::main]
async fn main(spawner: Spawner) -> ! {
    esp_println::logger::init_logger_from_env();

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    let timg0 = TimerGroup::new(peripherals.TIMG0);
    esp_rtos::start(timg0.timer0);

    info!("Embassy initialized!");

    test_matrix_math();
    test_matrix_math_ex();
    test_matrix_math_fixed();
    test_vector_math();
    test_image_math();

    benchmark_matrix_math();
    benchmark_matrix_math_ex();
    benchmark_matrix_math_fixed();
    benchmark_matrix_math_small_hot_loops();
    benchmark_vector_math();
    benchmark_image_math();

    let _ = spawner;

    loop {
        info!("Node idling...");
        Timer::after(Duration::from_secs(5)).await;
    }
}
