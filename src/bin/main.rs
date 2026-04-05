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
    EspMatrixMath,
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

fn test_vector_dotprod() {
    info!("--- FUNCTIONAL TEST (VECTOR DOT PRODUCT) ---");

    let size = 15;
    info!("Initializing vector of size {}...", size);

    let mut vec_a = AlignedDVec::<f32>::zeros(size);
    let mut vec_b = AlignedDVec::<f32>::zeros(size);

    for i in 0..size {
        vec_a[(i, 0)] = (i + 1) as f32;
        vec_b[(i, 0)] = (size - i) as f32;
    }

    let result_hardware = vec_a.esp_dot(&vec_b);
    let result_software = vec_a.dot(&vec_b);

    let diff = (result_hardware - result_software).abs();

    if diff > 0.0001 {
        error!(
            "FAILED: MISMATCH Hardware = {}, Software = {}",
            result_hardware, result_software
        );
    } else {
        info!("SUCCESS: Assembly SIMD dot product matches standard nalgebra dot");
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

fn benchmark_vector_dotprod() {
    info!("--- BENCHMARK TEST (VECTOR DOT PRODUCT) ---");
    let size = 1024;
    let iterations = 10000;

    info!(
        "Allocating {}x1 vectors for benchmarking over {} iterations...",
        size, iterations
    );

    let mut vec_a = AlignedDVec::<f32>::zeros(size);
    let mut vec_b = AlignedDVec::<f32>::zeros(size);

    for i in 0..size {
        vec_a[(i, 0)] = i as f32 % 7.0;
        vec_b[(i, 0)] = (size - i) as f32 % 11.0;
    }

    info!("Executing Hardware Accelerated Dot Product (Xtensa SIMD)...");
    let start_hw = Instant::now();
    let mut hw_res = 0.0;
    for _ in 0..iterations {
        hw_res = vec_a.esp_dot(&vec_b);
    }
    let duration_hw = start_hw.elapsed();
    info!(
        "-> Hardware Time: {} ms ({} microseconds)",
        duration_hw.as_millis(),
        duration_hw.as_micros()
    );

    info!("Executing ANSI Software Dot Product (nalgebra default)...");
    let start_sw = Instant::now();
    let mut sw_res = 0.0;
    for _ in 0..iterations {
        sw_res = vec_a.dot(&vec_b);
    }
    let duration_sw = start_sw.elapsed();
    info!(
        "-> Software Time: {} ms ({} microseconds)",
        duration_sw.as_millis(),
        duration_sw.as_micros()
    );

    let diff = (hw_res - sw_res).abs();
    info!(
        "Benchmark Result Consistency Check (Error magnitude): {}",
        diff
    );

    let perf_gain = duration_sw.as_micros() as f32 / duration_hw.as_micros() as f32;
    info!("===============================================");
    info!(
        ">>> DOT PRODUCT PERFORMANCE GAIN: {:.2}x FASTER <<<",
        perf_gain
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
    test_vector_dotprod();

    benchmark_matrix_math();
    benchmark_matrix_math_ex();
    benchmark_matrix_math_fixed();
    benchmark_matrix_math_small_hot_loops();
    benchmark_vector_dotprod();

    let _ = spawner;

    loop {
        info!("Node idling...");
        Timer::after(Duration::from_secs(5)).await;
    }
}
