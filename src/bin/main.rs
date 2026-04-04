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
use esp_vision::dsp::{AlignedDMat, AlignedDMatExt, EspMatrixMath};
use log::{error, info};

extern crate alloc;

esp_bootloader_esp_idf::esp_app_desc!();

fn test_matrix_math() {
    info!("--- FUNCTIONAL TEST ---");
    info!("Initializing 4x4 test matrices...");

    let mut mat_a = AlignedDMat::<f32>::zeros(4, 4);
    let mut mat_b = AlignedDMat::<f32>::zeros(4, 4);

    for i in 0..16 {
        mat_a[i] = (i + 1) as f32;
        mat_b[i] = (16 - i) as f32;
    }

    let result_hardware = mat_a.esp_mul(&mat_b);

    let mut result_software = AlignedDMat::<f32>::zeros(4, 4);
    result_software.gemm(1.0, &mat_a, &mat_b, 0.0);

    let mut success = true;
    for i in 0..16 {
        let diff = (result_hardware[i] - result_software[i]).abs();
        if diff > 0.0001 {
            error!(
                "MISMATCH at index {}: Hardware = {}, Software = {}",
                i, result_hardware[i], result_software[i]
            );
            success = false;
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

    let mut mat_a = AlignedDMat::<f32>::zeros(8, 8); // 64 elements
    let mut mat_b = AlignedDMat::<f32>::zeros(8, 8); // 64 elements

    let a_slice_mut = mat_a.as_mut_slice();
    for i in 0..64 {
        a_slice_mut[i] = (i + 1) as f32;
    }

    let b_slice_mut = mat_b.as_mut_slice();
    for i in 0..64 {
        b_slice_mut[i] = (64 - i) as f32;
    }

    let result_hardware = mat_a.esp_mul_ex(a_stride, &mat_b, b_stride, m, n, k);

    let mut result_software = AlignedDMat::<f32>::zeros(m, k);
    let c_slice = result_software.as_mut_slice();

    let a_slice = mat_a.as_slice();
    let b_slice = mat_b.as_slice();

    // Manual row-major math matching the ANSI C definition of mult_ex
    for i in 0..m {
        for j in 0..k {
            let mut sum = 0.0;
            for s in 0..n {
                sum += a_slice[i * a_stride + s] * b_slice[s * b_stride + j];
            }
            c_slice[i * k + j] = sum;
        }
    }

    let mut success = true;
    let res_hw_slice = result_hardware.as_slice();
    for i in 0..(m * k) {
        let diff = (res_hw_slice[i] - c_slice[i]).abs();
        if diff > 0.0001 {
            error!(
                "MISMATCH EX at index {}: Hardware = {}, Software = {}",
                i, res_hw_slice[i], c_slice[i]
            );
            success = false;
        }
    }

    if success {
        info!("SUCCESS: Assembly SIMD EX matches row-major baseline");
    } else {
        error!("FAILED: Assembly logic EX has a math or memory alignment error.");
    }
}

fn benchmark_matrix_math() {
    info!("--- BENCHMARK TEST ---");
    let size = 64;

    info!("Allocating {}x{} matrices for benchmarking...", size, size);

    let mut mat_a = AlignedDMat::<f32>::zeros(size, size);
    let mut mat_b = AlignedDMat::<f32>::zeros(size, size);

    for i in 0..(size * size) {
        mat_a[i] = i as f32 % 7.0;
        mat_b[i] = (size * size - i) as f32 % 11.0;
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
    for i in 0..(size * size) {
        diff_accum += (result_hw[i] - result_sw[i]).abs();
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

    let a_slice_mut = mat_a.as_mut_slice();
    for i in 0..(a_stride * a_stride) {
        a_slice_mut[i] = i as f32 % 7.0;
    }

    let b_slice_mut = mat_b.as_mut_slice();
    for i in 0..(b_stride * b_stride) {
        b_slice_mut[i] = (b_stride * b_stride - i) as f32 % 11.0;
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
    let c_slice = result_sw.as_mut_slice();
    let a_slice = mat_a.as_slice();
    let b_slice = mat_b.as_slice();

    let start_sw = Instant::now();
    for i in 0..m {
        for j in 0..k {
            let mut sum = 0.0;
            for s in 0..n {
                sum += a_slice[i * a_stride + s] * b_slice[s * b_stride + j];
            }
            c_slice[i * k + j] = sum;
        }
    }
    let duration_sw = start_sw.elapsed();
    info!(
        "-> Software EX Time: {} ms ({} microseconds)",
        duration_sw.as_millis(),
        duration_sw.as_micros()
    );

    let mut diff_accum = 0.0;
    let res_hw_slice = result_hw.as_slice();
    for i in 0..(m * k) {
        diff_accum += (res_hw_slice[i] - c_slice[i]).abs();
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
    benchmark_matrix_math();
    benchmark_matrix_math_ex();

    let _ = spawner;

    loop {
        info!("Node idling...");
        Timer::after(Duration::from_secs(5)).await;
    }
}
