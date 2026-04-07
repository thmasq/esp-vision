#![no_std]
#![no_main]
#![deny(
    clippy::mem_forget,
    reason = "mem::forget is generally not safe to do with esp_hal types, especially those \
    holding buffers for the duration of a data transfer."
)]
#![deny(clippy::large_stack_frames)]

use embassy_executor::Spawner;
use embassy_time::Instant;
use esp_backtrace as _;
use esp_hal::{
    clock::CpuClock,
    dma::{DmaRxBuf, DmaTxBuf},
    i2c::master::{Config as I2cConfig, I2c},
    lcd_cam::{
        LcdCam,
        cam::{Camera, Config as CamConfig},
    },
    spi::{Mode as SpiMode, slave::Spi},
    time::Rate,
    timer::timg::TimerGroup,
};

use esp_vision::apriltag::decode::AprilTagDetection;
use esp_vision::apriltag::unionfind::RleUnionFind;
use esp_vision::ov2640::{Ov2640, Resolution};
use log::info;

extern crate alloc;

esp_bootloader_esp_idf::esp_app_desc!();

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct SpiPayload {
    pub sequence: u32,
    pub timestamp_ms: u64,
    pub tag_count: u8,
    pub tags: [AprilTagDetection; 10],
}

impl SpiPayload {
    pub const fn empty() -> Self {
        Self {
            sequence: 0,
            timestamp_ms: 0,
            tag_count: 0,
            tags: [AprilTagDetection {
                id: 0,
                hamming: 0,
                rotation: 0,
                confidence: 0.0,
                center_x: 0.0,
                center_y: 0.0,
                yaw: 0.0,
                pitch: 0.0,
                roll: 0.0,
                distance_mm: 0.0,
            }; 10],
        }
    }
}

static mut SPI_TX_PAYLOAD: SpiPayload = SpiPayload::empty();

const CHUNK_LINES: usize = 16;
const CHUNK_SIZE: usize = 640 * CHUNK_LINES;
static mut SRAM_CHUNK_A: [core::mem::MaybeUninit<u8>; CHUNK_SIZE] =
    [core::mem::MaybeUninit::uninit(); CHUNK_SIZE];

#[allow(
    clippy::large_stack_frames,
    reason = "it's not unusual to allocate larger buffers etc. in main"
)]
#[esp_rtos::main]
async fn main(_spawner: Spawner) -> ! {
    esp_println::logger::init_logger_from_env();
    info!("Starting ESP-Vision Coprocessor...");

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    unsafe {
        esp_alloc::HEAP.add_region(esp_alloc::HeapRegion::new(
            0x3c00_0000 as *mut u8,
            8 * 1024 * 1024,
            esp_alloc::MemoryCapability::External.into(),
        ));
    }

    let timg0 = TimerGroup::new(peripherals.TIMG0);
    esp_rtos::start(timg0.timer0);

    let lcd_cam = LcdCam::new(peripherals.LCD_CAM);
    let cam_config = CamConfig::default().with_frequency(Rate::from_mhz(20));

    let mut camera = Camera::new(lcd_cam.cam, peripherals.DMA_CH0, cam_config)
        .unwrap()
        .with_master_clock(peripherals.GPIO15)
        .with_pixel_clock(peripherals.GPIO13)
        .with_vsync(peripherals.GPIO6)
        .with_h_enable(peripherals.GPIO7)
        .with_data0(peripherals.GPIO11)
        .with_data1(peripherals.GPIO9)
        .with_data2(peripherals.GPIO8)
        .with_data3(peripherals.GPIO10)
        .with_data4(peripherals.GPIO12)
        .with_data5(peripherals.GPIO18)
        .with_data6(peripherals.GPIO17)
        .with_data7(peripherals.GPIO16);

    let i2c = I2c::new(peripherals.I2C0, I2cConfig::default())
        .unwrap()
        .with_sda(peripherals.GPIO4)
        .with_scl(peripherals.GPIO5)
        .into_async();

    let mut camera_sensor = Ov2640::new(i2c);
    info!("Initializing OV2640...");
    camera_sensor
        .init_yuv422()
        .await
        .expect("Failed to init OV2640");
    camera_sensor
        .set_resolution(Resolution::Res640x480)
        .await
        .expect("Failed to set resolution");

    let spi_dma_channel = peripherals.DMA_CH2;
    let mut spi_slave = Spi::new(peripherals.SPI2, SpiMode::_0)
        .with_sck(peripherals.GPIO39)
        .with_mosi(peripherals.GPIO40)
        .with_miso(peripherals.GPIO41)
        .with_cs(peripherals.GPIO42)
        .with_dma(spi_dma_channel);

    info!("Pipeline initialized. Starting concurrent execution.");

    let spi_slave_future = async {
        let (tx_descriptors, _) = esp_hal::dma_descriptors!(core::mem::size_of::<SpiPayload>(), 0);

        let mut spi_tx_buf = DmaTxBuf::new(tx_descriptors, unsafe {
            core::slice::from_raw_parts_mut(
                core::ptr::addr_of_mut!(SPI_TX_PAYLOAD) as *mut u8,
                core::mem::size_of::<SpiPayload>(),
            )
        })
        .unwrap();

        loop {
            let transfer = match spi_slave.write(core::mem::size_of::<SpiPayload>(), spi_tx_buf) {
                Ok(t) => t,
                Err(_) => panic!("SPI Write initialization failed"),
            };

            while !transfer.is_done() {
                embassy_futures::yield_now().await;
            }

            let (reclaimed_spi, reclaimed_buf) = transfer.wait();
            spi_slave = reclaimed_spi;
            spi_tx_buf = reclaimed_buf;
        }
    };

    let vision_pipeline_future = async {
        let total_lines = 480;
        let chunks_count = total_lines / CHUNK_LINES;
        let mut global_uf = RleUnionFind::new(8000);

        let mut mono_image = esp_vision::apriltag::image::Image::new(640, 480, 640);

        let (rx_descriptors, _) = esp_hal::dma_descriptors!(614400, 0);

        let camera_yuyv_psram: &'static mut [u8] = alloc::vec![0u8; 614_400].leak();

        let camera_yuyv_ptr = camera_yuyv_psram.as_ptr();

        let mut dma_rx_buf = DmaRxBuf::new(rx_descriptors, camera_yuyv_psram).unwrap();

        let sram_chunk_slice = unsafe {
            core::slice::from_raw_parts_mut(
                core::ptr::addr_of_mut!(SRAM_CHUNK_A) as *mut u8,
                CHUNK_SIZE,
            )
        };

        loop {
            let capture_start = Instant::now();
            global_uf.clear();

            let transfer = match camera.receive(dma_rx_buf) {
                Ok(t) => t,
                Err(_) => panic!("Camera RX initialization failed"),
            };

            while !transfer.is_done() {
                embassy_futures::yield_now().await;
            }

            let (_, reclaimed_cam, reclaimed_buf) = transfer.wait();
            camera = reclaimed_cam;
            dma_rx_buf = reclaimed_buf;

            let camera_read_slice =
                unsafe { core::slice::from_raw_parts(camera_yuyv_ptr, 614_400) };

            let mono_psram = mono_image.as_mut_slice();
            for i in 0..(640 * 480) {
                mono_psram[i] = camera_read_slice[i * 2];
            }

            for i in 0..chunks_count {
                let y_offset = i * CHUNK_LINES;
                let offset = i * CHUNK_SIZE;

                unsafe {
                    core::ptr::copy_nonoverlapping(
                        mono_psram.as_ptr().add(offset),
                        sram_chunk_slice.as_mut_ptr(),
                        CHUNK_SIZE,
                    );
                }

                for p in sram_chunk_slice.iter_mut() {
                    *p = if *p > 125 { 255 } else { 0 };
                }

                global_uf.process_chunk(sram_chunk_slice, 640, CHUNK_LINES, y_offset);
            }

            global_uf.flatten();
            let blobs = global_uf.extract_valid_blobs(50, 10_000);

            let intrinsics = esp_vision::apriltag::pose::CameraIntrinsics {
                fx: 500.0,
                fy: 500.0,
                cx: 320.0,
                cy: 240.0,
                tag_size_mm: 165.0,
            };

            let mut valid_detections = [AprilTagDetection {
                id: 0,
                hamming: 0,
                rotation: 0,
                confidence: 0.0,
                center_x: 0.0,
                center_y: 0.0,
                yaw: 0.0,
                pitch: 0.0,
                roll: 0.0,
                distance_mm: 0.0,
            }; 10];
            let mut tag_count = 0;

            for blob in &blobs {
                let boundary =
                    esp_vision::apriltag::quad::extract_ordered_boundary(blob, &global_uf.segments);

                if let Some(quad) = esp_vision::apriltag::quad::find_quad_corners(&boundary) {
                    if let Some(detection) = esp_vision::apriltag::decode::extract_detection(
                        &mono_image,
                        &quad,
                        &intrinsics,
                    ) {
                        if tag_count < 10 {
                            valid_detections[tag_count] = detection;
                            tag_count += 1;
                        } else {
                            info!("Max tags (10) reached, dropping additional detections.");
                            break;
                        }
                    }
                }
            }

            unsafe {
                let next_seq = SPI_TX_PAYLOAD.sequence.wrapping_add(1) | 1;
                SPI_TX_PAYLOAD.sequence = next_seq;
                core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);

                SPI_TX_PAYLOAD.timestamp_ms = Instant::now().as_millis();
                SPI_TX_PAYLOAD.tag_count = tag_count as u8;
                SPI_TX_PAYLOAD.tags = valid_detections;

                core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
                SPI_TX_PAYLOAD.sequence = next_seq.wrapping_add(1);
            }

            info!(
                "Frame complete: {} tags in {}ms",
                tag_count,
                capture_start.elapsed().as_millis()
            );
        }
    };

    embassy_futures::join::join(spi_slave_future, vision_pipeline_future).await;
    unreachable!();
}
