#![allow(dead_code)]

use embedded_hal_async::delay::DelayNs;
use embedded_hal_async::i2c::I2c;

pub mod regs;
use regs::*;

const I2C_ADDR: u8 = 0x60 >> 1;

#[derive(Debug)]
pub enum Error<I2cErr> {
    I2c(I2cErr),
    InvalidResolution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Bank {
    Dsp = BANK_DSP,
    Sensor = BANK_SENSOR,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb565,
    Yuv422,
    Jpeg,
}

#[derive(Debug, Clone, Copy)]
pub enum FrameSize {
    /// 400 x 296
    Cif,
    /// 640 x 480
    Vga,
    /// 800 x 600
    Svga,
    /// 1600 x 1200
    Uxga,
}

impl FrameSize {
    fn dimensions(&self) -> (u16, u16) {
        match self {
            FrameSize::Cif => (400, 296),
            FrameSize::Vga => (640, 480),
            FrameSize::Svga => (800, 600),
            FrameSize::Uxga => (1600, 1200),
        }
    }
}

pub struct Ov2640<I2C> {
    i2c: I2C,
    current_bank: Option<Bank>,
    pixel_format: PixelFormat,
}

impl<I2C, I2cErr> Ov2640<I2C>
where
    I2C: I2c<Error = I2cErr>,
{
    pub fn new(i2c: I2C) -> Self {
        Self {
            i2c,
            current_bank: None,
            pixel_format: PixelFormat::Yuv422,
        }
    }

    pub async fn init(&mut self, delay: &mut impl DelayNs) -> Result<(), Error<I2cErr>> {
        self.write_reg(Bank::Sensor, 0x12, 0x80).await?;
        delay.delay_ms(10).await;

        self.write_regs(OV2640_SETTINGS_BASE).await?;
        Ok(())
    }

    pub async fn set_pixel_format(
        &mut self,
        format: PixelFormat,
        delay: &mut impl DelayNs,
    ) -> Result<(), Error<I2cErr>> {
        self.pixel_format = format;
        let regs = match format {
            PixelFormat::Rgb565 => OV2640_SETTINGS_RGB565,
            PixelFormat::Yuv422 => OV2640_SETTINGS_YUV422,
            PixelFormat::Jpeg => OV2640_SETTINGS_JPEG,
        };

        self.write_regs(regs).await?;
        delay.delay_ms(10).await;
        Ok(())
    }

    /// Configures the window sizes, downscalers, and clocks mathematically.
    pub async fn set_frame_size(
        &mut self,
        size: FrameSize,
        delay: &mut impl DelayNs,
    ) -> Result<(), Error<I2cErr>> {
        let (mut w, mut h) = size.dimensions();
        let (mut max_x, mut max_y) = (1600, 1200); // 4:3 base ratio
        let (mut offset_x, mut offset_y) = (0, 0);
        let mode: FrameSize;

        if matches!(size, FrameSize::Cif) {
            mode = FrameSize::Cif;
            max_x /= 4;
            max_y /= 4;
            offset_x /= 4;
            offset_y /= 4;
            if max_y > 296 {
                max_y = 296;
            }
        } else if matches!(size, FrameSize::Svga | FrameSize::Vga) {
            mode = FrameSize::Svga;
            max_x /= 2;
            max_y /= 2;
            offset_x /= 2;
            offset_y /= 2;
        } else {
            mode = FrameSize::Uxga;
        }

        max_x /= 4;
        max_y /= 4;
        w /= 4;
        h /= 4;

        let win_regs: &[(u8, u8)] = &[
            (BANK_SEL, BANK_DSP),
            (0x51, (max_x & 0xFF) as u8),    // HSIZE
            (0x52, (max_y & 0xFF) as u8),    // VSIZE
            (0x53, (offset_x & 0xFF) as u8), // XOFFL
            (0x54, (offset_y & 0xFF) as u8), // YOFFL
            (
                0x55,
                (((max_y >> 1) & 0x80)
                    | ((offset_y >> 4) & 0x70)
                    | ((max_x >> 5) & 0x08)
                    | ((offset_x >> 8) & 0x07)) as u8,
            ), // VHYX
            (0x57, ((max_x >> 2) & 0x80) as u8), // TEST
            (0x5A, (w & 0xFF) as u8),        // ZMOW
            (0x5B, (h & 0xFF) as u8),        // ZMOH
            (0x5C, (((h >> 6) & 0x04) | ((w >> 8) & 0x03)) as u8), // ZMHH
            (0, 0),
        ];

        let mut clk_div = 7;
        let mut pclk_div = 8;
        let mut clk_2x = 0;
        let mut pclk_auto = 1;

        if self.pixel_format == PixelFormat::Jpeg {
            clk_2x = 0;
            clk_div = 0;
            pclk_auto = 0;
            if matches!(mode, FrameSize::Uxga) {
                pclk_div = 12;
            }
        } else {
            if matches!(mode, FrameSize::Cif) {
                clk_div = 3;
            }
            if matches!(mode, FrameSize::Uxga) {
                pclk_div = 12;
            }
        }

        let pclk = ((pclk_div & 0x7F) | (pclk_auto << 7)) as u8;
        let clk = ((clk_div & 0x3F) | (clk_2x << 7)) as u8;

        let res_regs = match mode {
            FrameSize::Cif => OV2640_SETTINGS_TO_CIF,
            FrameSize::Svga => OV2640_SETTINGS_TO_SVGA,
            FrameSize::Uxga => OV2640_SETTINGS_TO_UXGA,
            FrameSize::Vga => OV2640_SETTINGS_TO_SVGA,
        };

        self.write_reg(Bank::Dsp, 0x05, 0x01).await?; // R_BYPASS_DSP_BYPAS
        self.write_regs(res_regs).await?;
        self.write_regs(win_regs).await?;

        self.write_reg(Bank::Sensor, 0x11, clk).await?; // CLKRC
        self.write_reg(Bank::Dsp, 0xD3, pclk).await?; // R_DVP_SP
        self.write_reg(Bank::Dsp, 0x05, 0x00).await?; // R_BYPASS_DSP_EN

        delay.delay_ms(10).await;

        let format_copy = self.pixel_format;
        self.set_pixel_format(format_copy, delay).await?;

        Ok(())
    }

    /// Internal helper: Switches active bank efficiently
    async fn set_bank(&mut self, bank: Bank) -> Result<(), Error<I2cErr>> {
        if self.current_bank != Some(bank) {
            self.i2c
                .write(I2C_ADDR, &[BANK_SEL, bank as u8])
                .await
                .map_err(Error::I2c)?;
            self.current_bank = Some(bank);
        }
        Ok(())
    }

    /// Internal helper: Writes an array of `(reg, value)` pairs, handling bank switches.
    /// Halts when encountering `(0, 0)`.
    async fn write_regs(&mut self, regs: &[(u8, u8)]) -> Result<(), Error<I2cErr>> {
        for &(reg, val) in regs {
            if reg == 0 && val == 0 {
                break;
            }
            if reg == BANK_SEL {
                let target = if val == BANK_DSP {
                    Bank::Dsp
                } else {
                    Bank::Sensor
                };
                self.set_bank(target).await?;
            } else {
                self.i2c
                    .write(I2C_ADDR, &[reg, val])
                    .await
                    .map_err(Error::I2c)?;
            }
        }
        Ok(())
    }

    /// Internal helper: Forces a bank switch and writes a specific register.
    async fn write_reg(&mut self, bank: Bank, reg: u8, val: u8) -> Result<(), Error<I2cErr>> {
        self.set_bank(bank).await?;
        self.i2c
            .write(I2C_ADDR, &[reg, val])
            .await
            .map_err(Error::I2c)?;
        Ok(())
    }
}
