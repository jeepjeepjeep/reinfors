//! Shared frame-rendering seam for pixel-observation games. Everything here is
//! crate-internal until a second consumer shapes the public surface; no tiny-skia
//! types cross this boundary.

use tiny_skia::{FillRule, Paint, PathBuilder, Pixmap, Transform};

/// A game-state-to-RGB renderer. `dst` is packed HWC u8, `frame_shape` is (h, w, 3).
// render() feeds the debug/PNG tooling (cfg(test) today); the encoder path uses the
// direct-CHW downsample instead.
#[allow(dead_code)]
pub(crate) trait FrameRenderer<S> {
    fn frame_shape(&self) -> (usize, usize, usize);
    fn render(&self, state: &S, dst: &mut [u8]);
}

/// Rasterization target at a supersample factor over the final frame.
pub(crate) struct Raster {
    pub pixmap: Pixmap,
    pub factor: u32,
}

impl Raster {
    pub fn new(final_w: u32, final_h: u32, factor: u32) -> Raster {
        Raster {
            pixmap: Pixmap::new(final_w * factor, final_h * factor)
                .expect("frame dimensions are nonzero"),
            factor,
        }
    }

    pub fn clear(&mut self) {
        self.pixmap.fill(tiny_skia::Color::TRANSPARENT);
    }

    pub fn fill_poly(&mut self, pts: &[[f32; 2]], rgb: [u8; 3]) {
        // Cull paths fully outside the frame before touching the path builder.
        let (mut lo_x, mut lo_y, mut hi_x, mut hi_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in pts {
            lo_x = lo_x.min(p[0]);
            lo_y = lo_y.min(p[1]);
            hi_x = hi_x.max(p[0]);
            hi_y = hi_y.max(p[1]);
        }
        let f = self.factor as f32;
        let (w, h) = (
            self.pixmap.width() as f32 / f,
            self.pixmap.height() as f32 / f,
        );
        if hi_x < 0.0 || hi_y < 0.0 || lo_x > w || lo_y > h {
            return;
        }
        let mut pb = PathBuilder::new();
        pb.move_to(pts[0][0] * f, pts[0][1] * f);
        for p in &pts[1..] {
            pb.line_to(p[0] * f, p[1] * f);
        }
        pb.close();
        let Some(path) = pb.finish() else { return };
        let mut paint = Paint::default();
        paint.set_color_rgba8(rgb[0], rgb[1], rgb[2], 255);
        paint.anti_alias = true;
        self.pixmap.fill_path(
            &path,
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );
    }

    /// Box-filter straight into a flat channel-major f32 row (raw 0-255 values),
    /// skipping the intermediate HWC buffer.
    pub fn downsample_chw_f32(&self, out: &mut Vec<f32>, final_w: usize, final_h: usize) {
        let f = self.factor as usize;
        let src = self.pixmap.data();
        let src_w = final_w * f;
        let area = (f * f) as u32;
        out.clear();
        out.reserve(3 * final_h * final_w);
        for c in 0..3 {
            for y in 0..final_h {
                for x in 0..final_w {
                    let mut acc = 0u32;
                    for sy in 0..f {
                        for sx in 0..f {
                            acc += u32::from(src[((y * f + sy) * src_w + x * f + sx) * 4 + c]);
                        }
                    }
                    out.push((acc / area) as f32);
                }
            }
        }
    }

    /// Box-filter down to the final resolution as packed RGB rows.
    pub fn downsample_into(&self, dst: &mut [u8], final_w: usize, final_h: usize) {
        let f = self.factor as usize;
        let src = self.pixmap.data();
        let src_w = final_w * f;
        let area = (f * f) as u32;
        for y in 0..final_h {
            for x in 0..final_w {
                let mut acc = [0u32; 3];
                for sy in 0..f {
                    for sx in 0..f {
                        let si = ((y * f + sy) * src_w + x * f + sx) * 4;
                        acc[0] += u32::from(src[si]);
                        acc[1] += u32::from(src[si + 1]);
                        acc[2] += u32::from(src[si + 2]);
                    }
                }
                let di = (y * final_w + x) * 3;
                dst[di] = (acc[0] / area) as u8;
                dst[di + 1] = (acc[1] / area) as u8;
                dst[di + 2] = (acc[2] / area) as u8;
            }
        }
    }
}

/// HWC u8 frame to the engine's flat channel-major f32 row (raw 0-255 values).
#[cfg(test)]
pub(crate) fn hwc_to_chw_f32(hwc: &[u8], h: usize, w: usize, out: &mut Vec<f32>) {
    out.clear();
    out.reserve(3 * h * w);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                out.push(f32::from(hwc[(y * w + x) * 3 + c]));
            }
        }
    }
}

/// 5x7 bitmap digits and minus sign, one bit per pixel, row-major from the top.
const GLYPHS: [(char, [u8; 7]); 11] = [
    (
        '0',
        [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
    ),
    (
        '1',
        [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
    ),
    (
        '2',
        [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
    ),
    (
        '3',
        [
            0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110,
        ],
    ),
    (
        '4',
        [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
    ),
    (
        '5',
        [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
    ),
    (
        '6',
        [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        '7',
        [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
    ),
    (
        '8',
        [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        '9',
        [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
    ),
    (
        '-',
        [
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ],
    ),
];

/// Draw `text` with the top-left corner at `(x, y)` in final-frame units.
pub(crate) fn draw_text(r: &mut Raster, text: &str, x: f32, y: f32, scale: f32, rgb: [u8; 3]) {
    let mut cx = x;
    for ch in text.chars() {
        if let Some((_, rows)) = GLYPHS.iter().find(|(g, _)| *g == ch) {
            for (ry, row) in rows.iter().enumerate() {
                for bx in 0..5 {
                    if row & (1 << (4 - bx)) != 0 {
                        let px = cx + bx as f32 * scale;
                        let py = y + ry as f32 * scale;
                        r.fill_poly(
                            &[
                                [px, py],
                                [px + scale, py],
                                [px + scale, py + scale],
                                [px, py + scale],
                            ],
                            rgb,
                        );
                    }
                }
            }
        }
        cx += 6.0 * scale;
    }
}

/// Debug/eyeball export; not part of any observation path.
#[cfg(test)]
pub(crate) fn write_png(path: &std::path::Path, hwc: &[u8], w: u32, h: u32) -> Result<(), String> {
    let mut pixmap = Pixmap::new(w, h).ok_or("zero-sized frame")?;
    let data = pixmap.data_mut();
    for i in 0..(w * h) as usize {
        data[i * 4] = hwc[i * 3];
        data[i * 4 + 1] = hwc[i * 3 + 1];
        data[i * 4 + 2] = hwc[i * 3 + 2];
        data[i * 4 + 3] = 255;
    }
    pixmap.save_png(path).map_err(|e| e.to_string())
}
