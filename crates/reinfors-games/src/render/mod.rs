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

    /// Flood the whole frame with an opaque color: identical bytes to path-filling a
    /// covering polygon (full coverage, opaque src-over), without the path machinery.
    pub fn fill_solid(&mut self, rgb: [u8; 3]) {
        self.pixmap
            .fill(tiny_skia::Color::from_rgba8(rgb[0], rgb[1], rgb[2], 255));
    }

    /// Alpha-blit a coverage bitmap (supersample-space coords) in a solid color.
    pub fn blit_alpha(&mut self, x0: i32, y0: i32, w: usize, h: usize, alpha: &[u8], rgb: [u8; 3]) {
        let (pw, ph) = (self.pixmap.width() as i32, self.pixmap.height() as i32);
        let data = self.pixmap.data_mut();
        for gy in 0..h as i32 {
            let py = y0 + gy;
            if py < 0 || py >= ph {
                continue;
            }
            for gx in 0..w as i32 {
                let px = x0 + gx;
                if px < 0 || px >= pw {
                    continue;
                }
                let a = u32::from(alpha[(gy as usize) * w + gx as usize]);
                if a == 0 {
                    continue;
                }
                let i = ((py * pw + px) * 4) as usize;
                // Premultiplied source-over: src premul is rgb*a, and the destination
                // alpha must advance by the same equation or transparent destinations
                // would end up with color exceeding alpha.
                for c in 0..3 {
                    let d = u32::from(data[i + c]);
                    let s = u32::from(rgb[c]);
                    data[i + c] = ((s * a + d * (255 - a) + 127) / 255) as u8;
                }
                let da = u32::from(data[i + 3]);
                data[i + 3] = ((255 * a + da * (255 - a) + 127) / 255) as u8;
            }
        }
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
    #[allow(dead_code)]
    pub fn downsample_chw_f32(&self, out: &mut Vec<f32>, final_w: usize, final_h: usize) {
        out.clear();
        out.resize(3 * final_h * final_w, 0.0);
        self.downsample_chw_f32_into(out, final_w, final_h);
    }

    /// `downsample_chw_f32` into caller-provided storage (the zero-copy arena path).
    pub fn downsample_chw_f32_into(&self, out: &mut [f32], final_w: usize, final_h: usize) {
        let f = self.factor as usize;
        let src = self.pixmap.data();
        let src_w = final_w * f;
        let area = (f * f) as u32;
        let plane = final_h * final_w;
        assert_eq!(out.len(), 3 * plane, "obs row width mismatch");
        let (rp, rest) = out.split_at_mut(plane);
        let (gp, bp) = rest.split_at_mut(plane);
        // Two passes per output row, both integer-exact vs the naive form: a
        // vertical widening sum over the f source rows (byte-linear, autovectorizes)
        // followed by a small horizontal reduction of each f-pixel group.
        let mut col = vec![0u16; src_w * 4];
        for y in 0..final_h {
            col.iter_mut().for_each(|v| *v = 0);
            for sy in 0..f {
                let row = &src[(y * f + sy) * src_w * 4..][..src_w * 4];
                for (a, &b) in col.iter_mut().zip(row) {
                    *a += u16::from(b);
                }
            }
            for x in 0..final_w {
                let group = &col[x * f * 4..][..f * 4];
                let mut a = [0u32; 3];
                for px in group.as_chunks::<4>().0 {
                    a[0] += u32::from(px[0]);
                    a[1] += u32::from(px[1]);
                    a[2] += u32::from(px[2]);
                }
                rp[y * final_w + x] = (a[0] / area) as f32;
                gp[y * final_w + x] = (a[1] / area) as f32;
                bp[y * final_w + x] = (a[2] / area) as f32;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blit_alpha_over_opaque_and_transparent_is_valid_premultiplied() {
        let mut r = Raster::new(4, 4, 1);
        r.fill_solid([0, 0, 0]);
        r.blit_alpha(0, 0, 1, 1, &[128], [255, 255, 255]);
        let d = r.pixmap.data();
        assert_eq!(
            &d[..4],
            &[128, 128, 128, 255],
            "opaque dst: color blends, alpha stays"
        );

        let mut r = Raster::new(4, 4, 1);
        r.clear();
        r.blit_alpha(0, 0, 1, 1, &[128], [255, 255, 255]);
        let d = r.pixmap.data();
        assert_eq!(
            &d[..4],
            &[128, 128, 128, 128],
            "transparent dst: alpha must advance with color (premultiplied validity)"
        );
        assert!(d[0] <= d[3], "premultiplied channel must not exceed alpha");
    }

    #[test]
    fn blit_alpha_clips_without_panicking() {
        let mut r = Raster::new(4, 4, 1);
        r.fill_solid([0, 0, 0]);
        let bitmap = vec![255u8; 8 * 8];
        r.blit_alpha(-6, -6, 8, 8, &bitmap, [255, 255, 255]);
        r.blit_alpha(2, 2, 8, 8, &bitmap, [255, 255, 255]);
        let d = r.pixmap.data();
        assert_eq!(
            &d[..4],
            &[255, 255, 255, 255],
            "top-left corner covered by first blit"
        );
        let last = (4 * 4 - 1) * 4;
        assert_eq!(
            &d[last..last + 4],
            &[255, 255, 255, 255],
            "bottom-right covered by second"
        );
        let mid = (4 + 1) * 4;
        assert_eq!(&d[mid..mid + 4], &[255, 255, 255, 255]);
    }
}
