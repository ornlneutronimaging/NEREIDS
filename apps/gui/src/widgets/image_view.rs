//! Shared image viewer widget: viridis-colormapped 2D array display with click-to-select.

use crate::state::RoiSelection;

/// Display a 2D f64 array as a viridis-colormapped image with click-to-select-pixel.
///
/// Normalizes the data range to [0, 1], maps through the viridis colormap,
/// and renders as an egui texture.  Returns `Some((row, col))` if the user
/// clicked on the image this frame, `None` otherwise.
///
/// Uses `ui.allocate_response` with `Sense::click()` so that clicks are
/// properly registered (unlike `ui.image()` which does not enable click
/// sensing by default).
pub fn show_viridis_image(
    ui: &mut egui::Ui,
    data: &ndarray::Array2<f64>,
    tex_id: &str,
) -> Option<(usize, usize)> {
    show_viridis_image_with_roi(ui, data, tex_id, None).0
}

/// Display a viridis-colormapped image with an optional ROI overlay.
///
/// Returns `(clicked_pixel, image_rect)`.  When `roi` is `Some`, a
/// semi-transparent rectangle is drawn over the corresponding region.
pub fn show_viridis_image_with_roi(
    ui: &mut egui::Ui,
    data: &ndarray::Array2<f64>,
    tex_id: &str,
    roi: Option<&RoiSelection>,
) -> (Option<(usize, usize)>, egui::Rect) {
    let (height, width) = (data.shape()[0], data.shape()[1]);
    if width == 0 || height == 0 {
        ui.label("(empty image)");
        return (None, egui::Rect::NOTHING);
    }

    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for &v in data.iter() {
        if v.is_finite() {
            vmin = vmin.min(v);
            vmax = vmax.max(v);
        }
    }

    let range = if (vmax - vmin).abs() < 1e-30 {
        1.0
    } else {
        vmax - vmin
    };

    let mut pixels = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        for x in 0..width {
            let v = data[[y, x]];
            let t = if v.is_finite() {
                ((v - vmin) / range).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (r, g, b) = viridis(t);
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
            pixels.push(255);
        }
    }

    let image = egui::ColorImage::from_rgba_unmultiplied([width, height], &pixels);
    let texture = ui
        .ctx()
        .load_texture(tex_id, image, egui::TextureOptions::NEAREST);

    let available_width = ui.available_width().min(512.0);
    let scale = available_width / width as f32;
    let display_size = egui::Vec2::new(width as f32 * scale, height as f32 * scale);

    // Allocate interactive rect with click sensing (ui.image() does not register clicks).
    let (response, painter) = ui.allocate_painter(display_size, egui::Sense::click());
    let image_rect = response.rect;
    painter.image(
        texture.id(),
        image_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );

    // Draw ROI overlay
    if let Some(roi) = roi {
        draw_roi_overlay(&painter, image_rect, (height, width), roi);
    }

    if response.clicked()
        && let Some(pos) = response.interact_pointer_pos()
    {
        let rel_x = ((pos.x - image_rect.left()) / image_rect.width()).clamp(0.0, 1.0);
        let rel_y = ((pos.y - image_rect.top()) / image_rect.height()).clamp(0.0, 1.0);
        let px_x = (rel_x * width as f32) as usize;
        let px_y = (rel_y * height as f32) as usize;
        return (
            Some((px_y.min(height - 1), px_x.min(width - 1))),
            image_rect,
        );
    }

    (None, image_rect)
}

/// Draw a semi-transparent ROI rectangle overlay on a displayed image.
fn draw_roi_overlay(
    painter: &egui::Painter,
    image_rect: egui::Rect,
    image_dims: (usize, usize),
    roi: &RoiSelection,
) {
    let (height, width) = image_dims;
    if width == 0 || height == 0 {
        return;
    }
    if roi.x_start >= roi.x_end || roi.y_start >= roi.y_end {
        return;
    }

    let x0 = image_rect.left() + (roi.x_start as f32 / width as f32) * image_rect.width();
    let x1 = image_rect.left() + (roi.x_end as f32 / width as f32) * image_rect.width();
    let y0 = image_rect.top() + (roi.y_start as f32 / height as f32) * image_rect.height();
    let y1 = image_rect.top() + (roi.y_end as f32 / height as f32) * image_rect.height();

    let roi_rect = egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1));

    // Semi-transparent fill
    painter.rect_filled(
        roi_rect,
        0.0,
        egui::Color32::from_rgba_unmultiplied(0, 120, 255, 40),
    );
    // Border
    painter.rect_stroke(
        roi_rect,
        0.0,
        egui::Stroke::new(1.5, egui::Color32::from_rgb(0, 120, 255)),
        egui::StrokeKind::Outside,
    );
}

/// 4-segment linear approximation of the viridis colormap.
pub fn viridis(t: f64) -> (u8, u8, u8) {
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (
            68.0 + s * (49.0 - 68.0),
            1.0 + s * (54.0 - 1.0),
            84.0 + s * (149.0 - 84.0),
        )
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (
            49.0 + s * (33.0 - 49.0),
            54.0 + s * (145.0 - 54.0),
            149.0 + s * (140.0 - 149.0),
        )
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (
            33.0 + s * (143.0 - 33.0),
            145.0 + s * (215.0 - 145.0),
            140.0 + s * (68.0 - 140.0),
        )
    } else {
        let s = (t - 0.75) / 0.25;
        (
            143.0 + s * (253.0 - 143.0),
            215.0 + s * (231.0 - 215.0),
            68.0 + s * (37.0 - 68.0),
        )
    };

    (r as u8, g as u8, b as u8)
}
