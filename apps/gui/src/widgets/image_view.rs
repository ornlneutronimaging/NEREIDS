//! Shared image viewer widget: colormapped 2D array display with click-to-select.
//!
//! Also provides an interactive ROI editor variant (`show_image_with_roi_editor`)
//! that supports drag-to-draw rectangle ROI on the image.

use crate::state::{Colormap, RoiSelection};

/// Transient drag origin, stored in egui per-widget temp memory during ROI drawing.
#[derive(Clone, Copy)]
struct RoiDragOrigin {
    y: usize,
    x: usize,
}

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
    show_colormapped_image(ui, data, tex_id, Colormap::Viridis)
}

/// Display a 2D f64 array with a configurable colormap and click-to-select-pixel.
pub fn show_colormapped_image(
    ui: &mut egui::Ui,
    data: &ndarray::Array2<f64>,
    tex_id: &str,
    colormap: Colormap,
) -> Option<(usize, usize)> {
    show_colormapped_image_with_roi(ui, data, tex_id, colormap, None, None).0
}

/// Display a viridis-colormapped image with ROI overlay and optional pixel marker.
///
/// Returns `(clicked_pixel, image_rect)`.  When `roi` is `Some`, a
/// semi-transparent rectangle is drawn over the corresponding region.
/// When `selected_pixel` is `Some`, an orange crosshair is drawn at that pixel.
pub fn show_viridis_image_with_roi(
    ui: &mut egui::Ui,
    data: &ndarray::Array2<f64>,
    tex_id: &str,
    roi: Option<&RoiSelection>,
    selected_pixel: Option<(usize, usize)>,
) -> (Option<(usize, usize)>, egui::Rect) {
    show_colormapped_image_with_roi(ui, data, tex_id, Colormap::Viridis, roi, selected_pixel)
}

/// Display a colormapped image with ROI overlay and optional pixel marker.
///
/// Returns `(clicked_pixel, image_rect)`.
pub fn show_colormapped_image_with_roi(
    ui: &mut egui::Ui,
    data: &ndarray::Array2<f64>,
    tex_id: &str,
    colormap: Colormap,
    roi: Option<&RoiSelection>,
    selected_pixel: Option<(usize, usize)>,
) -> (Option<(usize, usize)>, egui::Rect) {
    let (height, width) = (data.shape()[0], data.shape()[1]);
    if width == 0 || height == 0 {
        ui.label("(empty image)");
        return (None, egui::Rect::NOTHING);
    }

    let (response, painter, image_rect) =
        prepare_image_painter(ui, data, tex_id, colormap, egui::Sense::click());

    // Draw ROI overlay
    if let Some(roi) = roi {
        draw_roi_overlay(&painter, image_rect, (height, width), roi);
    }

    // Draw selected pixel marker
    if let Some((py, px)) = selected_pixel {
        draw_pixel_marker(&painter, image_rect, (height, width), py, px);
    }

    if response.clicked()
        && let Some(pos) = response.interact_pointer_pos()
    {
        let (py, px) = screen_to_pixel(pos, image_rect, (height, width));
        return (Some((py, px)), image_rect);
    }

    (None, image_rect)
}

/// Display a colormapped image with interactive drag-to-draw ROI editor.
///
/// The user can drag on the image to draw a new ROI rectangle. A rubber-band
/// rectangle (white) is shown during the drag; the committed ROI is drawn in
/// blue. Returns `Some(new_roi)` when the user finishes a drag, `None` otherwise.
pub fn show_image_with_roi_editor(
    ui: &mut egui::Ui,
    data: &ndarray::Array2<f64>,
    tex_id: &str,
    colormap: Colormap,
    current_roi: Option<&RoiSelection>,
) -> (Option<RoiSelection>, egui::Rect) {
    let (height, width) = (data.shape()[0], data.shape()[1]);
    if width == 0 || height == 0 {
        ui.label("(empty image)");
        return (None, egui::Rect::NOTHING);
    }

    let (response, painter, image_rect) =
        prepare_image_painter(ui, data, tex_id, colormap, egui::Sense::click_and_drag());

    let dims = (height, width);
    let drag_id = response.id;
    let mut committed_roi: Option<RoiSelection> = None;
    let mut is_dragging = false;

    // --- Drag state machine ---
    if response.drag_started()
        && let Some(pos) = response.interact_pointer_pos()
    {
        let (y, x) = screen_to_pixel(pos, image_rect, dims);
        ui.data_mut(|d| d.insert_temp(drag_id, RoiDragOrigin { y, x }));
    }

    if response.dragged()
        && let Some(origin) = ui.data(|d| d.get_temp::<RoiDragOrigin>(drag_id))
    {
        is_dragging = true;
        if let Some(pos) = response.interact_pointer_pos() {
            let (cy, cx) = screen_to_pixel(pos, image_rect, dims);
            let draft = make_roi(origin.y, origin.x, cy, cx, dims);
            draw_roi_draft_overlay(&painter, image_rect, dims, &draft);
        }
    }

    if response.drag_stopped()
        && let Some(origin) = ui.data(|d| d.get_temp::<RoiDragOrigin>(drag_id))
    {
        if let Some(pos) = response.interact_pointer_pos() {
            let (cy, cx) = screen_to_pixel(pos, image_rect, dims);
            let roi = make_roi(origin.y, origin.x, cy, cx, dims);
            if roi.y_end > roi.y_start && roi.x_end > roi.x_start {
                committed_roi = Some(roi);
            }
        }
        ui.data_mut(|d| d.remove::<RoiDragOrigin>(drag_id));
    }

    // Draw the committed ROI overlay when not actively dragging
    if !is_dragging && let Some(roi) = current_roi {
        draw_roi_overlay(&painter, image_rect, dims, roi);
    }

    // Crosshair cursor when hovering over the image
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    }

    (committed_roi, image_rect)
}

/// Prepare the colormapped texture and allocate an interactive painter.
///
/// Shared between `show_colormapped_image_with_roi` (click) and
/// `show_image_with_roi_editor` (click + drag).
fn prepare_image_painter(
    ui: &mut egui::Ui,
    data: &ndarray::Array2<f64>,
    tex_id: &str,
    colormap: Colormap,
    sense: egui::Sense,
) -> (egui::Response, egui::Painter, egui::Rect) {
    let (height, width) = (data.shape()[0], data.shape()[1]);
    let (vmin, vmax) = data_range(data);
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
            let (r, g, b) = apply_colormap(colormap, t);
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

    let (response, painter) = ui.allocate_painter(display_size, sense);
    let image_rect = response.rect;
    painter.image(
        texture.id(),
        image_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );

    (response, painter, image_rect)
}

/// Convert a screen position to pixel coordinates, clamped to image bounds.
fn screen_to_pixel(
    pos: egui::Pos2,
    image_rect: egui::Rect,
    (height, width): (usize, usize),
) -> (usize, usize) {
    let rel_x = ((pos.x - image_rect.left()) / image_rect.width()).clamp(0.0, 1.0);
    let rel_y = ((pos.y - image_rect.top()) / image_rect.height()).clamp(0.0, 1.0);
    let px_x = (rel_x * width as f32) as usize;
    let px_y = (rel_y * height as f32) as usize;
    (
        px_y.min(height.saturating_sub(1)),
        px_x.min(width.saturating_sub(1)),
    )
}

/// Build a `RoiSelection` from two corner points, auto-swapping so start <= end.
fn make_roi(
    y0: usize,
    x0: usize,
    y1: usize,
    x1: usize,
    (height, width): (usize, usize),
) -> RoiSelection {
    RoiSelection {
        y_start: y0.min(y1),
        y_end: (y0.max(y1) + 1).min(height),
        x_start: x0.min(x1),
        x_end: (x0.max(x1) + 1).min(width),
    }
}

/// Draw a white semi-transparent rectangle for the in-progress ROI draft.
fn draw_roi_draft_overlay(
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

    // White semi-transparent fill (distinct from blue committed overlay)
    painter.rect_filled(
        roi_rect,
        0.0,
        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 30),
    );
    // White border
    painter.rect_stroke(
        roi_rect,
        0.0,
        egui::Stroke::new(
            1.5,
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 180),
        ),
        egui::StrokeKind::Outside,
    );
}

/// Compute the (min, max) of finite values in a 2D array.
///
/// Returns `(0.0, 0.0)` when no finite values exist (empty or all-NaN).
pub fn data_range(data: &ndarray::Array2<f64>) -> (f64, f64) {
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for &v in data.iter() {
        if v.is_finite() {
            vmin = vmin.min(v);
            vmax = vmax.max(v);
        }
    }
    if vmin > vmax {
        // No finite values found
        (0.0, 0.0)
    } else {
        (vmin, vmax)
    }
}

/// Render a 2D f64 array to RGBA bytes using the specified colormap.
///
/// Returns a Vec of length `width * height * 4` in RGBA order.
pub fn render_to_rgba(data: &ndarray::Array2<f64>, colormap: Colormap) -> Vec<u8> {
    let (height, width) = (data.shape()[0], data.shape()[1]);
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let (vmin, vmax) = data_range(data);
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
            let (r, g, b) = apply_colormap(colormap, t);
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
            pixels.push(255);
        }
    }
    pixels
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

/// Draw an orange crosshair marker at the specified pixel location.
fn draw_pixel_marker(
    painter: &egui::Painter,
    rect: egui::Rect,
    (height, width): (usize, usize),
    py: usize,
    px: usize,
) {
    if py >= height || px >= width || width == 0 || height == 0 {
        return;
    }
    let frac_x = (px as f32 + 0.5) / width as f32;
    let frac_y = (py as f32 + 0.5) / height as f32;
    let center = egui::pos2(
        rect.left() + frac_x * rect.width(),
        rect.top() + frac_y * rect.height(),
    );
    let r = 4.0;
    let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 165, 0));
    painter.circle_stroke(center, r, stroke);
    painter.line_segment(
        [
            egui::pos2(center.x - r * 2.0, center.y),
            egui::pos2(center.x + r * 2.0, center.y),
        ],
        stroke,
    );
    painter.line_segment(
        [
            egui::pos2(center.x, center.y - r * 2.0),
            egui::pos2(center.x, center.y + r * 2.0),
        ],
        stroke,
    );
}

/// Dispatch to the appropriate colormap function.
pub fn apply_colormap(colormap: Colormap, t: f64) -> (u8, u8, u8) {
    match colormap {
        Colormap::Viridis => viridis(t),
        Colormap::Inferno => inferno(t),
        Colormap::Plasma => plasma(t),
        Colormap::Grayscale => grayscale(t),
    }
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

    (
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    )
}

/// 4-segment linear approximation of the inferno colormap.
/// Control points sampled from matplotlib's inferno at t = 0, 0.25, 0.5, 0.75, 1.0.
pub fn inferno(t: f64) -> (u8, u8, u8) {
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (
            0.0 + s * (87.0 - 0.0),
            0.0 + s * (16.0 - 0.0),
            4.0 + s * (110.0 - 4.0),
        )
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (
            87.0 + s * (188.0 - 87.0),
            16.0 + s * (55.0 - 16.0),
            110.0 + s * (84.0 - 110.0),
        )
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (
            188.0 + s * (249.0 - 188.0),
            55.0 + s * (142.0 - 55.0),
            84.0 + s * (9.0 - 84.0),
        )
    } else {
        let s = (t - 0.75) / 0.25;
        (
            249.0 + s * (252.0 - 249.0),
            142.0 + s * (255.0 - 142.0),
            9.0 + s * (164.0 - 9.0),
        )
    };

    (
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    )
}

/// 4-segment linear approximation of the plasma colormap.
/// Control points sampled from matplotlib's plasma at t = 0, 0.25, 0.5, 0.75, 1.0.
pub fn plasma(t: f64) -> (u8, u8, u8) {
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (
            13.0 + s * (126.0 - 13.0),
            8.0 + s * (3.0 - 8.0),
            135.0 + s * (168.0 - 135.0),
        )
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (
            126.0 + s * (203.0 - 126.0),
            3.0 + s * (71.0 - 3.0),
            168.0 + s * (119.0 - 168.0),
        )
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (
            203.0 + s * (248.0 - 203.0),
            71.0 + s * (149.0 - 71.0),
            119.0 + s * (40.0 - 119.0),
        )
    } else {
        let s = (t - 0.75) / 0.25;
        (
            248.0 + s * (240.0 - 248.0),
            149.0 + s * (249.0 - 149.0),
            40.0 + s * (33.0 - 40.0),
        )
    };

    (
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    )
}

/// Linear grayscale colormap (black to white).
pub fn grayscale(t: f64) -> (u8, u8, u8) {
    let v = (t.clamp(0.0, 1.0) * 255.0) as u8;
    (v, v, v)
}
