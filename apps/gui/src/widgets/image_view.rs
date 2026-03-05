//! Shared image viewer widget: colormapped 2D array display with click-to-select.
//!
//! Also provides an interactive ROI editor variant (`show_image_with_roi_editor`)
//! that supports drag-to-draw rectangle ROI on the image.

use crate::state::{Colormap, RoiSelection};

/// Result of an ROI editor interaction this frame.
pub enum RoiEditorResult {
    /// No interaction this frame.
    None,
    /// User finished drawing a new ROI rectangle.
    DrawnNew(RoiSelection),
    /// User finished moving an existing ROI to a new position.
    Moved { index: usize, new_roi: RoiSelection },
    /// User clicked on an existing ROI to select it.
    Selected(usize),
    /// User clicked on empty space, deselecting the current ROI.
    Deselected,
    /// User clicked on empty space with no ROI selected — pixel selection.
    ClickedPixel(usize, usize),
}

/// Transient drag state, stored in egui per-widget temp memory.
#[derive(Clone, Copy)]
enum RoiDragMode {
    /// Drawing a new ROI from an origin pixel.
    DrawNew { origin_y: usize, origin_x: usize },
    /// Moving an existing ROI (index, grab pixel, original ROI).
    MoveExisting {
        index: usize,
        grab_y: usize,
        grab_x: usize,
        orig: RoiSelection,
    },
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
    show_colormapped_image_with_roi(ui, data, tex_id, colormap, &[], None).0
}

/// Display a viridis-colormapped image with ROI overlays and optional pixel marker.
///
/// Returns `(clicked_pixel, image_rect)`.  Each ROI in the slice is drawn
/// as a semi-transparent blue rectangle overlay.
/// When `selected_pixel` is `Some`, an orange crosshair is drawn at that pixel.
pub fn show_viridis_image_with_roi(
    ui: &mut egui::Ui,
    data: &ndarray::Array2<f64>,
    tex_id: &str,
    rois: &[RoiSelection],
    selected_pixel: Option<(usize, usize)>,
) -> (Option<(usize, usize)>, egui::Rect) {
    show_colormapped_image_with_roi(ui, data, tex_id, Colormap::Viridis, rois, selected_pixel)
}

/// Display a colormapped image with ROI overlays and optional pixel marker.
///
/// Returns `(clicked_pixel, image_rect)`.
pub fn show_colormapped_image_with_roi(
    ui: &mut egui::Ui,
    data: &ndarray::Array2<f64>,
    tex_id: &str,
    colormap: Colormap,
    rois: &[RoiSelection],
    selected_pixel: Option<(usize, usize)>,
) -> (Option<(usize, usize)>, egui::Rect) {
    let (height, width) = (data.shape()[0], data.shape()[1]);
    if width == 0 || height == 0 {
        ui.label("(empty image)");
        return (None, egui::Rect::NOTHING);
    }

    let (response, painter, image_rect) =
        prepare_image_painter(ui, data, tex_id, colormap, egui::Sense::click());

    // Draw ROI overlays
    for roi in rois {
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

/// Display a colormapped image with interactive multi-ROI editor.
///
/// Supports:
/// - **Draw new**: drag on empty space to create a new ROI rectangle
/// - **Select**: click on an existing ROI to select it
/// - **Move**: Shift+drag on a selected ROI to reposition it
/// - **Select/Deselect ROI**: Shift+click on/off an ROI
/// - **Select pixel**: plain click (no Shift)
///
/// Returns `(RoiEditorResult, image_rect)`.
pub fn show_image_with_roi_editor(
    ui: &mut egui::Ui,
    data: &ndarray::Array2<f64>,
    tex_id: &str,
    colormap: Colormap,
    rois: &[RoiSelection],
    selected_roi: Option<usize>,
    selected_pixel: Option<(usize, usize)>,
) -> (RoiEditorResult, egui::Rect) {
    let (height, width) = (data.shape()[0], data.shape()[1]);
    if width == 0 || height == 0 {
        ui.label("(empty image)");
        return (RoiEditorResult::None, egui::Rect::NOTHING);
    }

    let (response, painter, image_rect) =
        prepare_image_painter(ui, data, tex_id, colormap, egui::Sense::click_and_drag());

    let dims = (height, width);
    let drag_id = response.id;
    let last_pixel_id = drag_id.with("last_drag_pixel");
    let mut result = RoiEditorResult::None;
    let mut is_dragging = false;

    // Shift key gates ROI operations; plain click selects a pixel.
    let shift_held = ui.input(|i| i.modifiers.shift);

    // --- Drag state machine (only when Shift is held) ---
    if shift_held
        && response.drag_started()
        && let Some(pos) = response.interact_pointer_pos()
    {
        let (py, px) = screen_to_pixel(pos, image_rect, dims);
        // Hit test: if we're on a selected ROI, start moving it
        if let Some(sel_idx) = selected_roi {
            if sel_idx < rois.len() && point_in_roi(py, px, &rois[sel_idx]) {
                ui.data_mut(|d| {
                    d.insert_temp(
                        drag_id,
                        RoiDragMode::MoveExisting {
                            index: sel_idx,
                            grab_y: py,
                            grab_x: px,
                            orig: rois[sel_idx],
                        },
                    )
                });
            } else {
                ui.data_mut(|d| {
                    d.insert_temp(
                        drag_id,
                        RoiDragMode::DrawNew {
                            origin_y: py,
                            origin_x: px,
                        },
                    )
                });
            }
        } else {
            ui.data_mut(|d| {
                d.insert_temp(
                    drag_id,
                    RoiDragMode::DrawNew {
                        origin_y: py,
                        origin_x: px,
                    },
                )
            });
        }
    }

    if response.dragged()
        && let Some(mode) = ui.data(|d| d.get_temp::<RoiDragMode>(drag_id))
    {
        is_dragging = true;
        if let Some(pos) = response.interact_pointer_pos() {
            let (cy, cx) = screen_to_pixel(pos, image_rect, dims);
            // Store last known pixel for fallback if pointer leaves image on release
            ui.data_mut(|d| d.insert_temp(last_pixel_id, (cy, cx)));
            match mode {
                RoiDragMode::DrawNew { origin_y, origin_x } => {
                    let draft = make_roi(origin_y, origin_x, cy, cx, dims);
                    draw_roi_draft_overlay(&painter, image_rect, dims, &draft);
                }
                RoiDragMode::MoveExisting {
                    orig,
                    grab_y,
                    grab_x,
                    ..
                } => {
                    let moved = move_roi(&orig, grab_y, grab_x, cy, cx, dims);
                    draw_roi_draft_overlay(&painter, image_rect, dims, &moved);
                }
            }
        }
    }

    if response.drag_stopped()
        && let Some(mode) = ui.data(|d| d.get_temp::<RoiDragMode>(drag_id))
    {
        // Use interact_pointer_pos, falling back to last stored pixel if pointer left image
        let end_pixel = response
            .interact_pointer_pos()
            .map(|pos| screen_to_pixel(pos, image_rect, dims))
            .or_else(|| ui.data(|d| d.get_temp::<(usize, usize)>(last_pixel_id)));
        if let Some((cy, cx)) = end_pixel {
            match mode {
                RoiDragMode::DrawNew { origin_y, origin_x } => {
                    if cy == origin_y && cx == origin_x {
                        // Shift+click — select or deselect ROI
                        if let Some(hit_idx) = hit_test_rois(cy, cx, rois) {
                            result = RoiEditorResult::Selected(hit_idx);
                        } else {
                            result = RoiEditorResult::Deselected;
                        }
                    } else {
                        let roi = make_roi(origin_y, origin_x, cy, cx, dims);
                        if roi.y_end > roi.y_start && roi.x_end > roi.x_start {
                            result = RoiEditorResult::DrawnNew(roi);
                        }
                    }
                }
                RoiDragMode::MoveExisting {
                    index,
                    orig,
                    grab_y,
                    grab_x,
                } => {
                    let moved = move_roi(&orig, grab_y, grab_x, cy, cx, dims);
                    if moved.y_end > moved.y_start && moved.x_end > moved.x_start {
                        result = RoiEditorResult::Moved {
                            index,
                            new_roi: moved,
                        };
                    }
                }
            }
        }
        ui.data_mut(|d| {
            d.remove::<RoiDragMode>(drag_id);
            d.remove::<(usize, usize)>(last_pixel_id);
        });
    }

    // Shift+click (no drag) → select/deselect ROI
    if shift_held
        && response.clicked()
        && let Some(pos) = response.interact_pointer_pos()
    {
        let (py, px) = screen_to_pixel(pos, image_rect, dims);
        if let Some(hit_idx) = hit_test_rois(py, px, rois) {
            result = RoiEditorResult::Selected(hit_idx);
        } else {
            result = RoiEditorResult::Deselected;
        }
    }

    // Plain click (no Shift) → select pixel
    if !shift_held
        && response.clicked()
        && let Some(pos) = response.interact_pointer_pos()
    {
        let (py, px) = screen_to_pixel(pos, image_rect, dims);
        result = RoiEditorResult::ClickedPixel(py, px);
    }

    // Draw all committed ROI overlays when not actively dragging
    if !is_dragging {
        for (i, roi) in rois.iter().enumerate() {
            if Some(i) == selected_roi {
                draw_roi_selected_overlay(&painter, image_rect, dims, roi);
            } else {
                draw_roi_overlay(&painter, image_rect, dims, roi);
            }
        }
    } else {
        // While dragging, draw non-dragged ROIs normally
        let dragged_idx = ui
            .data(|d| d.get_temp::<RoiDragMode>(drag_id))
            .and_then(|m| match m {
                RoiDragMode::MoveExisting { index, .. } => Some(index),
                _ => Option::None,
            });
        for (i, roi) in rois.iter().enumerate() {
            if Some(i) == dragged_idx {
                continue; // draft is drawn above
            }
            if Some(i) == selected_roi {
                draw_roi_selected_overlay(&painter, image_rect, dims, roi);
            } else {
                draw_roi_overlay(&painter, image_rect, dims, roi);
            }
        }
    }

    // Draw selected pixel marker
    if let Some((py, px)) = selected_pixel {
        draw_pixel_marker(&painter, image_rect, dims, py, px);
    }

    // Cursor: crosshair for ROI mode (Shift held), default for pixel mode
    if response.hovered()
        && let Some(pos) = ui.input(|i| i.pointer.hover_pos())
    {
        if shift_held {
            let (py, px) = screen_to_pixel(pos, image_rect, dims);
            if let Some(sel_idx) = selected_roi {
                if sel_idx < rois.len() && point_in_roi(py, px, &rois[sel_idx]) {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                } else {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
                }
            } else {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
            }
        } else {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
    }

    (result, image_rect)
}

/// Display a density map overlaid on a dimmed preview image.
///
/// Pixels inside any of the `fitting_rois` show the density colormap.
/// Pixels outside show a dimmed grayscale version of the preview.
/// Returns `(clicked_pixel, image_rect)`.
pub fn show_density_overlay(
    ui: &mut egui::Ui,
    preview: &ndarray::Array2<f64>,
    density: &ndarray::Array2<f64>,
    fitting_rois: &[RoiSelection],
    tex_id: &str,
    colormap: Colormap,
    selected_pixel: Option<(usize, usize)>,
) -> (Option<(usize, usize)>, egui::Rect) {
    let (height, width) = (preview.shape()[0], preview.shape()[1]);
    if width == 0 || height == 0 {
        ui.label("(empty image)");
        return (None, egui::Rect::NOTHING);
    }
    if density.shape() != preview.shape() {
        ui.label(format!(
            "(density/preview shape mismatch: {:?} vs {:?})",
            density.shape(),
            preview.shape()
        ));
        return (None, egui::Rect::NOTHING);
    }

    // Compute ranges for preview (grayscale) and density (colormap)
    let (p_min, p_max) = data_range(preview);
    let p_range = if (p_max - p_min).abs() < 1e-30 {
        1.0
    } else {
        p_max - p_min
    };
    let (d_min, d_max) = density_range_in_rois(density, fitting_rois);
    let d_range = if (d_max - d_min).abs() < 1e-30 {
        1.0
    } else {
        d_max - d_min
    };

    let mut pixels = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        for x in 0..width {
            let in_roi = fitting_rois
                .iter()
                .any(|r| y >= r.y_start && y < r.y_end && x >= r.x_start && x < r.x_end);
            if in_roi {
                // Density colormap
                let v = density[[y, x]];
                let t = if v.is_finite() {
                    ((v - d_min) / d_range).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let (r, g, b) = apply_colormap(colormap, t);
                pixels.push(r);
                pixels.push(g);
                pixels.push(b);
                pixels.push(255);
            } else {
                // Dimmed grayscale preview
                let v = preview[[y, x]];
                let t = if v.is_finite() {
                    ((v - p_min) / p_range).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let gray = (t * 80.0 + 30.0).clamp(0.0, 255.0) as u8;
                pixels.push(gray);
                pixels.push(gray);
                pixels.push(gray);
                pixels.push(255);
            }
        }
    }

    let image = egui::ColorImage::from_rgba_unmultiplied([width, height], &pixels);
    let texture = ui
        .ctx()
        .load_texture(tex_id, image, egui::TextureOptions::NEAREST);

    let available_width = ui.available_width();
    let available_height = ui.available_height();
    let scale_w = available_width / width as f32;
    let scale = if available_height > 2000.0 {
        scale_w
    } else {
        let scale_h = available_height / height as f32;
        scale_w.min(scale_h)
    }
    .max(0.5);
    let display_size = egui::Vec2::new(width as f32 * scale, height as f32 * scale);

    let (response, painter) = ui.allocate_painter(display_size, egui::Sense::click());
    // Center within allocation to preserve aspect ratio (see prepare_image_painter).
    let image_rect = egui::Rect::from_center_size(response.rect.center(), display_size);
    painter.image(
        texture.id(),
        image_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );

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

/// Compute (min, max) of finite density values within ROI regions only.
fn density_range_in_rois(density: &ndarray::Array2<f64>, rois: &[RoiSelection]) -> (f64, f64) {
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    let (height, width) = (density.shape()[0], density.shape()[1]);
    for roi in rois {
        for y in roi.y_start..roi.y_end.min(height) {
            for x in roi.x_start..roi.x_end.min(width) {
                let v = density[[y, x]];
                if v.is_finite() {
                    vmin = vmin.min(v);
                    vmax = vmax.max(v);
                }
            }
        }
    }
    if vmin > vmax {
        (0.0, 0.0)
    } else {
        (vmin, vmax)
    }
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

    let available_width = ui.available_width();
    let available_height = ui.available_height();
    let scale_w = available_width / width as f32;
    // In ScrollArea contexts, available_height is infinite — use width-only
    // scaling so tiles don't grow unbounded.
    let scale = if available_height > 2000.0 {
        scale_w
    } else {
        let scale_h = available_height / height as f32;
        scale_w.min(scale_h)
    }
    .max(0.5); // never shrink below 0.5px/texel
    let display_size = egui::Vec2::new(width as f32 * scale, height as f32 * scale);

    let (response, painter) = ui.allocate_painter(display_size, sense);
    // egui column layouts may stretch the allocated rect wider than
    // display_size.  Center the image within the allocation to
    // preserve the data's aspect ratio.
    let image_rect = egui::Rect::from_center_size(response.rect.center(), display_size);
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
    if image_rect.width() <= 0.0 || image_rect.height() <= 0.0 {
        return (0, 0);
    }
    let rel_x = ((pos.x - image_rect.left()) / image_rect.width()).clamp(0.0, 1.0);
    let rel_y = ((pos.y - image_rect.top()) / image_rect.height()).clamp(0.0, 1.0);
    let px_x = (rel_x * width as f32) as usize;
    let px_y = (rel_y * height as f32) as usize;
    (
        px_y.min(height.saturating_sub(1)),
        px_x.min(width.saturating_sub(1)),
    )
}

/// Test whether a pixel coordinate falls inside an ROI rectangle.
fn point_in_roi(py: usize, px: usize, roi: &RoiSelection) -> bool {
    py >= roi.y_start && py < roi.y_end && px >= roi.x_start && px < roi.x_end
}

/// Hit-test a pixel coordinate against a list of ROIs.
/// Returns the index of the first ROI containing the point, or `None`.
fn hit_test_rois(py: usize, px: usize, rois: &[RoiSelection]) -> Option<usize> {
    rois.iter().position(|roi| point_in_roi(py, px, roi))
}

/// Compute a moved version of an ROI, shifting it by the delta between
/// the grab point and the current cursor position, clamped to image bounds.
fn move_roi(
    orig: &RoiSelection,
    grab_y: usize,
    grab_x: usize,
    cur_y: usize,
    cur_x: usize,
    (height, width): (usize, usize),
) -> RoiSelection {
    let roi_h = orig.y_end.saturating_sub(orig.y_start);
    let roi_w = orig.x_end.saturating_sub(orig.x_start);

    // Compute signed delta
    let dy = cur_y as isize - grab_y as isize;
    let dx = cur_x as isize - grab_x as isize;

    // Apply delta to origin, clamp so ROI stays in bounds
    let new_y_start = (orig.y_start as isize + dy)
        .max(0)
        .min((height.saturating_sub(roi_h)) as isize) as usize;
    let new_x_start = (orig.x_start as isize + dx)
        .max(0)
        .min((width.saturating_sub(roi_w)) as isize) as usize;

    RoiSelection {
        y_start: new_y_start,
        y_end: new_y_start + roi_h,
        x_start: new_x_start,
        x_end: new_x_start + roi_w,
    }
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

/// Draw a selected ROI overlay (green border, brighter fill) to distinguish from
/// unselected ROIs.
fn draw_roi_selected_overlay(
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

    // Green semi-transparent fill
    painter.rect_filled(
        roi_rect,
        0.0,
        egui::Color32::from_rgba_unmultiplied(0, 200, 80, 50),
    );
    // Green border (thicker)
    painter.rect_stroke(
        roi_rect,
        0.0,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(0, 200, 80)),
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
