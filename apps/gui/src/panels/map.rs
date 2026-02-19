//! Density map visualization panel.
//!
//! Displays 2D density maps as false-color images using egui textures.

use crate::state::AppState;

/// Draw the map visualization panel.
pub fn map_panel(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Density Maps");

    let result = match state.spatial_result {
        Some(ref r) => r,
        None => {
            // Show preview image if available
            if let Some(ref preview) = state.preview_image {
                ui.label("Preview (summed counts):");
                show_grayscale_image(ui, preview, "preview_tex");
            } else {
                ui.label("Run spatial mapping to see density maps.");
            }
            return;
        }
    };

    // Isotope selector
    let n_isotopes = result.density_maps.len();
    if n_isotopes > 1 {
        ui.horizontal(|ui| {
            ui.label("Isotope:");
            for i in 0..n_isotopes {
                let name = state
                    .isotope_entries
                    .iter()
                    .filter(|e| e.enabled && e.resonance_data.is_some())
                    .nth(i)
                    .map(|e| e.symbol.as_str())
                    .unwrap_or("?");
                ui.selectable_value(&mut state.map_display_isotope, i, name);
            }
        });
    }

    let idx = state.map_display_isotope.min(n_isotopes.saturating_sub(1));

    // Show density map
    ui.label("Density (atoms/barn):");
    show_grayscale_image(ui, &result.density_maps[idx], "density_tex");

    ui.add_space(8.0);

    // Show convergence info
    ui.label("Convergence map:");
    let conv_f64 = result.converged_map.mapv(|b| if b { 1.0 } else { 0.0 });
    show_grayscale_image(ui, &conv_f64, "conv_tex");

    ui.label(format!(
        "{}/{} pixels converged",
        result.n_converged, result.n_total
    ));
}

/// Render a 2D f64 array as a grayscale image in egui.
fn show_grayscale_image(ui: &mut egui::Ui, data: &ndarray::Array2<f64>, tex_id: &str) {
    let (height, width) = (data.shape()[0], data.shape()[1]);
    if width == 0 || height == 0 {
        ui.label("(empty image)");
        return;
    }

    // Find min/max for normalization
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

    // Build RGBA pixels with a viridis-like colormap
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

    // Scale to fit in available width, preserving aspect ratio
    let available_width = ui.available_width().min(512.0);
    let scale = available_width / width as f32;
    let display_size = egui::Vec2::new(width as f32 * scale, height as f32 * scale);

    let response = ui.image(egui::load::SizedTexture::new(texture.id(), display_size));

    // Click on image to select pixel
    if response.clicked()
        && let Some(pos) = response.interact_pointer_pos()
    {
        let rect = response.rect;
        let rel_x = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
        let rel_y = ((pos.y - rect.top()) / rect.height()).clamp(0.0, 1.0);
        let px_x = (rel_x * width as f32) as usize;
        let px_y = (rel_y * height as f32) as usize;
        ui.ctx().data_mut(|d| {
            d.insert_temp(
                egui::Id::new("clicked_pixel"),
                (px_y.min(height - 1), px_x.min(width - 1)),
            );
        });
    }
}

/// Simple viridis-like colormap: maps t in [0, 1] to RGB bytes.
fn viridis(t: f64) -> (u8, u8, u8) {
    // Simplified 5-stop viridis approximation
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
