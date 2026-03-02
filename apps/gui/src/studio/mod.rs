//! Studio mode — power-user three-pane result explorer.
//!
//! Layout: tile gallery (left) | full-size viewer (center) | inspector/controls (right).
//! Requires spatial mapping results; shows a placeholder otherwise.

use crate::guided::result_widgets;
use crate::state::{AppState, Colormap};
use crate::theme::ThemeColors;
use crate::widgets::image_view::show_colormapped_image;

/// Render the Studio mode content.
pub fn studio_content(ctx: &egui::Context, state: &mut AppState) {
    // Guard: no results — show centered placeholder
    if state.spatial_result.is_none() {
        let colors = ThemeColors::from_ctx(ctx);
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(colors.bg))
            .show(ctx, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new(
                            "No results yet — run spatial mapping in Guided mode first.",
                        )
                        .heading()
                        .color(colors.fg3),
                    );
                });
            });
        return;
    }

    // Ensure tile_display is populated
    if let Some(ref r) = state.spatial_result {
        if state.tile_display.len() < r.density_maps.len() + 1 {
            state.init_tile_display(r.density_maps.len());
        }
    }

    // Left panel: tile gallery
    egui::SidePanel::left("studio_gallery")
        .resizable(true)
        .default_width(160.0)
        .min_width(120.0)
        .show(ctx, |ui| {
            tile_gallery(ui, state);
        });

    // Right panel: inspector + controls
    egui::SidePanel::right("studio_inspector")
        .resizable(true)
        .default_width(280.0)
        .min_width(200.0)
        .show(ctx, |ui| {
            inspector_panel(ui, state);
        });

    // Center: full-size viewer
    egui::CentralPanel::default().show(ctx, |ui| {
        full_viewer(ui, state);
    });
}

// ---------------------------------------------------------------------------
// Left panel: tile gallery
// ---------------------------------------------------------------------------

/// Scrollable vertical gallery of density map thumbnails.
fn tile_gallery(ui: &mut egui::Ui, state: &mut AppState) {
    ui.label(egui::RichText::new("Gallery").strong());
    ui.separator();

    // Extract tile data before mutable borrow
    let (n_density, symbols, density_maps, conv_f64) = match state.spatial_result {
        Some(ref r) => {
            let symbols: Vec<String> = state
                .isotope_entries
                .iter()
                .filter(|e| e.enabled && e.resonance_data.is_some())
                .map(|e| e.symbol.clone())
                .collect();
            let n_density = r.density_maps.len().min(symbols.len());
            let density_maps = r.density_maps.clone();
            let conv = r.converged_map.mapv(|b| if b { 1.0 } else { 0.0 });
            (n_density, symbols, density_maps, conv)
        }
        None => return,
    };

    egui::ScrollArea::vertical().show(ui, |ui| {
        for i in 0..=n_density {
            let is_convergence = i == n_density;
            let label = if is_convergence {
                "Convergence".to_string()
            } else {
                symbols.get(i).cloned().unwrap_or_else(|| format!("Isotope {i}"))
            };
            let data = if is_convergence {
                &conv_f64
            } else {
                &density_maps[i]
            };
            let tex_id = if is_convergence {
                "studio_gallery_conv".to_string()
            } else {
                format!("studio_gallery_{i}")
            };
            let colormap = state
                .tile_display
                .get(i)
                .map_or(Colormap::Viridis, |t| t.colormap);
            let selected = state.studio_selected_tile == i;

            // Highlight selected tile with a colored frame
            let frame = if selected {
                egui::Frame::group(ui.style())
                    .inner_margin(egui::Margin::same(4))
                    .stroke(egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 149, 237)))
            } else {
                egui::Frame::group(ui.style()).inner_margin(egui::Margin::same(4))
            };

            let resp = frame
                .show(ui, |ui| {
                    ui.label(egui::RichText::new(&label).small().strong());
                    let _ = show_colormapped_image(ui, data, &tex_id, colormap);
                })
                .response;

            if resp.clicked() {
                state.studio_selected_tile = i;
            }

            ui.add_space(4.0);
        }
    });
}

// ---------------------------------------------------------------------------
// Center panel: full-size viewer
// ---------------------------------------------------------------------------

/// Full-size viewer for the selected tile with colorbar and toolbelt.
fn full_viewer(ui: &mut egui::Ui, state: &mut AppState) {
    // Extract tile data
    let (n_density, data, label, conv_f64) = match state.spatial_result {
        Some(ref r) => {
            let symbols: Vec<String> = state
                .isotope_entries
                .iter()
                .filter(|e| e.enabled && e.resonance_data.is_some())
                .map(|e| e.symbol.clone())
                .collect();
            let n_density = r.density_maps.len().min(symbols.len());

            // Clamp selected tile to valid range
            let max_tile = n_density; // n_density = convergence index
            let tile_idx = state.studio_selected_tile.min(max_tile);
            if tile_idx != state.studio_selected_tile {
                state.studio_selected_tile = tile_idx;
            }

            let is_convergence = tile_idx == n_density;
            let conv = r.converged_map.mapv(|b| if b { 1.0 } else { 0.0 });

            if is_convergence {
                let label = format!("Convergence ({}/{})", r.n_converged, r.n_total);
                (n_density, conv.clone(), label, conv)
            } else {
                let label = symbols
                    .get(tile_idx)
                    .cloned()
                    .unwrap_or_else(|| format!("Isotope {tile_idx}"));
                let data = r.density_maps[tile_idx].clone();
                (n_density, data, format!("{label} density"), conv)
            }
        }
        None => return,
    };

    let tile_idx = state.studio_selected_tile;
    let colormap = state
        .tile_display
        .get(tile_idx)
        .map_or(Colormap::Viridis, |t| t.colormap);
    let show_bar = state
        .tile_display
        .get(tile_idx)
        .is_some_and(|t| t.show_colorbar);

    ui.label(egui::RichText::new(&label).heading());
    ui.add_space(4.0);

    // Navigation: prev / next buttons
    ui.horizontal(|ui| {
        if ui
            .add_enabled(tile_idx > 0, egui::Button::new("< Prev"))
            .clicked()
        {
            state.studio_selected_tile = tile_idx.saturating_sub(1);
        }
        if ui
            .add_enabled(tile_idx < n_density, egui::Button::new("Next >"))
            .clicked()
        {
            state.studio_selected_tile = tile_idx + 1;
        }
    });
    ui.add_space(4.0);

    // Image with optional colorbar
    egui::ScrollArea::both().show(ui, |ui| {
        if show_bar {
            ui.horizontal(|ui| {
                if let Some((y, x)) =
                    show_colormapped_image(ui, &data, "studio_viewer", colormap)
                {
                    state.selected_pixel = Some((y, x));
                    state.pixel_fit_result = None;
                }
                result_widgets::draw_colorbar(ui, &data, colormap);
            });
        } else if let Some((y, x)) =
            show_colormapped_image(ui, &data, "studio_viewer", colormap)
        {
            state.selected_pixel = Some((y, x));
            state.pixel_fit_result = None;
        }

        // Toolbelt below the image
        // Clone the label to avoid borrow conflict with state
        let short_label: String = if tile_idx == n_density {
            "convergence".to_string()
        } else {
            state
                .isotope_entries
                .iter()
                .filter(|e| e.enabled && e.resonance_data.is_some())
                .nth(tile_idx)
                .map_or_else(|| "unknown".to_string(), |e| e.symbol.clone())
        };
        result_widgets::tile_toolbelt(ui, &data, tile_idx, &short_label, state);
    });

    // Drop the cloned data — make conv_f64 usable if needed
    let _ = conv_f64;
}

// ---------------------------------------------------------------------------
// Right panel: inspector + controls
// ---------------------------------------------------------------------------

/// Scrollable inspector panel with pixel stats, summary, export, and provenance.
fn inspector_panel(ui: &mut egui::Ui, state: &mut AppState) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        // Pixel Inspector
        result_widgets::pixel_inspector(ui, state);
        ui.add_space(12.0);

        // Summary Statistics
        if let Some(ref result) = state.spatial_result {
            result_widgets::summary_card(ui, result, &state.isotope_entries);
        }
        ui.add_space(12.0);

        // Export
        result_widgets::export_panel(ui, state);
        ui.add_space(12.0);

        // Provenance
        result_widgets::provenance_section(ui, state);
    });
}
