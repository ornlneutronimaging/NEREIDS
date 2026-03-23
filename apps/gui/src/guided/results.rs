//! Step 5: Results — compact summary + "Open in Studio" entry point.
//!
//! The Results step shows a quick-look summary (convergence stats, density
//! table, thumbnail map).  Detailed analysis (pixel inspector, export,
//! colormaps, per-tile toolbelt) lives exclusively in Studio mode.

use super::result_widgets;
use crate::state::{AppState, Colormap, StudioDocTab, UiMode};
use crate::widgets::design;
use crate::widgets::image_view::show_colormapped_image;

/// Draw the Results step content.
pub fn results_step(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(ui, "Results", "Analysis summary");

    match state.spatial_result {
        Some(ref r) => {
            let has_temp = r.temperature_map.is_some();
            let needed = r.density_maps.len() + 1 + has_temp as usize;
            if state.tile_display.len() < needed {
                let n = r.density_maps.len();
                state.init_tile_display(n);
            }
        }
        None => {
            ui.label("Run spatial mapping (Analyze step) to see results here.");
            if design::nav_buttons(
                ui,
                Some("\u{2190} Back"),
                "Open in Studio",
                false,
                "Run analysis first",
            ) == design::NavAction::Back
            {
                state.nav_prev();
            }
            return;
        }
    }

    let result = state.spatial_result.as_ref().unwrap();

    // -- Summary Statistics Card --
    result_widgets::summary_card(ui, result, state.uncertainty_is_estimated);
    ui.add_space(4.0);

    // -- Stat Row --
    {
        let conv_pct = if result.n_total > 0 {
            100.0 * result.n_converged as f64 / result.n_total as f64
        } else {
            0.0
        };
        let (chi2_sum, chi2_count) = result
            .chi_squared_map
            .iter()
            .zip(result.converged_map.iter())
            .filter(|(v, c)| **c && v.is_finite())
            .fold((0.0_f64, 0usize), |(s, n), (v, _)| (s + *v, n + 1));
        let mean_chi2 = if chi2_count == 0 {
            0.0
        } else {
            chi2_sum / chi2_count as f64
        };
        let n_iso = result.density_maps.len();
        let pct = format!("{conv_pct:.1}%");
        let chi2 = format!("{mean_chi2:.2}");
        let n_iso_str = format!("{n_iso}");
        design::stat_row(
            ui,
            &[
                (&pct, "Converged"),
                (&chi2, "Mean \u{03C7}\u{00B2}\u{1D63}"),
                (&n_iso_str, "Isotopes"),
            ],
        );
    }
    ui.add_space(12.0);

    // -- Thumbnail: first density map (compact, no toolbelt) --
    if let Some(data) = result.density_maps.first() {
        design::card_with_header(ui, "Density Map Preview", None, |ui| {
            if let Some(sym) = result.isotope_labels.first() {
                ui.label(egui::RichText::new(sym).small());
            }
            let _ = show_colormapped_image(ui, data, "results_thumb", Colormap::Viridis);
        });
    }
    ui.add_space(12.0);

    // -- Navigation: Back + Open in Studio --
    match design::nav_buttons(
        ui,
        Some("\u{2190} Back"),
        "Open in Studio \u{2192}",
        true,
        "",
    ) {
        design::NavAction::Back => state.nav_prev(),
        design::NavAction::Continue => {
            state.studio_doc_tab = StudioDocTab::Analysis;
            state.ui_mode = UiMode::Studio;
        }
        design::NavAction::None => {}
    }
}
