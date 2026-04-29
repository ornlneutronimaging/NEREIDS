//! Step 4: Analyze -- solver configuration, fit execution, spectrum/map display.
//!
//! Layout: asymmetric analysis cockpit: square image workbench and run controls
//! on the left, spectrum + resonance tracks in the wide center workspace, and
//! fit results in an on-demand drawer.  The Guided sidebar and app
//! toolbar/status live outside this page; `analyze_step` works inside the
//! remaining central viewport.

use crate::state::{
    AppState, EndfStatus, InputMode, IsotopeEntry, RoiSelection, SolverMethod, SpectrumAxis,
};
use crate::widgets::design::{self, NavAction};
use crate::widgets::image_view::{
    RoiEditorResult, apply_colormap, data_range, show_image_with_roi_editor, show_viridis_image,
    show_viridis_image_with_roi,
};
use egui_plot::{Corner, Line, Plot, PlotPoints, PlotTransform, Points, VLine};
use ndarray::Axis;
use nereids_pipeline::pipeline::{
    BackgroundConfig, InputData, SolverConfig, SpectrumFitResult, UnifiedFitConfig,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

struct TickRow<'a> {
    z: u32,
    a: u32,
    label: String,
    color: egui::Color32,
    data: &'a nereids_endf::resonance::ResonanceData,
}

enum ImageToolStripCenter {
    None,
    Tof { n_bins: usize },
    Isotope { labels: Vec<String> },
}

#[derive(Clone)]
struct DraggedResonanceRuler {
    label: String,
    color: egui::Color32,
    axis_x: f64,
    energy_ev: f64,
}

struct TickCandidate {
    axis_x: f64,
    energy_ev: f64,
    x_pos: f32,
}

/// Draw the Analyze step content.
///
/// Two-column cockpit layout:
/// ```text
/// +-- Image workbench + run controls --+-- Spectrum + isotope tracks -------+
/// | density/preview + colorbar          | measured · c·OB · fit lines       |
/// |  · click pixel / shift+drag ROI     | linked resonance tick strips      |
/// |  · solver/run controls              | shared energy / TOF axis          |
/// +-------------------------------------+-----------------------------------+
/// ```
pub fn analyze_step(ui: &mut egui::Ui, state: &mut AppState) {
    // Auto-prepare pre-normalized data if the user skipped the Normalize step.
    if matches!(
        state.input_mode,
        InputMode::TransmissionTiff | InputMode::Hdf5Histogram | InputMode::Hdf5Event
    ) && state.normalized.is_none()
        && state.sample_data.is_some()
        && state.spectrum_values.is_some()
    {
        if state.open_beam_data.is_some() {
            crate::guided::normalize::normalize_hdf5_with_ob(state);
        } else {
            crate::guided::normalize::prepare_transmission(state);
        }
    }
    ensure_selected_pixel(state);

    let available_width = ui.available_width();
    let spacing_x = ui.spacing().item_spacing.x;

    // `col_height` is the available room above the nav strip. Keep this bounded
    // by the real viewport height; Analyze is intentionally not inside a page
    // ScrollArea so fixed-aspect image sizing can use `available_height()`.
    let col_height = (ui.available_height() - 40.0).max(120.0);
    let (image_width, spectrum_width) = cockpit_column_widths(available_width, spacing_x);

    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(image_width, col_height),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                image_panel(ui, state);
            },
        );

        ui.separator();

        ui.allocate_ui_with_layout(
            egui::vec2(spectrum_width, col_height),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                spectrum_panel(ui, state);
            },
        );
    });

    fit_info_drawer(ui.ctx(), state);
    isotope_track_picker(ui.ctx(), state);

    // -- Navigation --
    let can_continue = state.spatial_result.is_some();
    match design::nav_buttons(
        ui,
        Some("\u{2190} Back"),
        "Results \u{2192}",
        can_continue,
        "Run analysis to continue",
    ) {
        NavAction::Back => state.nav_prev(),
        NavAction::Continue => state.nav_next(),
        NavAction::None => {}
    }
}

fn cockpit_column_widths(available_width: f32, spacing_x: f32) -> (f32, f32) {
    const IMAGE_MIN: f32 = 340.0;
    const IMAGE_MAX: f32 = 720.0;
    const CENTER_MIN: f32 = 520.0;

    // Account for the separator and surrounding horizontal layout spacing.
    let chrome = spacing_x * 4.0;
    let usable = (available_width - chrome).max(120.0);
    let minimum_total = CENTER_MIN + IMAGE_MIN;

    if usable < minimum_total {
        // Below the minimum: split proportionally and accept that both
        // panels will be tight.
        let image = usable * 0.42;
        let center = usable - image;
        return (image, center);
    }

    // Above the minimum: clamp the image to [IMAGE_MIN, IMAGE_MAX] and
    // give the rest to the center.  Because IMAGE_MIN + CENTER_MIN ==
    // minimum_total and image grows at 0.38 × usable (center at 0.62 ×
    // usable), `center >= CENTER_MIN` is guaranteed throughout this
    // branch — no deficit redistribution is needed.
    let image = (usable * 0.38).clamp(IMAGE_MIN, IMAGE_MAX);
    let center = usable - image;
    (image, center)
}

fn ensure_selected_pixel(state: &mut AppState) {
    if state.selected_pixel.is_some() {
        return;
    }

    let Some(ref norm) = state.normalized else {
        return;
    };

    let shape = norm.transmission.shape();
    if shape.len() < 3 || shape[1] == 0 || shape[2] == 0 {
        return;
    }

    state.selected_pixel = Some((shape[1] / 2, shape[2] / 2));
}

// ---- Fit Controls ----

fn fit_controls(ui: &mut egui::Ui, state: &mut AppState, available_height_hint: f32) {
    // -- Solver configuration --
    let method_text = match state.solver_method {
        SolverMethod::LevenbergMarquardt => "Levenberg-Marquardt",
        SolverMethod::PoissonKL => "Poisson KL",
    };
    let draw_method = |ui: &mut egui::Ui, state: &mut AppState| {
        ui.label("Method:");
        egui::ComboBox::from_id_salt("solver_method")
            .selected_text(method_text)
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut state.solver_method,
                    SolverMethod::LevenbergMarquardt,
                    "Levenberg-Marquardt",
                );
                ui.selectable_value(
                    &mut state.solver_method,
                    SolverMethod::PoissonKL,
                    "Poisson KL",
                );
            });
    };
    let draw_max_iter = |ui: &mut egui::Ui, state: &mut AppState| {
        ui.label("Max iter:");
        ui.add(egui::DragValue::new(&mut state.lm_config.max_iter).range(1..=10000));
    };

    if ui.available_width() >= 330.0 {
        ui.horizontal(|ui| {
            draw_method(ui, state);
            ui.separator();
            draw_max_iter(ui, state);
        });
    } else {
        ui.horizontal(|ui| draw_method(ui, state));
        ui.horizontal(|ui| draw_max_iter(ui, state));
    }

    // Advanced solver controls (collapsible)
    let auto_show_advanced = available_height_hint > 300.0;
    let show_advanced = auto_show_advanced || state.show_advanced_solver;
    let gear = if show_advanced {
        "\u{2699} Advanced \u{25b2}"
    } else {
        "\u{2699} Advanced \u{25bc}"
    };
    if ui
        .add(egui::Button::new(egui::RichText::new(gear).small()).frame(false))
        .clicked()
    {
        state.show_advanced_solver = !state.show_advanced_solver;
    }

    if show_advanced {
        // Snapshot solver controls so we can detect a change after the
        // panel runs and invalidate cached fit results.  egui's
        // `selectable_value` mutates state in place with no change
        // callback, so the previous-value-capture pattern (per MEMORY.md
        // "GUI state management lesson") is the cleanest way.
        let prev_kl_background_enabled = state.kl_background_enabled;
        let prev_kl_c_ratio = state.kl_c_ratio;
        let prev_kl_polish = state.kl_enable_polish_override;
        let prev_fit_temperature = state.fit_temperature;
        let prev_fit_energy_scale = state.fit_energy_scale;
        let prev_lm_background_enabled = state.lm_background_enabled;
        let prev_compute_covariance = state.lm_config.compute_covariance;
        let prev_tol_param = state.lm_config.tol_param;
        let prev_lambda_init = state.lm_config.lambda_init;

        let draw_advanced_solver = |ui: &mut egui::Ui, state: &mut AppState| {
            ui.indent("advanced_solver", |ui| {
                // Fit temperature and Fit energy scale are mutually exclusive
                // (pipeline returns a hard error if both are true — see
                // `pipeline.rs` ~L977).  Grey out each when the other is on so
                // the constraint is visible in the UI rather than only at fit
                // time.
                ui.add_enabled(
                    !state.fit_energy_scale,
                    egui::Checkbox::new(
                        &mut state.fit_temperature,
                        "Fit temperature (slow for Spatial Map)",
                    ),
                );
                ui.add_enabled(
                    !state.fit_temperature,
                    egui::Checkbox::new(
                        &mut state.fit_energy_scale,
                        "Fit energy scale (TZERO t\u{2080} + L_scale, SAMMY equivalent)",
                    ),
                )
                .on_hover_text(
                    "Adds the residual time-zero offset t\u{2080} (\u{03BC}s) \
                 and flight-path scale L_scale (dimensionless) as free \
                 parameters during the fit.  Both seed at identity \
                 (t\u{2080} = 0.0, L_scale = 1.0) — the nominal Delay \
                 has already been subtracted when building the energy \
                 grid, and L_scale multiplies the configured Flight \
                 Path.  Optimiser bound: t\u{2080} \u{2208} \u{00B1}10 \
                 \u{03BC}s, L_scale \u{2208} [0.99, 1.01].  Mutually \
                 exclusive with Fit temperature.",
                );
                if matches!(state.solver_method, SolverMethod::LevenbergMarquardt) {
                    ui.checkbox(
                        &mut state.lm_background_enabled,
                        "LM background (SAMMY: Anorm + BackA/B/C)",
                    );
                }
                if matches!(state.solver_method, SolverMethod::PoissonKL) {
                    ui.checkbox(
                        &mut state.kl_background_enabled,
                        "KL background (SAMMY: A\u{2099} + B_A + B_B/\u{221A}E + B_C\u{221A}E)",
                    );
                    ui.horizontal(|ui| {
                        ui.label("c = Q_s / Q_ob:");
                        ui.add(
                            egui::DragValue::new(&mut state.kl_c_ratio)
                                .speed(0.01)
                                .range(1e-4..=100.0),
                        )
                        .on_hover_text(
                            "Proton-charge ratio for the counts-KL solver \
                         (memo 35 §P1.3).  Leave at 1.0 when the \
                         caller has already PC-normalized the flux.",
                        );
                    });
                    // Polish override (memo 38 §6).  For spatial maps the
                    // default (None = auto-disable when n_pixels > 1) is the
                    // right choice.  Exposing an explicit toggle is
                    // research-oriented; wire as a tri-state via ComboBox
                    // so the distinction between "auto" and "forced off" is
                    // visible.
                    ui.horizontal(|ui| {
                        ui.label("Polish:");
                        let selected = match state.kl_enable_polish_override {
                            None => "Auto (on for single / off for spatial)",
                            Some(true) => "On (forced)",
                            Some(false) => "Off (forced)",
                        };
                        egui::ComboBox::from_id_salt("kl_polish_override")
                            .selected_text(selected)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut state.kl_enable_polish_override,
                                    None,
                                    "Auto",
                                );
                                ui.selectable_value(
                                    &mut state.kl_enable_polish_override,
                                    Some(true),
                                    "On (forced)",
                                );
                                ui.selectable_value(
                                    &mut state.kl_enable_polish_override,
                                    Some(false),
                                    "Off (forced)",
                                );
                            });
                    });
                }
                ui.checkbox(
                    &mut state.lm_config.compute_covariance,
                    "Compute covariance (single-pixel/ROI only)",
                );
                ui.horizontal(|ui| {
                    ui.label("Tol (param):");
                    ui.add(
                        egui::DragValue::new(&mut state.lm_config.tol_param)
                            .speed(1e-9)
                            .range(1e-12..=1.0),
                    );
                });
                if state.solver_method == SolverMethod::LevenbergMarquardt {
                    ui.horizontal(|ui| {
                        ui.label("Lambda init:");
                        ui.add(
                            egui::DragValue::new(&mut state.lm_config.lambda_init)
                                .speed(1e-4)
                                .range(1e-10..=1e6),
                        );
                    });
                }
            });
        };

        if auto_show_advanced {
            draw_advanced_solver(ui, state);
        } else {
            egui::ScrollArea::vertical()
                .id_salt("advanced_solver_scroll")
                .max_height(160.0)
                .auto_shrink([false, false])
                .show(ui, |ui| draw_advanced_solver(ui, state));
        }

        // If any solver control changed, downstream fit results no
        // longer reflect the active configuration — invalidate them so
        // the Results panel doesn't show stale densities/D-per-dof.
        // Compare bit-pattern for the f64 to avoid the +0.0 == -0.0
        // edge case, then exact compare for the bool / Option<bool>.
        let solver_changed = state.kl_background_enabled != prev_kl_background_enabled
            || state.kl_c_ratio.to_bits() != prev_kl_c_ratio.to_bits()
            || state.kl_enable_polish_override != prev_kl_polish
            || state.fit_temperature != prev_fit_temperature
            || state.fit_energy_scale != prev_fit_energy_scale
            || state.lm_background_enabled != prev_lm_background_enabled
            || state.lm_config.compute_covariance != prev_compute_covariance
            || state.lm_config.tol_param.to_bits() != prev_tol_param.to_bits()
            || state.lm_config.lambda_init.to_bits() != prev_lambda_init.to_bits();
        if solver_changed {
            clear_analyze_downstream(state);
        }
    }

    if state.is_fitting {
        if let Some(ref fp) = state.fitting_progress {
            let done = fp.done();
            let total = fp.total();
            let frac = fp.fraction();
            crate::widgets::design::progress_mini(ui, frac, &format!("{done}/{total} px"));
        } else {
            ui.spinner();
            ui.label("Fitting...");
        }
    }
}

fn run_action_buttons(ui: &mut egui::Ui, state: &mut AppState) {
    let has_enabled_iso = state
        .isotope_entries
        .iter()
        .any(|e| e.enabled && e.resonance_data.is_some());
    let has_enabled_grp = state
        .isotope_groups
        .iter()
        .any(|g| g.enabled && g.overall_status() == EndfStatus::Loaded);
    let ready = state.normalized.is_some()
        && state.energies.is_some()
        && (has_enabled_iso || has_enabled_grp);

    let can_run = ready && !state.is_fitting;

    ui.horizontal(|ui| {
        if ui
            .add_enabled(can_run, egui::Button::new("●"))
            .on_hover_text("Fit selected pixel")
            .clicked()
        {
            fit_pixel(state);
        }
        if ui
            .add_enabled(can_run, egui::Button::new("▣"))
            .on_hover_text("Fit ROI average")
            .clicked()
        {
            fit_roi(state);
        }
        if ui
            .add_enabled(can_run, egui::Button::new("▦"))
            .on_hover_text("Fit spatial map for all pixels")
            .clicked()
        {
            run_spatial_map(state);
        }
    });
}

fn image_tool_strip(ui: &mut egui::Ui, state: &mut AppState, center: ImageToolStripCenter) {
    ui.horizontal(|ui| {
        let has_sel = state.selected_roi.is_some_and(|i| i < state.rois.len());
        if ui
            .add_enabled(has_sel, egui::Button::new("⌫"))
            .on_hover_text("Delete selected ROI")
            .clicked()
            && let Some(idx) = state.selected_roi
            && idx < state.rois.len()
        {
            state.log_provenance(
                crate::state::ProvenanceEventKind::ConfigChanged,
                format!("ROI #{} deleted", idx + 1),
            );
            state.rois.remove(idx);
            state.selected_roi = match state.selected_roi {
                Some(s) if s == idx => None,
                Some(s) if s > idx => Some(s - 1),
                other => other,
            };
            clear_analyze_downstream(state);
        }

        if ui
            .add_enabled(!state.rois.is_empty(), egui::Button::new("⊘"))
            .on_hover_text("Clear all ROIs")
            .clicked()
        {
            let n = state.rois.len();
            state.log_provenance(
                crate::state::ProvenanceEventKind::ConfigChanged,
                format!("Cleared all {n} ROIs"),
            );
            state.rois.clear();
            state.selected_roi = None;
            clear_analyze_downstream(state);
        }

        ui.separator();

        match center {
            ImageToolStripCenter::None => {}
            ImageToolStripCenter::Tof { n_bins } => {
                let slider_width = (ui.available_width() - 118.0).clamp(180.0, 360.0);
                let slider = egui::Slider::new(
                    &mut state.analyze_tof_slice_index,
                    0..=n_bins.saturating_sub(1),
                )
                .show_value(true);
                ui.add_sized([slider_width, 18.0], slider)
                    .on_hover_text("TOF slice shown in the image");
            }
            ImageToolStripCenter::Isotope { labels } => {
                let n_isotopes = labels.len();
                if n_isotopes > 1 {
                    let idx = state.map_display_isotope.min(n_isotopes - 1);
                    let current = labels.get(idx).map(String::as_str).unwrap_or("?");
                    egui::ComboBox::from_id_salt("isotope_map_select")
                        .selected_text(current)
                        .width(88.0)
                        .show_ui(ui, |ui| {
                            for (i, name) in labels.iter().enumerate() {
                                ui.selectable_value(&mut state.map_display_isotope, i, name);
                            }
                        })
                        .response
                        .on_hover_text("Density map isotope");
                }
            }
        }

        ui.separator();
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            run_action_buttons(ui, state);
        });
    });
}

fn has_fit_details(state: &AppState) -> bool {
    if state.pixel_fit_result.is_some() || state.last_fit_feedback.is_some() {
        return true;
    }

    let Some((y, x)) = state.selected_pixel else {
        return false;
    };
    state.spatial_result.as_ref().is_some_and(|r| {
        y < r.converged_map.shape()[0] && x < r.converged_map.shape()[1] && r.converged_map[[y, x]]
    })
}

fn fit_info_button(ui: &mut egui::Ui, state: &mut AppState) {
    let enabled = has_fit_details(state);
    let response = ui.add_enabled(
        enabled,
        egui::Button::new(egui::RichText::new("i").strong()).small(),
    );
    if response.clicked() {
        state.show_analyze_fit_info = true;
    }
    response.on_hover_text(if enabled {
        "Show fit details"
    } else {
        "Run a fit to enable fit details"
    });
}

fn fit_info_drawer(ctx: &egui::Context, state: &mut AppState) {
    if !has_fit_details(state) {
        state.show_analyze_fit_info = false;
        return;
    }
    if !state.show_analyze_fit_info {
        return;
    }

    let mut open = state.show_analyze_fit_info;
    egui::Window::new("Fit Details")
        .open(&mut open)
        .default_width(360.0)
        .default_height(420.0)
        .resizable(true)
        .collapsible(false)
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-24.0, 96.0))
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("analyze_fit_details_drawer")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    fit_feedback_panel(ui, state);

                    let selected_fit = state
                        .selected_pixel
                        .and_then(|(y, x)| selected_pixel_fit_result_for_overlay(state, y, x));
                    if let Some(ref result) = selected_fit {
                        ui.add_space(8.0);
                        ui.separator();
                        ui.label(egui::RichText::new("Selected Fit").strong());
                        ui.add_space(4.0);
                        fit_results_panel(ui, state, result);
                    }

                    convergence_summary(ui, state);
                });
        });
    state.show_analyze_fit_info = open;
}

/// Floating popover that lets the user hide / show individual isotope tick
/// strips when more isotopes are loaded than fit on-screen at once. Mirrors
/// the iteration order of `collect_all_resonance_data_with_mapping` so the
/// picker rows match the strip block's ordering exactly.
fn isotope_track_picker(ctx: &egui::Context, state: &mut AppState) {
    if !state.show_isotope_track_picker {
        return;
    }

    // Snapshot the list of pickable isotopes BEFORE entering the closure so
    // we don't hold simultaneous shared + mutable borrows on `state`.
    struct PickerRow {
        z: u32,
        a: u32,
        label: String,
        color: egui::Color32,
    }
    let mut entries: Vec<PickerRow> = Vec::new();
    for entry in &state.isotope_entries {
        if entry.enabled && entry.resonance_data.is_some() {
            entries.push(PickerRow {
                z: entry.z,
                a: entry.a,
                label: isotope_track_label(&entry.symbol, entry.a),
                color: design::isotope_dot_color(&entry.symbol),
            });
        }
    }
    for group in &state.isotope_groups {
        if group.enabled && group.overall_status() == EndfStatus::Loaded {
            for member in &group.members {
                if member.resonance_data.is_some() {
                    entries.push(PickerRow {
                        z: group.z,
                        a: member.a,
                        label: isotope_track_label(&member.symbol, member.a),
                        color: design::isotope_dot_color(&member.symbol),
                    });
                }
            }
        }
    }

    if entries.is_empty() {
        state.show_isotope_track_picker = false;
        return;
    }

    let mut open = state.show_isotope_track_picker;
    egui::Window::new("Tick Strips")
        .open(&mut open)
        .default_width(280.0)
        .default_height(320.0)
        .resizable(true)
        .collapsible(false)
        .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-24.0, -120.0))
        .show(ctx, |ui| {
            ui.label(egui::RichText::new("Show diagnostic tick strip for:").small());
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.small_button("Show all").clicked() {
                    state.hidden_isotope_tracks.clear();
                }
                if ui.small_button("Hide all").clicked() {
                    state.hidden_isotope_tracks = entries.iter().map(|r| (r.z, r.a)).collect();
                }
            });
            ui.separator();
            egui::ScrollArea::vertical()
                .id_salt("analyze_track_picker")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for row in &entries {
                        let key = (row.z, row.a);
                        let mut visible = !state.hidden_isotope_tracks.contains(&key);
                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut visible, "").changed() {
                                if visible {
                                    state.hidden_isotope_tracks.remove(&key);
                                } else {
                                    state.hidden_isotope_tracks.insert(key);
                                }
                            }
                            // Colored dot matching the strip's color.
                            let (rect, _) = ui
                                .allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                            ui.painter().circle_filled(rect.center(), 4.0, row.color);
                            ui.label(egui::RichText::new(&row.label).color(row.color));
                        });
                    }
                });
        });
    state.show_isotope_track_picker = open;
}

fn fit_feedback_panel(ui: &mut egui::Ui, state: &AppState) {
    let Some(ref fb) = state.last_fit_feedback else {
        return;
    };

    let (border_color, bg_color) = if fb.success {
        (
            egui::Color32::from_rgb(60, 160, 60),
            egui::Color32::from_rgba_premultiplied(30, 80, 30, 40),
        )
    } else {
        (
            egui::Color32::from_rgb(200, 60, 60),
            egui::Color32::from_rgba_premultiplied(80, 30, 30, 40),
        )
    };
    egui::Frame::default()
        .fill(bg_color)
        .stroke(egui::Stroke::new(1.5, border_color))
        .corner_radius(4.0)
        .inner_margin(6.0)
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(&fb.summary)
                    .color(border_color)
                    .strong(),
            );
            for (symbol, density) in &fb.densities {
                ui.label(format!("{symbol}: {density:.4e} at/barn"));
            }
            if let Some(t) = fb.temperature_k {
                ui.label(format!("T = {t:.1} K"));
            }
        });
}

/// Small convergence map + summary, shown in the controls column after spatial map.
fn convergence_summary(ui: &mut egui::Ui, state: &AppState) {
    let result = match state.spatial_result {
        Some(ref r) => r,
        None => return,
    };

    ui.add_space(12.0);
    ui.label(egui::RichText::new("Convergence").strong());
    let conv_f64 = result.converged_map.mapv(|b| if b { 1.0 } else { 0.0 });
    let _ = show_viridis_image(ui, &conv_f64, "conv_tex");
    ui.label(
        egui::RichText::new(format!(
            "{}/{} converged",
            result.n_converged, result.n_total
        ))
        .size(11.0),
    );
}

// ---- Image Panel (Column 2) ----

/// Shows spatial result density maps if available, otherwise the preview image
/// with interactive ROI editor. Click to select pixel, Shift+drag to draw ROIs.
fn image_panel(ui: &mut egui::Ui, state: &mut AppState) {
    if let Some(ref result) = state.spatial_result {
        // -- Density map display (read-only ROI overlay) --
        let n_isotopes = result.density_maps.len();
        if n_isotopes == 0 {
            ui.label("No density maps available.");
            return;
        }

        let idx = state.map_display_isotope.min(n_isotopes - 1);
        let isotope_labels = result
            .isotope_labels
            .iter()
            .map(|s| compact_isotope_label_from_name(s))
            .collect();

        let (clicked, image_rect) = show_viridis_image_with_roi(
            ui,
            &result.density_maps[idx],
            "density_tex",
            &state.rois,
            if state.selected_roi.is_some() {
                None
            } else {
                state.selected_pixel
            },
        );
        image_color_bar(
            ui,
            &result.density_maps[idx],
            crate::state::Colormap::Viridis,
            image_rect.width(),
        );
        image_tool_strip(
            ui,
            state,
            ImageToolStripCenter::Isotope {
                labels: isotope_labels,
            },
        );
        if let Some((y, x)) = clicked {
            state.selected_pixel = Some((y, x));
            state.selected_roi = None;
            state.pixel_fit_result = None;
            state.residuals_cache = None;
            state.last_fit_feedback = None;
        }
    } else if let Some(ref norm) = state.normalized {
        // -- TOF-sliced preview with interactive ROI editor --
        let n_tof = norm.transmission.shape()[0];
        if n_tof == 0 {
            ui.label("(no data)");
        } else {
            if state.analyze_tof_slice_index >= n_tof {
                state.analyze_tof_slice_index = n_tof - 1;
            }

            // Display source follows the input data: counts when both
            // sample + OB are loaded (TiffPair, Hdf5* with OB), else
            // the normalised transmission ratio.  TransmissionTiff is
            // always transmission (the input itself is a ratio).
            let show_counts = display_as_counts(state);
            let (label, slice) = if show_counts {
                let s = state
                    .sample_data
                    .as_ref()
                    .unwrap()
                    .index_axis(Axis(0), state.analyze_tof_slice_index)
                    .to_owned();
                ("Sample counts (TOF slice):", s)
            } else {
                let s = norm
                    .transmission
                    .index_axis(Axis(0), state.analyze_tof_slice_index)
                    .to_owned();
                ("Transmission (TOF slice):", s)
            };
            ui.label(label);

            let sel_roi = state.selected_roi;
            let sel_px = if sel_roi.is_some() {
                None
            } else {
                state.selected_pixel
            };
            let (editor_result, image_rect) = show_image_with_roi_editor(
                ui,
                &slice,
                "analyze_preview_tex",
                crate::state::Colormap::Viridis,
                &state.rois,
                sel_roi,
                sel_px,
            );
            apply_roi_editor_result(state, editor_result);
            image_color_bar(
                ui,
                &slice,
                crate::state::Colormap::Viridis,
                image_rect.width(),
            );
            image_tool_strip(ui, state, ImageToolStripCenter::Tof { n_bins: n_tof });
        }
    } else if let Some(ref preview) = state.preview_image {
        // -- Raw preview with interactive ROI editor --
        ui.label("Preview (summed counts):");
        let sel_roi = state.selected_roi;
        let sel_px = if sel_roi.is_some() {
            None
        } else {
            state.selected_pixel
        };
        let (editor_result, image_rect) = show_image_with_roi_editor(
            ui,
            preview,
            "preview_tex",
            crate::state::Colormap::Viridis,
            &state.rois,
            sel_roi,
            sel_px,
        );
        image_color_bar(
            ui,
            preview,
            crate::state::Colormap::Viridis,
            image_rect.width(),
        );
        image_tool_strip(ui, state, ImageToolStripCenter::None);
        apply_roi_editor_result(state, editor_result);
    } else {
        ui.label("Load and normalize data to see preview.");
    }

    ui.add_space(8.0);
    ui.separator();
    let fit_controls_height = (ui.available_height() - 4.0).max(120.0);
    egui::ScrollArea::vertical()
        .id_salt("analyze_image_fit_controls")
        .max_height(fit_controls_height)
        .show(ui, |ui| fit_controls(ui, state, fit_controls_height));
}

fn image_color_bar(
    ui: &mut egui::Ui,
    data: &ndarray::Array2<f64>,
    colormap: crate::state::Colormap,
    target_width: f32,
) {
    let width = target_width.min(ui.available_width()).max(96.0).round();
    if !width.is_finite() {
        return;
    }
    let (vmin, vmax) = data_range(data);
    let (response, painter) = ui.allocate_painter(egui::vec2(width, 10.0), egui::Sense::hover());
    let steps = 96;
    for i in 0..steps {
        let t0 = i as f32 / steps as f32;
        let t1 = (i + 1) as f32 / steps as f32;
        let (r, g, b) = apply_colormap(colormap, t0 as f64);
        let x0 = response.rect.left() + t0 * response.rect.width();
        let x1 = response.rect.left() + t1 * response.rect.width();
        painter.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(x0, response.rect.top()),
                egui::pos2(x1, response.rect.bottom()),
            ),
            0.0,
            egui::Color32::from_rgb(r, g, b),
        );
    }
    painter.rect_stroke(
        response.rect,
        0.0,
        egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color),
        egui::StrokeKind::Inside,
    );
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(format_value(vmin)).small().weak());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(format_value(vmax)).small().weak());
        });
    });
}

fn format_value(value: f64) -> String {
    if !value.is_finite() {
        "NaN".to_string()
    } else if value.abs() >= 1.0e4 || (value != 0.0 && value.abs() < 1.0e-3) {
        format!("{value:.3e}")
    } else {
        format!("{value:.4}")
    }
}

/// Apply the result of the ROI editor interaction to the state.
fn apply_roi_editor_result(state: &mut AppState, result: RoiEditorResult) {
    match result {
        RoiEditorResult::DrawnNew(roi) => {
            state.rois.push(roi);
            state.selected_roi = Some(state.rois.len() - 1);
            clear_analyze_downstream(state);
            state.log_provenance(
                crate::state::ProvenanceEventKind::ConfigChanged,
                format!(
                    "ROI #{} drawn: y=[{}, {}] x=[{}, {}]",
                    state.rois.len(),
                    roi.y_start,
                    roi.y_end,
                    roi.x_start,
                    roi.x_end,
                ),
            );
        }
        RoiEditorResult::Moved { index, new_roi } => {
            if index < state.rois.len() {
                state.rois[index] = new_roi;
                clear_analyze_downstream(state);
                state.selected_roi = Some(index);
                state.log_provenance(
                    crate::state::ProvenanceEventKind::ConfigChanged,
                    format!(
                        "ROI #{} moved: y=[{}, {}] x=[{}, {}]",
                        index + 1,
                        new_roi.y_start,
                        new_roi.y_end,
                        new_roi.x_start,
                        new_roi.x_end,
                    ),
                );
            }
        }
        RoiEditorResult::Moving { index, new_roi } => {
            if index < state.rois.len() {
                state.rois[index] = new_roi;
                state.selected_roi = Some(index);
                state.pixel_fit_result = None;
                state.residuals_cache = None;
                state.last_fit_feedback = None;
                state.show_analyze_fit_info = false;
            }
        }
        RoiEditorResult::Selected(idx) => {
            state.selected_roi = Some(idx);
            state.pixel_fit_result = None;
            state.residuals_cache = None;
            state.last_fit_feedback = None;
            state.show_analyze_fit_info = false;
        }
        RoiEditorResult::Deselected => {
            state.selected_roi = None;
        }
        RoiEditorResult::ClickedPixel(y, x) => {
            state.selected_pixel = Some((y, x));
            state.selected_roi = None;
            state.pixel_fit_result = None;
            state.residuals_cache = None;
            state.last_fit_feedback = None;
        }
        RoiEditorResult::None => {}
    }
}

/// Clear downstream fit results when ROI changes.
fn clear_analyze_downstream(state: &mut AppState) {
    state.pixel_fit_result = None;
    state.spatial_result = None;
    state.last_fit_feedback = None;
    state.fitting_rois.clear();
    state.show_analyze_fit_info = false;
}

// ---- Spectrum Panel (Column 3) ----

/// Whether the Analyze panels should display raw sample counts +
/// open-beam reference rather than the normalised transmission ratio.
///
/// The rule is **input-data-driven**, not solver-driven: KL works with
/// either domain.  Counts mode requires both sample + OB to be loaded
/// (so the c·OB reference line and counts-scale fit overlay are
/// computable), and the input mode must not be `TransmissionTiff` (a
/// pre-normalised transmission stack — its raw input *is* a ratio).
fn display_as_counts(state: &AppState) -> bool {
    state.sample_data.is_some()
        && state.open_beam_data.is_some()
        && !matches!(state.input_mode, InputMode::TransmissionTiff)
}

fn selected_pixel_fit_result_for_overlay(
    state: &AppState,
    y: usize,
    x: usize,
) -> Option<SpectrumFitResult> {
    if let Some(result) = state.pixel_fit_result.clone() {
        return Some(result);
    }

    let result = state.spatial_result.as_ref()?;
    let shape = result.converged_map.shape();
    if y >= shape[0] || x >= shape[1] || !result.converged_map[[y, x]] {
        return None;
    }

    let densities: Vec<f64> = result.density_maps.iter().map(|map| map[[y, x]]).collect();
    if densities.is_empty() || densities.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let uncertainties = if result.uncertainty_maps.len() == densities.len() {
        Some(
            result
                .uncertainty_maps
                .iter()
                .map(|map| map[[y, x]])
                .collect(),
        )
    } else {
        None
    };

    let background = result.background_maps.as_ref().map_or([0.0; 3], |maps| {
        [maps[0][[y, x]], maps[1][[y, x]], maps[2][[y, x]]]
    });

    Some(SpectrumFitResult {
        densities,
        uncertainties,
        reduced_chi_squared: result.chi_squared_map[[y, x]],
        converged: true,
        iterations: 0,
        temperature_k: result.temperature_map.as_ref().map(|map| map[[y, x]]),
        temperature_k_unc: result
            .temperature_uncertainty_map
            .as_ref()
            .map(|map| map[[y, x]]),
        anorm: result.anorm_map.as_ref().map_or(1.0, |map| map[[y, x]]),
        background,
        back_d: 0.0,
        back_f: 0.0,
        t0_us: result.t0_us_map.as_ref().map(|map| map[[y, x]]),
        l_scale: result.l_scale_map.as_ref().map(|map| map[[y, x]]),
        deviance_per_dof: result.deviance_per_dof_map.as_ref().map(|map| map[[y, x]]),
    })
}

fn isotope_track_label(symbol: &str, a: u32) -> String {
    let bare_symbol = symbol.split_once('-').map_or(symbol, |(s, _)| s);
    format!("{}{}", superscript_u32(a), bare_symbol)
}

fn compact_isotope_label_from_name(name: &str) -> String {
    if let Some((symbol, mass)) = name.split_once('-')
        && let Ok(a) = mass.parse::<u32>()
    {
        return format!("{}{}", superscript_u32(a), symbol);
    }
    name.to_string()
}

fn superscript_u32(value: u32) -> String {
    value
        .to_string()
        .chars()
        .map(|c| match c {
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            '9' => '⁹',
            _ => c,
        })
        .collect()
}

#[derive(Clone, Copy)]
enum SpectrumSelection {
    Pixel { y: usize, x: usize },
    Roi { index: usize, roi: RoiSelection },
}

fn active_spectrum_selection(state: &AppState) -> Option<SpectrumSelection> {
    if let Some(index) = state.selected_roi
        && let Some(&roi) = state.rois.get(index)
        && roi.y_end > roi.y_start
        && roi.x_end > roi.x_start
    {
        return Some(SpectrumSelection::Roi { index, roi });
    }

    state
        .selected_pixel
        .map(|(y, x)| SpectrumSelection::Pixel { y, x })
}

fn roi_mean_at(data: &ndarray::Array3<f64>, i: usize, roi: RoiSelection) -> f64 {
    let shape = data.shape();
    let y0 = roi.y_start.min(shape[1]);
    let y1 = roi.y_end.min(shape[1]);
    let x0 = roi.x_start.min(shape[2]);
    let x1 = roi.x_end.min(shape[2]);
    if y0 >= y1 || x0 >= x1 {
        return f64::NAN;
    }

    let mut sum = 0.0;
    let mut n = 0usize;
    for yy in y0..y1 {
        for xx in x0..x1 {
            let v = data[[i, yy, xx]];
            if v.is_finite() {
                sum += v;
                n += 1;
            }
        }
    }
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

fn spectrum_panel(ui: &mut egui::Ui, state: &mut AppState) {
    let selection = active_spectrum_selection(state);

    // Active selection selector. Pixel coordinate editing is intentionally
    // disabled while an ROI is selected because the spectrum is ROI-averaged.
    ui.horizontal(|ui| {
        ui.label("Selection:");
        match selection {
            Some(SpectrumSelection::Roi { index, roi }) => {
                ui.label(format!(
                    "ROI {}  y:{}-{} x:{}-{}",
                    index + 1,
                    roi.y_start,
                    roi.y_end.saturating_sub(1),
                    roi.x_start,
                    roi.x_end.saturating_sub(1)
                ));
            }
            Some(SpectrumSelection::Pixel { y, x }) => {
                ui.label(format!("Pixel ({y}, {x})"));
            }
            None => {
                ui.label("(none)");
            }
        }
        if let Some(ref norm) = state.normalized {
            let shape = norm.transmission.shape();
            let height = shape[1];
            let width = shape[2];

            let mut y_val = state.selected_pixel.map_or(0, |(y, _)| y);
            let mut x_val = state.selected_pixel.map_or(0, |(_, x)| x);
            let pixel_mode = !matches!(selection, Some(SpectrumSelection::Roi { .. }));

            let y_changed = ui
                .add_enabled(
                    pixel_mode,
                    egui::DragValue::new(&mut y_val)
                        .prefix("y: ")
                        .range(0..=height.saturating_sub(1)),
                )
                .changed();
            let x_changed = ui
                .add_enabled(
                    pixel_mode,
                    egui::DragValue::new(&mut x_val)
                        .prefix("x: ")
                        .range(0..=width.saturating_sub(1)),
                )
                .changed();

            if y_changed || x_changed {
                state.selected_pixel = Some((y_val, x_val));
                state.selected_roi = None;
                state.pixel_fit_result = None;
                state.residuals_cache = None;
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            fit_info_button(ui, state);
        });
    });

    // Axis toggle + resonance dips
    ui.horizontal(|ui| {
        ui.label("Axis:");
        ui.selectable_value(
            &mut state.analyze_spectrum_axis,
            SpectrumAxis::EnergyEv,
            "Energy (eV)",
        );
        ui.selectable_value(
            &mut state.analyze_spectrum_axis,
            SpectrumAxis::TofMicroseconds,
            "TOF (\u{03bc}s)",
        );
        ui.separator();
        ui.checkbox(&mut state.show_resonance_dips, "Resonance tracks");
    });

    let norm = match state.normalized {
        Some(ref n) => n,
        None => {
            ui.label("No normalized data available.");
            return;
        }
    };

    let n_tof = norm.transmission.shape()[0];

    // Build x-axis values
    let Some((x_values, x_label)) = design::build_spectrum_x_axis(&design::SpectrumXAxisParams {
        axis: state.analyze_spectrum_axis,
        energies: state.energies.as_deref(),
        spectrum_values: state.spectrum_values.as_ref().map(|v| v.as_slice()),
        spectrum_unit: state.spectrum_unit,
        spectrum_kind: state.spectrum_kind,
        flight_path_m: state.beamline.flight_path_m,
        delay_us: state.beamline.delay_us,
        n_tof,
    }) else {
        ui.label("Load and normalize data to see spectrum.");
        return;
    };

    let selection = match selection {
        Some(selection) => selection,
        None => {
            ui.label("Select a pixel or ROI to view its spectrum.");
            return;
        }
    };

    let shape = norm.transmission.shape();
    if let SpectrumSelection::Pixel { y, x } = selection
        && (y >= shape[1] || x >= shape[2])
    {
        ui.label("Selected pixel is out of bounds. Click the image to select a new pixel.");
        return;
    }

    let n_plot = n_tof.min(x_values.len());
    if n_plot == 0 {
        return;
    }

    // Display source follows the input data type, not the solver
    // (KL works with both transmission and counts).  Counts mode is
    // active iff the user has both sample + OB loaded for a non-
    // TransmissionTiff mode.
    let show_counts = display_as_counts(state);

    // Per-bin multiplier `c · OB[i, y, x]` — used to scale both the
    // fit overlay (T_fit → expected sample counts) and the OB
    // reference line.  Empty when not in counts mode.
    let counts_scale: Vec<f64> = if show_counts {
        let ob = state.open_beam_data.as_ref().unwrap();
        let c = state.kl_c_ratio;
        (0..n_plot)
            .map(|i| match selection {
                SpectrumSelection::Pixel { y, x } => c * ob[[i, y, x]],
                SpectrumSelection::Roi { roi, .. } => c * roi_mean_at(ob, i, roi),
            })
            .collect()
    } else {
        Vec::new()
    };

    // Measured spectrum (skip points where x is non-finite, e.g. from
    // non-positive energies).  In counts mode plot raw sample counts;
    // else plot the normalised transmission ratio.
    let measured_points: PlotPoints = if show_counts {
        let sample = state.sample_data.as_ref().unwrap();
        (0..n_plot)
            .filter(|&i| x_values[i].is_finite())
            .map(|i| {
                let y_value = match selection {
                    SpectrumSelection::Pixel { y, x } => sample[[i, y, x]],
                    SpectrumSelection::Roi { roi, .. } => roi_mean_at(sample, i, roi),
                };
                [x_values[i], y_value]
            })
            .collect()
    } else {
        (0..n_plot)
            .filter(|&i| x_values[i].is_finite())
            .map(|i| {
                let y_value = match selection {
                    SpectrumSelection::Pixel { y, x } => norm.transmission[[i, y, x]],
                    SpectrumSelection::Roi { roi, .. } => roi_mean_at(&norm.transmission, i, roi),
                };
                [x_values[i], y_value]
            })
            .collect()
    };
    let measured_points = Points::new("Measured", measured_points)
        .radius(1.4)
        .color(crate::theme::semantic::RED);

    // OB reference line (counts mode only): the c·OB curve the KL
    // model multiplies T against to predict sample counts.  Showing
    // it makes the data flow into the solver visible.
    let ob_line = if show_counts {
        let pts: PlotPoints = (0..n_plot)
            .filter(|&i| x_values[i].is_finite())
            .map(|i| [x_values[i], counts_scale[i]])
            .collect();
        Some(Line::new("c \u{00B7} OB", pts))
    } else {
        None
    };

    // Fit result (if available).  In counts mode, scale T_fit by
    // `c · OB[i, y, x]` so the overlay sits on the same axis as the
    // raw sample-counts measured line.  This omits the fitted Anorm
    // + LM background polynomial (BackA + BackB/√E + BackC·√E) — both
    // are typically small, and a fully-correct counts overlay needs
    // the joint-Poisson forward model.  Tracked as a follow-up.
    let fit_result_for_overlay = match selection {
        SpectrumSelection::Pixel { y, x } => selected_pixel_fit_result_for_overlay(state, y, x),
        SpectrumSelection::Roi { .. } => state.pixel_fit_result.clone(),
    };
    let fit_line = fit_result_for_overlay.as_ref().and_then(|result| {
        let energies = state.energies.as_ref()?;
        let (all_rd, density_indices, density_ratios) =
            design::collect_all_resonance_data_with_mapping(state);
        let instrument = design::build_resolution_function(
            state.resolution_enabled,
            &state.resolution_mode,
            state.beamline.flight_path_m,
        )
        .ok()
        .flatten()
        .map(|r| Arc::new(nereids_physics::transmission::InstrumentParams { resolution: r }));
        design::build_fit_line(&design::FitLineParams {
            result,
            resonance_data: &all_rd,
            density_indices: &density_indices,
            density_ratios: &density_ratios,
            energies,
            temperature_k: state.temperature_k,
            x_values: &x_values,
            n_plot,
            instrument,
            y_multiplier: if show_counts {
                Some(&counts_scale)
            } else {
                None
            },
        })
    });

    // TOF position marker x-value
    let tof_marker_x = x_values
        .get(state.analyze_tof_slice_index.min(n_plot.saturating_sub(1)))
        .copied();

    // Per-isotope resonance tick strips (GSAS-II-style fit-diagnostic panel,
    // issue #510).  Each loaded isotope gets a thin row below the spectrum
    // showing its resolved-resonance positions, color-keyed to match the
    // isotope chips elsewhere in the UI.  Strips share the spectrum's
    // x-axis via egui_plot's `link_axis`, so panning/zooming the spectrum
    // moves all strip ticks in lockstep.  Gated on `show_resonance_dips`
    // (same toggle that controls the on-main-plot dip overlay) so a single
    // user action turns ALL resonance indicators on or off.
    //
    // Iteration order MIRRORS `design::collect_all_resonance_data_with_mapping`:
    // individual isotope_entries first, then enabled-and-fully-loaded group
    // members.  Keeping the same gate (`g.enabled && overall_status() ==
    // Loaded`) ensures the strips show exactly the resonances the fit + dip
    // overlay are using — otherwise users fitting an element group would see
    // dips on the spectrum with no per-isotope row to attribute them to.
    const STRIP_HEIGHT: f32 = 18.0;
    const MAX_VISIBLE_STRIPS: usize = 6;
    let mut strip_rows: Vec<TickRow<'_>> = Vec::new();
    if state.show_resonance_dips {
        for entry in &state.isotope_entries {
            if entry.enabled
                && let Some(data) = entry.resonance_data.as_ref()
            {
                strip_rows.push(TickRow {
                    z: entry.z,
                    a: entry.a,
                    label: isotope_track_label(&entry.symbol, entry.a),
                    color: design::isotope_dot_color(&entry.symbol),
                    data,
                });
            }
        }
        for group in &state.isotope_groups {
            if group.enabled && group.overall_status() == EndfStatus::Loaded {
                for member in &group.members {
                    if let Some(data) = member.resonance_data.as_ref() {
                        strip_rows.push(TickRow {
                            z: group.z,
                            a: member.a,
                            label: isotope_track_label(&member.symbol, member.a),
                            color: design::isotope_dot_color(&member.symbol),
                            data,
                        });
                    }
                }
            }
        }
    }
    // Apply the user's visibility filter (set via the picker popover).  Hidden
    // tracks are still part of the fit — this only suppresses the diagnostic
    // strip so the visible list stays focused when many isotopes are loaded.
    let total_strip_count = strip_rows.len();
    let visible_indices: Vec<usize> = strip_rows
        .iter()
        .enumerate()
        .filter(|(_, r)| !state.hidden_isotope_tracks.contains(&(r.z, r.a)))
        .map(|(i, _)| i)
        .take(MAX_VISIBLE_STRIPS)
        .collect();
    let n_strips = visible_indices.len();
    let overflow_count = total_strip_count - n_strips;
    let footer_height = if total_strip_count > 0
        && (overflow_count > 0 || !state.hidden_isotope_tracks.is_empty())
    {
        STRIP_HEIGHT + 4.0
    } else {
        0.0
    };
    let strips_total_height = if n_strips > 0 || footer_height > 0.0 {
        // +4.0 padding for the separator above the strip block.
        (n_strips as f32) * STRIP_HEIGHT + 4.0 + footer_height
    } else {
        0.0
    };

    // Reserve space below the plot for the tick strips so neither the plot nor
    // the track viewport pushes past the cockpit column. Fit results live in
    // the right inspector, not below the spectrum.
    let plot_height = (ui.available_height() - 20.0 - strips_total_height).max(220.0);

    // Plot
    let y_label = if show_counts {
        "Counts"
    } else {
        "Transmission"
    };
    let plot_response = Plot::new("spectrum_plot")
        .height(plot_height)
        .x_axis_label(x_label)
        .y_axis_label(y_label)
        .legend(egui_plot::Legend::default().position(Corner::RightBottom))
        .link_axis("analyze_xaxis", [true, false])
        .show(ui, |plot_ui| {
            plot_ui.points(measured_points);
            if let Some(ob) = ob_line {
                plot_ui.line(ob);
            }
            if let Some(fit) = fit_line {
                plot_ui.line(fit);
            }

            // Current TOF position marker
            if let Some(xv) = tof_marker_x {
                plot_ui.vline(
                    VLine::new("TOF position", xv)
                        .color(egui::Color32::from_rgb(255, 165, 0))
                        .style(egui_plot::LineStyle::dashed_dense()),
                );
            }
        });

    // Tick strips below the main plot.  Render up to `MAX_VISIBLE_STRIPS`
    // visible (non-hidden) tracks directly into the spectrum column so the
    // strip plot widths match the spectrum's plot frame pixel-for-pixel —
    // no `ScrollArea` chrome to narrow the rightmost ticks.  Overflow and
    // user-hidden tracks are reachable via the footer chip → picker popover.
    let mut dragged_ruler = None;
    let mut open_picker = false;
    if n_strips > 0 {
        ui.add_space(2.0);
        for (slot, &source_index) in visible_indices.iter().enumerate() {
            let row = &strip_rows[source_index];
            if let Some(ruler) =
                draw_aligned_tick_row(ui, slot, row, &plot_response.transform, state)
            {
                dragged_ruler = Some(ruler);
            }
        }
    }
    if total_strip_count > 0 && (overflow_count > 0 || !state.hidden_isotope_tracks.is_empty()) {
        let hidden_count = state.hidden_isotope_tracks.len();
        let label = if overflow_count > 0 && hidden_count > 0 {
            format!("+{overflow_count} more  ({hidden_count} hidden)  [pick…]")
        } else if overflow_count > 0 {
            format!("+{overflow_count} more  [pick…]")
        } else {
            format!("{hidden_count} hidden  [pick…]")
        };
        ui.horizontal(|ui| {
            ui.add_space(8.0);
            if ui
                .add(egui::Button::new(egui::RichText::new(label).small()).frame(false))
                .clicked()
            {
                open_picker = true;
            }
        });
    }
    if let Some(ruler) = dragged_ruler {
        draw_dragged_resonance_ruler(ui, &plot_response.transform, &ruler, state);
    }
    if open_picker {
        state.show_isotope_track_picker = !state.show_isotope_track_picker;
    }
}

fn draw_aligned_tick_row(
    ui: &mut egui::Ui,
    row_index: usize,
    row: &TickRow<'_>,
    transform: &PlotTransform,
    state: &AppState,
) -> Option<DraggedResonanceRuler> {
    const STRIP_HEIGHT: f32 = 18.0;

    let available_width = ui.available_width();
    let (response, painter) = ui.allocate_painter(
        egui::vec2(available_width, STRIP_HEIGHT),
        egui::Sense::click_and_drag(),
    );
    let plot_frame = *transform.frame();
    let track_left = plot_frame.left().max(response.rect.left());
    let track_right = plot_frame.right().min(response.rect.right());
    if track_right <= track_left {
        return None;
    }

    let track_rect = egui::Rect::from_min_max(
        egui::pos2(track_left, response.rect.top() + 2.0),
        egui::pos2(track_right, response.rect.bottom() - 2.0),
    );
    painter.rect_stroke(
        track_rect,
        2.0,
        egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color),
        egui::StrokeKind::Inside,
    );

    let bounds = transform.bounds();
    let x_lo = bounds.min()[0];
    let x_hi = bounds.max()[0];
    if !x_lo.is_finite() || !x_hi.is_finite() || x_lo >= x_hi {
        return None;
    }

    let mut candidates = Vec::new();
    for range in row.data.ranges.iter().filter(|r| r.resolved) {
        for lg in &range.l_groups {
            for res in &lg.resonances {
                let Some(x) = design::resonance_energy_to_axis(
                    res.energy,
                    state.analyze_spectrum_axis,
                    state.spectrum_unit,
                    state.beamline.flight_path_m,
                    state.beamline.delay_us,
                ) else {
                    continue;
                };
                if x < x_lo || x > x_hi {
                    continue;
                }
                let x_pos = transform
                    .position_from_point_x(x)
                    .clamp(track_rect.left(), track_rect.right());
                candidates.push(TickCandidate {
                    axis_x: x,
                    energy_ev: res.energy,
                    x_pos,
                });
                painter.line_segment(
                    [
                        egui::pos2(x_pos, track_rect.top()),
                        egui::pos2(x_pos, track_rect.bottom()),
                    ],
                    egui::Stroke::new(1.0, row.color),
                );
            }
        }
    }

    let label_right = (track_left - 6.0).max(response.rect.left());
    if label_right > response.rect.left() + 18.0 {
        let label_rect = egui::Rect::from_min_max(
            egui::pos2(response.rect.left(), response.rect.top()),
            egui::pos2(label_right, response.rect.bottom()),
        );
        painter.circle_filled(
            egui::pos2(label_rect.left() + 7.0, label_rect.center().y),
            3.0,
            row.color,
        );
        painter.text(
            egui::pos2(label_rect.left() + 14.0, label_rect.center().y),
            egui::Align2::LEFT_CENTER,
            &row.label,
            egui::FontId::proportional(12.0),
            row.color,
        );
    }
    let response = response.on_hover_text(format!(
        "{} resonance track #{}\nDrag a tick upward to compare it against the spectrum",
        row.label,
        row_index + 1
    ));
    dragged_resonance_from_response(ui, response, &candidates, row)
}

fn dragged_resonance_from_response(
    ui: &mut egui::Ui,
    response: egui::Response,
    candidates: &[TickCandidate],
    row: &TickRow<'_>,
) -> Option<DraggedResonanceRuler> {
    let drag_key = egui::Id::new("analyze_dragged_resonance_ruler");

    if response.drag_started()
        && let Some(pointer) = response.interact_pointer_pos()
        && let Some(candidate) = candidates
            .iter()
            .min_by(|a, b| {
                (a.x_pos - pointer.x)
                    .abs()
                    .total_cmp(&(b.x_pos - pointer.x).abs())
            })
            .filter(|candidate| (candidate.x_pos - pointer.x).abs() <= 12.0)
    {
        ui.data_mut(|data| {
            data.insert_temp(
                drag_key,
                DraggedResonanceRuler {
                    label: row.label.clone(),
                    color: row.color,
                    axis_x: candidate.axis_x,
                    energy_ev: candidate.energy_ev,
                },
            );
        });
    }

    if response.drag_stopped() {
        ui.data_mut(|data| data.remove::<DraggedResonanceRuler>(drag_key));
        return None;
    }

    if response.dragged() {
        return ui.data(|data| data.get_temp::<DraggedResonanceRuler>(drag_key));
    }

    None
}

fn draw_dragged_resonance_ruler(
    ui: &egui::Ui,
    transform: &PlotTransform,
    ruler: &DraggedResonanceRuler,
    state: &AppState,
) {
    let frame = *transform.frame();
    let x = transform
        .position_from_point_x(ruler.axis_x)
        .clamp(frame.left(), frame.right());
    let pointer_y = ui
        .input(|i| i.pointer.hover_pos())
        .map_or(frame.top() + 14.0, |pos| {
            pos.y.clamp(frame.top() + 14.0, frame.bottom() - 14.0)
        });
    let label = match state.analyze_spectrum_axis {
        SpectrumAxis::EnergyEv => format!("{}  {:.4} eV", ruler.label, ruler.energy_ev),
        SpectrumAxis::TofMicroseconds => {
            format!(
                "{}  {:.4} \u{03bc}s ({:.4} eV)",
                ruler.label, ruler.axis_x, ruler.energy_ev
            )
        }
    };

    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("analyze_dragged_resonance_ruler_layer"),
    ));
    painter.line_segment(
        [egui::pos2(x, frame.top()), egui::pos2(x, frame.bottom())],
        egui::Stroke::new(2.0, ruler.color),
    );
    painter.text(
        egui::pos2(x + 6.0, pointer_y),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(12.0),
        ruler.color,
    );
}

fn fit_results_panel(ui: &mut egui::Ui, state: &AppState, result: &SpectrumFitResult) {
    ui.horizontal_wrapped(|ui| {
        let (label, color) = if result.converged {
            ("Converged", crate::theme::semantic::GREEN)
        } else {
            ("NOT converged", crate::theme::semantic::RED)
        };
        ui.label(egui::RichText::new(label).color(color).strong());
        // Memo 35 §P1.2: when the joint-Poisson solver populated
        // deviance_per_dof, label as D/dof; else keep chi2_r.
        let gof_label = if result.deviance_per_dof.is_some() {
            "D/dof"
        } else {
            "chi2_r"
        };
        if state.uncertainty_is_estimated {
            ui.label(
                egui::RichText::new(format!(
                    "{} = {:.4} (approx.)",
                    gof_label, result.reduced_chi_squared
                ))
                .color(crate::theme::semantic::ORANGE),
            );
        } else {
            ui.label(format!("{} = {:.4}", gof_label, result.reduced_chi_squared));
        }
        ui.label(format!("iter = {}", result.iterations));
        if let Some(t) = result.temperature_k {
            if !state.uncertainty_is_estimated {
                if let Some(u) = result.temperature_k_unc {
                    ui.label(format!("T = {t:.1} \u{00b1} {u:.1} K"));
                } else {
                    ui.label(format!("T = {t:.1} K"));
                }
            } else {
                ui.label(format!("T = {t:.1} K"));
            }
        }
    });

    ui.add_space(4.0);

    // Display per-entity density results — build labels in the same order
    // as build_fit_config (individuals first, then groups).
    let mut fit_labels: Vec<String> = state
        .isotope_entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .map(|e| e.symbol.clone())
        .collect();
    for g in &state.isotope_groups {
        if g.enabled && g.overall_status() == EndfStatus::Loaded {
            fit_labels.push(g.name.clone());
        }
    }
    egui::Grid::new("analyze_fit_result_grid")
        .num_columns(3)
        .spacing([8.0, 4.0])
        .show(ui, |ui| {
            ui.label(egui::RichText::new("Entity").small().strong());
            ui.label(egui::RichText::new("Density").small().strong());
            ui.label(egui::RichText::new("Unc.").small().strong());
            ui.end_row();

            for i in 0..result.densities.len() {
                let name = fit_labels.get(i).map(|s| s.as_str()).unwrap_or("?");
                ui.label(name);
                ui.label(format!("{:.6e}", result.densities[i]));
                if state.uncertainty_is_estimated {
                    ui.label("approx.");
                } else {
                    let unc_str = result
                        .uncertainties
                        .as_ref()
                        .and_then(|u| u.get(i))
                        .map_or("N/A".to_string(), |u| format!("{:.2e}", u));
                    ui.label(unc_str);
                }
                ui.end_row();
            }
        });
}

// ---- Fit Helpers ----

fn build_fit_config(state: &AppState) -> Result<UnifiedFitConfig, String> {
    let energies = state
        .energies
        .as_ref()
        .ok_or_else(|| "No energy grid loaded".to_string())?
        .clone();

    let enabled: Vec<&IsotopeEntry> = state
        .isotope_entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .collect();

    // Collect enabled groups with all members loaded
    let enabled_groups: Vec<_> = state
        .isotope_groups
        .iter()
        .filter(|g| g.enabled && g.overall_status() == EndfStatus::Loaded)
        .collect();

    if enabled.is_empty() && enabled_groups.is_empty() {
        return Err("No enabled isotopes with resonance data".into());
    }

    let resolution = design::build_resolution_function(
        state.resolution_enabled,
        &state.resolution_mode,
        state.beamline.flight_path_m,
    )
    .map_err(|e| format!("{e} \u{2014} load a resolution file or disable broadening"))?;

    // When groups are present, use with_groups() to build the config.
    // Individual isotopes are wrapped as single-member groups for uniformity.
    let mut config = if !enabled_groups.is_empty() {
        use nereids_core::types::{Isotope, IsotopeGroup};
        use nereids_endf::resonance::ResonanceData;

        let mut group_specs: Vec<(IsotopeGroup, Vec<ResonanceData>)> = Vec::new();
        let mut group_densities: Vec<f64> = Vec::new();

        // Wrap individual isotopes as single-member groups
        for iso in &enabled {
            let isotope = Isotope::new(iso.z, iso.a)
                .map_err(|e| format!("Invalid isotope {}: {e}", iso.symbol))?;
            let group = IsotopeGroup::custom(iso.symbol.clone(), vec![(isotope, 1.0)])
                .map_err(|e| format!("Group wrap error for {}: {e}", iso.symbol))?;
            group_specs.push((group, vec![iso.resonance_data.clone().unwrap()]));
            group_densities.push(iso.initial_density);
        }

        // Add actual groups
        for g in &enabled_groups {
            let members: Vec<(Isotope, f64)> = g
                .members
                .iter()
                .map(|m| Isotope::new(g.z, m.a).map(|iso| (iso, m.ratio)))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Group {} isotope error: {e}", g.name))?;
            let group = IsotopeGroup::custom(g.name.clone(), members)
                .map_err(|e| format!("Group config error for {}: {e}", g.name))?;
            let rd: Vec<ResonanceData> = g
                .members
                .iter()
                .filter_map(|m| m.resonance_data.clone())
                .collect();
            group_specs.push((group, rd));
            group_densities.push(g.initial_density);
        }

        let refs: Vec<(&IsotopeGroup, &[ResonanceData])> = group_specs
            .iter()
            .map(|(g, rd)| (g, rd.as_slice()))
            .collect();

        // Guard: group_specs must have at least one group with at least one member.
        let (first_group, first_rd) = group_specs
            .first()
            .filter(|(_, rd)| !rd.is_empty())
            .ok_or_else(|| {
                "No groups with loaded resonance data — cannot build fit config".to_string()
            })?;

        // Build a base config using the first group's real data, then replace via with_groups.
        // with_groups() replaces everything, so the base values are overwritten immediately,
        // but using real data avoids sentinel values and documents the provenance.
        let base_rd = vec![first_rd[0].clone()];
        let base_names = vec![first_group.name().to_string()];
        let base_densities = vec![group_densities[0]];
        let base = UnifiedFitConfig::new(
            energies,
            base_rd,
            base_names,
            state.temperature_k,
            resolution,
            base_densities,
        )
        .map_err(|e| format!("Config validation error: {e}"))?;

        base.with_groups(&refs, group_densities)
            .map_err(|e| format!("Group config error: {e}"))?
    } else {
        // No groups — use the standard per-isotope path
        let resonance_data: Vec<_> = enabled
            .iter()
            .filter_map(|e| e.resonance_data.clone())
            .collect();
        let isotope_names: Vec<_> = enabled.iter().map(|e| e.symbol.clone()).collect();
        let initial_densities: Vec<_> = enabled.iter().map(|e| e.initial_density).collect();

        UnifiedFitConfig::new(
            energies,
            resonance_data,
            isotope_names,
            state.temperature_k,
            resolution,
            initial_densities,
        )
        .map_err(|e| format!("Config validation error: {e}"))?
    };

    config = config.with_compute_covariance(state.lm_config.compute_covariance);

    let solver = if state.solver_method == SolverMethod::PoissonKL {
        SolverConfig::PoissonKL(nereids_fitting::poisson::PoissonConfig {
            max_iter: state.lm_config.max_iter,
            ..Default::default()
        })
    } else {
        SolverConfig::LevenbergMarquardt(state.lm_config.clone())
    };
    config = config.with_solver(solver);

    if state.fit_temperature {
        config = config.with_fit_temperature(true);
    }

    // Energy-scale calibration: TZERO t₀ (μs) + flight-path L_scale
    // (dimensionless) as free parameters.  Pipeline rejects the
    // combination with fit_temperature; the UI grey-out should
    // prevent that path.
    //
    // **Why `t0_init_us = 0.0`** (not `state.beamline.delay_us`):
    // `tof_edges_to_energy` (`nereids-io::tof`) already subtracts
    // the configured Delay from raw TOF before building the nominal
    // energy grid the model sees.  The fit `t0` parameter therefore
    // represents the *residual* timing offset on top of that
    // already-corrected grid — its physical zero is 0.0 μs, and the
    // optimiser bound is ±10 μs (`pipeline::add_energy_scale_params`).
    // Seeding at the configured Delay would double-subtract and, for
    // VENUS-typical delays > 10 μs, fall outside the bound.
    //
    // `L_scale = 1.0` (identity) seeds the multiplicative correction
    // on top of the nominal `flight_path_m` the grid was built with.
    if state.fit_energy_scale {
        config = config.with_energy_scale(0.0, 1.0, state.beamline.flight_path_m);
    }

    // Background: solver-aware.  Both LM and counts-KL use the SAMMY
    // 4-term wrapper (Anorm + BackA + BackB/√E + BackC·√E).  Post P2.2,
    // the KL path routes through joint-Poisson with this wrapper too.
    let bg_enabled = match state.solver_method {
        SolverMethod::LevenbergMarquardt => state.lm_background_enabled,
        SolverMethod::PoissonKL => state.kl_background_enabled,
    };
    if bg_enabled {
        config = config.with_transmission_background(BackgroundConfig::default());
    }

    // Counts-KL options: proton-charge ratio + polish override.  Both
    // are no-ops for the LM dispatch.
    if matches!(state.solver_method, SolverMethod::PoissonKL) {
        if (state.kl_c_ratio - 1.0).abs() > 1e-12 {
            config =
                config.with_counts_background(nereids_pipeline::pipeline::CountsBackgroundConfig {
                    c: state.kl_c_ratio,
                    ..Default::default()
                });
        }
        if state.kl_enable_polish_override.is_some() {
            config = config.with_counts_enable_polish(state.kl_enable_polish_override);
        }
    }

    Ok(config)
}

fn fit_pixel(state: &mut AppState) {
    state.last_fit_feedback = None;

    let config = match build_fit_config(state) {
        Ok(c) => c,
        Err(e) => {
            state.last_fit_feedback = Some(crate::state::FitFeedback {
                success: false,
                summary: e.clone(),
                densities: vec![],
                temperature_k: None,
            });
            state.status_message = e;
            return;
        }
    };

    let (y, x) = match state.selected_pixel {
        Some(px) => px,
        None => {
            let msg = "No pixel selected".to_string();
            state.last_fit_feedback = Some(crate::state::FitFeedback {
                success: false,
                summary: msg.clone(),
                densities: vec![],
                temperature_k: None,
            });
            state.status_message = msg;
            return;
        }
    };

    let norm = match state.normalized {
        Some(ref n) => n,
        None => return,
    };

    let shape = norm.transmission.shape();
    if y >= shape[1] || x >= shape[2] {
        let msg = "Selected pixel is out of bounds".to_string();
        state.last_fit_feedback = Some(crate::state::FitFeedback {
            success: false,
            summary: msg.clone(),
            densities: vec![],
            temperature_k: None,
        });
        state.status_message = msg;
        return;
    }

    let n_energies = shape[0];
    let t_spectrum: Vec<f64> = (0..n_energies)
        .map(|e| norm.transmission[[e, y, x]])
        .collect();
    let sigma: Vec<f64> = (0..n_energies)
        .map(|e| norm.uncertainty[[e, y, x]].max(1e-10))
        .collect();

    let mut enabled_symbols: Vec<String> = state
        .isotope_entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .map(|e| e.symbol.clone())
        .collect();
    // Append group names in the same order as build_fit_config
    // (individuals first as single-member groups, then actual groups).
    for g in &state.isotope_groups {
        if g.enabled && g.overall_status() == EndfStatus::Loaded {
            enabled_symbols.push(g.name.clone());
        }
    }

    // Use counts-domain input only when the solver is Poisson KL AND both
    // sample and open beam are available. LM always uses Transmission.
    let use_counts = matches!(state.solver_method, SolverMethod::PoissonKL);
    let input = if use_counts
        && let (Some(sample), Some(open_beam)) = (&state.sample_data, &state.open_beam_data)
    {
        let sample_counts: Vec<f64> = (0..n_energies).map(|e| sample[[e, y, x]]).collect();
        let open_beam_counts: Vec<f64> = (0..n_energies).map(|e| open_beam[[e, y, x]]).collect();
        InputData::Counts {
            sample_counts,
            open_beam_counts,
        }
    } else {
        InputData::Transmission {
            transmission: t_spectrum,
            uncertainty: sigma,
        }
    };

    let result = match nereids_pipeline::pipeline::fit_spectrum_typed(&input, &config) {
        Ok(r) => r,
        Err(e) => {
            let msg = format!("Fit error: {e}");
            state.last_fit_feedback = Some(crate::state::FitFeedback {
                success: false,
                summary: msg.clone(),
                densities: vec![],
                temperature_k: None,
            });
            state.status_message = msg;
            return;
        }
    };

    let chi2_suffix = if state.uncertainty_is_estimated {
        " (approx.)"
    } else {
        ""
    };
    let summary = if result.converged {
        format!(
            "Pixel ({},{}) converged, \u{03C7}\u{00B2}\u{1D63} = {:.4}{}",
            y, x, result.reduced_chi_squared, chi2_suffix
        )
    } else {
        format!("Pixel ({},{}) did NOT converge", y, x)
    };

    let densities: Vec<(String, f64)> = enabled_symbols
        .iter()
        .zip(result.densities.iter())
        .map(|(s, &d)| (s.clone(), d))
        .collect();

    let fitted_temp = result.temperature_k;
    state.last_fit_feedback = Some(crate::state::FitFeedback {
        success: result.converged,
        summary: summary.clone(),
        densities,
        temperature_k: fitted_temp,
    });
    state.status_message = summary;
    state.pixel_fit_result = Some(result);
    state.fit_result_gen = state.fit_result_gen.wrapping_add(1);
}

fn fit_roi(state: &mut AppState) {
    state.last_fit_feedback = None;

    let config = match build_fit_config(state) {
        Ok(c) => c,
        Err(e) => {
            state.last_fit_feedback = Some(crate::state::FitFeedback {
                success: false,
                summary: e.clone(),
                densities: vec![],
                temperature_k: None,
            });
            state.status_message = e;
            return;
        }
    };

    let norm = match state.normalized {
        Some(ref n) => n,
        None => return,
    };

    if state.rois.is_empty() {
        let msg = "No ROI defined".to_string();
        state.last_fit_feedback = Some(crate::state::FitFeedback {
            success: false,
            summary: msg.clone(),
            densities: vec![],
            temperature_k: None,
        });
        state.status_message = msg;
        return;
    }

    let shape = norm.transmission.shape();
    let (n_tof, height, width) = (shape[0], shape[1], shape[2]);

    // Build union mask: average transmission across all pixels inside any ROI
    let mut sum_t = vec![0.0f64; n_tof];
    let mut sum_w = vec![0.0f64; n_tof]; // inverse-variance weights
    let mut n_pixels = 0usize;
    let mut pixels: Vec<(usize, usize)> = Vec::new();

    for y in 0..height {
        for x in 0..width {
            if !state.rois.iter().any(|r| r.contains(y, x)) {
                continue;
            }
            n_pixels += 1;
            pixels.push((y, x));
            for t in 0..n_tof {
                let val = norm.transmission[[t, y, x]];
                let sig = norm.uncertainty[[t, y, x]];
                if val.is_finite() && sig.is_finite() && sig > 0.0 {
                    let w = 1.0 / (sig * sig);
                    sum_t[t] += val * w;
                    sum_w[t] += w;
                }
            }
        }
    }

    if n_pixels == 0 {
        let msg = "No valid pixels in ROI".to_string();
        state.last_fit_feedback = Some(crate::state::FitFeedback {
            success: false,
            summary: msg.clone(),
            densities: vec![],
            temperature_k: None,
        });
        state.status_message = msg;
        return;
    }

    // Weighted average transmission and propagated uncertainty
    let avg_t: Vec<f64> = sum_t
        .iter()
        .zip(sum_w.iter())
        .map(|(&s, &w)| if w > 0.0 { s / w } else { 0.0 })
        .collect();
    // Bins with no valid pixels get negligible weight so the fitter ignores them.
    let sigma: Vec<f64> = sum_w
        .iter()
        .map(|&w| if w > 0.0 { 1.0 / w.sqrt() } else { 1.0e30 })
        .collect();

    let mut enabled_symbols: Vec<String> = state
        .isotope_entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .map(|e| e.symbol.clone())
        .collect();
    // Append group names in the same order as build_fit_config
    for g in &state.isotope_groups {
        if g.enabled && g.overall_status() == EndfStatus::Loaded {
            enabled_symbols.push(g.name.clone());
        }
    }

    // Use counts-domain only when the solver is Poisson KL AND both
    // sample and open beam are available. LM always uses Transmission.
    let use_counts = matches!(state.solver_method, SolverMethod::PoissonKL);
    let roi_input = if use_counts
        && let (Some(sample), Some(open_beam)) = (&state.sample_data, &state.open_beam_data)
    {
        let sample_counts: Vec<f64> = (0..shape[0])
            .map(|t| {
                pixels
                    .iter()
                    .map(|&(y, x)| sample[[t, y, x]].max(0.0))
                    .sum::<f64>()
            })
            .collect();
        let open_beam_counts: Vec<f64> = (0..shape[0])
            .map(|t| {
                pixels
                    .iter()
                    .map(|&(y, x)| open_beam[[t, y, x]].max(0.0))
                    .sum::<f64>()
            })
            .collect();
        InputData::Counts {
            sample_counts,
            open_beam_counts,
        }
    } else {
        InputData::Transmission {
            transmission: avg_t,
            uncertainty: sigma,
        }
    };

    let result = match nereids_pipeline::pipeline::fit_spectrum_typed(&roi_input, &config) {
        Ok(r) => r,
        Err(e) => {
            let msg = format!("ROI fit error: {e}");
            state.last_fit_feedback = Some(crate::state::FitFeedback {
                success: false,
                summary: msg.clone(),
                densities: vec![],
                temperature_k: None,
            });
            state.status_message = msg;
            return;
        }
    };

    let chi2_suffix = if state.uncertainty_is_estimated {
        " (approx.)"
    } else {
        ""
    };
    let summary = if result.converged {
        format!(
            "ROI fit ({n_pixels} px) converged, \u{03C7}\u{00B2}\u{1D63} = {:.4}{}",
            result.reduced_chi_squared, chi2_suffix
        )
    } else {
        format!("ROI fit ({n_pixels} px) did NOT converge")
    };

    let densities: Vec<(String, f64)> = enabled_symbols
        .iter()
        .zip(result.densities.iter())
        .map(|(s, d)| (s.clone(), *d))
        .collect();

    let fitted_temp = result.temperature_k;
    state.last_fit_feedback = Some(crate::state::FitFeedback {
        success: result.converged,
        summary: summary.clone(),
        densities,
        temperature_k: fitted_temp,
    });
    state.status_message = summary;
    state.pixel_fit_result = Some(result);
    state.fit_result_gen = state.fit_result_gen.wrapping_add(1);
}

pub fn run_spatial_map(state: &mut AppState) {
    state.last_fit_feedback = None;

    let config = match build_fit_config(state) {
        Ok(c) => c,
        Err(e) => {
            state.status_message = e;
            return;
        }
    };

    let norm = match state.normalized {
        Some(ref n) => Arc::clone(n),
        None => return,
    };

    // Combine dead_pixels with ROI mask: pixels outside all ROIs are treated as dead.
    let shape = norm.transmission.shape();
    let (height, width) = (shape[1], shape[2]);
    let dead_pixels = if !state.rois.is_empty() {
        let mut mask = ndarray::Array2::from_elem((height, width), true);
        for roi in &state.rois {
            for y in roi.y_start..roi.y_end.min(height) {
                for x in roi.x_start..roi.x_end.min(width) {
                    mask[[y, x]] = false;
                }
            }
        }
        // Merge with existing dead_pixels
        if let Some(ref dp) = state.dead_pixels {
            ndarray::Zip::from(&mut mask)
                .and(dp)
                .for_each(|m, &d| *m = *m || d);
        }
        Some(mask)
    } else {
        state.dead_pixels.clone()
    };

    // Snapshot ROIs at fit time for overlay rendering in Results
    state.fitting_rois = state.rois.clone();

    let (tx, rx) = mpsc::channel();
    state.pending_spatial = Some(rx);
    state.is_fitting = true;
    state.status_message = "Running spatial mapping...".into();
    let cancel = Arc::clone(&state.cancel_token);

    // Progress: single FittingProgress struct holds the Arc<AtomicUsize> counter
    // and the pixel total.  Display code reads the atomic directly each frame.
    let n_live = match dead_pixels {
        Some(ref dp) => dp.iter().filter(|&&d| !d).count(),
        None => height * width,
    };
    let (fp, progress) = crate::state::FittingProgress::new(n_live);
    state.fitting_progress = Some(fp);
    let sample_data = state.sample_data.clone();
    let open_beam_data = state.open_beam_data.clone();

    // Clone the egui context so the background thread can poke the GUI
    // event loop directly via ctx.request_repaint().  This sends an
    // OS-level wake signal — far more reliable than timer-based repaints
    // when rayon saturates the CPU.
    let ctx = state.egui_ctx.clone();

    // Build a DEDICATED rayon pool for per-pixel fitting.
    //
    // spatial_map uses par_iter over pixels, and each pixel's forward-model
    // evaluation calls broadened_cross_sections / unbroadened_cross_sections
    // which ALSO use par_iter on the global rayon pool.  If both the outer
    // pixel loop and the inner physics functions share the global pool, all
    // pool threads block on inner par_iter tasks that can only run when outer
    // tasks yield — creating massive contention and effectively deadlocking
    // the pool (especially with heavy temperature fitting).
    //
    // By running the outer pixel loop on a dedicated pool, the inner physics
    // par_iter calls dispatch to the *global* pool instead.  The two pools
    // are independent, so there is no nested contention.
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(1);
    let pool = match rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
    {
        Ok(p) => p,
        Err(e) => {
            state.status_message = format!("Failed to create thread pool: {e}");
            state.is_fitting = false;
            state.fitting_progress = None;
            return;
        }
    };

    std::thread::spawn(move || {
        // Watcher thread: poke the GUI every 100ms so the progress bar
        // repaints.  Uses near-zero CPU (sleep + one syscall per wake).
        let done_flag = Arc::new(AtomicBool::new(false));
        let watcher = {
            let done = Arc::clone(&done_flag);
            let repaint_ctx = ctx.clone();
            std::thread::spawn(move || {
                while !done.load(Ordering::Relaxed) {
                    if let Some(ref c) = repaint_ctx {
                        c.request_repaint();
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                // One final poke so the completion frame renders immediately.
                if let Some(ref c) = repaint_ctx {
                    c.request_repaint();
                }
            })
        };

        // Run spatial_map_typed on the dedicated pool so its par_iter doesn't
        // share the global pool with inner physics par_iter calls.
        // Use counts-domain only when the solver is Poisson KL AND both
        // sample and open beam are available. LM always uses Transmission.
        let use_counts = matches!(config.solver(), SolverConfig::PoissonKL(_));
        let input = if use_counts
            && let (Some(sample), Some(open_beam)) = (&sample_data, &open_beam_data)
        {
            nereids_pipeline::spatial::InputData3D::Counts {
                sample_counts: sample.view(),
                open_beam_counts: open_beam.view(),
            }
        } else {
            nereids_pipeline::spatial::InputData3D::Transmission {
                transmission: norm.transmission.view(),
                uncertainty: norm.uncertainty.view(),
            }
        };
        let result = pool.install(|| {
            nereids_pipeline::spatial::spatial_map_typed(
                &input,
                &config,
                dead_pixels.as_ref(),
                Some(&cancel),
                Some(&progress),
            )
        });

        // Signal watcher to stop, then send result.
        done_flag.store(true, Ordering::Relaxed);
        watcher.join().ok();

        match result {
            Ok(r) => {
                if !cancel.load(Ordering::Relaxed) {
                    let _ = tx.send(Ok(r));
                }
            }
            Err(nereids_pipeline::error::PipelineError::Cancelled) => {}
            Err(e) => {
                let _ = tx.send(Err(format!("{e}")));
            }
        }
    });
}
