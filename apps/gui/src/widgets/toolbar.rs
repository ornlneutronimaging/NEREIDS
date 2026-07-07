//! Top toolbar: logo, mode toggle, studio tools, progress, home, theme.
//!
//! Prototype: `.top-toolbar { height: 48px; backdrop-filter: blur(20px); }`

use crate::state::{AppState, GuidedStep, ThemePreference, UiMode};
use crate::theme::ThemeColors;
use crate::widgets::design;

/// Render the top toolbar.
pub fn toolbar(ui: &mut egui::Ui, state: &mut AppState) {
    let ctx = ui.ctx().clone();
    let colors = ThemeColors::from_ctx(&ctx);
    egui::Panel::top("toolbar")
        .exact_size(48.0)
        .frame(
            egui::Frame::NONE
                .fill(colors.bg2)
                .inner_margin(egui::Margin::symmetric(12, 6))
                .stroke(egui::Stroke::new(1.0, colors.border)),
        )
        .show_inside(ui, |ui| {
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 10.0;

                // Logo
                let logo = egui::Image::from_bytes(
                    "bytes://nereids-logo.svg",
                    include_bytes!("../../../../nereids-logo.svg"),
                )
                .fit_to_exact_size(egui::vec2(22.0, 22.0));
                ui.add(logo);

                // App name
                ui.label(egui::RichText::new("NEREIDS").strong().size(14.0));

                ui.add_space(20.0);

                // Mode toggle
                ui.selectable_value(&mut state.ui_mode, UiMode::Guided, "Guided");
                ui.selectable_value(&mut state.ui_mode, UiMode::Studio, "Studio");

                // Help menu (log folder + log path) — issue #524
                help_menu(&ctx, ui);

                // Trailing controls right-aligned
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Theme toggle (rightmost) — cycles ☀ → ☽ → A
                    let icon = match state.theme_preference {
                        ThemePreference::Light => "\u{2600}", // ☀
                        ThemePreference::Dark => "\u{263D}",  // ☽
                        ThemePreference::Auto => "A",
                    };
                    if design::btn_icon(ui, icon, false).clicked() {
                        state.theme_preference = match state.theme_preference {
                            ThemePreference::Light => ThemePreference::Dark,
                            ThemePreference::Dark => ThemePreference::Auto,
                            ThemePreference::Auto => ThemePreference::Light,
                        };
                    }

                    // Home button — returns to Landing page
                    if design::btn_primary(ui, "\u{2302} Home").clicked() {
                        state.guided_step = GuidedStep::Landing;
                        state.ui_mode = UiMode::Guided;
                    }

                    // Save button — visible when spatial results exist
                    let has_results = state.spatial_result.is_some();
                    if has_results {
                        if state.is_saving {
                            ui.add_enabled(false, egui::Button::new("\u{1F4BE} Saving..."));
                        } else if design::btn_primary(ui, "\u{1F4BE} Save").clicked() {
                            crate::project::save_project_dialog(state);
                        }
                    }

                    // Open button — always visible
                    if design::btn_primary(ui, "\u{1F4C2} Open").clicked() {
                        crate::project::load_project_dialog(state);
                    }

                    // Progress indicator
                    if state.is_fitting {
                        if let Some(ref fp) = state.fitting_progress {
                            let frac = fp.fraction();
                            let done = fp.done();
                            let total = fp.total();
                            design::progress_mini(
                                ui,
                                frac,
                                &format!("{:.0}% \u{2014} {done}/{total}", frac * 100.0),
                            );
                        } else {
                            design::progress_mini(ui, 0.0, "Fitting...");
                        }
                    } else if state.is_fetching_endf {
                        design::progress_mini(ui, 0.0, "Fetching ENDF...");
                    } else if state.is_saving {
                        design::progress_mini(ui, 0.0, "Saving...");
                    }
                });
            });
        });
}

/// Help dropdown: reveal the log folder or copy the active log file path.
/// Added for issue #524 (file-based logging for user troubleshooting).
fn help_menu(ctx: &egui::Context, ui: &mut egui::Ui) {
    ui.menu_button("Help", |ui| {
        if ui.button("Open log folder").clicked() {
            // Spawn off the GUI thread because `opener::reveal` on Linux
            // does a synchronous D-Bus round-trip to the FileManager1
            // interface, and even `opener::open` shells out — neither
            // should block frame rendering.
            let file = crate::logging::log_file_path();
            let dir = crate::logging::log_dir();
            std::thread::spawn(move || reveal_log_in_file_manager(&file, &dir));
            ui.close();
        }
        if ui.button("Copy log path").clicked() {
            let path = crate::logging::log_file_path().display().to_string();
            ctx.copy_text(path);
            ui.close();
        }
    });
}

/// Reveal `file` in the platform file manager, falling back to opening
/// the parent `dir` if reveal isn't possible. Runs on a detached worker
/// thread to keep the GUI responsive.
fn reveal_log_in_file_manager(file: &std::path::Path, dir: &std::path::Path) {
    if file.exists() {
        match opener::reveal(file) {
            Ok(()) => return,
            Err(err) => {
                tracing::warn!(error = %err, "opener::reveal failed; falling back to open(dir)");
            }
        }
    } else {
        tracing::info!("today's log file not created yet; opening log directory");
    }
    if let Err(open_err) = opener::open(dir) {
        tracing::error!(error = %open_err, "opener::open failed");
    }
}
