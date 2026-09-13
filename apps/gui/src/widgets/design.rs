//! Design system widgets: cards, underline tabs, badges, buttons, drop zones.
//!
//! Theme-aware widgets resolve colors from the egui context (`ThemeColors`)
//! or from `semantic` constants.
//! Prototype reference: `.prototypes/D_hybrid_v4.html` CSS §Cards & Forms,
//! §Tabs, §Badges, §Content Area, §Toolbar, §Drop Zones.

use crate::state::{
    AppState, EndfFetchResult, EndfStatus, FetchTarget, GuidedStep, ResolutionMode, SpectrumAxis,
};
use crate::theme::{ThemeColors, semantic};
use egui::{Color32, CornerRadius, Margin, Rect, Response, RichText, Sense, Shadow, Stroke, Ui};
use egui_plot::{Line, PlotPoints, VLine};
use nereids_core::types::Isotope;
use nereids_endf::retrieval::EndfLibrary;
use nereids_io::spectrum::{SpectrumUnit, SpectrumValueKind};
use nereids_physics::resolution::{ResolutionFunction, ResolutionParams, TabulatedResolution};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};

/// Semi-transparent red used for resonance energy dip markers on spectrum plots.
/// Values are manually premultiplied: RGBA(180, 80, 80, 50) → (35, 15, 15, 50).
pub const RESONANCE_DIP_COLOR: Color32 = Color32::from_rgba_premultiplied(35, 15, 15, 50);

// ── Content Header ──────────────────────────────────────────────

/// Page-level title (22px bold) + subtitle (13px fg2).
///
/// Prototype: `.content-header h1` + `.content-header p`
pub fn content_header(ui: &mut Ui, title: &str, subtitle: &str) {
    let tc = ThemeColors::from_ctx(ui.ctx());
    ui.label(RichText::new(title).size(22.0).strong());
    if !subtitle.is_empty() {
        ui.label(RichText::new(subtitle).size(13.0).color(tc.fg2));
    }
    ui.add_space(12.0);
}

// ── Card ────────────────────────────────────────────────────────

/// Styled card container: bg2 fill, 10px radius, 16px padding, border, shadow.
/// Adds 14px bottom spacing after the card.
///
/// Prototype: `.card { background: var(--bg2); border-radius: 10px; padding: 16px;
///   box-shadow: var(--card-shadow); border: 1px solid var(--border); }`
pub fn card(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui)) {
    let tc = ThemeColors::from_ctx(ui.ctx());
    let shadow_alpha = if ui.style().visuals.dark_mode { 76 } else { 20 };
    egui::Frame::NONE
        .fill(tc.bg2)
        .stroke(Stroke::new(1.0_f32, tc.border))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::same(16))
        .shadow(Shadow {
            offset: [0, 1],
            blur: 3,
            spread: 0,
            color: Color32::from_black_alpha(shadow_alpha),
        })
        .show(ui, add_contents);
    ui.add_space(14.0);
}

/// Card with header row: 14px bold title (left) + optional badge (right).
/// Then 8px gap before the content closure.
///
/// Prototype: `.card` + `.card-header`
pub fn card_with_header(
    ui: &mut Ui,
    title: &str,
    header_badge: Option<(&str, BadgeVariant)>,
    add_contents: impl FnOnce(&mut Ui),
) {
    card(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(title).size(14.0).strong());
            if let Some((text, variant)) = header_badge {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    badge(ui, text, variant);
                });
            }
        });
        ui.add_space(8.0);
        add_contents(ui);
    });
}

// ── Underline Tabs ──────────────────────────────────────────────

/// Underline tab bar: accent-colored 2px bottom border on active tab.
/// Returns `true` if the selection changed (for cache invalidation).
///
/// Prototype: `.tab-row { border-bottom: 1px solid var(--border); }`
///            `.tab-item.active { color: var(--accent); border-bottom-color: var(--accent); }`
pub fn underline_tabs(ui: &mut Ui, labels: &[&str], selected: &mut usize) -> bool {
    let tc = ThemeColors::from_ctx(ui.ctx());
    let prev = *selected;

    // Track only the active tab rect — no need to collect all rects.
    let mut active_rect = None;

    let row = ui.horizontal(|ui| {
        for (i, label) in labels.iter().enumerate() {
            let active = i == *selected;
            let color = if active { tc.accent } else { tc.fg3 };
            let text = RichText::new(*label).size(12.0).color(color);
            let response = ui.add(egui::Label::new(text).sense(Sense::click()));
            if active {
                active_rect = Some(response.rect);
            }
            if response.clicked() {
                *selected = i;
            }
        }
    });

    // Draw underlines on a second pass so they don't get clipped by horizontal layout.
    let row_rect = row.response.rect;
    let baseline = row_rect.bottom() + 2.0;
    let painter = ui.painter();

    // Full-width 1px border (prototype: border-bottom on .tab-row)
    painter.line_segment(
        [
            egui::pos2(row_rect.left(), baseline),
            egui::pos2(row_rect.right(), baseline),
        ],
        Stroke::new(1.0_f32, tc.border),
    );

    // 2px accent underline on active tab
    if let Some(rect) = active_rect {
        painter.line_segment(
            [
                egui::pos2(rect.left(), baseline),
                egui::pos2(rect.right(), baseline),
            ],
            Stroke::new(2.0_f32, tc.accent),
        );
    }

    ui.add_space(10.0);
    *selected != prev
}

// ── Badge ───────────────────────────────────────────────────────

/// Badge color variant.
#[derive(Clone, Copy)]
pub enum BadgeVariant {
    Green,
    Orange,
    Red,
}

/// Colored pill badge: 4px radius, 10px bold font, semi-transparent background.
///
/// Prototype: `.badge { padding: 2px 7px; border-radius: 4px; font-size: 10px; }`
pub fn badge(ui: &mut Ui, text: &str, variant: BadgeVariant) {
    let fg = match variant {
        BadgeVariant::Green => semantic::GREEN,
        BadgeVariant::Orange => semantic::ORANGE,
        BadgeVariant::Red => semantic::RED,
    };
    let bg = Color32::from_rgba_unmultiplied(fg.r(), fg.g(), fg.b(), 38);
    egui::Frame::NONE
        .fill(bg)
        .corner_radius(CornerRadius::same(4))
        .inner_margin(Margin::symmetric(7, 2))
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(10.0).color(fg).strong());
        });
}

// ── Buttons ─────────────────────────────────────────────────────

/// Accent-filled action button: white bold text on accent background.
///
/// Prototype: `.btn-action { background: var(--accent); color: white;
///   border-radius: 6px; font-size: 11px; font-weight: 600; }`
pub fn btn_primary(ui: &mut Ui, text: &str) -> Response {
    let tc = ThemeColors::from_ctx(ui.ctx());
    ui.add(
        egui::Button::new(
            RichText::new(text)
                .size(11.0)
                .strong()
                .color(Color32::WHITE),
        )
        .fill(tc.accent)
        .corner_radius(CornerRadius::same(6)),
    )
}

/// Small icon/label button for toolbars. Active state gets accent fill.
///
/// Prototype: `.tool-btn { background: var(--bg2); border: 1px solid var(--border); }`
///            `.tool-btn.active { background: var(--accent); color: white; }`
pub fn btn_icon(ui: &mut Ui, label: &str, active: bool) -> Response {
    let tc = ThemeColors::from_ctx(ui.ctx());
    let (fill, text_color, stroke) = if active {
        (tc.accent, Color32::WHITE, Stroke::NONE)
    } else {
        (tc.bg2, tc.fg2, Stroke::new(1.0_f32, tc.border))
    };
    ui.add(
        egui::Button::new(RichText::new(label).size(11.0).strong().color(text_color))
            .fill(fill)
            .stroke(stroke)
            .corner_radius(CornerRadius::same(5)),
    )
}

// ── Drop Zone ───────────────────────────────────────────────────

/// Clickable file drop zone with loaded/unloaded visual states.
///
/// Returns a `Response` — callers check `.clicked()` to open a file dialog.
///
/// Prototype: `.drop-zone { border: 2px dashed var(--border); border-radius: 10px; }`
/// Note: egui renders a solid border as an approximation (no native dashed support).
///            `.drop-zone.loaded { border-style: solid; border-color: var(--green); }`
pub fn drop_zone(ui: &mut Ui, loaded: bool, label: &str, hint: &str) -> Response {
    let tc = ThemeColors::from_ctx(ui.ctx());
    let height = 72.0;
    let width = ui.available_width();

    let (response, painter) = ui.allocate_painter(egui::Vec2::new(width, height), Sense::click());
    let rect = response.rect;

    let (border_color, border_width, fill) = if loaded {
        (
            semantic::GREEN,
            2.0_f32,
            Color32::from_rgba_unmultiplied(
                semantic::GREEN.r(),
                semantic::GREEN.g(),
                semantic::GREEN.b(),
                13,
            ),
        )
    } else {
        (tc.fg3, 1.5_f32, Color32::TRANSPARENT)
    };

    // Background + border
    painter.rect_filled(rect, CornerRadius::same(10), fill);
    painter.rect_stroke(
        rect,
        CornerRadius::same(10),
        Stroke::new(border_width, border_color),
        egui::StrokeKind::Inside,
    );

    // Label text (centered)
    let label_color = if loaded { semantic::GREEN } else { tc.fg2 };
    let label_y = rect.center().y - 6.0;
    painter.text(
        egui::pos2(rect.center().x, label_y),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(13.0),
        label_color,
    );

    // Hint text (below label)
    if !hint.is_empty() {
        painter.text(
            egui::pos2(rect.center().x, label_y + 16.0),
            egui::Align2::CENTER_CENTER,
            hint,
            egui::FontId::proportional(10.0),
            tc.fg3,
        );
    }

    response
}

// ── Progress Mini ───────────────────────────────────────────────

/// Compact progress bar (80×3px) + text label for toolbar display.
///
/// Prototype: `.progress-bar-sm { width: 80px; height: 3px; }`
pub fn progress_mini(ui: &mut Ui, fraction: f32, text: &str) {
    let tc = ThemeColors::from_ctx(ui.ctx());
    let bar_width = 80.0_f32;
    let bar_height = 3.0_f32;

    ui.horizontal(|ui| {
        let (rect, _) =
            ui.allocate_exact_size(egui::Vec2::new(bar_width, bar_height), Sense::hover());
        let painter = ui.painter();

        // Background track
        painter.rect_filled(rect, CornerRadius::same(2), tc.bg3);

        // Fill
        let fill_width = bar_width * fraction.clamp(0.0, 1.0);
        if fill_width > 0.0 {
            let fill_rect = Rect::from_min_size(rect.min, egui::Vec2::new(fill_width, bar_height));
            painter.rect_filled(fill_rect, CornerRadius::same(2), tc.accent);
        }

        ui.label(RichText::new(text).size(10.0).color(tc.fg3));
    });
}

// ── Stat Row ────────────────────────────────────────────────────

/// Horizontal row of summary stat boxes: bold value + small label.
///
/// Each box: bg3-filled, 6px radius, 12×8 padding.
/// Prototype: stat summary boxes below normalization controls.
pub fn stat_row(ui: &mut Ui, stats: &[(&str, &str)]) {
    let tc = ThemeColors::from_ctx(ui.ctx());
    ui.horizontal(|ui| {
        for (value, label) in stats {
            egui::Frame::NONE
                .fill(tc.bg3)
                .corner_radius(CornerRadius::same(6))
                .inner_margin(Margin::symmetric(12, 8))
                .show(ui, |ui| {
                    ui.vertical(|ui| {
                        ui.label(RichText::new(*value).size(16.0).strong());
                        ui.label(RichText::new(*label).size(10.0).color(tc.fg3));
                    });
                });
        }
    });
}

// ── Isotope Chip ────────────────────────────────────────────────

/// Action returned by an isotope chip interaction.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ChipAction {
    None,
    Remove,
    ToggleEnabled,
}

/// Compact isotope chip: colored dot + symbol + density + ENDF badge + remove button.
///
/// Pill shape with bg3 fill (enabled) or bg2 fill (disabled).
/// Prototype: inline isotope tags in the Configure step.
pub fn isotope_chip(
    ui: &mut Ui,
    symbol: &str,
    density: f64,
    endf_status: EndfStatus,
    enabled: bool,
    id: egui::Id,
) -> ChipAction {
    let tc = ThemeColors::from_ctx(ui.ctx());
    let mut action = ChipAction::None;

    let fill = if enabled { tc.bg3 } else { tc.bg2 };
    egui::Frame::NONE
        .fill(fill)
        .stroke(Stroke::new(1.0_f32, tc.border))
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::symmetric(8, 4))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;

                // Colored dot (hash-based color)
                let dot_color = isotope_dot_color(symbol);
                let (dot_rect, _) = ui.allocate_exact_size(egui::Vec2::splat(8.0), Sense::hover());
                ui.painter()
                    .circle_filled(dot_rect.center(), 4.0, dot_color);

                // Enable/disable toggle via clicking the symbol
                let sym_resp = ui.add(
                    egui::Label::new(RichText::new(symbol).size(11.0).strong())
                        .sense(Sense::click()),
                );
                if sym_resp.clicked() {
                    action = ChipAction::ToggleEnabled;
                }

                // Density
                ui.label(
                    RichText::new(format!("{:.4}", density))
                        .size(10.0)
                        .color(tc.fg2),
                );

                // ENDF status badge
                match endf_status {
                    EndfStatus::Pending => badge(ui, "ENDF", BadgeVariant::Orange),
                    EndfStatus::Fetching => {
                        ui.spinner();
                    }
                    EndfStatus::Loaded => badge(ui, "ENDF", BadgeVariant::Green),
                    EndfStatus::Failed => badge(ui, "FAIL", BadgeVariant::Red),
                }

                // Remove button
                let x_resp = ui.add(
                    egui::Button::new(RichText::new("✕").size(9.0).color(tc.fg3)).frame(false),
                );
                if x_resp.clicked() {
                    action = ChipAction::Remove;
                }
            });
        });

    let _ = id; // reserved for density edit popup tracking
    action
}

/// Compact group chip: colored dot + group name + member count badge + density + ENDF badge + X.
///
/// Same visual pattern as `isotope_chip` but with a member count label
/// ("3 iso") to distinguish groups from individual isotopes.
pub fn group_chip(
    ui: &mut Ui,
    name: &str,
    n_members: usize,
    density: f64,
    endf_status: EndfStatus,
    enabled: bool,
    id: egui::Id,
) -> ChipAction {
    let tc = ThemeColors::from_ctx(ui.ctx());
    let mut action = ChipAction::None;

    let fill = if enabled { tc.bg3 } else { tc.bg2 };
    egui::Frame::NONE
        .fill(fill)
        .stroke(Stroke::new(1.0_f32, tc.border))
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::symmetric(8, 4))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;

                // Colored dot (hash-based color from group name)
                let dot_color = isotope_dot_color(name);
                let (dot_rect, _) = ui.allocate_exact_size(egui::Vec2::splat(8.0), Sense::hover());
                ui.painter()
                    .circle_filled(dot_rect.center(), 4.0, dot_color);

                // Enable/disable toggle via clicking the name
                let sym_resp = ui.add(
                    egui::Label::new(RichText::new(name).size(11.0).strong()).sense(Sense::click()),
                );
                if sym_resp.clicked() {
                    action = ChipAction::ToggleEnabled;
                }

                // Member count badge
                ui.label(
                    RichText::new(format!("{n_members} iso"))
                        .size(9.0)
                        .color(tc.fg3),
                );

                // Density
                ui.label(
                    RichText::new(format!("{density:.4}"))
                        .size(10.0)
                        .color(tc.fg2),
                );

                // ENDF status badge
                match endf_status {
                    EndfStatus::Pending => badge(ui, "ENDF", BadgeVariant::Orange),
                    EndfStatus::Fetching => {
                        ui.spinner();
                    }
                    EndfStatus::Loaded => badge(ui, "ENDF", BadgeVariant::Green),
                    EndfStatus::Failed => badge(ui, "FAIL", BadgeVariant::Red),
                }

                // Remove button
                let x_resp = ui.add(
                    egui::Button::new(RichText::new("\u{2715}").size(9.0).color(tc.fg3))
                        .frame(false),
                );
                if x_resp.clicked() {
                    action = ChipAction::Remove;
                }
            });
        });

    let _ = id; // reserved for density edit popup tracking
    action
}

/// Deterministic dot color for an isotope symbol (hash-based hue).
pub fn isotope_dot_color(symbol: &str) -> Color32 {
    let mut hash: u32 = 5381;
    for b in symbol.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(u32::from(b));
    }
    let hue = (hash % 360) as f32;
    hsl_to_rgb(hue, 0.70, 0.55)
}

/// Convert HSL to `Color32` (hue in degrees, s/l in 0..1).
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Color32 {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r1, g1, b1) = match h as u32 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Color32::from_rgb(
        ((r1 + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((g1 + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((b1 + m) * 255.0).clamp(0.0, 255.0) as u8,
    )
}

// ── Navigation Buttons ──────────────────────────────────────────

/// Action returned by the navigation button bar.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NavAction {
    None,
    Back,
    Continue,
}

/// Back/Continue navigation bar with optional guard.
///
/// `back_label`: `Some("← Back")` or `None` to hide.
/// `continue_label`: e.g. `"Continue →"`.
/// `can_continue`: `false` disables the Continue button.
/// `hint`: shown when Continue is disabled.
pub fn nav_buttons(
    ui: &mut Ui,
    back_label: Option<&str>,
    continue_label: &str,
    can_continue: bool,
    hint: &str,
) -> NavAction {
    let tc = ThemeColors::from_ctx(ui.ctx());
    let mut action = NavAction::None;

    ui.add_space(8.0);
    ui.horizontal(|ui| {
        if let Some(label) = back_label
            && ui.button(label).clicked()
        {
            action = NavAction::Back;
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_enabled_ui(can_continue, |ui| {
                if btn_primary(ui, continue_label).clicked() {
                    action = NavAction::Continue;
                }
            });
            if !can_continue && !hint.is_empty() {
                ui.label(RichText::new(hint).size(10.0).color(tc.fg3));
            }
        });
    });

    action
}

// ── Resolution Card ────────────────────────────────────────────

/// Result from the resolution card widget.
pub struct ResolutionCardResult {
    pub changed: bool,
}

/// Shared resolution broadening card: Gaussian parametric or tabulated file.
///
/// Used identically in Configure, Forward Model, and Detectability —
/// `target` tells the dialog dispatcher which card's state a picked
/// resolution file belongs to. `changed` covers the inline edits only;
/// file picks resolve through `file_dialog::dispatch_results`, which
/// applies the same per-card invalidation as the callers.
pub fn resolution_card(
    ui: &mut Ui,
    enabled: &mut bool,
    mode: &mut ResolutionMode,
    flight_path_m: f64,
    dialogs: &mut crate::file_dialog::FileDialogs,
    target: crate::file_dialog::ResolutionTarget,
) -> ResolutionCardResult {
    let mut changed = false;

    card_with_header(ui, "Instrument Resolution", None, |ui| {
        let prev_enabled = *enabled;
        ui.checkbox(enabled, "Enable broadening");
        if *enabled != prev_enabled {
            changed = true;
        }

        if !*enabled {
            return;
        }

        // Mode selector: Gaussian vs Tabulated
        let is_gaussian = matches!(mode, ResolutionMode::Gaussian { .. });
        ui.horizontal(|ui| {
            if ui.radio(is_gaussian, "Gaussian (parametric)").clicked() && !is_gaussian {
                *mode = ResolutionMode::default();
                changed = true;
            }
            if ui.radio(!is_gaussian, "From file").clicked() && is_gaussian {
                *mode = ResolutionMode::Tabulated {
                    path: std::path::PathBuf::new(),
                    data: None,
                    error: None,
                };
                changed = true;
            }
        });

        ui.add_space(4.0);

        match mode {
            ResolutionMode::Gaussian {
                delta_t_us,
                delta_l_m,
            } => {
                ui.horizontal(|ui| {
                    ui.label(format!("Flight path: {flight_path_m:.2} m"));
                    ui.label("(from Beamline)");
                });
                let prev_dt = *delta_t_us;
                let prev_dl = *delta_l_m;
                ui.horizontal(|ui| {
                    ui.label("\u{0394}t (\u{03bc}s):");
                    ui.add(
                        egui::DragValue::new(delta_t_us)
                            .speed(0.1)
                            .range(0.0..=100.0),
                    );
                    ui.label("\u{0394}L (m):");
                    ui.add(
                        egui::DragValue::new(delta_l_m)
                            .speed(0.001)
                            .range(0.0..=1.0),
                    );
                });
                if *delta_t_us != prev_dt || *delta_l_m != prev_dl {
                    changed = true;
                }
            }
            ResolutionMode::Tabulated { path, data, error } => {
                ui.horizontal(|ui| {
                    if ui.button("Load resolution file\u{2026}").clicked() {
                        dialogs.pick_file(
                            crate::file_dialog::DialogIntent::ResolutionFile(target),
                            crate::file_dialog::DialogOptions {
                                filters: vec![("Resolution", &["txt", "dat"])],
                                ..Default::default()
                            },
                        );
                    }
                    if let Some(name) = path.file_name() {
                        ui.label(
                            RichText::new(name.to_string_lossy().to_string())
                                .monospace()
                                .size(11.0),
                        );
                    }
                });
                // Show summary if loaded, or error if parse failed
                if let Some(tab) = data {
                    let n = tab.ref_energies().len();
                    let e_min = tab.ref_energies().first().copied().unwrap_or(0.0);
                    let e_max = tab.ref_energies().last().copied().unwrap_or(0.0);
                    ui.label(
                        RichText::new(format!(
                            "{n} reference energies, {e_min:.4}\u{2013}{e_max:.1} eV"
                        ))
                        .size(11.0)
                        .color(crate::theme::semantic::GREEN),
                    );
                } else if let Some(err) = error {
                    ui.colored_label(crate::theme::semantic::RED, format!("Parse error: {err}"));
                } else if !path.as_os_str().is_empty() {
                    ui.colored_label(
                        crate::theme::semantic::RED,
                        "File not loaded \u{2014} select a valid resolution file",
                    );
                }
            }
        }
    });

    ResolutionCardResult { changed }
}

/// Apply a picked tabulated-resolution file to `mode` (dispatch handler
/// for `DialogIntent::ResolutionFile`). Overwrites `mode` with a fresh
/// `Tabulated` variant so a pick that resolves on a later frame wins
/// even if the user toggled the radio in between. Returns `true` when
/// the mode changed (always — parse failures are recorded in `error`,
/// matching the previous inline behaviour).
pub fn apply_resolution_file(
    mode: &mut ResolutionMode,
    file: std::path::PathBuf,
    flight_path_m: f64,
) -> bool {
    let (data, error) = match file.to_str() {
        Some(path_str) => match TabulatedResolution::from_file(path_str, flight_path_m) {
            Ok(tab) => (Some(Arc::new(tab)), None),
            Err(e) => (None, Some(format!("{e}"))),
        },
        None => (
            None,
            Some("File path is not valid UTF-8; please choose a different file".into()),
        ),
    };
    *mode = ResolutionMode::Tabulated {
        path: file,
        data,
        error,
    };
    true
}

// ── ENDF Library Name ──────────────────────────────────────────

/// Map an `EndfLibrary` variant to its display name.
pub fn library_name(lib: EndfLibrary) -> &'static str {
    match lib {
        EndfLibrary::EndfB8_0 => "ENDF/B-VIII.0",
        EndfLibrary::EndfB8_1 => "ENDF/B-VIII.1",
        EndfLibrary::Jeff3_3 => "JEFF-3.3",
        EndfLibrary::Jendl5 => "JENDL-5",
        EndfLibrary::Tendl2023 => "TENDL-2023",
        EndfLibrary::Cendl3_2 => "CENDL-3.2",
    }
}

// ── Teleport Pill ──────────────────────────────────────────────

/// Accent-filled pill button that navigates to another guided step.
pub fn teleport_pill(ui: &mut Ui, label: &str, target: GuidedStep, state: &mut AppState) {
    let accent = ThemeColors::from_ctx(ui.ctx()).accent;
    let btn = egui::Button::new(RichText::new(label).small().color(Color32::WHITE))
        .fill(accent)
        .corner_radius(12.0);
    if ui.add(btn).clicked() {
        state.status_message = String::new();
        state.guided_step = target;
    }
}

// ── Shared Helpers ─────────────────────────────────────────────────

/// Work item for the ENDF fetch worker.
///
/// `target` identifies which list the result should be routed to:
/// Configure, ForwardModel, DetectMatrix, or DetectTrace.
pub(crate) struct EndfWorkItem {
    pub z: u32,
    pub a: u32,
    pub target: FetchTarget,
    pub isotope: Isotope,
    pub symbol: String,
    pub library: EndfLibrary,
}

/// Background worker for ENDF data fetching.
///
/// Runs inside a `std::thread::spawn` closure — iterates `work` items,
/// fetches + parses each, and sends results on `tx`. Supports cancellation.
/// Results are keyed by `(z, a)` so the receiver can match them to entries
/// regardless of list mutations during the fetch.
pub(crate) fn endf_fetch_worker(
    work: Vec<EndfWorkItem>,
    cancel: Arc<AtomicBool>,
    tx: mpsc::Sender<EndfFetchResult>,
) {
    let retriever = nereids_endf::retrieval::EndfRetriever::new();
    for (idx, item) in work.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let Some(mat) = nereids_endf::retrieval::mat_number(&item.isotope, item.library) else {
            // Defensive: should be unreachable — callers pre-filter by mat_number
            let _ = tx.send(EndfFetchResult {
                z: item.z,
                a: item.a,
                target: item.target,
                symbol: item.symbol.clone(),
                result: Err(format!(
                    "No MAT number for {} — isotope not in ENDF database",
                    item.symbol
                )),
            });
            continue;
        };
        let result = match retriever.get_endf_file(&item.isotope, item.library, mat) {
            Ok((_path, endf_text)) => match nereids_endf::parser::parse_endf_file2(&endf_text) {
                Ok(data) => {
                    // Mixed evaluation: the file loads, but any skipped spans
                    // contribute zero cross-section. Pure non-evaluable files
                    // never reach here — the parser rejects them with a hard
                    // error.
                    crate::project::warn_unevaluated_ranges(&item.symbol, &data);
                    Ok(data)
                }
                Err(e) => Err(format!("Parse error for {}: {e}", item.symbol)),
            },
            Err(e) => {
                let blocked = e.is_remote_access_blocked();
                let msg = format!("Fetch error for {}: {e}", item.symbol);
                if blocked {
                    let _ = tx.send(EndfFetchResult {
                        z: item.z,
                        a: item.a,
                        target: item.target,
                        symbol: item.symbol.clone(),
                        result: Err(msg.clone()),
                    });
                    for skipped in &work[idx + 1..] {
                        if cancel.load(Ordering::Relaxed) {
                            break;
                        }
                        let _ = tx.send(EndfFetchResult {
                            z: skipped.z,
                            a: skipped.a,
                            target: skipped.target,
                            symbol: skipped.symbol.clone(),
                            result: Err(format!(
                                "Skipped {} because the upstream ENDF server blocked this batch; retry later.",
                                skipped.symbol
                            )),
                        });
                    }
                    break;
                }
                Err(msg)
            }
        };
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let _ = tx.send(EndfFetchResult {
            z: item.z,
            a: item.a,
            target: item.target,
            symbol: item.symbol.clone(),
            result,
        });
    }
}

/// Build an `Option<ResolutionFunction>` from the current resolution settings.
///
/// Returns `Ok(None)` if resolution is disabled, `Ok(Some(..))` on success,
/// or `Err(msg)` if parameters are invalid or a file isn't loaded.
pub(crate) fn build_resolution_function(
    enabled: bool,
    mode: &ResolutionMode,
    flight_path_m: f64,
) -> Result<Option<ResolutionFunction>, String> {
    if !enabled {
        return Ok(None);
    }
    match mode {
        ResolutionMode::Gaussian {
            delta_t_us,
            delta_l_m,
        } => {
            let params = ResolutionParams::new(flight_path_m, *delta_t_us, *delta_l_m, 0.0)
                .map_err(|e| format!("Invalid resolution parameters: {e}"))?;
            Ok(Some(ResolutionFunction::Gaussian(params)))
        }
        ResolutionMode::Tabulated {
            data: Some(tab), ..
        } => Ok(Some(ResolutionFunction::Tabulated(Arc::clone(tab)))),
        ResolutionMode::Tabulated { data: None, .. } => Err("Resolution file not loaded".into()),
    }
}

/// Parameters for [`build_spectrum_x_axis`].
pub(crate) struct SpectrumXAxisParams<'a> {
    pub axis: SpectrumAxis,
    pub energies: Option<&'a [f64]>,
    pub spectrum_values: Option<&'a [f64]>,
    pub spectrum_unit: SpectrumUnit,
    pub spectrum_kind: SpectrumValueKind,
    pub flight_path_m: f64,
    pub delay_us: f64,
    pub n_tof: usize,
}

/// Build x-axis values and label for a spectrum plot.
///
/// Handles all combinations of `SpectrumAxis × SpectrumUnit × SpectrumValueKind`,
/// including energy-to-TOF conversion with NaN guards for non-positive energies.
/// Returns `None` if the required data is missing (e.g., no energy grid for EnergyEv axis).
pub(crate) fn build_spectrum_x_axis(
    p: &SpectrumXAxisParams<'_>,
) -> Option<(Vec<f64>, &'static str)> {
    match p.axis {
        SpectrumAxis::EnergyEv => Some((p.energies?.to_vec(), "Energy (eV)")),
        SpectrumAxis::TofMicroseconds => {
            let v = match p.spectrum_values {
                Some(v) => v,
                None => {
                    return Some(((0..p.n_tof).map(|i| i as f64).collect(), "Frame index"));
                }
            };
            let can_convert = p.flight_path_m.is_finite() && p.flight_path_m > 0.0;
            match (p.spectrum_unit, p.spectrum_kind) {
                (SpectrumUnit::TofMicroseconds, SpectrumValueKind::BinEdges) => {
                    let centers: Vec<f64> = v
                        .windows(2)
                        .take(p.n_tof)
                        .map(|w| 0.5 * (w[0] + w[1]))
                        .collect();
                    Some((centers, "TOF (\u{03bc}s)"))
                }
                (SpectrumUnit::TofMicroseconds, SpectrumValueKind::BinCenters) => {
                    Some((v.iter().take(p.n_tof).copied().collect(), "TOF (\u{03bc}s)"))
                }
                (SpectrumUnit::EnergyEv, SpectrumValueKind::BinEdges) => {
                    if can_convert {
                        let tof_vals: Vec<f64> = v
                            .windows(2)
                            .take(p.n_tof)
                            .map(|w| {
                                let center = 0.5 * (w[0] + w[1]);
                                if center > 0.0 {
                                    nereids_core::constants::energy_to_tof(center, p.flight_path_m)
                                        + p.delay_us
                                } else {
                                    f64::NAN
                                }
                            })
                            .collect();
                        Some((tof_vals, "TOF (\u{03bc}s)"))
                    } else {
                        let centers: Vec<f64> = v
                            .windows(2)
                            .take(p.n_tof)
                            .map(|w| 0.5 * (w[0] + w[1]))
                            .collect();
                        Some((centers, "Energy (eV)"))
                    }
                }
                (SpectrumUnit::EnergyEv, SpectrumValueKind::BinCenters) => {
                    if can_convert {
                        let tof_vals: Vec<f64> = v
                            .iter()
                            .take(p.n_tof)
                            .map(|&e| {
                                if e > 0.0 {
                                    nereids_core::constants::energy_to_tof(e, p.flight_path_m)
                                        + p.delay_us
                                } else {
                                    f64::NAN
                                }
                            })
                            .collect();
                        Some((tof_vals, "TOF (\u{03bc}s)"))
                    } else {
                        Some((v.iter().take(p.n_tof).copied().collect(), "Energy (eV)"))
                    }
                }
            }
        }
    }
}

/// Draw resonance energy dip markers on a spectrum plot (energy axis only).
///
/// Iterates all provided resonance data sets and draws a `VLine`
/// for each resonance within the data x-range. The caller is responsible
/// for collecting resonance data from both individual entries and group members.
pub(crate) fn draw_resonance_dips(
    plot_ui: &mut egui_plot::PlotUi,
    all_resonance_data: &[nereids_endf::resonance::ResonanceData],
    x_values: &[f64],
) {
    let (x_min, x_max) = x_values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    for res_data in all_resonance_data {
        for range in &res_data.ranges {
            for lg in &range.l_groups {
                for res in &lg.resonances {
                    if res.energy >= x_min && res.energy <= x_max {
                        plot_ui.vline(
                            VLine::new("", res.energy)
                                .color(RESONANCE_DIP_COLOR)
                                .width(0.5_f32),
                        );
                    }
                }
            }
        }
    }
}

/// Convert a resonance energy (eV) into the spectrum's active x-axis units.
///
/// Mirrors the energy→TOF transform in [`build_spectrum_x_axis`] **including
/// the fallback branch**: when the TOF axis is selected but `flight_path_m`
/// is invalid (zero, negative, or non-finite), `build_spectrum_x_axis` falls
/// back to plotting energy values whenever the input data is in `EnergyEv`
/// units (and silently relabels the axis). To keep tick positions aligned
/// with the actual rendered x-axis in that fallback case, this helper does
/// the same — returns `Some(energy_ev)` when the TOF conversion is undefined
/// AND the spectrum input is energy-valued.
///
/// Returns `None` only when no axis position can be derived: TOF axis with
/// invalid `flight_path_m` AND spectrum input is already in TOF units (in
/// which case the spectrum cannot be aligned with energy-input resonances
/// without a flight path).
pub(crate) fn resonance_energy_to_axis(
    energy_ev: f64,
    axis: SpectrumAxis,
    spectrum_unit: SpectrumUnit,
    flight_path_m: f64,
    delay_us: f64,
) -> Option<f64> {
    match axis {
        SpectrumAxis::EnergyEv => Some(energy_ev),
        SpectrumAxis::TofMicroseconds => {
            let can_convert = energy_ev > 0.0 && flight_path_m.is_finite() && flight_path_m > 0.0;
            if can_convert {
                Some(nereids_core::constants::energy_to_tof(energy_ev, flight_path_m) + delay_us)
            } else if matches!(spectrum_unit, SpectrumUnit::EnergyEv) {
                // Mirror `build_spectrum_x_axis` fallback: when TOF
                // conversion is unavailable but input is energy-valued,
                // the spectrum falls back to plotting energy. Strips
                // need to do the same to stay aligned.
                Some(energy_ev)
            } else {
                None
            }
        }
    }
}

/// Parameters for building a fit overlay line.
pub(crate) struct FitLineParams<'a> {
    pub result: &'a nereids_pipeline::pipeline::SpectrumFitResult,
    pub resonance_data: &'a [nereids_endf::resonance::ResonanceData],
    pub density_indices: &'a [usize],
    pub density_ratios: &'a [f64],
    pub energies: &'a [f64],
    pub temperature_k: f64,
    pub x_values: &'a [f64],
    pub n_plot: usize,
    /// Instrument resolution for the fit overlay.
    /// When `Some`, the overlay applies resolution after Beer-Lambert.
    pub instrument: Option<std::sync::Arc<nereids_physics::transmission::InstrumentParams>>,
    /// Optional per-bin multiplier applied to the fitted transmission
    /// before plotting.  Use this to scale T_fit to expected sample
    /// counts in counts-mode display: `multiplier[i] = c · OB[i, y, x]`.
    /// Length must be ≥ `n_plot`; ignored when `None`.
    pub y_multiplier: Option<&'a [f64]>,
}

/// The energy grid the fitted physics was evaluated on, for `nominal` —
/// the fit's slice of the loaded grid.
///
/// An energy-scale fit (SAMMY TZERO) solves for `(t0, L_scale)` and
/// evaluates the resonance physics at the CALIBRATED energies, never at the
/// nominal ones. Every redraw of such a fit has to map the same grid
/// through the same transform or it shows a curve the fit never computed,
/// displaced wherever the calibration is not the identity. The transform
/// comes from the result itself — [`nereids_pipeline::pipeline::SpectrumFitResult::corrected_energies`]
/// reuses the fitter's own `corrected_energy_grid` with the flight path the
/// fit was configured with — so no redraw re-derives it.
///
/// Returns the nominal grid unchanged when the energy scale was not fitted.
///
/// # Errors
/// A user-facing line when the stored calibration does not map this grid to
/// physical energies. The fitted curve cannot be reproduced then, and
/// saying so is better than drawing the un-calibrated one. Every rejection
/// of `corrected_energy_grid` reaches the caller this way, not only the
/// degenerate `t0` at or past the grid's shortest flight time: a non-finite
/// `t0`; an `l_scale` that is not finite and positive; a flight path that
/// is not finite and positive; an empty grid; any grid energy that is not
/// finite and positive; and a grid that is not strictly ascending.
pub(crate) fn fitted_physics_energies(
    result: &nereids_pipeline::pipeline::SpectrumFitResult,
    nominal: &[f64],
) -> Result<Vec<f64>, String> {
    match result.corrected_energies(nominal) {
        None => Ok(nominal.to_vec()),
        Some(Ok(corrected)) => Ok(corrected),
        Some(Err(e)) => Err(format!(
            "Fit curve hidden: the fitted energy scale (t0 = {t0} \u{b5}s, L_scale = {l_scale}) \
             does not map this grid to physical energies ({e})",
            t0 = result.t0_us.unwrap_or(f64::NAN),
            l_scale = result.l_scale.unwrap_or(f64::NAN),
        )),
    }
}

/// The fitted forward model's curve on the fit's grid:
///
/// ```text
/// y(E) = B(E)·[Anorm·T(E) + BackA + BackB/√E + BackC·√E + BackD·exp(−BackF/√E)]
/// ```
///
/// `transmission` is the inner physics model's output, evaluated on the
/// CALIBRATED grid (see [`fitted_physics_energies`]).
/// `nominal_energies` are the NOMINAL energies of the same bins, because
/// the fit composes the SAMMY background polynomial and the `ln(E/E_ref)`
/// baseline on the nominal grid — `fit_transmission_lm` wraps the physics
/// model in `NormalizedTransmissionModel` / `MultiplicativeBaselineModel`
/// with `config.energies()` — while only the inner physics moves with the
/// fitted energy scale.
///
/// Shared by the spectrum overlay and the residual dock so the two cannot
/// disagree about what the fit predicted: a residual taken against bare
/// Beer-Lambert transmission carries the whole background and baseline as
/// fake signal.
pub(crate) fn compose_fitted_curve(
    result: &nereids_pipeline::pipeline::SpectrumFitResult,
    nominal_energies: &[f64],
    transmission: &[f64],
) -> Vec<f64> {
    let n = nominal_energies.len().min(transmission.len());
    (0..n)
        .map(|i| {
            let e = nominal_energies[i];
            let bg_poly = if e > 0.0 && e.is_finite() {
                let sqrt_e = e.sqrt();
                // `back_d` / `back_f` are `Option<f64>`: `None` means
                // "exponential tail not fit" (counts-KL path, LM with
                // `fit_back_d=false`, or a project reload that dropped
                // the unpersisted maps).  Drop the tail term entirely
                // in that case.  SAMMY pairs BackD/BackF, so
                // `(Some, Some)` is the only arm that fires in
                // practice.
                let exp_tail = match (result.back_d, result.back_f) {
                    (Some(bd), Some(bf)) => bd * (-bf / sqrt_e).exp(),
                    _ => 0.0,
                };
                result.background[0]
                    + result.background[1] / sqrt_e
                    + result.background[2] * sqrt_e
                    + exp_tail
            } else {
                // Non-positive or non-finite energy: skip the polynomial.
                // BackB/√E and BackD·exp(-BackF/√E) blow up at E ≤ 0, and
                // the spectrum already filters non-finite x via
                // build_spectrum_x_axis.
                0.0
            };
            let y = result.anorm * transmission[i] + bg_poly;
            // Issue #635: the multiplicative baseline is OUTERMOST —
            // y = B(E)·[Anorm·T + additive background] — reconstructed with
            // the EXACT E_ref the fit used (stored on the result).  `None`
            // (baseline not fit, or a reload that dropped it) leaves the
            // curve unchanged.
            match (result.baseline, result.baseline_e_ref_ev) {
                (Some(b), Some(e_ref)) if e > 0.0 && e.is_finite() && e_ref > 0.0 => {
                    let z = (e / e_ref).ln();
                    (b[0] + b[1] * z + b[2] * z * z) * y
                }
                _ => y,
            }
        })
        .collect()
}

/// What the residual dock shows for one fit: the per-bin residuals and
/// their summary statistics.
pub(crate) struct FitResiduals {
    /// `(nominal energy eV, measured − fitted)` for every finite residual.
    pub residuals: Vec<(f64, f64)>,
    pub rms: f64,
    pub max_abs: f64,
}

/// Residuals of `measured` against the fitted forward model — measured
/// minus the SAME curve the overlay draws (see [`compose_fitted_curve`]),
/// so the dock's plot, RMS and Max|r| describe the fit that was actually
/// performed. Bins whose residual is not finite are dropped, as they
/// cannot be plotted or summarised.
pub(crate) fn fit_residuals(
    result: &nereids_pipeline::pipeline::SpectrumFitResult,
    nominal_energies: &[f64],
    transmission: &[f64],
    measured: &[f64],
) -> FitResiduals {
    let fitted = compose_fitted_curve(result, nominal_energies, transmission);
    let n = fitted.len().min(measured.len());
    let mut residuals = Vec::with_capacity(n);
    let mut sum_sq = 0.0;
    let mut max_abs = 0.0_f64;
    for i in 0..n {
        let res = measured[i] - fitted[i];
        if res.is_finite() {
            residuals.push((nominal_energies[i], res));
            sum_sq += res * res;
            max_abs = max_abs.max(res.abs());
        }
    }
    let rms = if residuals.is_empty() {
        0.0
    } else {
        (sum_sq / residuals.len() as f64).sqrt()
    };
    FitResiduals {
        residuals,
        rms,
        max_abs,
    }
}

/// The model that redraws a fit result, and whether its Doppler routes are
/// the ones the fit disclosed.
pub(crate) struct OverlayModel {
    /// The forward model at the displayed temperature, gated the way the
    /// fit gated.
    pub model: nereids_fitting::transmission_model::TransmissionFitModel,
    /// A user-facing line when the overlay's routes differ from the routes
    /// the result disclosed (the isotope set, grid or resolution changed
    /// since the fit): the overlay is drawn, but it is not the fitted curve.
    ///
    /// Also set when the result discloses NO routes and this redraw
    /// broadens something: the comparison cannot run then, and an
    /// unchecked redraw must not look like a checked one.
    pub route_mismatch: Option<String>,
}

/// The model that redraws a fit result: the forward model at the displayed
/// temperature with the Doppler routes decided the way the fit decided them.
///
/// `energies` must be the grid the fit evaluated the physics on: the
/// fit-energy slice, not the whole loaded grid (see
/// `analyze::fit_grid_range`), mapped through the fitted energy scale (see
/// [`fitted_physics_energies`]) — because the
/// route gate reads the grid: an acquisition that runs past the resolved
/// range demotes an isotope on the whole grid while the fit range inside
/// it stays continuous. A free-temperature fit gates its routes once at
/// the fit's upper bound; redrawing at the fitted temperature must gate
/// there too, or an isotope the fit demoted to the sampled table near a
/// range edge would be redrawn through the continuous integral. A
/// fixed-temperature fit gated at that temperature. The plan's routes are
/// compared with the routes the result disclosed and a disagreement is
/// returned as [`OverlayModel::route_mismatch`] for the caller to show.
///
/// `disclosed_routes` is `None` for a result that carries no disclosure —
/// a per-pixel result assembled from a spatial map is the case that
/// reaches the GUI, because a map discloses one map-wide route set and
/// none per pixel, and withholds even that when its converged pixels
/// disagree. The comparison cannot run then; a broadening redraw returns
/// that absence as [`OverlayModel::route_mismatch`] rather than passing
/// unchecked for checked.
///
/// Returns `None` when the model or its plan cannot be built (the caller
/// then draws no overlay, as before).
pub(crate) fn build_overlay_model(
    energies: Vec<f64>,
    resonance_data: Vec<nereids_endf::resonance::ResonanceData>,
    temperature_k: f64,
    instrument: Option<Arc<nereids_physics::transmission::InstrumentParams>>,
    density_mapping: (Vec<usize>, Vec<f64>),
    free_temperature: bool,
    disclosed_routes: Option<&[nereids_physics::doppler_route::IsotopeDopplerRoute]>,
) -> Option<OverlayModel> {
    use nereids_fitting::transmission_model::{
        TEMPERATURE_FIT_UPPER_BOUND_K, TransmissionFitModel,
    };
    use nereids_physics::transmission::DopplerPlan;

    let gate_temperature_k = if free_temperature {
        TEMPERATURE_FIT_UPPER_BOUND_K
    } else {
        temperature_k
    };
    let plan = DopplerPlan::new(
        &energies,
        &resonance_data,
        gate_temperature_k,
        instrument.as_deref(),
        None,
    )
    .ok()?;
    // Label the plan's routes with the isotopes they were decided for, as
    // the fit's own disclosure is labelled: a positional comparison of the
    // bare routes answers "same kind of route" for a DIFFERENT isotope,
    // so swapping one Reich-Moore isotope for another would agree and the
    // redrawn curve would be presented as the fitted one.
    let overlay_routes: Vec<nereids_physics::doppler_route::IsotopeDopplerRoute> = resonance_data
        .iter()
        .zip(plan.routes())
        .map(
            |(rd, route)| nereids_physics::doppler_route::IsotopeDopplerRoute {
                isotope: rd.isotope,
                route: route.clone(),
            },
        )
        .collect();
    let describe = |routes: &mut dyn Iterator<Item = String>| routes.collect::<Vec<_>>().join("; ");
    let route_mismatch = match disclosed_routes {
        Some(disclosed) => {
            let agree = disclosed.len() == overlay_routes.len()
                && disclosed
                    .iter()
                    .zip(&overlay_routes)
                    .all(|(d, o)| d.same_kind(o));
            if agree {
                None
            } else {
                let message = format!(
                    "Fit overlay is not the fitted curve: it redraws [{}] where the fit disclosed \
                     [{}] (route gate at {gate_temperature_k} K); the isotope set, grid or \
                     resolution changed since the fit",
                    describe(&mut overlay_routes.iter().map(ToString::to_string)),
                    describe(&mut disclosed.iter().map(ToString::to_string)),
                );
                tracing::warn!("{message}");
                Some(message)
            }
        }
        // No disclosure to compare against. Silence would present an
        // unchecked redraw exactly like a checked one, so the absence is
        // reported wherever the disagreement would have been — but only
        // when this redraw actually broadens something, since an
        // unbroadened redraw has no route to get wrong.
        None if overlay_routes.iter().any(|r| {
            !matches!(
                r.route,
                nereids_physics::doppler_route::DopplerRoute::Unbroadened
            )
        }) =>
        {
            let message = format!(
                "Fit overlay is unchecked: it redraws [{}] (route gate at {gate_temperature_k} K) \
                 but the fit result discloses no routes, so nothing says these are the ones the \
                 fit took. A spatial map discloses one map-wide route set and none per pixel, and \
                 withholds even that when its converged pixels disagree",
                describe(&mut overlay_routes.iter().map(ToString::to_string)),
            );
            tracing::warn!("{message}");
            Some(message)
        }
        None => None,
    };
    let model = TransmissionFitModel::new(
        energies,
        resonance_data,
        temperature_k,
        instrument,
        density_mapping,
        None,
        None,
    )
    .ok()?
    .with_doppler_plan(Arc::new(plan));
    Some(OverlayModel {
        model,
        route_mismatch,
    })
}

/// Counts and instrument resolution cannot share the transmission-only fit
/// overlay until the detector response has separate open/sample arms.
pub(crate) fn counts_resolution_overlay_unsupported(
    shows_counts: bool,
    has_instrument_resolution: bool,
) -> bool {
    shows_counts && has_instrument_resolution
}

pub(crate) const COUNTS_RESOLUTION_OVERLAY_MESSAGE: &str = "Count fit overlay hidden: instrument resolution needs the exact separate-arm model \
     R[Phi] and R[Phi*T]. Multiplying c*OB by R[T] is not a physical count model, so \
     the transmission-only overlay cannot represent a resolved count fit.";

/// A fit overlay line and the warning that goes with it, if any.
pub(crate) struct FitLine {
    /// `None` when the fitted curve cannot be reproduced — currently a
    /// degenerate energy-scale calibration, which `warning` then explains.
    /// Drawing nothing is the honest outcome: the alternative is a curve
    /// the fit never computed.
    pub line: Option<Line<'static>>,
    /// A user-facing line to show beside the plot: an
    /// [`OverlayModel::route_mismatch`], or the reason no curve is drawn.
    pub warning: Option<String>,
}

/// Build a fit overlay line from a `SpectrumFitResult`.
///
/// Returns `None` if the fit didn't converge, no resonance data is provided,
/// or model construction fails.
///
/// `energies`, `x_values` and `y_multiplier` are the fit's slice of the
/// loaded grid, index-aligned with each other (see
/// `analyze::fit_grid_range`). `resonance_data` should contain data for ALL
/// fitted entities (individual isotopes + group members). `density_indices`
/// and `density_ratios` map each resonance data entry to a density
/// parameter index and its abundance ratio.
pub(crate) fn build_fit_line(p: &FitLineParams<'_>) -> Option<FitLine> {
    if counts_resolution_overlay_unsupported(p.y_multiplier.is_some(), p.instrument.is_some()) {
        return None;
    }
    if !p.result.converged {
        return None;
    }
    if p.resonance_data.is_empty() {
        return None;
    }
    let resonance_data: Vec<_> = p.resonance_data.to_vec();
    let overlay_temp = p.result.temperature_k.unwrap_or(p.temperature_k);
    // The physics is redrawn on the grid the fit evaluated it on — the
    // nominal slice mapped through the fitted energy scale — while the
    // x-axis, the SAMMY background polynomial and the ln(E/E_ref) baseline
    // stay on the NOMINAL grid, because that is how the fit composed them
    // (see `compose_fitted_curve`).
    let physics_energies = match fitted_physics_energies(p.result, p.energies) {
        Ok(grid) => grid,
        Err(message) => {
            tracing::warn!("{message}");
            return Some(FitLine {
                line: None,
                warning: Some(message),
            });
        }
    };
    let OverlayModel {
        model,
        route_mismatch,
    } = build_overlay_model(
        physics_energies,
        resonance_data,
        overlay_temp,
        p.instrument.clone(),
        (p.density_indices.to_vec(), p.density_ratios.to_vec()),
        p.result.temperature_k.is_some(),
        p.result.doppler_routes.as_deref(),
    )?;

    use nereids_fitting::lm::FitModel;
    let fitted_t = model.evaluate(&p.result.densities).ok()?;
    // Compose the fitted Anorm, SAMMY background and multiplicative
    // baseline so the overlay is the forward model the solver fitted, not
    // bare Beer-Lambert transmission.  `y_multiplier` (= c·OB[i] in counts
    // mode) then scales the result to the displayed y-axis.
    let composed = compose_fitted_curve(p.result, p.energies, &fitted_t);
    let n_fit = p.n_plot.min(composed.len()).min(p.x_values.len());
    let fit_points: PlotPoints = (0..n_fit)
        .filter(|&i| p.x_values[i].is_finite())
        .map(|i| {
            let y = match p.y_multiplier {
                Some(m) if i < m.len() => m[i] * composed[i],
                _ => composed[i],
            };
            [p.x_values[i], y]
        })
        .collect();
    Some(FitLine {
        line: Some(
            Line::new("Fit", fit_points)
                .width(1.25_f32)
                .color(egui::Color32::from_rgba_unmultiplied(0, 122, 255, 170)),
        ),
        warning: route_mismatch,
    })
}

// ── Resonance Data Collection ──────────────────────────────────

/// Collect all resonance data and density mapping from enabled isotopes + groups.
///
/// Order matches `build_fit_config()`: individuals first, then group members.
/// Returns:
/// - `all_rd`: flat list of ResonanceData (one per individual isotope + one per group member)
/// - `density_indices`: maps each rd to a density parameter index
/// - `density_ratios`: abundance ratio for each rd (1.0 for individuals)
pub(crate) fn collect_all_resonance_data_with_mapping(
    state: &AppState,
) -> (
    Vec<nereids_endf::resonance::ResonanceData>,
    Vec<usize>,
    Vec<f64>,
) {
    let mut all_rd = Vec::new();
    let mut indices = Vec::new();
    let mut ratios = Vec::new();
    let mut density_idx = 0usize;

    for e in &state.isotope_entries {
        if e.enabled && e.resonance_data.is_some() {
            all_rd.push(e.resonance_data.clone().unwrap());
            indices.push(density_idx);
            ratios.push(1.0);
            density_idx += 1;
        }
    }
    for g in &state.isotope_groups {
        if g.enabled && g.overall_status() == EndfStatus::Loaded {
            for m in &g.members {
                if let Some(rd) = &m.resonance_data {
                    all_rd.push(rd.clone());
                    indices.push(density_idx);
                    ratios.push(m.ratio);
                }
            }
            density_idx += 1;
        }
    }
    (all_rd, indices, ratios)
}

#[cfg(test)]
mod tests {
    use super::{
        FitLineParams, build_fit_line, build_overlay_model, compose_fitted_curve,
        counts_resolution_overlay_unsupported, fit_residuals, fitted_physics_energies,
    };
    use egui_plot::{PlotGeometry, PlotItem};
    use nereids_endf::resonance::ResonanceFormalism;
    use nereids_endf::resonance::test_support::u238_with_formalism;
    use nereids_fitting::lm::{FitModel, LmConfig};
    use nereids_fitting::transmission_model::TransmissionFitModel;
    use nereids_physics::doppler_route::{DopplerRoute, IsotopeDopplerRoute, SampledTableReason};
    use nereids_pipeline::pipeline::{
        InputData, SolverConfig, SpectrumFitResult, UnifiedFitConfig, fit_spectrum_typed,
    };

    /// A converged single-density result whose SAMMY energy scale is
    /// `(t0_us, L_scale = 1)` on a 25 m flight path, with nothing else
    /// composed on top of the transmission.
    ///
    /// It discloses the route a real fit of U-238 MLBW on [`line_grid`] at
    /// 293.6 K discloses — the continuous integral — so a redraw of it is
    /// CHECKED against that disclosure. A result carrying no disclosure is
    /// redrawn unchecked, which `build_overlay_model` reports rather than
    /// hides (see `an_undisclosed_route_set_is_reported_not_assumed`).
    fn energy_scale_result(t0_us: f64) -> SpectrumFitResult {
        SpectrumFitResult {
            densities: vec![0.001],
            uncertainties: None,
            reduced_chi_squared: 1.0,
            converged: true,
            iterations: 3,
            temperature_k: None,
            temperature_k_unc: None,
            anorm: 1.0,
            background: [0.0; 3],
            back_d: None,
            back_f: None,
            t0_us: Some(t0_us),
            l_scale: Some(1.0),
            energy_scale_flight_path_m: Some(25.0),
            deviance_per_dof: None,
            baseline: None,
            baseline_e_ref_ev: None,
            warnings: Vec::new(),
            doppler_routes: Some(vec![IsotopeDopplerRoute {
                isotope: u238_with_formalism(ResonanceFormalism::MLBW).isotope,
                route: DopplerRoute::Continuous {
                    formalisms: vec![ResonanceFormalism::MLBW],
                },
            }]),
        }
    }

    /// A grid across the U-238 6.674 eV line.
    fn line_grid() -> Vec<f64> {
        (0..161).map(|i| 6.0 + (i as f64) * 0.01).collect()
    }

    /// The U-238 MLBW transmission at density 1e-3 on `grid`, 293.6 K.
    fn curve_on(rd: &nereids_endf::resonance::ResonanceData, grid: Vec<f64>) -> Vec<f64> {
        TransmissionFitModel::new(
            grid,
            vec![rd.clone()],
            293.6,
            None,
            (vec![0], vec![1.0]),
            None,
            None,
        )
        .unwrap()
        .evaluate(&[0.001])
        .unwrap()
    }

    fn drawn_y(line: &egui_plot::Line<'static>) -> Vec<f64> {
        match line.geometry() {
            PlotGeometry::Points(points) => points.iter().map(|p| p.y).collect(),
            _ => panic!("a fit overlay line is built from explicit points"),
        }
    }

    fn fit_line_of(
        result: &SpectrumFitResult,
        rd: &nereids_endf::resonance::ResonanceData,
        nominal: &[f64],
    ) -> super::FitLine {
        build_fit_line(&FitLineParams {
            result,
            resonance_data: std::slice::from_ref(rd),
            density_indices: &[0],
            density_ratios: &[1.0],
            energies: nominal,
            temperature_k: 293.6,
            x_values: nominal,
            n_plot: nominal.len(),
            instrument: None,
            y_multiplier: None,
        })
        .expect("a converged result with resonance data draws something")
    }

    #[test]
    fn count_overlay_is_suppressed_only_with_active_resolution() {
        assert!(counts_resolution_overlay_unsupported(true, true));
        assert!(!counts_resolution_overlay_unsupported(true, false));
        assert!(!counts_resolution_overlay_unsupported(false, true));
        assert!(!counts_resolution_overlay_unsupported(false, false));
    }

    /// MLBW whose range ends at 8 eV on a 4-6.9 eV grid: continuous at
    /// 293.6 K, demoted at the free-temperature gate. The overlay of a
    /// free-temperature result must take the demoted route at the fitted
    /// temperature, and the fixed-temperature overlay must be the forward
    /// model exactly.
    #[test]
    fn overlay_model_gates_its_routes_the_way_the_fit_did() {
        let mut near_edge = u238_with_formalism(ResonanceFormalism::MLBW);
        near_edge.ranges[0].energy_high = 8.0;
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.0145).collect();
        let build = |free_temperature: bool| {
            build_overlay_model(
                energies.clone(),
                vec![near_edge.clone()],
                293.6,
                None,
                (vec![0], vec![1.0]),
                free_temperature,
                None,
            )
            .unwrap()
            .model
        };

        let free = build(true);
        assert!(matches!(
            free.doppler_routes().unwrap().unwrap()[0],
            DopplerRoute::SampledTable {
                reason: SampledTableReason::WindowCrossesRangeBoundary { .. }
            }
        ));
        let fixed = build(false);
        assert_eq!(
            fixed.doppler_routes().unwrap().unwrap()[0],
            DopplerRoute::Continuous {
                formalisms: vec![ResonanceFormalism::MLBW]
            }
        );
        let forward = TransmissionFitModel::new(
            energies,
            vec![near_edge],
            293.6,
            None,
            (vec![0], vec![1.0]),
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            fixed.evaluate(&[0.001]).unwrap(),
            forward.evaluate(&[0.001]).unwrap()
        );
        let free_curve = free.evaluate(&[0.001]).unwrap();
        assert!(
            free_curve
                .iter()
                .zip(forward.evaluate(&[0.001]).unwrap())
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "the two routes must differ for the gate to matter"
        );
    }

    /// A loaded grid whose tail (9-9.3 eV) leaves an MLBW range ending at
    /// 8 eV demotes the isotope on the whole grid; a fit restricted to
    /// 4-6.9 eV is continuous. The overlay built on the fit's grid takes
    /// the fit's routes; built on the display grid it would not, and the
    /// helper says so instead of drawing a different curve silently.
    #[test]
    fn overlay_on_the_fit_grid_takes_the_fit_routes() {
        let mut near_edge = u238_with_formalism(ResonanceFormalism::MLBW);
        near_edge.ranges[0].energy_high = 8.0;
        let mut display_grid: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.0145).collect();
        display_grid.extend((0..4).map(|i| 9.0 + (i as f64) * 0.1));
        let fit_grid = display_grid[..201].to_vec();

        let truth = TransmissionFitModel::new(
            fit_grid.clone(),
            vec![near_edge.clone()],
            300.0,
            None,
            (vec![0], vec![1.0]),
            None,
            None,
        )
        .unwrap();
        let transmission = truth.evaluate(&[0.001]).unwrap();
        let uncertainty = vec![0.01; transmission.len()];
        let config = UnifiedFitConfig::new(
            fit_grid.clone(),
            vec![near_edge.clone()],
            vec!["U-238".into()],
            300.0,
            None,
            vec![0.0012],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig {
            max_iter: 5,
            ..LmConfig::default()
        }));
        let result = fit_spectrum_typed(
            &InputData::Transmission {
                transmission,
                uncertainty,
            },
            &config,
        )
        .unwrap();
        let disclosed = result.doppler_routes.as_deref().unwrap();
        assert_eq!(
            disclosed[0].route,
            DopplerRoute::Continuous {
                formalisms: vec![ResonanceFormalism::MLBW]
            }
        );

        let overlay = |grid: Vec<f64>| {
            build_overlay_model(
                grid,
                vec![near_edge.clone()],
                300.0,
                None,
                (vec![0], vec![1.0]),
                result.temperature_k.is_some(),
                Some(disclosed),
            )
            .unwrap()
        };
        let on_fit_grid = overlay(fit_grid);
        assert_eq!(on_fit_grid.route_mismatch, None);
        let routes = on_fit_grid.model.doppler_routes().unwrap().unwrap();
        assert_eq!(routes.len(), disclosed.len());
        assert!(
            routes
                .iter()
                .zip(disclosed)
                .all(|(overlay, fit)| fit.route.same_kind(overlay))
        );

        let on_display_grid = overlay(display_grid);
        let mismatch = on_display_grid
            .route_mismatch
            .expect("the display grid demotes the isotope, which the fit did not");
        assert!(
            mismatch.contains("grid energy 9.00e0 eV lies outside the resolved MLBW range"),
            "{mismatch}"
        );
        assert!(
            mismatch.contains("continuous free-gas integral"),
            "{mismatch}"
        );
    }

    /// An energy-scale fit evaluated the physics at the CALIBRATED energies.
    /// The drawn curve must be that one — on a U-238 line a 1 µs t₀ moves
    /// the transmission by more than 0.05, so redrawing on the nominal grid
    /// is a visibly different curve presented as the fit.
    #[test]
    fn fit_line_redraws_the_physics_on_the_fitted_energy_scale() {
        let rd = u238_with_formalism(ResonanceFormalism::MLBW);
        let nominal = line_grid();
        let result = energy_scale_result(1.0);

        let calibrated = curve_on(
            &rd,
            result
                .corrected_energies(&nominal)
                .expect("the energy scale was fitted")
                .expect("and is not degenerate"),
        );
        let uncalibrated = curve_on(&rd, nominal.clone());
        let displacement = calibrated
            .iter()
            .zip(&uncalibrated)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            displacement > 0.05,
            "the calibrated and nominal grids must give visibly different curves for this \
             test to mean anything, got {displacement}"
        );

        let fit = fit_line_of(&result, &rd, &nominal);
        assert_eq!(fit.warning, None);
        assert_eq!(drawn_y(fit.line.as_ref().unwrap()), calibrated);
    }

    /// The drawn curve is the whole fitted forward model — Anorm, the SAMMY
    /// background and the multiplicative baseline — and it is
    /// `compose_fitted_curve` that produces it, which is what the residual
    /// dock subtracts from the measurement.
    #[test]
    fn the_drawn_curve_is_the_fitted_composition_not_bare_transmission() {
        let rd = u238_with_formalism(ResonanceFormalism::MLBW);
        let nominal = line_grid();
        let mut result = energy_scale_result(0.0);
        result.anorm = 0.97;
        result.background = [0.02, 0.01, 0.003];
        result.back_d = Some(0.05);
        result.back_f = Some(1.5);
        result.baseline = Some([1.01, -0.02, 0.004]);
        result.baseline_e_ref_ev = Some(6.7);

        // The identity energy scale isolates the composition.
        let transmission = curve_on(&rd, nominal.clone());
        let composed = compose_fitted_curve(&result, &nominal, &transmission);
        let gap = composed
            .iter()
            .zip(&transmission)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            gap > 0.05,
            "the composition must differ from bare transmission for this test to mean \
             anything, got {gap}"
        );

        let fit = fit_line_of(&result, &rd, &nominal);
        assert_eq!(drawn_y(fit.line.as_ref().unwrap()), composed);
    }

    /// A `t0` past the grid's shortest flight time maps it to non-physical
    /// energies. The fitted curve cannot be reproduced then, so none is
    /// drawn and the caller is told why — the alternative is the
    /// un-calibrated curve, silently.
    #[test]
    fn a_degenerate_energy_scale_hides_the_curve_and_says_why() {
        let rd = u238_with_formalism(ResonanceFormalism::MLBW);
        let nominal = line_grid();
        let result = energy_scale_result(1.0e4);
        assert!(fitted_physics_energies(&result, &nominal).is_err());

        let fit = fit_line_of(&result, &rd, &nominal);
        assert!(fit.line.is_none());
        let warning = fit.warning.expect("the refusal reaches the caller");
        assert!(
            warning.contains("does not map this grid to physical energies"),
            "{warning}"
        );
    }

    /// A grid with no energy scale fitted is its own physics grid — `None`
    /// on the result means "not fitted", not "fitted to the identity".
    #[test]
    fn an_unfitted_energy_scale_leaves_the_grid_alone() {
        let mut result = energy_scale_result(1.0);
        result.t0_us = None;
        result.l_scale = None;
        result.energy_scale_flight_path_m = None;
        let nominal = line_grid();
        assert_eq!(fitted_physics_energies(&result, &nominal).unwrap(), nominal);
    }

    /// The residual dock subtracts the same composed curve the overlay
    /// draws. Against bare Beer-Lambert transmission the whole background
    /// and baseline would land in the residual — and in the RMS and Max|r|
    /// the dock reports — for any fit with those boxes ticked.
    #[test]
    fn residuals_are_taken_against_the_composed_fit() {
        let rd = u238_with_formalism(ResonanceFormalism::MLBW);
        let nominal = line_grid();
        let mut result = energy_scale_result(0.0);
        result.anorm = 0.97;
        result.background = [0.1, 0.01, 0.003];

        // A measurement that IS the fitted model: every residual is zero.
        let transmission = curve_on(&rd, nominal.clone());
        let measured = compose_fitted_curve(&result, &nominal, &transmission);
        let stats = fit_residuals(&result, &nominal, &transmission, &measured);
        assert_eq!(stats.residuals.len(), nominal.len());
        assert_eq!(stats.max_abs, 0.0, "a perfect fit has no residual");
        assert_eq!(stats.rms, 0.0);

        // The same measurement against bare transmission would report the
        // background as signal.
        let bare = fit_residuals(
            &SpectrumFitResult {
                anorm: 1.0,
                background: [0.0; 3],
                ..energy_scale_result(0.0)
            },
            &nominal,
            &transmission,
            &measured,
        );
        assert!(
            bare.max_abs > 0.05,
            "the omitted composition must be a visible error, got {}",
            bare.max_abs
        );
    }

    /// A result that discloses no routes cannot be checked against the
    /// redraw. That is the per-pixel result of a spatial map whose
    /// converged pixels disagreed — the map then withholds the map-wide
    /// set and every pixel carries `None`. The redraw still happens, so
    /// the missing check is reported; an unbroadened redraw has no route
    /// to have got wrong and stays silent.
    #[test]
    fn an_undisclosed_route_set_is_reported_not_assumed() {
        let rd = u238_with_formalism(ResonanceFormalism::MLBW);
        let energies = line_grid();
        let overlay = |temperature_k: f64| {
            build_overlay_model(
                energies.clone(),
                vec![rd.clone()],
                temperature_k,
                None,
                (vec![0], vec![1.0]),
                false,
                None,
            )
            .unwrap()
        };

        let message = overlay(293.6)
            .route_mismatch
            .expect("a broadening redraw with no disclosure to check it against");
        assert!(message.contains("discloses no routes"), "{message}");
        assert!(
            message.contains("U-238") && message.contains("continuous free-gas integral"),
            "the redrawn route is named so the user can judge it: {message}"
        );

        // 0 K: every route is `Unbroadened`, and an unbroadened redraw
        // cannot have taken the wrong broadening route.
        assert_eq!(overlay(0.0).route_mismatch, None);
    }

    /// Routes are compared WITH their isotopes: two different isotopes can
    /// take the same kind of route, and a curve computed for another
    /// isotope is not the fitted curve however much the routes agree.
    #[test]
    fn overlay_routes_are_compared_with_their_isotopes() {
        let rd = u238_with_formalism(ResonanceFormalism::MLBW);
        let energies = line_grid();
        let mlbw = || DopplerRoute::Continuous {
            formalisms: vec![ResonanceFormalism::MLBW],
        };
        let overlay = |disclosed: &[IsotopeDopplerRoute]| {
            build_overlay_model(
                energies.clone(),
                vec![rd.clone()],
                293.6,
                None,
                (vec![0], vec![1.0]),
                false,
                Some(disclosed),
            )
            .unwrap()
        };

        let other_isotope = [IsotopeDopplerRoute {
            isotope: nereids_core::types::Isotope::new(72, 178).unwrap(),
            route: mlbw(),
        }];
        let message = overlay(&other_isotope)
            .route_mismatch
            .expect("another isotope on the same kind of route is not the fitted curve");
        assert!(
            message.contains("U-238") && message.contains("Hf-178"),
            "the message must name the isotope on both sides: {message}"
        );

        let same_isotope = [IsotopeDopplerRoute {
            isotope: rd.isotope,
            route: mlbw(),
        }];
        assert_eq!(overlay(&same_isotope).route_mismatch, None);
    }
}
