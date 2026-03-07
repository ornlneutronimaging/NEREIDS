//! Design system widgets: cards, underline tabs, badges, buttons, drop zones.
//!
//! Theme-aware widgets resolve colors from the egui context (`ThemeColors`)
//! or from `semantic` constants.
//! Prototype reference: `.prototypes/D_hybrid_v4.html` CSS §Cards & Forms,
//! §Tabs, §Badges, §Content Area, §Toolbar, §Drop Zones.

use crate::state::{
    AppState, EndfFetchResult, EndfStatus, GuidedStep, IsotopeEntry, ResolutionMode, SpectrumAxis,
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
        .stroke(Stroke::new(1.0, tc.border))
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
        Stroke::new(1.0, tc.border),
    );

    // 2px accent underline on active tab
    if let Some(rect) = active_rect {
        painter.line_segment(
            [
                egui::pos2(rect.left(), baseline),
                egui::pos2(rect.right(), baseline),
            ],
            Stroke::new(2.0, tc.accent),
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
        (tc.bg2, tc.fg2, Stroke::new(1.0, tc.border))
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
            2.0,
            Color32::from_rgba_unmultiplied(
                semantic::GREEN.r(),
                semantic::GREEN.g(),
                semantic::GREEN.b(),
                13,
            ),
        )
    } else {
        (tc.fg3, 1.5, Color32::TRANSPARENT)
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
        .stroke(Stroke::new(1.0, tc.border))
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
/// Used identically in Configure, Forward Model, and Detectability.
pub fn resolution_card(
    ui: &mut Ui,
    enabled: &mut bool,
    mode: &mut ResolutionMode,
    flight_path_m: f64,
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
                    if ui.button("Load resolution file\u{2026}").clicked()
                        && let Some(file) = rfd::FileDialog::new()
                            .add_filter("Resolution", &["txt", "dat"])
                            .pick_file()
                    {
                        if let Some(path_str) = file.to_str() {
                            match TabulatedResolution::from_file(path_str, flight_path_m) {
                                Ok(tab) => {
                                    *path = file;
                                    *data = Some(Arc::new(tab));
                                    *error = None;
                                    changed = true;
                                }
                                Err(e) => {
                                    *path = file;
                                    *data = None;
                                    *error = Some(format!("{e}"));
                                    changed = true;
                                }
                            }
                        } else {
                            *path = file;
                            *data = None;
                            *error = Some(
                                "File path is not valid UTF-8; please choose a different file"
                                    .into(),
                            );
                            changed = true;
                        }
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

// ── ENDF Library Name ──────────────────────────────────────────

/// Map an `EndfLibrary` variant to its display name.
pub fn library_name(lib: EndfLibrary) -> &'static str {
    match lib {
        EndfLibrary::EndfB8_0 => "ENDF/B-VIII.0",
        EndfLibrary::EndfB8_1 => "ENDF/B-VIII.1",
        EndfLibrary::Jeff3_3 => "JEFF-3.3",
        EndfLibrary::Jendl5 => "JENDL-5",
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

/// Background worker for ENDF data fetching.
///
/// Runs inside a `std::thread::spawn` closure — iterates `work` items,
/// fetches + parses each, and sends results on `tx`. Supports cancellation.
pub(crate) fn endf_fetch_worker(
    work: Vec<(usize, Isotope, String, EndfLibrary)>,
    cancel: Arc<AtomicBool>,
    tx: mpsc::Sender<EndfFetchResult>,
) {
    let retriever = nereids_endf::retrieval::EndfRetriever::new();
    for (index, isotope, symbol, library) in work {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let Some(mat) = nereids_endf::retrieval::mat_number(&isotope) else {
            continue;
        };
        let result = match retriever.get_endf_file(&isotope, library, mat) {
            Ok((_path, endf_text)) => match nereids_endf::parser::parse_endf_file2(&endf_text) {
                Ok(data) => Ok(data),
                Err(e) => Err(format!("Parse error for {symbol}: {e}")),
            },
            Err(e) => Err(format!("Fetch error for {symbol}: {e}")),
        };
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let _ = tx.send(EndfFetchResult {
            index,
            symbol,
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
            let params = ResolutionParams::new(flight_path_m, *delta_t_us, *delta_l_m)
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
/// Iterates all enabled isotope entries with resonance data and draws a `VLine`
/// for each resonance within the visible x-axis range.
pub(crate) fn draw_resonance_dips(
    plot_ui: &mut egui_plot::PlotUi,
    entries: &[IsotopeEntry],
    x_values: &[f64],
) {
    let (x_min, x_max) = x_values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    for entry in entries {
        if !entry.enabled {
            continue;
        }
        let Some(ref res_data) = entry.resonance_data else {
            continue;
        };
        for range in &res_data.ranges {
            for lg in &range.l_groups {
                for res in &lg.resonances {
                    if res.energy >= x_min && res.energy <= x_max {
                        plot_ui.vline(
                            VLine::new(format!("{} {:.1}eV", entry.symbol, res.energy), res.energy)
                                .color(RESONANCE_DIP_COLOR)
                                .width(0.5),
                        );
                    }
                }
            }
        }
    }
}

/// Build a fit overlay line from a `SpectrumFitResult`.
///
/// Returns `None` if the fit didn't converge, no enabled isotopes have resonance data,
/// or model construction fails.
pub(crate) fn build_fit_line(
    result: &nereids_pipeline::pipeline::SpectrumFitResult,
    entries: &[IsotopeEntry],
    energies: &[f64],
    temperature_k: f64,
    x_values: &[f64],
    n_plot: usize,
) -> Option<Line<'static>> {
    if !result.converged {
        return None;
    }
    let resonance_data: Vec<_> = entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .filter_map(|e| e.resonance_data.clone())
        .collect();
    if resonance_data.is_empty() {
        return None;
    }
    let overlay_temp = result.temperature_k.unwrap_or(temperature_k);
    let model = nereids_fitting::transmission_model::TransmissionFitModel::new(
        energies.to_vec(),
        resonance_data,
        overlay_temp,
        None,
        (0..result.densities.len()).collect(),
        None,
        None,
    )
    .ok()?;

    use nereids_fitting::lm::FitModel;
    let fitted_t = model.evaluate(&result.densities);
    let n_fit = n_plot.min(fitted_t.len());
    let fit_points: PlotPoints = (0..n_fit)
        .filter(|&i| x_values[i].is_finite())
        .map(|i| [x_values[i], fitted_t[i]])
        .collect();
    Some(Line::new("Fit", fit_points).width(2.0))
}
