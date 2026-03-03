//! Design system widgets: cards, underline tabs, badges, buttons, drop zones.
//!
//! Theme-aware widgets resolve colors from the egui context (`ThemeColors`)
//! or from `semantic` constants.
//! Prototype reference: `.prototypes/D_hybrid_v4.html` CSS §Cards & Forms,
//! §Tabs, §Badges, §Content Area, §Toolbar, §Drop Zones.

use crate::theme::{ThemeColors, semantic};
use egui::{Color32, CornerRadius, Margin, Rect, Response, RichText, Sense, Shadow, Stroke, Ui};

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
