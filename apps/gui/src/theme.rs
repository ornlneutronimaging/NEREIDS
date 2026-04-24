//! Theme system: dark/light/auto palettes, semantic colors, font configuration.

use egui::Color32;

/// Dark mode palette.
pub mod dark {
    use egui::Color32;
    pub const BG: Color32 = Color32::from_rgb(0x1c, 0x1c, 0x1e);
    pub const BG2: Color32 = Color32::from_rgb(0x2c, 0x2c, 0x2e);
    pub const BG3: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x3c);
    pub const FG: Color32 = Color32::from_rgb(0xf5, 0xf5, 0xf7);
    pub const FG2: Color32 = Color32::from_rgb(0x98, 0x98, 0x9d);
    pub const FG3: Color32 = Color32::from_rgb(0x63, 0x63, 0x66);
    pub const ACCENT: Color32 = Color32::from_rgb(0x0a, 0x84, 0xff);
    pub const BORDER: Color32 = Color32::from_rgb(0x48, 0x48, 0x4a);
}

/// Light mode palette.
pub mod light {
    use egui::Color32;
    pub const BG: Color32 = Color32::from_rgb(0xf5, 0xf5, 0xf7);
    pub const BG2: Color32 = Color32::from_rgb(0xff, 0xff, 0xff);
    pub const BG3: Color32 = Color32::from_rgb(0xe8, 0xe8, 0xed);
    pub const FG: Color32 = Color32::from_rgb(0x1d, 0x1d, 0x1f);
    pub const FG2: Color32 = Color32::from_rgb(0x6e, 0x6e, 0x73);
    pub const FG3: Color32 = Color32::from_rgb(0x86, 0x86, 0x8b);
    pub const ACCENT: Color32 = Color32::from_rgb(0x00, 0x71, 0xe3);
    pub const BORDER: Color32 = Color32::from_rgb(0xd2, 0xd2, 0xd7);
}

/// Semantic colors (fixed across themes).
pub mod semantic {
    use egui::Color32;
    pub const GREEN: Color32 = Color32::from_rgb(0x34, 0xc7, 0x59);
    pub const RED: Color32 = Color32::from_rgb(0xff, 0x3b, 0x30);
    pub const ORANGE: Color32 = Color32::from_rgb(0xff, 0x95, 0x00);
    pub const YELLOW: Color32 = Color32::from_rgb(0xff, 0xcc, 0x00);
}

/// Resolved theme colors for the current mode.
#[derive(Clone, Copy)]
pub struct ThemeColors {
    pub bg: Color32,
    pub bg2: Color32,
    pub bg3: Color32,
    pub fg: Color32,
    pub fg2: Color32,
    pub fg3: Color32,
    pub accent: Color32,
    pub border: Color32,
}

impl ThemeColors {
    pub fn from_dark_mode(is_dark: bool) -> Self {
        if is_dark {
            Self {
                bg: dark::BG,
                bg2: dark::BG2,
                bg3: dark::BG3,
                fg: dark::FG,
                fg2: dark::FG2,
                fg3: dark::FG3,
                accent: dark::ACCENT,
                border: dark::BORDER,
            }
        } else {
            Self {
                bg: light::BG,
                bg2: light::BG2,
                bg3: light::BG3,
                fg: light::FG,
                fg2: light::FG2,
                fg3: light::FG3,
                accent: light::ACCENT,
                border: light::BORDER,
            }
        }
    }

    pub fn from_ctx(ctx: &egui::Context) -> Self {
        Self::from_dark_mode(ctx.style().visuals.dark_mode)
    }
}

use crate::state::ThemePreference;

/// Resolve the theme preference to a concrete dark/light boolean.
///
/// For `Auto`, uses the system theme if available, otherwise falls back
/// to the current egui visuals.
pub fn resolve_dark_mode(ctx: &egui::Context, preference: ThemePreference) -> bool {
    match preference {
        ThemePreference::Dark => true,
        ThemePreference::Light => false,
        ThemePreference::Auto => match ctx.system_theme() {
            Some(theme) => theme == egui::Theme::Dark,
            None => ctx.style().visuals.dark_mode,
        },
    }
}

/// Apply the theme to the egui context based on user preference.
pub fn apply_theme(ctx: &egui::Context, preference: ThemePreference) {
    let is_dark = resolve_dark_mode(ctx, preference);

    let colors = ThemeColors::from_dark_mode(is_dark);
    let mut visuals = if is_dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };

    visuals.override_text_color = Some(colors.fg);
    visuals.panel_fill = colors.bg;
    visuals.window_fill = colors.bg2;
    visuals.faint_bg_color = colors.bg2;

    visuals.widgets.noninteractive.bg_fill = colors.bg2;
    visuals.widgets.noninteractive.fg_stroke.color = colors.fg;
    visuals.widgets.noninteractive.bg_stroke.color = colors.border;

    visuals.widgets.inactive.fg_stroke.color = colors.fg2;
    visuals.widgets.hovered.bg_stroke.color = colors.accent;
    visuals.widgets.active.bg_fill = colors.accent;

    visuals.selection.bg_fill = colors.accent.gamma_multiply(0.3);
    visuals.selection.stroke.color = colors.accent;

    // Apple design language: 6px rounding
    let corner_radius = egui::CornerRadius::same(6);
    visuals.widgets.noninteractive.corner_radius = corner_radius;
    visuals.widgets.inactive.corner_radius = corner_radius;
    visuals.widgets.hovered.corner_radius = corner_radius;
    visuals.widgets.active.corner_radius = corner_radius;
    visuals.widgets.open.corner_radius = corner_radius;
    visuals.window_corner_radius = corner_radius;
    visuals.menu_corner_radius = corner_radius;

    ctx.set_visuals(visuals);
}

/// Configure font sizes (called once at startup).
pub fn configure_fonts(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::new(13.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::new(16.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Small,
        egui::FontId::new(11.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Monospace,
        egui::FontId::new(12.0, egui::FontFamily::Monospace),
    );
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 5.0);
    ctx.set_style(style);
}
