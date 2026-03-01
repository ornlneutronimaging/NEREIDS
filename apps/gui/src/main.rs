//! NEREIDS — Neutron Resonance Imaging Data Analysis
//!
//! Desktop GUI application for interactive neutron transmission analysis,
//! isotope identification, and spatial composition mapping.

mod app;
mod guided;
mod state;
mod studio;
mod theme;
mod widgets;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("NEREIDS — Neutron Resonance Imaging"),
        ..Default::default()
    };

    eframe::run_native(
        "NEREIDS",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            theme::configure_fonts(&cc.egui_ctx);
            Ok(Box::new(app::NereidsApp::new(cc)))
        }),
    )
}
