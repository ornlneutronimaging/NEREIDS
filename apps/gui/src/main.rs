//! NEREIDS — Neutron Resonance Imaging Data Analysis
//!
//! Desktop GUI application for interactive neutron transmission analysis,
//! isotope identification, and spatial composition mapping.

mod app;
mod guided;
mod pipeline;
mod state;
mod studio;
mod telemetry;
mod theme;
mod widgets;

fn main() -> eframe::Result {
    // Reserve one CPU core for the GUI thread.  Without this, rayon's global
    // pool uses all cores during spatial mapping, starving the main thread
    // and freezing the progress bar.
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok(); // harmless if already initialised

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
