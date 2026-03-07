//! NEREIDS — Neutron Resonance Imaging Data Analysis
//!
//! Desktop GUI application for interactive neutron transmission analysis,
//! isotope identification, and spatial composition mapping.

mod app;
mod guided;
mod pipeline;
mod project;
mod state;
mod studio;
mod telemetry;
mod theme;
mod widgets;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("NEREIDS — Neutron Resonance Imaging"),
        ..Default::default()
    };

    // Check for a project file argument (e.g. `nereids-gui project.nrd.h5`)
    let project_arg = std::env::args().nth(1).map(std::path::PathBuf::from);

    eframe::run_native(
        "NEREIDS",
        options,
        Box::new(move |cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            theme::configure_fonts(&cc.egui_ctx);
            let mut app = app::NereidsApp::new(cc);
            if let Some(ref path) = project_arg
                && path.exists()
            {
                project::load_project_from_path(&mut app.state, path);
            }
            Ok(Box::new(app))
        }),
    )
}
