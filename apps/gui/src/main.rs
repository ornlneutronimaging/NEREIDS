//! NEREIDS — Neutron Resonance Imaging Data Analysis
//!
//! Desktop GUI application for interactive neutron transmission analysis,
//! isotope identification, and spatial composition mapping.

mod app;
mod guided;
mod logging;
mod pipeline;
mod project;
mod state;
mod studio;
mod telemetry;
mod theme;
mod widgets;

fn main() -> eframe::Result {
    // Init logging first so panics during option/storage/font setup
    // are captured. The non-blocking writer's WorkerGuard is stashed
    // inside the logging module; `logging::shutdown()` (called from
    // `app::on_exit` / the macOS close-requested handler) drops it
    // before `std::process::exit(0)` so buffered records flush.
    logging::init();

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
            if let Some(ref path) = project_arg {
                if path.exists() {
                    tracing::info!(path = %path.display(), "loading project from CLI arg");
                    project::load_project_from_path(&mut app.state, path);
                } else {
                    tracing::warn!(
                        path = %path.display(),
                        "CLI project arg does not exist; ignoring"
                    );
                }
            }
            Ok(Box::new(app))
        }),
    )
}
