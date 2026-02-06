//! NEREIDS GUI application.

mod app;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("NEREIDS"),
        ..Default::default()
    };

    eframe::run_native(
        "NEREIDS",
        options,
        Box::new(|cc| Ok(Box::new(app::NereidsApp::new(cc)))),
    )
}
