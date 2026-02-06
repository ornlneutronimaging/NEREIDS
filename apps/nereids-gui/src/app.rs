//! Main application state and egui update loop.

pub struct NereidsApp;

impl NereidsApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self
    }
}

impl eframe::App for NereidsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("NEREIDS");
            ui.label("NEutron REsonance Imaging Diagnostic Suite");
            ui.separator();
            ui.label("GUI under development.");
        });
    }
}
