//! Main application structure and egui App implementation.

use crate::panels;
use crate::state::{AppState, Tab};

/// NEREIDS desktop application.
pub struct NereidsApp {
    pub state: AppState,
}

impl NereidsApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            state: AppState::default(),
        }
    }
}

impl eframe::App for NereidsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check for pixel clicks from map panel
        if let Some((y, x)) = ctx.data(|d| d.get_temp::<(usize, usize)>(egui::Id::new("clicked_pixel"))) {
            self.state.selected_pixel = Some((y, x));
            self.state.pixel_fit_result = None;
            ctx.data_mut(|d| {
                d.remove_temp::<(usize, usize)>(egui::Id::new("clicked_pixel"));
            });
        }

        // Left sidebar: data loading
        egui::SidePanel::left("data_panel")
            .default_width(280.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    panels::data::data_panel(ui, &mut self.state);
                });
            });

        // Right sidebar: fitting controls
        egui::SidePanel::right("fitting_panel")
            .default_width(260.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    panels::fitting::fitting_panel(ui, &mut self.state);
                });
            });

        // Bottom status bar
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.state.status_message);
            });
        });

        // Central panel: spectrum or map
        egui::CentralPanel::default().show(ctx, |ui| {
            // Tab bar
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.state.active_tab, Tab::Spectrum, "Spectrum");
                ui.selectable_value(&mut self.state.active_tab, Tab::Map, "Map");
            });
            ui.separator();

            match self.state.active_tab {
                Tab::Spectrum => panels::spectrum::spectrum_panel(ui, &mut self.state),
                Tab::Map => panels::map::map_panel(ui, &mut self.state),
            }
        });
    }
}
