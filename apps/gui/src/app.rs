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
        // Poll background tasks
        poll_pending_tasks(&mut self.state);

        // Keep repainting while background work is in progress
        if self.state.is_fitting || self.state.is_fetching_endf {
            ctx.request_repaint();
        }

        // Check for pixel clicks from map panel
        if let Some((y, x)) =
            ctx.data(|d| d.get_temp::<(usize, usize)>(egui::Id::new("clicked_pixel")))
        {
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

/// Poll background task channels and apply results to state.
fn poll_pending_tasks(state: &mut AppState) {
    // Poll spatial map result
    if let Some(ref rx) = state.pending_spatial {
        match rx.try_recv() {
            Ok(result) => {
                state.status_message = format!(
                    "Spatial map: {}/{} converged",
                    result.n_converged, result.n_total
                );
                state.spatial_result = Some(result);
                state.is_fitting = false;
                state.active_tab = Tab::Map;
                state.pending_spatial = None;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                state.status_message = "Spatial map task failed".into();
                state.is_fitting = false;
                state.pending_spatial = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {} // Still running
        }
    }

    // Poll ENDF fetch results (streamed one per isotope)
    if let Some(ref rx) = state.pending_endf {
        let mut disconnected = false;
        // Drain all available results this frame
        loop {
            match rx.try_recv() {
                Ok(fetch) => {
                    if let Some(entry) = state.isotope_entries.get_mut(fetch.index) {
                        match fetch.result {
                            Ok(data) => {
                                entry.resonance_data = Some(data);
                                state.status_message = format!("Loaded {}", fetch.symbol);
                            }
                            Err(msg) => {
                                state.status_message = msg;
                            }
                        }
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        // Only finalize when the sender is dropped (thread finished)
        if disconnected {
            if !state
                .isotope_entries
                .iter()
                .any(|e| e.enabled && e.resonance_data.is_none())
            {
                state.status_message = "All ENDF data loaded".into();
            }
            state.is_fetching_endf = false;
            state.pending_endf = None;
        }
    }
}
