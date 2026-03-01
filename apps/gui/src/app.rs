//! Main application structure and egui App implementation.

use crate::guided;
use crate::state::{AppState, Tab, UiMode};
use crate::studio;
use crate::theme;
use crate::widgets;

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
        // Apply theme
        theme::apply_theme(ctx, self.state.theme_preference);

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

        // Top toolbar
        widgets::toolbar::toolbar(ctx, &mut self.state);

        // Bottom status bar
        widgets::statusbar::status_bar(ctx, &self.state);

        // Main content area
        match self.state.ui_mode {
            UiMode::Guided => {
                guided::sidebar::guided_sidebar(ctx, &mut self.state);
                egui::CentralPanel::default().show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        guided::guided_content(ui, &mut self.state);
                    });
                });
            }
            UiMode::Studio => {
                studio::studio_content(ctx, &mut self.state);
            }
        }
    }
}

/// Poll background task channels and apply results to state.
fn poll_pending_tasks(state: &mut AppState) {
    // Poll spatial map result
    if let Some(ref rx) = state.pending_spatial {
        match rx.try_recv() {
            Ok(Ok(result)) => {
                state.status_message = format!(
                    "Spatial map: {}/{} converged",
                    result.n_converged, result.n_total
                );
                state.spatial_result = Some(result);
                state.is_fitting = false;
                state.active_tab = Tab::Map;
                state.pending_spatial = None;
            }
            Ok(Err(err_msg)) => {
                state.status_message = format!("Spatial map error: {err_msg}");
                state.is_fitting = false;
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
