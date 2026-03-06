//! Memory telemetry for process-level resource monitoring.
//!
//! Adapted from rustpix's `MemoryTelemetry` pattern.  Refreshes RSS
//! at a fixed interval (750 ms) to avoid per-frame overhead.

use crate::state::AppState;
use sysinfo::{Pid, ProcessesToUpdate, System, get_current_pid};

/// Process memory telemetry, refreshed at a fixed interval.
pub struct MemoryTelemetry {
    system: System,
    pid: Option<Pid>,
    last_refresh: f64,
    pub rss_bytes: u64,
}

impl MemoryTelemetry {
    pub fn new() -> Self {
        let mut system = System::new();
        let pid = get_current_pid().ok();
        // Initial refresh so we have data immediately.
        if let Some(pid) = pid {
            system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        }
        let rss = pid
            .and_then(|p| system.process(p))
            .map_or(0, |p| p.memory());
        Self {
            system,
            pid,
            last_refresh: -1.0,
            rss_bytes: rss,
        }
    }

    /// Refresh process memory stats if the interval has elapsed.
    pub fn refresh(&mut self, now: f64) {
        const INTERVAL: f64 = 0.75;
        if now - self.last_refresh < INTERVAL {
            return;
        }
        self.last_refresh = now;
        if let Some(pid) = self.pid {
            self.system
                .refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
            if let Some(process) = self.system.process(pid) {
                self.rss_bytes = process.memory();
            }
        }
    }
}

/// Estimated buffer sizes for the tooltip breakdown.
pub fn memory_breakdown(state: &AppState) -> Vec<(&'static str, u64)> {
    let mut items = Vec::new();

    if let Some(ref data) = state.sample_data {
        let s = data.shape();
        items.push(("Sample data", (s[0] * s[1] * s[2] * 8) as u64));
    }
    if let Some(ref data) = state.open_beam_data {
        let s = data.shape();
        items.push(("Open beam", (s[0] * s[1] * s[2] * 8) as u64));
    }
    if let Some(ref norm) = state.normalized {
        let s = norm.transmission.shape();
        // transmission + uncertainty = 2 arrays
        items.push(("Normalized", (s[0] * s[1] * s[2] * 8 * 2) as u64));
    }
    if let Some(ref result) = state.spatial_result {
        for map in &result.density_maps {
            let s = map.shape();
            items.push(("Density map", (s[0] * s[1] * 8) as u64));
        }
        let s = result.converged_map.shape();
        items.push(("Convergence", (s[0] * s[1]) as u64)); // bool = 1 byte
        let s = result.chi_squared_map.shape();
        items.push(("Chi² map", (s[0] * s[1] * 8) as u64));
    }
    if let Some(ref e) = state.energies {
        items.push(("Energy grid", (e.len() * 8) as u64));
    }
    if let Some(ref v) = state.spectrum_values {
        items.push(("Spectrum values", (v.len() * 8) as u64));
    }

    items
}

/// Format bytes as human-readable string (KB/MB/GB).
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1 KB");
        assert_eq!(format_bytes(1536), "2 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(1073741824), "1.0 GB");
        assert_eq!(format_bytes(1610612736), "1.5 GB");
    }

    #[test]
    fn test_memory_breakdown_empty_state() {
        let state = AppState::default();
        let breakdown = memory_breakdown(&state);
        assert!(breakdown.is_empty());
    }
}
