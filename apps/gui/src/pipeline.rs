//! Unified pipeline executor for re-running stages from a dirty point.
//!
//! The pipeline walks the active `state.pipeline` entries starting from
//! `from_step` and dispatches each stage.  Synchronous stages (Rebin,
//! Normalize) run inline; the Analyze stage spawns a background thread
//! (same as `analyze::run_spatial_map`).
//!
//! This module is the single source of truth for "re-run from step X" logic,
//! shared by Guided mode's step-by-step flow and Studio's re-run button.

use crate::state::{AppState, GuidedStep};

/// Execute the pipeline from the earliest dirty stage.
///
/// Returns `Ok(false)` if nothing is dirty (no re-run needed).
/// Returns `Ok(true)` if stages were dispatched.
pub fn run_from_dirty(state: &mut AppState) -> Result<bool, String> {
    match state.dirty_from {
        Some(step) => {
            run_pipeline(state, step)?;
            Ok(true)
        }
        None => Ok(false),
    }
}

/// Execute the pipeline from `from_step` forward.
///
/// - Walks `state.pipeline` entries whose `step` is ≥ `from_step`.
/// - Synchronous stages (Rebin, Normalize) run inline.
/// - Analyze launches a background task via `run_spatial_map`.
/// - Clears `dirty_from` on successful dispatch.
///
/// Returns `Err` with a user-facing message if a stage fails.
pub fn run_pipeline(state: &mut AppState, from_step: GuidedStep) -> Result<(), String> {
    let from_order = from_step as u8;

    // Collect the stages we need to run (clone to avoid borrow issues).
    let stages: Vec<GuidedStep> = state
        .pipeline
        .iter()
        .map(|e| e.step)
        .filter(|&s| (s as u8) >= from_order)
        .collect();

    // Track whether Normalize was explicitly dispatched.
    let mut normalized_ran = false;

    for step in &stages {
        match step {
            GuidedStep::Rebin => {
                run_rebin(state);
            }
            GuidedStep::Normalize => {
                run_normalize(state)?;
                normalized_ran = true;
            }
            GuidedStep::Analyze => {
                // Transmission pipelines omit the Normalize step, but
                // beamline edits still mark dirty from Normalize because
                // the energy grid depends on flight_path / delay.  Ensure
                // normalization + energy grid is fresh before Analyze.
                if !normalized_ran && from_order <= (GuidedStep::Normalize as u8) {
                    run_normalize(state)?;
                }

                // Analyze is async — launch and return.  The GUI polls the
                // channel for completion.  Dirty state is cleared in
                // poll_pending_tasks on success; on error/cancel it stays
                // dirty so the user can re-run.
                crate::guided::analyze::run_spatial_map(state);
                return Ok(());
            }
            // Other stages (Configure, Load, Bin, Results, etc.) are either
            // UI-only or not re-runnable, so we skip them.
            _ => {}
        }
    }

    state.clear_dirty();
    Ok(())
}

/// Re-run the rebin stage.
///
/// Only applies if `rebin_factor > 1` and rebin hasn't already been applied.
/// In a re-run scenario the data has already been loaded from disk, so we
/// just need to apply the rebinning again.
fn run_rebin(_state: &mut AppState) {
    // Re-running rebin requires reloading original data first, then
    // re-applying the factor.  Currently a no-op — the pipeline falls
    // through to Normalize/Analyze which use whatever data is loaded.
}

/// Execute the normalize stage synchronously.
///
/// Delegates to `prepare_transmission` for pre-normalized data or
/// computes normalization for TiffPair.
///
/// Always clears cached normalization + energy grid first so that
/// parameter changes (beamline, etc.) take effect.
fn run_normalize(state: &mut AppState) -> Result<(), String> {
    use crate::state::InputMode;

    // Force recomputation — the whole point of re-running is that upstream
    // parameters changed.  Without this, the `is_none()` guards short-circuit
    // and we silently keep stale data.
    state.normalized = None;
    state.energies = None;

    match state.input_mode {
        InputMode::TransmissionTiff | InputMode::Hdf5Histogram | InputMode::Hdf5Event => {
            if state.sample_data.is_some() {
                crate::guided::normalize::prepare_transmission(state);
            }
            if state.normalized.is_none() {
                return Err("Failed to prepare transmission data".into());
            }
        }
        InputMode::TiffPair => {
            if state.sample_data.is_none() || state.open_beam_data.is_none() {
                return Err("Missing sample or open beam data for normalization".into());
            }
            crate::guided::normalize::normalize_data(state);
            if state.normalized.is_none() {
                return Err("Normalization failed — check status message".into());
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_ordering() {
        assert!((GuidedStep::Normalize as u8) < (GuidedStep::Analyze as u8));
        assert!((GuidedStep::Rebin as u8) < (GuidedStep::Normalize as u8));
        assert!((GuidedStep::Load as u8) < (GuidedStep::Rebin as u8));
        assert!((GuidedStep::Configure as u8) < (GuidedStep::Load as u8));
    }

    #[test]
    fn test_mark_dirty_min_semantics() {
        let mut state = AppState::default();

        // First mark: sets dirty_from
        state.mark_dirty(GuidedStep::Analyze);
        assert_eq!(state.dirty_from, Some(GuidedStep::Analyze));

        // Earlier stage: overwrites
        state.mark_dirty(GuidedStep::Normalize);
        assert_eq!(state.dirty_from, Some(GuidedStep::Normalize));

        // Later stage: keeps earlier
        state.mark_dirty(GuidedStep::Analyze);
        assert_eq!(state.dirty_from, Some(GuidedStep::Normalize));
    }

    #[test]
    fn test_clear_dirty() {
        let mut state = AppState::default();
        state.mark_dirty(GuidedStep::Analyze);
        assert!(state.dirty_from.is_some());
        state.clear_dirty();
        assert!(state.dirty_from.is_none());
    }

    #[test]
    fn test_invalidate_results_clears_dirty() {
        let mut state = AppState::default();
        state.mark_dirty(GuidedStep::Normalize);
        state.invalidate_results();
        assert!(state.dirty_from.is_none());
    }

    #[test]
    fn test_run_from_dirty_noop_when_clean() {
        let mut state = AppState::default();
        let result = run_from_dirty(&mut state);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_mark_dirty_same_stage_idempotent() {
        let mut state = AppState::default();
        state.mark_dirty(GuidedStep::Analyze);
        state.mark_dirty(GuidedStep::Analyze);
        assert_eq!(state.dirty_from, Some(GuidedStep::Analyze));
    }

    #[test]
    fn test_stage_order_all_distinct() {
        let steps = [
            GuidedStep::Landing,
            GuidedStep::Wizard,
            GuidedStep::Configure,
            GuidedStep::Load,
            GuidedStep::Bin,
            GuidedStep::Rebin,
            GuidedStep::Normalize,
            GuidedStep::Analyze,
            GuidedStep::Results,
            GuidedStep::ForwardModel,
            GuidedStep::Detectability,
        ];
        for i in 0..steps.len() {
            for j in (i + 1)..steps.len() {
                assert!(
                    (steps[i] as u8) < (steps[j] as u8),
                    "{:?} should be < {:?}",
                    steps[i],
                    steps[j]
                );
            }
        }
    }
}
