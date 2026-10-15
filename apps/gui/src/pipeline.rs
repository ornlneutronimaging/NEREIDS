//! Guided-step ordering invariants.
//!
//! The pipeline executor was removed with the fit path; the step ordering
//! still defines navigation and display order.

#[cfg(test)]
mod tests {
    use crate::state::GuidedStep;

    #[test]
    fn test_stage_ordering() {
        assert!((GuidedStep::Normalize as u8) < (GuidedStep::Analyze as u8));
        assert!((GuidedStep::Rebin as u8) < (GuidedStep::Normalize as u8));
        assert!((GuidedStep::Load as u8) < (GuidedStep::Rebin as u8));
        assert!((GuidedStep::Configure as u8) < (GuidedStep::Load as u8));
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
