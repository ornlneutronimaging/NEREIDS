//! DASlogs run-log reading and beam-state interval derivation (issue #637).
//!
//! SNS/HFIR NeXus files record slow-control PVs under
//! `/entry/DASlogs/<pv>/{time, value}`.  These are **transition logs**, not
//! uniformly-sampled time series: each `value[i]` takes effect at `time[i]`
//! (seconds relative to run start) and persists until `time[i+1]` (the last
//! value persists to the end of the run).  Averaging the `value` array
//! directly is therefore wrong whenever entries are unevenly spaced — on a
//! real VENUS run the entry-mean of the `pause` log read 0.43 while the
//! time-weighted truth was 0.90.  [`intervals_where`] encodes the correct
//! step-function semantics once, so callers never re-derive them.
//!
//! Real SNS logs occasionally contain **corrupt records**: device
//! reconnects write entries with `time = 0.0` and an uninitialized-memory
//! payload (a subnormal double, ~6.9e-310), observed both mid-log (where
//! the time jumps backward) and as the leading entry (where it does not)
//! on VENUS furnace channels `BL10:SE:ND1:*` in 3 of 59 IPTS-37432 runs.
//! [`read_run_log`] drops entries whose time jumps backward or whose
//! value is subnormal and reports the count in
//! [`RunLog::n_dropped_corrupt`], so the step function is never fed
//! garbage and the anomaly stays visible.
//!
//! The derived `(t_start, t_end)` interval lists feed
//! [`crate::nexus::load_nexus_bank_spectrum`]'s `keep_intervals` event
//! filter and compose across PVs via [`intervals_intersect`] (e.g.
//! `pause == 0` ∩ `beam_power > 1.5 MW`).  Facility-specific PV names never
//! appear in this crate — they belong to caller code and docstrings.

use crate::error::IoError;
use std::path::Path;

/// One slow-control PV read from `/entry/DASlogs/<pv>` plus the run length.
#[derive(Debug, Clone)]
pub struct RunLog {
    /// Transition times in seconds relative to run start (ascending).
    pub times: Vec<f64>,
    /// Value taking effect at the matching `times` entry.
    pub values: Vec<f64>,
    /// Total run duration in seconds (`/entry/duration`), the implicit end
    /// of the last transition segment.
    pub duration_s: f64,
    /// ISO-8601 epoch of the `time` axis (the `start` or `offset` attribute),
    /// when recorded.  Compare with
    /// [`crate::nexus::BankSpectrum::pulse_time_offset_iso`] to confirm the
    /// log clock and the pulse clock share a zero point (at SNS both are
    /// seconds since run start and the attributes match exactly).
    pub offset_iso: Option<String>,
    /// Entries dropped as corrupt reconnect records — backward time jump
    /// or subnormal value payload (see the module docs).  Non-zero is
    /// worth a mention in run-health screens; retained entries are
    /// unaffected.
    pub n_dropped_corrupt: usize,
}

/// Read a DASlogs PV as a transition log (issue #637).
///
/// Reads `/entry/DASlogs/<pv>/time` and `/value` (numeric) and
/// `/entry/duration`.  Times are seconds since run start (the same epoch
/// the NXevent_data `event_time_zero` values are relative to, so derived
/// intervals feed [`crate::nexus::load_nexus_bank_spectrum`] directly).
/// Corrupt reconnect records — backward/NaN time or subnormal value
/// payload — are dropped and counted in [`RunLog::n_dropped_corrupt`]
/// (real SNS files contain both shapes; module docs).
///
/// # Errors
/// [`IoError::FileNotFound`] when the file cannot be opened;
/// [`IoError::InvalidParameter`] when the PV group or datasets are missing
/// or non-numeric; [`IoError::ShapeMismatch`] when `time` and `value`
/// lengths differ.
pub fn read_run_log(path: &Path, pv: &str) -> Result<RunLog, IoError> {
    let file = hdf5::File::open(path).map_err(|e| {
        IoError::FileNotFound(
            path.display().to_string(),
            std::io::Error::other(e.to_string()),
        )
    })?;
    let group = file.group(&format!("entry/DASlogs/{pv}")).map_err(|e| {
        IoError::InvalidParameter(format!("Missing /entry/DASlogs/{pv} group: {e}"))
    })?;
    let time_ds = group
        .dataset("time")
        .map_err(|e| IoError::InvalidParameter(format!("Missing {pv}/time dataset: {e}")))?;
    let times: Vec<f64> = time_ds
        .read_1d()
        .map_err(|e| IoError::InvalidParameter(format!("Failed to read {pv}/time: {e}")))?
        .to_vec();
    let offset_iso = match crate::nexus::read_string_attr(&time_ds, "start")? {
        Some(v) => Some(v),
        None => crate::nexus::read_string_attr(&time_ds, "offset")?,
    };
    let values: Vec<f64> = group
        .dataset("value")
        .map_err(|e| IoError::InvalidParameter(format!("Missing {pv}/value dataset: {e}")))?
        .read_1d()
        .map_err(|e| {
            IoError::InvalidParameter(format!(
                "Failed to read {pv}/value as a 1-D numeric array (string-valued and \
                 multi-dimensional PVs are not supported): {e}"
            ))
        })?
        .to_vec();
    if times.len() != values.len() {
        return Err(IoError::ShapeMismatch(format!(
            "{pv}: time has {} entries but value has {}",
            times.len(),
            values.len()
        )));
    }
    // Drop corrupt reconnect records — see module docs.  Two signatures:
    // a backward-jumping (or NaN) time, and a subnormal value payload
    // (0 < |v| < f64::MIN_POSITIVE — uninitialized memory, not a
    // measurement; observed as the LEADING entry where no backward jump
    // exists to catch it).  The count is reported, never hidden.
    let mut clean_times = Vec::with_capacity(times.len());
    let mut clean_values = Vec::with_capacity(values.len());
    let mut running_max = f64::NEG_INFINITY;
    for (&t, &v) in times.iter().zip(&values) {
        let subnormal_payload = v != 0.0 && v.abs() < f64::MIN_POSITIVE;
        if t.is_finite() && t >= running_max && !subnormal_payload {
            running_max = t;
            clean_times.push(t);
            clean_values.push(v);
        }
    }
    let n_dropped_corrupt = times.len() - clean_times.len();
    let (times, values) = (clean_times, clean_values);
    let duration_s: f64 = {
        let ds = file
            .dataset("entry/duration")
            .map_err(|e| IoError::InvalidParameter(format!("Missing /entry/duration: {e}")))?;
        // SNS writes duration as shape (1,); tolerate a scalar () too.
        if ds.ndim() == 0 {
            ds.read_scalar::<f64>()
                .map_err(|e| IoError::InvalidParameter(format!("Failed to read duration: {e}")))?
        } else {
            let v: Vec<f64> = ds
                .read_1d()
                .map_err(|e| IoError::InvalidParameter(format!("Failed to read duration: {e}")))?
                .to_vec();
            *v.first()
                .ok_or_else(|| IoError::InvalidParameter("/entry/duration is empty".to_string()))?
        }
    };
    Ok(RunLog {
        times,
        values,
        duration_s,
        offset_iso,
        n_dropped_corrupt,
    })
}

/// Derive the run-time intervals on which a transition-log PV satisfies
/// `min_value <= value <= max_value` (either bound optional), using the
/// correct step-function semantics (issue #637): `values[i]` holds on
/// `[times[i], times[i+1])`, the last value holds to `duration_s`.
///
/// Time before the first transition entry is treated as **not** matching
/// (the state is unrecorded; excluding it is the conservative choice for a
/// keep-filter).  `NaN` values never match.  Adjacent/overlapping matching
/// segments are merged; empty segments (`t_end <= t_start`) are dropped.
/// The final segment's end is padded by one f32 ULP above `duration_s`
/// because SNS records `/entry/duration` in float32 while pulse times are
/// float64 — without the pad, a keep-everything filter drops the final
/// pulse of roughly half of real runs.
///
/// # Errors
/// [`IoError::InvalidParameter`] on non-finite/negative `duration_s`,
/// descending `times`, mismatched lengths, or a bound pair with
/// `min_value > max_value`.
pub fn intervals_where(
    times: &[f64],
    values: &[f64],
    duration_s: f64,
    min_value: Option<f64>,
    max_value: Option<f64>,
) -> Result<Vec<(f64, f64)>, IoError> {
    if !duration_s.is_finite() || duration_s < 0.0 {
        return Err(IoError::InvalidParameter(format!(
            "duration_s must be finite and non-negative, got {duration_s}"
        )));
    }
    if times.len() != values.len() {
        return Err(IoError::ShapeMismatch(format!(
            "times has {} entries but values has {}",
            times.len(),
            values.len()
        )));
    }
    // Explicit finiteness check so a single-entry log gets the same
    // validation as longer ones (a windows(2)-only check would let a lone
    // NaN time through, to be silently clamped by max(0.0) below).
    if let Some(i) = times.iter().position(|t| !t.is_finite()) {
        return Err(IoError::InvalidParameter(format!(
            "times[{i}] is not finite ({})",
            times[i]
        )));
    }
    if times.windows(2).any(|w| w[1] < w[0]) {
        return Err(IoError::InvalidParameter(
            "times must be ascending (transition log)".to_string(),
        ));
    }
    if let (Some(lo), Some(hi)) = (min_value, max_value)
        && lo > hi
    {
        return Err(IoError::InvalidParameter(format!(
            "min_value ({lo}) must not exceed max_value ({hi})"
        )));
    }
    let matches = |v: f64| -> bool {
        v.is_finite() && min_value.is_none_or(|lo| v >= lo) && max_value.is_none_or(|hi| v <= hi)
    };
    // /entry/duration is float32-quantized in SNS files while pulse times
    // (event_time_zero) are float64: the final pulse of ~half of surveyed
    // real runs is time-stamped within one f32 ULP ABOVE the recorded
    // duration.  Pad the run end by one f32 ULP so the final segment does
    // not spuriously exclude it (issue #637).
    let run_end = duration_s.max(((duration_s as f32).next_up()) as f64);
    let mut out: Vec<(f64, f64)> = Vec::new();
    for i in 0..times.len() {
        if !matches(values[i]) {
            continue;
        }
        let t0 = times[i].max(0.0);
        let t1 = if i + 1 < times.len() {
            times[i + 1].min(duration_s)
        } else {
            run_end
        };
        if t1 <= t0 {
            continue;
        }
        match out.last_mut() {
            // Merge segments that touch (shared transition point).
            Some(last) if t0 <= last.1 => last.1 = last.1.max(t1),
            _ => out.push((t0, t1)),
        }
    }
    Ok(out)
}

/// Intersect two interval lists — e.g. `pause == 0` ∩ `beam_power > 1.5 MW`.
///
/// Inputs may be unsorted/overlapping (each side is normalised by
/// sort-and-merge first, the same policy as
/// [`crate::nexus::load_nexus_bank_spectrum`]'s `keep_intervals`); every
/// pair must be finite with `t_end > t_start`.  Output is sorted,
/// non-overlapping, and drops empty intersections.
///
/// # Errors
/// [`IoError::InvalidParameter`] on a non-finite or empty/inverted pair.
pub fn intervals_intersect(a: &[(f64, f64)], b: &[(f64, f64)]) -> Result<Vec<(f64, f64)>, IoError> {
    let a = normalize_intervals(a)?;
    let b = normalize_intervals(b)?;
    let mut out = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let lo = a[i].0.max(b[j].0);
        let hi = a[i].1.min(b[j].1);
        if hi > lo {
            out.push((lo, hi));
        }
        if a[i].1 <= b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    Ok(out)
}

/// Validate interval pairs (finite, `t_end > t_start`) and normalise the
/// list to sorted non-overlapping form by sort-and-merge.
pub(crate) fn normalize_intervals(raw: &[(f64, f64)]) -> Result<Vec<(f64, f64)>, IoError> {
    for &(a, b) in raw {
        if !a.is_finite() || !b.is_finite() || b <= a {
            return Err(IoError::InvalidParameter(format!(
                "interval entries must be finite with t_end > t_start, got ({a}, {b})"
            )));
        }
    }
    let mut v = raw.to_vec();
    v.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (a, b) in v {
        match merged.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The motivating real-world case (issue #637): a `pause` transition log
    /// [0, 1, 0, 1, 0, 1, 0] whose entry-mean (0.43) wildly understates the
    /// time-weighted pause fraction (0.90).  `intervals_where(pause == 0)`
    /// must return the four short LIVE segments, not 57 % of the run.
    #[test]
    fn step_semantics_pause_transition_log() {
        let times = [0.0, 1000.0, 20000.0, 21500.0, 42589.0, 44100.0, 44338.0];
        // value:    0     1        0        1        0        1        0
        let values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let live = intervals_where(&times, &values, 44339.0, Some(-0.5), Some(0.5)).unwrap();
        assert_eq!(live.len(), 4);
        assert_eq!(live[0], (0.0, 1000.0));
        assert_eq!(live[1], (20000.0, 21500.0));
        assert_eq!(live[2], (42589.0, 44100.0));
        assert_eq!(live[3].0, 44338.0);
        assert!(live[3].1 >= 44339.0); // padded run end (f32 ULP)
        let live_total: f64 = live.iter().map(|(a, b)| b - a).sum();
        let entry_mean = values.iter().sum::<f64>() / values.len() as f64;
        // Entry-mean says ~43 % paused; time-weighting says ~90 % paused.
        assert!((entry_mean - 3.0 / 7.0).abs() < 1e-12);
        assert!(live_total / 44339.0 < 0.11, "live fraction {live_total}");
    }

    #[test]
    fn last_value_persists_to_duration_and_pre_log_time_excluded() {
        // Log starts at t = 100 s: state before that is unrecorded -> excluded.
        let iv = intervals_where(&[100.0], &[1.0], 500.0, Some(0.5), None).unwrap();
        assert_eq!(iv.len(), 1);
        assert_eq!(iv[0].0, 100.0);
        assert!(iv[0].1 >= 500.0 && iv[0].1 < 500.001); // padded run end
    }

    #[test]
    fn adjacent_matching_segments_merge() {
        let iv =
            intervals_where(&[0.0, 10.0, 20.0], &[1.0, 2.0, 0.0], 30.0, Some(0.5), None).unwrap();
        assert_eq!(iv, vec![(0.0, 20.0)]);
    }

    #[test]
    fn nan_values_never_match_and_bounds_validate() {
        let iv = intervals_where(&[0.0, 10.0], &[f64::NAN, 1.0], 20.0, Some(0.0), None).unwrap();
        assert_eq!(iv[0].0, 10.0);
        assert!(iv[0].1 >= 20.0);
        assert!(intervals_where(&[0.0], &[1.0], 10.0, Some(2.0), Some(1.0)).is_err());
        assert!(intervals_where(&[0.0], &[1.0], f64::NAN, None, None).is_err());
        assert!(intervals_where(&[10.0, 5.0], &[1.0, 1.0], 20.0, None, None).is_err());
        // Single-entry logs get the same finiteness validation.
        assert!(intervals_where(&[f64::NAN], &[1.0], 10.0, None, None).is_err());
    }

    fn create_test_daslogs(path: &std::path::Path, pv: &str, times: &[f64], values: &[f64]) {
        let file = hdf5::File::create(path).expect("create test file");
        let entry = file.create_group("entry").expect("entry");
        entry
            .new_dataset_builder()
            .with_data(&[44339.29_f64])
            .create("duration")
            .expect("duration");
        let g = entry
            .create_group(&format!("DASlogs/{pv}"))
            .expect("pv group");
        let t = g
            .new_dataset_builder()
            .with_data(times)
            .create("time")
            .expect("time");
        t.new_attr::<hdf5::types::VarLenUnicode>()
            .create("start")
            .expect("attr")
            .write_scalar(
                &"2026-06-22T19:01:07.183368667-04:00"
                    .parse::<hdf5::types::VarLenUnicode>()
                    .unwrap(),
            )
            .expect("write");
        g.new_dataset_builder()
            .with_data(values)
            .create("value")
            .expect("value");
    }

    #[test]
    fn corrupt_reconnect_record_dropped_and_counted() {
        // Mirror of VENUS run 19383 BL10:SE:ND1:CH1:PV — device reconnects
        // write (time=0.0, value=subnormal garbage) records BOTH as the
        // leading entry (no backward jump to catch it) and mid-log; the
        // clock resumes on the run clock immediately after.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reconnect.h5");
        create_test_daslogs(
            &path,
            "ch1",
            &[0.0, 2.0, 1194.96, 1226.96, 0.0, 1228.99],
            &[6.9e-310, 27.7, 27.75, 27.79, 6.9e-310, 27.78],
        );
        let log = read_run_log(&path, "ch1").expect("must read despite reconnect records");
        assert_eq!(log.n_dropped_corrupt, 2);
        assert_eq!(log.times.len(), 4);
        assert!(!log.values.contains(&6.9e-310));
        // The cleaned log is valid intervals_where input.
        let iv = intervals_where(&log.times, &log.values, log.duration_s, Some(27.0), None)
            .expect("cleaned log must be ascending");
        assert_eq!(iv.len(), 1);
        // The state on [0, 2) was recorded only by the garbage record, so
        // it is treated as unrecorded — the interval starts at the first
        // clean entry.
        assert_eq!(iv[0].0, 2.0);
    }

    #[test]
    fn read_run_log_round_trip_and_missing_pv() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("das.h5");
        create_test_daslogs(&path, "pause", &[0.0, 3622.9, 43720.9], &[0.0, 1.0, 0.0]);
        let log = read_run_log(&path, "pause").expect("read");
        assert_eq!(log.times.len(), 3);
        assert_eq!(log.n_dropped_corrupt, 0); // exact 0.0 values are NOT corrupt
        assert_eq!(log.values, vec![0.0, 1.0, 0.0]);
        assert!((log.duration_s - 44339.29).abs() < 1e-9);
        assert!(log.offset_iso.unwrap().starts_with("2026-06-22"));
        // End-to-end: transition log -> live intervals (pause == 0).
        let live = intervals_where(
            &log.times,
            &log.values,
            log.duration_s,
            Some(-0.5),
            Some(0.5),
        )
        .unwrap();
        assert_eq!(live.len(), 2);
        assert!((live[0].1 - 3622.9).abs() < 1e-9);
        assert!(live[1].1 >= 44339.29); // padded run end
        // Missing PV is a clear error.
        assert!(read_run_log(&path, "no_such_pv").is_err());
    }

    #[test]
    fn final_pulse_survives_f32_duration_quantization() {
        // Real shape (VENUS run 19373): /entry/duration is f32
        // (10.883509635925293 after promotion) while the last pulse's
        // event_time_zero is f64 10.88351 — strictly above it.  The final
        // segment must still contain that pulse.
        let duration_f32 = 10.883_51_f32 as f64; // 10.883509635925293
        let last_pulse = 10.883_51_f64;
        assert!(last_pulse > duration_f32);
        let iv = intervals_where(&[0.0], &[0.0], duration_f32, Some(-0.5), Some(0.5)).unwrap();
        assert_eq!(iv.len(), 1);
        assert!(
            iv[0].1 > last_pulse,
            "padded end {} <= pulse {last_pulse}",
            iv[0].1
        );
    }

    #[test]
    fn intersect_basic_disjoint_nested_touching_and_unsorted() {
        let a = [(0.0, 10.0), (20.0, 30.0)];
        let b = [(5.0, 25.0)];
        assert_eq!(
            intervals_intersect(&a, &b).unwrap(),
            vec![(5.0, 10.0), (20.0, 25.0)]
        );
        assert_eq!(intervals_intersect(&a, &[]).unwrap(), vec![]);
        // Touching endpoints produce no (empty) interval.
        assert_eq!(
            intervals_intersect(&[(0.0, 5.0)], &[(5.0, 9.0)]).unwrap(),
            vec![]
        );
        // Nested.
        assert_eq!(
            intervals_intersect(&[(0.0, 100.0)], &[(10.0, 20.0), (30.0, 40.0)]).unwrap(),
            vec![(10.0, 20.0), (30.0, 40.0)]
        );
        // Unsorted input is normalised, not silently corrupted.
        assert_eq!(
            intervals_intersect(&[(20.0, 30.0), (0.0, 10.0)], &[(5.0, 25.0)]).unwrap(),
            vec![(5.0, 10.0), (20.0, 25.0)]
        );
        // Sorted-but-overlapping input merges; postcondition holds.
        assert_eq!(
            intervals_intersect(&[(0.0, 10.0), (5.0, 20.0)], &[(0.0, 20.0)]).unwrap(),
            vec![(0.0, 20.0)]
        );
        // Invalid pairs error.
        assert!(intervals_intersect(&[(5.0, 5.0)], &[(0.0, 1.0)]).is_err());
        assert!(intervals_intersect(&[(f64::NAN, 1.0)], &[(0.0, 1.0)]).is_err());
    }
}
