//! DASlogs-based run-health summary for SNS NeXus files (`hdf5` feature).
//!
//! A VENUS acquisition can be paused mid-run or suffer accelerator beam
//! dips; both silently reduce the effective exposure of the summed TIFF
//! stack.  [`run_health`] reads the slow-control logs under
//! `/entry/DASlogs/<pv>/{time,value}` and reports the time-weighted
//! fraction of the run spent paused and the fraction spent with the beam
//! power dipped below a threshold.
//!
//! ## Transition logs, not samples
//!
//! DASlogs PVs log *transitions*: an entry is written when the value
//! changes, not on a regular clock.  Taking the plain mean of the logged
//! entries is therefore wrong (a run paused for 90 % of its duration may
//! contain just two pause entries).  All statistics here use
//! **last-value-held** time-weighted integration instead: `value[0]` holds
//! from `t = 0`, each `value[i]` holds until the next log entry, and the
//! final value holds to the end of the run.  Intervals are clamped to
//! `[0, duration]` (DASlogs timestamps can precede the run start).
//!
//! ## Run duration
//!
//! The window length comes from the `/entry/duration` scalar when present.
//! Otherwise it falls back to the latest log timestamp across the PVs that
//! were read — a *lower bound* on the true run duration (the run continued
//! after the last logged transition), which makes the reported fractions
//! upper bounds.  A non-positive or non-finite window is a hard error,
//! never a NaN fraction.
//!
//! ## SNS PV names
//!
//! The defaults target SNS: `pause` (nonzero while the DAQ is paused) and
//! `proton_charge` (per-pulse beam power proxy).  Other facilities pass
//! their own PV names via [`RunHealthOptions`].

use std::path::Path;

use crate::error::IoError;

/// Default beam-dip threshold as a fraction of the median power.
///
/// A dip is counted when the power PV drops below
/// `power_dip_fraction × median(power)`.  0.5 sits between nominal
/// source jitter (a few percent around the median) and true beam-off
/// dips (power → 0), so it flags real outages without firing on noise.
pub const DEFAULT_POWER_DIP_FRACTION: f64 = 0.5;

/// Options for [`run_health`]: PV names and the beam-dip threshold.
#[derive(Debug, Clone)]
pub struct RunHealthOptions {
    /// DASlogs PV that is nonzero while the DAQ is paused
    /// (SNS: `"pause"`).
    pub pause_pv: String,
    /// DASlogs PV proxying beam power (SNS: `"proton_charge"`).
    pub power_pv: String,
    /// Beam-dip threshold as a fraction of the median power
    /// (default [`DEFAULT_POWER_DIP_FRACTION`]).
    pub power_dip_fraction: f64,
}

impl Default for RunHealthOptions {
    fn default() -> Self {
        Self {
            pause_pv: "pause".into(),
            power_pv: "proton_charge".into(),
            power_dip_fraction: DEFAULT_POWER_DIP_FRACTION,
        }
    }
}

/// Run-health summary computed from DASlogs.
///
/// Fields are `None` when the corresponding quantity *cannot be
/// computed*: the PV (or the whole `DASlogs` group) is absent from the
/// file, the PV is present but logged zero entries, or — for
/// [`beam_dip_fraction`](Self::beam_dip_fraction) only — the dip
/// threshold is undefined.  None of these is an error; they simply mean
/// the facility did not log that quantity (or logged nothing usable).
#[derive(Debug, Clone, PartialEq)]
pub struct RunHealth {
    /// Time-weighted fraction of the run spent paused (pause PV nonzero).
    pub pause_fraction: Option<f64>,
    /// Time-weighted fraction of the run with power below
    /// `power_dip_fraction × median(power)`.
    ///
    /// `None` when the power PV is absent or empty, **or when the dip
    /// threshold is undefined because the sample median of the power
    /// entries is non-positive** (e.g. the beam was off for at least half
    /// the log entries → median = 0, so the strict `< threshold` predicate
    /// could never fire and would misreport the worst runs as dip-free) —
    /// check [`median_power`](Self::median_power), which is co-reported.
    pub beam_dip_fraction: Option<f64>,
    /// Sample median of the power PV entries (median of the logged
    /// values, *not* time-weighted — documented deliberately: it is the
    /// threshold anchor, not an exposure estimate).
    pub median_power: Option<f64>,
    /// Run duration in seconds: `/entry/duration` when present, else the
    /// latest log timestamp across the PVs read (a lower bound).
    pub duration_s: Option<f64>,
    /// Number of pause-PV log entries read (0 when absent or empty).
    pub n_pause_entries: usize,
    /// Number of power-PV log entries read (0 when absent or empty).
    pub n_power_entries: usize,
}

/// Compute a run-health summary from `/entry/DASlogs` of a NeXus file.
///
/// See the [module docs](self) for the last-value-held semantics, the
/// duration fallback, and the SNS PV-name defaults.
///
/// # Errors
/// * [`IoError::Hdf5Error`] when the file or `/entry` cannot be opened.
/// * [`IoError::InvalidParameter`] when a PV is *present but malformed*
///   (time/value length mismatch, non-finite entries, negative power
///   values, decreasing timestamps), `/entry/duration` is present but
///   non-positive or non-finite, the integration window is non-positive,
///   or `power_dip_fraction` is not a positive finite number.
///
/// Absent PVs (or an absent `DASlogs` group) are *not* errors — the
/// corresponding fields are `None`, as is a present PV with zero log
/// entries ("no entries logged" carries no integrable information).
/// Absence vs malformed is decided by link existence (`member_names`),
/// the `read_dead_pixel_mask` idiom: collapsing "not there" and "there
/// but unreadable" into one path would mask real file corruption as
/// absence.
///
/// `beam_dip_fraction` is additionally `None` (with `median_power` still
/// reported) when the sample median of the power entries is non-positive:
/// the dip threshold `power_dip_fraction × median` is then undefined —
/// see [`RunHealth::beam_dip_fraction`].
pub fn run_health(path: &Path, options: &RunHealthOptions) -> Result<RunHealth, IoError> {
    if !options.power_dip_fraction.is_finite() || options.power_dip_fraction <= 0.0 {
        return Err(IoError::InvalidParameter(format!(
            "power_dip_fraction must be a positive finite number, got {}",
            options.power_dip_fraction,
        )));
    }

    let file = hdf5::File::open(path)
        .map_err(|e| IoError::Hdf5Error(format!("Cannot open HDF5 file: {e}")))?;
    let entry = file
        .group("entry")
        .map_err(|e| IoError::Hdf5Error(format!("Cannot open /entry: {e}")))?;
    let entry_members = entry
        .member_names()
        .map_err(|e| IoError::InvalidParameter(format!("Failed to list /entry members: {e}")))?;

    // /entry/duration: absent → fall back to log timestamps below;
    // present but malformed → hard error.
    let file_duration = if entry_members.iter().any(|n| n == "duration") {
        let ds = entry.dataset("duration").map_err(|e| {
            IoError::InvalidParameter(format!(
                "/entry/duration is present but is not a readable dataset: {e}"
            ))
        })?;
        let d = read_scalar_f64(&ds, "/entry/duration")?;
        if !d.is_finite() || d <= 0.0 {
            return Err(IoError::InvalidParameter(format!(
                "/entry/duration must be positive and finite, got {d}"
            )));
        }
        Some(d)
    } else {
        None
    };

    // /entry/DASlogs: absent → all-None health (valid file without logs).
    let daslogs = if entry_members.iter().any(|n| n == "DASlogs") {
        Some(entry.group("DASlogs").map_err(|e| {
            IoError::InvalidParameter(format!(
                "/entry/DASlogs is present but is not a readable group: {e}"
            ))
        })?)
    } else {
        None
    };

    // The pause PV is a state flag (any nonzero value means "paused", sign
    // included), but beam power is physically non-negative — a negative
    // entry is malformed data, not a dip, so it takes the hard-error path.
    let pause = match &daslogs {
        Some(group) => read_pv_series(group, &options.pause_pv, false)?,
        None => None,
    };
    let power = match &daslogs {
        Some(group) => read_pv_series(group, &options.power_pv, true)?,
        None => None,
    };

    let n_pause_entries = pause.as_ref().map_or(0, |(t, _)| t.len());
    let n_power_entries = power.as_ref().map_or(0, |(t, _)| t.len());

    // Duration: file value, else the latest timestamp across PVs read
    // (a lower bound — see module docs).
    let duration_s = file_duration.or_else(|| {
        let last_times = [
            pause.as_ref().and_then(|(t, _)| t.last().copied()),
            power.as_ref().and_then(|(t, _)| t.last().copied()),
        ];
        last_times.into_iter().flatten().fold(None, |acc, t| {
            Some(match acc {
                None => t,
                Some(a) if t > a => t,
                Some(a) => a,
            })
        })
    });

    // Nothing to integrate → report what we know (all-None fractions).
    if pause.is_none() && power.is_none() {
        return Ok(RunHealth {
            pause_fraction: None,
            beam_dip_fraction: None,
            median_power: None,
            duration_s,
            n_pause_entries: 0,
            n_power_entries: 0,
        });
    }

    // A PV was read, so integration needs a usable window.  Non-positive
    // windows (e.g. a single log entry at t = 0 and no /entry/duration)
    // are a hard error — dividing by them would fabricate NaN/∞ fractions.
    let duration = duration_s.ok_or_else(|| {
        IoError::InvalidParameter("Cannot determine run duration for DASlogs integration".into())
    })?;
    if !duration.is_finite() || duration <= 0.0 {
        return Err(IoError::InvalidParameter(format!(
            "Run duration for DASlogs integration must be positive, got {duration}"
        )));
    }

    let pause_fraction = pause
        .as_ref()
        .map(|(time, value)| lvh_fraction(time, value, duration, |v| v != 0.0));

    let (median_power, beam_dip_fraction) = match &power {
        Some((time, value)) => {
            let median = sample_median(value);
            // The dip threshold `power_dip_fraction × median` is undefined
            // when the median is non-positive (e.g. the beam was off for at
            // least half the log entries → median = 0): the strict `<`
            // predicate could never fire, misreporting exactly the worst
            // runs as beam_dip_fraction = 0.0.  Report "cannot compute"
            // (None) instead — the struct's established signal — with
            // `median_power` co-reported so callers can see why.  Values
            // are validated finite and non-negative on read, so the
            // is_finite() arm is defense-in-depth only.
            if !median.is_finite() || median <= 0.0 {
                (Some(median), None)
            } else {
                let threshold = options.power_dip_fraction * median;
                let dip = lvh_fraction(time, value, duration, |v| v < threshold);
                (Some(median), Some(dip))
            }
        }
        None => (None, None),
    };

    Ok(RunHealth {
        pause_fraction,
        beam_dip_fraction,
        median_power,
        duration_s: Some(duration),
        n_pause_entries,
        n_power_entries,
    })
}

/// Read a scalar f64 from a dataset that may be stored 0-dimensional or as
/// a 1-element vector (both encodings occur in the wild for
/// `/entry/duration`).
fn read_scalar_f64(ds: &hdf5::Dataset, what: &str) -> Result<f64, IoError> {
    if ds.ndim() == 0 {
        return ds
            .read_scalar::<f64>()
            .map_err(|e| IoError::InvalidParameter(format!("Failed to read {what}: {e}")));
    }
    let values: Vec<f64> = ds
        .read_raw()
        .map_err(|e| IoError::InvalidParameter(format!("Failed to read {what}: {e}")))?;
    if values.len() != 1 {
        return Err(IoError::InvalidParameter(format!(
            "{what} must be a scalar, got {} values",
            values.len()
        )));
    }
    Ok(values[0])
}

/// Parallel `time`/`value` series of one DASlogs PV.
type PvSeries = (Vec<f64>, Vec<f64>);

/// Read `<group>/<pv>/{time,value}` as parallel 1D f64 series.
///
/// Returns `Ok(None)` when the PV link is absent (valid file without that
/// log) **or** when the PV is present but both series are empty — "no
/// entries logged" carries no integrable information, so it is treated
/// like absence rather than corruption (a hard error here would discard
/// the other PV's summary).  Returns `Err` when the PV is present but
/// malformed: unreadable group/datasets, length-mismatched series,
/// non-finite entries, negative values when `require_non_negative` is set
/// (physically non-negative PVs such as beam power), or decreasing
/// timestamps.  Duplicate (equal) timestamps are allowed — real SNS logs
/// contain them; they contribute zero-width intervals.
fn read_pv_series(
    daslogs: &hdf5::Group,
    pv: &str,
    require_non_negative: bool,
) -> Result<Option<PvSeries>, IoError> {
    let members = daslogs.member_names().map_err(|e| {
        IoError::InvalidParameter(format!("Failed to list /entry/DASlogs members: {e}"))
    })?;
    if !members.iter().any(|n| n == pv) {
        return Ok(None);
    }
    let group = daslogs.group(pv).map_err(|e| {
        IoError::InvalidParameter(format!(
            "/entry/DASlogs/{pv} is present but is not a readable group: {e}"
        ))
    })?;

    let read_series = |name: &str| -> Result<Vec<f64>, IoError> {
        let ds = group.dataset(name).map_err(|e| {
            IoError::InvalidParameter(format!(
                "/entry/DASlogs/{pv}/{name} is missing or unreadable: {e}"
            ))
        })?;
        ds.read_raw::<f64>().map_err(|e| {
            IoError::InvalidParameter(format!("Failed to read /entry/DASlogs/{pv}/{name}: {e}"))
        })
    };
    let time = read_series("time")?;
    let value = read_series("value")?;

    // Present-but-empty (zero entries in BOTH series) is "no entries
    // logged", semantically closer to an absent PV than to corruption —
    // map to Ok(None) so the other PV's summary survives.  An empty time
    // with a non-empty value (or vice versa) still falls through to the
    // length-mismatch error below: that IS corruption.
    if time.is_empty() && value.is_empty() {
        return Ok(None);
    }
    if time.len() != value.len() {
        return Err(IoError::InvalidParameter(format!(
            "/entry/DASlogs/{pv}: time has {} entries but value has {}",
            time.len(),
            value.len(),
        )));
    }
    for (i, &t) in time.iter().enumerate() {
        if !t.is_finite() {
            return Err(IoError::InvalidParameter(format!(
                "/entry/DASlogs/{pv}/time[{i}] is non-finite ({t})"
            )));
        }
    }
    for (i, &v) in value.iter().enumerate() {
        if !v.is_finite() {
            return Err(IoError::InvalidParameter(format!(
                "/entry/DASlogs/{pv}/value[{i}] is non-finite ({v})"
            )));
        }
        if require_non_negative && v < 0.0 {
            return Err(IoError::InvalidParameter(format!(
                "/entry/DASlogs/{pv}/value[{i}] is negative ({v}); \
                 this PV is physically non-negative (e.g. beam power), so \
                 a negative entry is malformed data"
            )));
        }
    }
    for window in time.windows(2) {
        if window[1] < window[0] {
            return Err(IoError::InvalidParameter(format!(
                "/entry/DASlogs/{pv}/time must be ascending, but found {} after {}",
                window[1], window[0],
            )));
        }
    }

    Ok(Some((time, value)))
}

/// Last-value-held time-weighted fraction of `[0, duration]` where
/// `pred(value)` holds.
///
/// `value[0]` holds from `t = 0` (the value before the first logged
/// transition is taken to be the first logged value), `value[i]` holds
/// over `[time[i], time[i+1])`, and the final value holds to `duration`.
/// Interior interval bounds are clamped to `[0, duration]`, so the
/// interval widths telescope to exactly `duration` and the result lies in
/// `[0, 1]`.
fn lvh_fraction(time: &[f64], value: &[f64], duration: f64, pred: impl Fn(f64) -> bool) -> f64 {
    debug_assert!(!time.is_empty() && time.len() == value.len());
    debug_assert!(duration.is_finite() && duration > 0.0);
    let n = time.len();
    let mut held = 0.0;
    for i in 0..n {
        let start = if i == 0 {
            0.0
        } else {
            time[i].clamp(0.0, duration)
        };
        let end = if i + 1 < n {
            time[i + 1].clamp(0.0, duration)
        } else {
            duration
        };
        if pred(value[i]) {
            held += (end - start).max(0.0);
        }
    }
    held / duration
}

/// Sample median of the logged values (mean of the two middle order
/// statistics for even counts).  This is the median of the log *entries*,
/// not a time-weighted median — it anchors the dip threshold and must not
/// be read as an exposure-weighted average.
fn sample_median(values: &[f64]) -> f64 {
    debug_assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("values validated finite"));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal NeXus file with an optional `/entry/duration` and
    /// the given DASlogs PVs as parallel `time`/`value` series.
    fn write_run(path: &Path, duration: Option<f64>, pvs: &[(&str, &[f64], &[f64])]) {
        let file = hdf5::File::create(path).expect("create test file");
        let entry = file.create_group("entry").expect("create entry");
        if let Some(d) = duration {
            entry
                .new_dataset::<f64>()
                .shape(())
                .create("duration")
                .expect("create duration")
                .write_scalar(&d)
                .expect("write duration");
        }
        let das = entry.create_group("DASlogs").expect("create DASlogs");
        for (name, time, value) in pvs {
            let group = das.create_group(name).expect("create pv group");
            group
                .new_dataset::<f64>()
                .shape([time.len()])
                .create("time")
                .expect("create time")
                .write_raw(time)
                .expect("write time");
            group
                .new_dataset::<f64>()
                .shape([value.len()])
                .create("value")
                .expect("create value")
                .write_raw(value)
                .expect("write value");
        }
    }

    /// T32: pause 0@0 / 1@10 / 0@20 over a 100 s run → 10 % paused.
    #[test]
    fn test_pause_fraction_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.h5");
        write_run(
            &path,
            Some(100.0),
            &[("pause", &[0.0, 10.0, 20.0], &[0.0, 1.0, 0.0])],
        );

        let health = run_health(&path, &RunHealthOptions::default()).unwrap();
        assert_eq!(health.pause_fraction, Some(0.1));
        assert_eq!(health.duration_s, Some(100.0));
        assert_eq!(health.n_pause_entries, 3);
        assert_eq!(health.beam_dip_fraction, None);
        assert_eq!(health.median_power, None);
        assert_eq!(health.n_power_entries, 0);
    }

    /// T33: absent pause PV → pause fields None, not an error.
    #[test]
    fn test_absent_pause_pv_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.h5");
        write_run(
            &path,
            Some(100.0),
            &[("proton_charge", &[0.0, 50.0], &[10.0, 10.0])],
        );

        let health = run_health(&path, &RunHealthOptions::default()).unwrap();
        assert_eq!(health.pause_fraction, None);
        assert_eq!(health.n_pause_entries, 0);
        assert!(health.beam_dip_fraction.is_some());
    }

    /// T34: a single pause=1 entry holds over the whole window → 1.0.
    #[test]
    fn test_single_entry_holds_whole_window() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.h5");
        write_run(&path, Some(100.0), &[("pause", &[10.0], &[1.0])]);

        let health = run_health(&path, &RunHealthOptions::default()).unwrap();
        assert_eq!(health.pause_fraction, Some(1.0));
        assert_eq!(health.n_pause_entries, 1);
    }

    /// T35: power dips below half the median are counted, and the sample
    /// median is reported.
    #[test]
    fn test_beam_dip_fraction_and_median() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.h5");
        // 10 → dip to 1 over [10, 20) → back to 10; median of
        // [10, 1, 10, 10] = 10 (middle two of the sorted values).
        write_run(
            &path,
            Some(40.0),
            &[(
                "proton_charge",
                &[0.0, 10.0, 20.0, 30.0],
                &[10.0, 1.0, 10.0, 10.0],
            )],
        );

        let health = run_health(&path, &RunHealthOptions::default()).unwrap();
        assert_eq!(health.median_power, Some(10.0));
        assert_eq!(health.beam_dip_fraction, Some(0.25));
        assert_eq!(health.n_power_entries, 4);
    }

    /// T36: time/value length mismatch is malformed, not absent.
    #[test]
    fn test_length_mismatch_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.h5");
        write_run(
            &path,
            Some(100.0),
            &[("pause", &[0.0, 10.0, 20.0], &[0.0, 1.0])],
        );

        let err = run_health(&path, &RunHealthOptions::default()).unwrap_err();
        assert!(
            matches!(err, IoError::InvalidParameter(_)),
            "Expected InvalidParameter, got: {:?}",
            err,
        );
        assert!(format!("{err}").contains("time has 3 entries but value has 2"));
    }

    /// T37: decreasing timestamps are malformed.
    #[test]
    fn test_non_ascending_time_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.h5");
        write_run(
            &path,
            Some(100.0),
            &[("pause", &[0.0, 20.0, 10.0], &[0.0, 1.0, 0.0])],
        );

        let err = run_health(&path, &RunHealthOptions::default()).unwrap_err();
        assert!(
            format!("{err}").contains("ascending"),
            "Expected ascending-time error, got: {err}",
        );
    }

    /// T38: without /entry/duration the window falls back to the latest
    /// log timestamp; a non-positive fallback window is a hard error.
    #[test]
    fn test_duration_fallback_and_zero_window() {
        let dir = tempfile::tempdir().unwrap();

        // Fallback: last timestamp (20 s) becomes the window → pause
        // fraction 10/20.
        let path = dir.path().join("fallback.h5");
        write_run(
            &path,
            None,
            &[("pause", &[0.0, 10.0, 20.0], &[0.0, 1.0, 0.0])],
        );
        let health = run_health(&path, &RunHealthOptions::default()).unwrap();
        assert_eq!(health.duration_s, Some(20.0));
        assert_eq!(health.pause_fraction, Some(0.5));

        // Zero window: single entry at t = 0 and no duration → error,
        // never a NaN fraction.
        let path = dir.path().join("zero.h5");
        write_run(&path, None, &[("pause", &[0.0], &[1.0])]);
        let err = run_health(&path, &RunHealthOptions::default()).unwrap_err();
        assert!(
            matches!(err, IoError::InvalidParameter(_)),
            "Expected InvalidParameter, got: {:?}",
            err,
        );
    }

    /// T39: a file without a DASlogs group reports all-None health.
    #[test]
    fn test_missing_daslogs_group_all_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.h5");
        let file = hdf5::File::create(&path).unwrap();
        let entry = file.create_group("entry").unwrap();
        entry
            .new_dataset::<f64>()
            .shape(())
            .create("duration")
            .unwrap()
            .write_scalar(&100.0)
            .unwrap();
        drop(file);

        let health = run_health(&path, &RunHealthOptions::default()).unwrap();
        assert_eq!(health.pause_fraction, None);
        assert_eq!(health.beam_dip_fraction, None);
        assert_eq!(health.median_power, None);
        assert_eq!(health.duration_s, Some(100.0));
        assert_eq!(health.n_pause_entries, 0);
        assert_eq!(health.n_power_entries, 0);
    }

    /// Malformed /entry/duration (non-positive) is a hard error.
    #[test]
    fn test_nonpositive_duration_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.h5");
        write_run(&path, Some(0.0), &[("pause", &[0.0], &[0.0])]);

        let err = run_health(&path, &RunHealthOptions::default()).unwrap_err();
        assert!(format!("{err}").contains("duration"));
    }

    /// T53: a mostly-zero power log (median 0 — beam off for at least
    /// half the entries) makes the dip threshold undefined: the strict
    /// `< 0` predicate could never fire, so beam_dip_fraction must be
    /// None (not a false 0.0) with median_power still co-reported.
    #[test]
    fn test_zero_median_power_dip_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.h5");
        // Beam off (0) for 3 of 4 entries → sample median 0.
        write_run(
            &path,
            Some(40.0),
            &[(
                "proton_charge",
                &[0.0, 10.0, 20.0, 30.0],
                &[0.0, 0.0, 0.0, 10.0],
            )],
        );

        let health = run_health(&path, &RunHealthOptions::default()).unwrap();
        assert_eq!(health.median_power, Some(0.0));
        assert_eq!(health.beam_dip_fraction, None);
        assert_eq!(health.n_power_entries, 4);
        assert_eq!(health.duration_s, Some(40.0));
    }

    /// T54: a negative power value is malformed (beam power is physically
    /// non-negative) and hard-errors; a negative *pause* value is not an
    /// error — the pause PV is a state flag where any nonzero value
    /// (sign included) means "paused".
    #[test]
    fn test_negative_power_errors_negative_pause_allowed() {
        let dir = tempfile::tempdir().unwrap();

        let path = dir.path().join("neg_power.h5");
        write_run(
            &path,
            Some(100.0),
            &[("proton_charge", &[0.0, 10.0], &[10.0, -1.0])],
        );
        let err = run_health(&path, &RunHealthOptions::default()).unwrap_err();
        assert!(
            format!("{err}").contains("negative"),
            "Expected negative-power error, got: {err}",
        );

        let path = dir.path().join("neg_pause.h5");
        write_run(
            &path,
            Some(100.0),
            &[("pause", &[0.0, 10.0, 20.0], &[0.0, -1.0, 0.0])],
        );
        let health = run_health(&path, &RunHealthOptions::default()).unwrap();
        assert_eq!(health.pause_fraction, Some(0.1));
    }

    /// T55: a present-but-EMPTY PV (zero entries logged) is treated like
    /// an absent PV — None fields, no error — and the other PV's summary
    /// survives.
    #[test]
    fn test_empty_pv_is_none_other_pv_survives() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.h5");
        write_run(
            &path,
            Some(100.0),
            &[
                ("pause", &[], &[]),
                ("proton_charge", &[0.0, 50.0], &[10.0, 10.0]),
            ],
        );

        let health = run_health(&path, &RunHealthOptions::default()).unwrap();
        assert_eq!(health.pause_fraction, None);
        assert_eq!(health.n_pause_entries, 0);
        assert_eq!(health.median_power, Some(10.0));
        assert_eq!(health.beam_dip_fraction, Some(0.0));
        assert_eq!(health.n_power_entries, 2);
    }

    /// Invalid power_dip_fraction is rejected up-front.
    #[test]
    fn test_invalid_dip_fraction_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.h5");
        write_run(&path, Some(100.0), &[("pause", &[0.0], &[0.0])]);

        let options = RunHealthOptions {
            power_dip_fraction: f64::NAN,
            ..Default::default()
        };
        let err = run_health(&path, &options).unwrap_err();
        assert!(format!("{err}").contains("power_dip_fraction"));
    }
}
