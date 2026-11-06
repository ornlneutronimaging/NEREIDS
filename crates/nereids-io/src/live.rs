//! Live fractions of a Timepix-1 frame-mode run.
//!
//! A pixel records one count per shutter, so a neutron in bin `k` is recorded
//! only if the pixel had no count earlier in that shutter:
//! `L_k = 1 − Σ_{j<k} C_j / S`, with `C_j` the pixel's counts in bin `j` of the
//! shutter and `S` its triggers.
//!
//! # References
//! A. S. Tremsin et al., JINST 9 (2014) C05026;
//! K. Watanabe et al., Nucl. Instrum. Methods A 861 (2017) 55, Eqs. 1–5.

use ndarray::{Array3, ArrayView3, Axis, Zip};

use crate::error::IoError;
use crate::fits_stack::Shutter;

/// Each frame's shutter: the last one opening by the frame's start, within rounding.
///
/// A frame that ends after the next shutter opens is refused: its counts mix
/// two shutters.
fn shutter_of_frames(edges_us: &[f64], shutters: &[Shutter]) -> Result<Vec<usize>, IoError> {
    let width = edges_us
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min);
    let slack = 1e-6 * width;
    edges_us
        .windows(2)
        .enumerate()
        .map(|(k, w)| {
            let i = shutters
                .iter()
                .rposition(|s| s.start_us - slack <= w[0])
                .filter(|&i| w[0] < shutters[i].end_us)
                .ok_or_else(|| {
                    IoError::InvalidParameter(format!(
                        "frame {k} starts at {} µs, outside every shutter",
                        w[0]
                    ))
                })?;
            match shutters.get(i + 1) {
                Some(next) if w[1] > next.start_us + slack => {
                    Err(IoError::InvalidParameter(format!(
                        "frame {k} ends at {} µs, after shutter {} opens at {} µs",
                        w[1],
                        i + 1,
                        next.start_us
                    )))
                }
                _ => Ok(i),
            }
        })
        .collect()
}

/// Live fractions of each pixel in each bin, by the counts before it in its shutter.
///
/// `shutters[i].triggers` should count the pulses that reached the detector
/// while shutter `i` was open; triggers without beam make `L` too large.
///
/// # Arguments
/// * `counts` — Raw counts in (time bin, row, column), finite and 0 or more.
/// * `edges_us` — Time-bin edges in µs, ascending, one more than the bins.
/// * `shutters` — The run's shutters; each bin starts inside one and ends by the next.
///
/// # Returns
/// `L` in (time bin, row, column), each in (0, 1].
///
/// # Errors
/// [`IoError::ShapeMismatch`] if `counts` is empty or `edges_us` does not fit it;
/// [`IoError::InvalidParameter`] if a count is invalid, a bin starts outside
/// every shutter or ends after the next opens, shutters overlap, or a pixel's
/// counts in a shutter reach its triggers.
pub fn live_fractions(
    counts: ArrayView3<f64>,
    edges_us: &[f64],
    shutters: &[Shutter],
) -> Result<Array3<f64>, IoError> {
    let bins = counts.len_of(Axis(0));
    if edges_us.len() != bins + 1 || counts.is_empty() {
        return Err(IoError::ShapeMismatch(format!(
            "{} time-bin edges for {bins} bins",
            edges_us.len()
        )));
    }
    if let Some(v) = counts.iter().find(|v| !(v.is_finite() && **v >= 0.0)) {
        return Err(IoError::InvalidParameter(format!(
            "counts must be finite and 0 or more, got {v}"
        )));
    }
    if edges_us.iter().any(|t| !t.is_finite()) || edges_us.windows(2).any(|w| w[1] <= w[0]) {
        return Err(IoError::InvalidParameter(
            "time-bin edges must be finite and ascending".into(),
        ));
    }
    if let Some(s) = shutters.iter().find(|s| {
        !(s.start_us.is_finite()
            && s.end_us.is_finite()
            && s.start_us < s.end_us
            && s.triggers.is_finite()
            && s.triggers > 0.0)
    }) {
        return Err(IoError::InvalidParameter(format!(
            "a shutter needs finite times, start before end, and triggers, got {s:?}"
        )));
    }
    if shutters.windows(2).any(|w| w[1].start_us < w[0].end_us) {
        return Err(IoError::InvalidParameter(
            "shutters must be in time order and must not overlap".into(),
        ));
    }
    let owner = shutter_of_frames(edges_us, shutters)?;

    let mut live = Array3::zeros(counts.raw_dim());
    let mut before = ndarray::Array2::<f64>::zeros((counts.shape()[1], counts.shape()[2]));
    for k in 0..bins {
        let shutter = &shutters[owner[k]];
        if k > 0 && owner[k] != owner[k - 1] {
            before.fill(0.0);
        }
        Zip::from(live.index_axis_mut(Axis(0), k))
            .and(&before)
            .for_each(|l, &b| *l = 1.0 - b / shutter.triggers);
        if let Some(((row, column), l)) = live
            .index_axis(Axis(0), k)
            .indexed_iter()
            .find(|(_, l)| **l <= 0.0)
        {
            return Err(IoError::InvalidParameter(format!(
                "pixel row {row}, column {column} has live fraction {l} in bin {k}: \
                 its counts before the bin reach the shutter's {} triggers",
                shutter.triggers
            )));
        }
        before += &counts.index_axis(Axis(0), k);
        if let Some(((row, column), b)) =
            before.indexed_iter().find(|(_, b)| **b > shutter.triggers)
        {
            return Err(IoError::InvalidParameter(format!(
                "pixel row {row}, column {column} has {b} counts by bin {k}, \
                 more than the shutter's {} triggers",
                shutter.triggers
            )));
        }
    }
    Ok(live)
}

/// The standard deviation of a live fraction `L` from a binomial count of hits.
///
/// # References
/// K. Watanabe et al., Nucl. Instrum. Methods A 861 (2017) 55, Eqs. 4–5.
pub fn live_fraction_sigma(live: f64, triggers: f64) -> f64 {
    (live * (1.0 - live) / triggers).sqrt()
}

/// A region's live fraction in each bin: its counts over their arrivals, `Σ C / Σ (C / L)`.
///
/// A bin with no counts in the region takes the mean `L` of its pixels.
///
/// # Arguments
/// * `counts` — Raw counts in (time bin, row, column), finite and 0 or more.
/// * `live` — Live fractions in (0, 1] of `counts`, the same shape.
/// * `cells` — The region's pixels as `[row, column]`, at least one, each once.
///
/// # Errors
/// [`IoError::ShapeMismatch`] if the shapes differ or a cell is outside them;
/// [`IoError::InvalidParameter`] if `cells` or the bins are empty, a cell
/// repeats, a region
/// count is not finite and 0 or more or its live fraction not in (0, 1], or
/// the counts overflow the sums.
pub fn region_live(
    counts: ArrayView3<f64>,
    live: ArrayView3<f64>,
    cells: &[[usize; 2]],
) -> Result<Vec<f64>, IoError> {
    if counts.shape() != live.shape() {
        return Err(IoError::ShapeMismatch(format!(
            "counts {:?} and live fractions {:?}",
            counts.shape(),
            live.shape()
        )));
    }
    if cells.is_empty() || counts.is_empty() {
        return Err(IoError::InvalidParameter(
            "a region needs a pixel and a bin".into(),
        ));
    }
    let (rows, columns) = (counts.shape()[1], counts.shape()[2]);
    if let Some(cell) = cells.iter().find(|[r, c]| *r >= rows || *c >= columns) {
        return Err(IoError::ShapeMismatch(format!(
            "cell {cell:?} is outside the {rows} × {columns} image"
        )));
    }
    let mut sorted = cells.to_vec();
    sorted.sort_unstable();
    if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
        return Err(IoError::InvalidParameter(format!(
            "cell {:?} is in the region twice",
            w[0]
        )));
    }
    for k in 0..counts.len_of(Axis(0)) {
        for &[r, c] in cells {
            let (n, l) = (counts[[k, r, c]], live[[k, r, c]]);
            if !(n.is_finite() && n >= 0.0 && l > 0.0 && l <= 1.0) {
                return Err(IoError::InvalidParameter(format!(
                    "bin {k}, cell {:?}: count {n} and live fraction {l}",
                    [r, c]
                )));
            }
        }
    }
    (0..counts.len_of(Axis(0)))
        .map(|k| {
            let (mut recorded, mut arrived, mut sum) = (0.0, 0.0, 0.0);
            for &[r, c] in cells {
                let (n, l) = (counts[[k, r, c]], live[[k, r, c]]);
                recorded += n;
                arrived += n / l;
                sum += l;
            }
            let l = if recorded > 0.0 {
                recorded / arrived
            } else {
                sum / cells.len() as f64
            };
            if l > 0.0 && l <= 1.0 {
                Ok(l)
            } else {
                Err(IoError::InvalidParameter(format!(
                    "bin {k}: the region's live fraction {l} is out of range; its counts overflow"
                )))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn shutter(start_us: f64, end_us: f64, triggers: f64) -> Shutter {
        Shutter {
            start_us,
            end_us,
            triggers,
        }
    }

    #[test]
    fn a_bin_sees_the_counts_before_it_in_its_shutter() {
        let counts = Array3::from_shape_vec((4, 1, 1), vec![2.0, 3.0, 5.0, 4.0]).unwrap();
        let shutters = [shutter(0.0, 3.0, 10.0), shutter(3.0, 4.0, 8.0)];
        let live = live_fractions(counts.view(), &[0.0, 1.0, 2.0, 3.0, 4.0], &shutters).unwrap();
        assert_eq!(live.as_slice().unwrap(), &[1.0, 0.8, 0.5, 1.0]);

        let near = [shutter(0.0, 1.0, 10.0), shutter(1.0, 2.0, 10.0)];
        let two = Array3::from_shape_vec((2, 1, 1), vec![2.0, 3.0]).unwrap();
        assert!(live_fractions(two.view(), &[0.0, 0.9995, 1.9995], &near).is_err());

        let with_gap = [shutter(0.0, 0.9, 10.0), shutter(1.0, 2.0, 10.0)];
        let live = live_fractions(two.view(), &[0.0, 0.9999995, 1.9999995], &with_gap).unwrap();
        assert_eq!(live[[1, 0, 0]], 1.0);
    }

    #[test]
    fn counts_that_reach_the_triggers_or_a_bin_outside_the_shutters_are_refused() {
        let counts = Array3::from_shape_vec((2, 1, 1), vec![10.0, 1.0]).unwrap();
        let edges = [0.0, 1.0, 2.0];
        assert!(live_fractions(counts.view(), &edges, &[shutter(0.0, 2.0, 10.0)]).is_err());
        assert!(live_fractions(counts.view(), &edges, &[shutter(0.0, 1.0, 20.0)]).is_err());
        assert!(live_fractions(counts.view(), &edges[..2], &[shutter(0.0, 2.0, 20.0)]).is_err());
        let last = Array3::from_shape_vec((2, 1, 1), vec![6.0, 6.0]).unwrap();
        assert!(live_fractions(last.view(), &edges, &[shutter(0.0, 2.0, 10.0)]).is_err());
        let overlap = [shutter(0.0, 2.0, 10.0), shutter(1.0, 3.0, 100.0)];
        assert!(live_fractions(counts.view(), &edges, &overlap).is_err());
    }

    #[test]
    fn a_region_weights_each_pixel_by_its_arrivals() {
        let counts = Array3::from_shape_vec((2, 1, 2), vec![6.0, 2.0, 0.0, 0.0]).unwrap();
        let live = Array3::from_shape_vec((2, 1, 2), vec![0.5, 1.0, 0.6, 0.8]).unwrap();
        let region = region_live(counts.view(), live.view(), &[[0, 0], [0, 1]]).unwrap();
        assert!((region[0] - 8.0 / 14.0).abs() < 1e-15);
        assert!((region[1] - 0.7).abs() < 1e-15);
        assert!(region_live(counts.view(), live.view(), &[[1, 0]]).is_err());
        let dead = Array3::from_shape_vec((2, 1, 2), vec![0.0, 1.0, 0.6, 0.8]).unwrap();
        assert!(region_live(counts.view(), dead.view(), &[[0, 0]]).is_err());
        assert!(region_live(counts.view(), live.view(), &[[0, 0], [0, 1], [0, 0]]).is_err());
    }

    /// Recorded counts of a pixel that keeps the first neutron of each shutter.
    fn first_hits(rates: &[f64], shutters: &[(usize, usize, u64)], seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut uniform = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut counts = vec![0.0; rates.len()];
        for &(first, end, triggers) in shutters {
            let cumulative: Vec<f64> = rates[first..end]
                .iter()
                .scan(0.0, |sum, r| {
                    *sum += r;
                    Some(*sum)
                })
                .collect();
            for _ in 0..triggers {
                let arrival = -(1.0 - uniform()).ln();
                let k = cumulative.partition_point(|&c| c <= arrival);
                if k < cumulative.len() {
                    counts[first + k] += 1.0;
                }
            }
        }
        counts
    }

    #[test]
    fn the_live_fraction_recovers_the_arrivals_of_first_hit_counts() {
        let bins = 400;
        let rates: Vec<f64> = (0..bins)
            .map(|k| match k {
                0..5 => 0.01,
                5..200 => 1.2e-3,
                _ => 1.5e-3,
            })
            .collect();
        let triggers = [400_000_u64, 300_000];
        let recorded = first_hits(
            &rates,
            &[(0, 200, triggers[0]), (200, bins, triggers[1])],
            7,
        );
        let shutters = [
            shutter(0.0, 200.0, triggers[0] as f64),
            shutter(200.0, bins as f64, triggers[1] as f64),
        ];
        let edges: Vec<f64> = (0..=bins).map(|k| k as f64).collect();
        let counts = Array3::from_shape_vec((bins, 1, 1), recorded.clone()).unwrap();
        let live = live_fractions(counts.view(), &edges, &shutters).unwrap();
        assert_eq!(live[[200, 0, 0]], 1.0);

        // A recorded count in bin k needs one arrival there, P = 1 − exp(−r_k).
        for group in (0..bins).step_by(20) {
            let range = group..group + 20;
            let s = if group < 200 {
                triggers[0]
            } else {
                triggers[1]
            } as f64;
            let truth: f64 = range.clone().map(|k| s * -(-rates[k]).exp_m1()).sum();
            let found: f64 = range.clone().map(|k| recorded[k] / live[[k, 0, 0]]).sum();
            let raw: f64 = range.clone().map(|k| recorded[k]).sum();
            let sigma = raw.sqrt() * found / raw;
            assert!(
                (found - truth).abs() < 4.0 * sigma,
                "bins {range:?}: {found} arrivals found, {truth} expected, σ {sigma}"
            );
            if (60..200).contains(&group) || group >= 260 {
                assert!(
                    (raw - truth).abs() > 5.0 * sigma,
                    "bins {range:?} lose too little"
                );
            }
        }
    }
}
