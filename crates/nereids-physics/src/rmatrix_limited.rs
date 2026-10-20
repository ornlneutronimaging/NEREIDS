//! R-Matrix Limited (LRF=7) cross sections in the Reich-Moore form (KRM=3).
//!
//! Each spin group uses X = P^½ (I − R L)⁻¹ R P^½, which equals Eq. (D.47)
//! without a division by P, over its open particle channels.
//!
//! # References
//! - ENDF-6 Formats Manual, ENDF-102 (2023), Appendix D.1.7, Eqs. (D.37)–(D.54),
//!   (D.77) and (D.82).
//! - N. M. Larson, SAMMY Users' Guide, ORNL/TM-9179/R8 (2008), §II.A and §II.B.1.

use num_complex::Complex64;

use nereids_endf::resonance::{RmlBackground, RmlRange};

use crate::channel;
use crate::penetrability;
use crate::reich_moore::CrossSections;

/// Energy-independent data of one LRF=7 range.
#[derive(Debug, Clone)]
pub(crate) struct RmlPlan {
    /// Target-to-neutron mass ratio A of the incident pair (MB/MA).
    awr: f64,
    groups: Vec<GroupPlan>,
}

#[derive(Debug, Clone)]
struct GroupPlan {
    /// Spin statistical factor g_J of the incident pair, Eq. (D.38).
    g_j: f64,
    /// Particle channels; the eliminated photon channel is not among them.
    channels: Vec<ChannelPlan>,
    levels: Vec<Level>,
    /// Background R-functions as (index in `channels`, parameters).
    background: Vec<(usize, RmlBackground)>,
}

#[derive(Debug, Clone)]
struct ChannelPlan {
    /// Whether the channel's pair is the incident pair.
    entrance: bool,
    /// Lab-energy threshold in eV, −Q·(A+1)/A; the channel is open above it.
    threshold: f64,
    /// Reduced mass of the pair in neutron masses.
    reduced_mass: f64,
    l: u32,
    penetrability: bool,
    /// Boundary condition B when the pair calculates the shift factor.
    boundary: Option<f64>,
    radius_true: f64,
    radius_phase: f64,
}

#[derive(Debug, Clone)]
struct Level {
    energy: f64,
    /// Half the eliminated photon width, |Γ_λγ|/2, in eV.
    half_photon_width: f64,
    /// Reduced-width amplitude γ_λc per particle channel, in √eV.
    amplitudes: Vec<f64>,
}

impl ChannelPlan {
    /// Channel wave number in fm⁻¹ at lab energy `energy_ev` above threshold, Eq. (D.82).
    fn wave_number(&self, energy_ev: f64, cm_fraction: f64) -> f64 {
        channel::wave_number_from_cm(
            (energy_ev - self.threshold) * cm_fraction,
            self.reduced_mass,
        )
    }
}

impl RmlPlan {
    /// Build the plan of `rml`: thresholds, wave-number constants and
    /// reduced-width amplitudes, γ = sign(Γ)·√(|Γ| / 2P(|E_λ − E_c|)).
    ///
    /// # Panics
    /// Panics if an amplitude is not finite, which the parser's threshold
    /// check excludes.
    pub(crate) fn new(rml: &RmlRange) -> Self {
        let incident = &rml.pairs[rml.incident_pair];
        let awr = incident.mass_b / incident.mass_a;
        let cm_fraction = awr / (awr + 1.0);
        let groups = rml
            .spin_groups
            .iter()
            .map(|group| {
                let mut channels = Vec::new();
                let mut photon = None;
                for (index, ch) in group.channels.iter().enumerate() {
                    if ch.pair == rml.photon_pair {
                        photon = Some(index);
                        continue;
                    }
                    let pair = &rml.pairs[ch.pair];
                    channels.push((
                        index,
                        ChannelPlan {
                            entrance: ch.pair == rml.incident_pair,
                            threshold: -pair.q_value / cm_fraction,
                            reduced_mass: pair.mass_a * pair.mass_b / (pair.mass_a + pair.mass_b),
                            l: ch.l,
                            penetrability: pair.penetrability,
                            boundary: pair.shift.then_some(ch.boundary),
                            radius_true: ch.radius_true,
                            radius_phase: ch.radius_phase,
                        },
                    ));
                }
                let photon = photon.expect("the parser requires one photon channel per group");
                let levels = group
                    .resonances
                    .iter()
                    .map(|res| Level {
                        energy: res.energy,
                        half_photon_width: 0.5 * res.widths[photon].abs(),
                        amplitudes: channels
                            .iter()
                            .map(|(index, ch)| {
                                amplitude(res.widths[*index], res.energy, ch, cm_fraction)
                            })
                            .collect(),
                    })
                    .collect();
                let background = group
                    .background
                    .iter()
                    .map(|b| {
                        let index = channels
                            .iter()
                            .position(|(i, _)| *i == b.channel)
                            .expect("the parser keeps backgrounds off the photon channel");
                        (index, b.clone())
                    })
                    .collect();
                let spin_a = incident.spin_a.abs();
                let spin_b = incident.spin_b.abs();
                GroupPlan {
                    g_j: (2.0 * group.j + 1.0) / ((2.0 * spin_a + 1.0) * (2.0 * spin_b + 1.0)),
                    channels: channels.into_iter().map(|(_, ch)| ch).collect(),
                    levels,
                    background,
                }
            })
            .collect();
        Self { awr, groups }
    }

    /// Cross sections in barns at lab energy `energy_ev` > 0 eV, Eqs. (D.50)–(D.54).
    ///
    /// Channels closed at `energy_ev` decouple: the parser admits only closed
    /// channels without a shift factor, for which L⁻¹ is infinite.
    ///
    /// # Panics
    /// Panics if the level matrix I − R·L is singular.
    pub(crate) fn evaluate(&self, energy_ev: f64) -> CrossSections {
        let cm_fraction = self.awr / (self.awr + 1.0);
        let four_pi_over_k2 = 4.0 * channel::pi_over_k_squared_barns(energy_ev, self.awr);
        let mut xs = CrossSections::default();
        for group in &self.groups {
            let open: Vec<usize> = (0..group.channels.len())
                .filter(|&c| energy_ev > group.channels[c].threshold)
                .collect();
            let n = open.len();
            if n == 0 {
                continue;
            }

            let mut sqrt_p = vec![0.0; n];
            let mut l_matrix = vec![Complex64::new(0.0, 0.0); n];
            let mut phase = vec![0.0; n];
            for (i, &c) in open.iter().enumerate() {
                let ch = &group.channels[c];
                if ch.penetrability {
                    let k = ch.wave_number(energy_ev, cm_fraction);
                    let rho = k * ch.radius_true;
                    let p = penetrability::penetrability(ch.l, rho);
                    let s_minus_b = ch
                        .boundary
                        .map_or(0.0, |b| penetrability::shift_factor(ch.l, rho) - b);
                    sqrt_p[i] = p.sqrt();
                    l_matrix[i] = Complex64::new(s_minus_b, p);
                    phase[i] = penetrability::phase_shift(ch.l, k * ch.radius_phase);
                } else {
                    sqrt_p[i] = 1.0;
                    l_matrix[i] = Complex64::new(0.0, 1.0);
                }
            }

            // R_cc' = Σ_λ γ_λc γ_λc' / (E_λ − E − iΓ_λγ/2) + δ_cc' R^bkg_c, Eq. (D.44).
            let mut r = vec![Complex64::new(0.0, 0.0); n * n];
            for level in &group.levels {
                let de = level.energy - energy_ev;
                let hw = level.half_photon_width;
                let denominator = de * de + hw * hw;
                assert!(
                    denominator > 0.0,
                    "LRF=7 energy {energy_ev} eV hits a level with zero photon width"
                );
                let pole = Complex64::new(de / denominator, hw / denominator);
                for (i, &ci) in open.iter().enumerate() {
                    let gi = level.amplitudes[ci];
                    if gi == 0.0 {
                        continue;
                    }
                    for (j, &cj) in open.iter().enumerate() {
                        r[i * n + j] += pole * (gi * level.amplitudes[cj]);
                    }
                }
            }
            for (c, background) in &group.background {
                if let Some(i) = open.iter().position(|o| o == c) {
                    r[i * n + i] += background.evaluate(energy_ev);
                }
            }

            // Y = (I − R L)⁻¹ R, then X = P^½ Y P^½.
            let mut m = vec![Complex64::new(0.0, 0.0); n * n];
            for i in 0..n {
                for j in 0..n {
                    let delta = if i == j { 1.0 } else { 0.0 };
                    m[i * n + j] = delta - r[i * n + j] * l_matrix[j];
                }
            }
            let mut x = r;
            assert!(
                solve_in_place(&mut m, &mut x, n),
                "LRF=7 level matrix I - R L is singular at {energy_ev} eV"
            );
            for i in 0..n {
                for j in 0..n {
                    x[i * n + j] *= sqrt_p[i] * sqrt_p[j];
                }
            }

            let scale = four_pi_over_k2 * group.g_j;
            for (i, &ci) in open.iter().enumerate() {
                if !group.channels[ci].entrance {
                    continue;
                }
                let xii = x[i * n + i];
                let (sin_2phi, cos_2phi) = (2.0 * phase[i]).sin_cos();
                let sin2_phi = phase[i].sin().powi(2);
                let mut to_entrance = 0.0;
                let mut to_exit = 0.0;
                for (j, &cj) in open.iter().enumerate() {
                    let flux = x[i * n + j].norm_sqr();
                    if group.channels[cj].entrance {
                        to_entrance += flux;
                    } else {
                        to_exit += flux;
                    }
                }
                xs.total += scale * (sin2_phi + xii.im * cos_2phi - xii.re * sin_2phi);
                xs.elastic +=
                    scale * (sin2_phi * (1.0 - 2.0 * xii.im) - xii.re * sin_2phi + to_entrance);
                xs.capture += scale * (xii.im - to_entrance - to_exit);
                xs.reaction += scale * to_exit;
            }
        }
        xs
    }
}

/// Reduced-width amplitude of width `width` (eV) at resonance energy `energy` (eV).
fn amplitude(width: f64, energy: f64, ch: &ChannelPlan, cm_fraction: f64) -> f64 {
    if width == 0.0 {
        return 0.0;
    }
    let p = if ch.penetrability {
        let k = channel::wave_number_from_cm(
            (energy - ch.threshold).abs() * cm_fraction,
            ch.reduced_mass,
        );
        penetrability::penetrability(ch.l, k * ch.radius_true)
    } else {
        1.0
    };
    let gamma = (0.5 * width.abs() / p).sqrt().copysign(width);
    assert!(
        gamma.is_finite(),
        "LRF=7 amplitude is not finite: width {width} eV at {energy} eV, P = {p}"
    );
    gamma
}

/// Solve `a · y = b` for the `n × n` row-major matrices, overwriting `b` with
/// `y`; partial pivoting. Returns `false` for a zero or non-finite pivot.
fn solve_in_place(a: &mut [Complex64], b: &mut [Complex64], n: usize) -> bool {
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&p, &q| a[p * n + col].norm().total_cmp(&a[q * n + col].norm()))
            .expect("non-empty column");
        let pivot_value = a[pivot * n + col];
        if pivot_value.norm() == 0.0 || !pivot_value.is_finite() {
            return false;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
                b.swap(col * n + k, pivot * n + k);
            }
        }
        let inverse = 1.0 / a[col * n + col];
        for row in col + 1..n {
            let factor = a[row * n + col] * inverse;
            if factor == Complex64::new(0.0, 0.0) {
                continue;
            }
            for k in col..n {
                let v = a[col * n + k];
                a[row * n + k] -= factor * v;
            }
            for k in 0..n {
                let v = b[col * n + k];
                b[row * n + k] -= factor * v;
            }
        }
    }
    for col in (0..n).rev() {
        let inverse = 1.0 / a[col * n + col];
        for k in 0..n {
            b[col * n + k] *= inverse;
        }
        for row in 0..col {
            let factor = a[row * n + col];
            if factor == Complex64::new(0.0, 0.0) {
                continue;
            }
            for k in 0..n {
                let v = b[col * n + k];
                b[row * n + k] -= factor * v;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_endf::parser::parse_endf_file2;
    use nereids_endf::resonance::{
        LGroup, ParticlePair, Resonance, ResonanceData, ResonanceFormalism, ResonanceRange,
        RmlChannel, RmlResonance, RmlSpinGroup,
    };

    use crate::reich_moore::cross_sections_on_grid;

    const A: f64 = 180.0;

    fn photon() -> ParticlePair {
        ParticlePair {
            mass_a: 0.0,
            mass_b: A + 1.0,
            charge_a: 0.0,
            charge_b: 74.0,
            spin_a: 1.0,
            spin_b: 0.0,
            q_value: 0.0,
            mt: 102,
            penetrability: false,
            shift: false,
        }
    }

    fn neutron() -> ParticlePair {
        ParticlePair {
            mass_a: 1.0,
            mass_b: A,
            charge_a: 0.0,
            charge_b: 74.0,
            spin_a: 0.5,
            spin_b: 0.0,
            q_value: 0.0,
            mt: 2,
            penetrability: true,
            shift: false,
        }
    }

    fn channel(pair: usize, l: u32, radius: f64) -> RmlChannel {
        RmlChannel {
            pair,
            l,
            channel_spin: 0.5,
            boundary: 0.0,
            radius_phase: radius,
            radius_true: radius,
        }
    }

    fn group(j: f64, channels: Vec<RmlChannel>, resonances: Vec<RmlResonance>) -> RmlSpinGroup {
        RmlSpinGroup {
            j,
            parity: 1,
            channels,
            resonances,
            background: vec![],
        }
    }

    fn data(rml: RmlRange, awr: f64, energy_high: f64) -> ResonanceData {
        ResonanceData {
            isotope: nereids_core::types::Isotope::new(74, 180).unwrap(),
            za: 74180,
            awr,
            smooth: Default::default(),
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high,
                resolved: true,
                formalism: ResonanceFormalism::RMatrixLimited,
                target_spin: 0.0,
                scattering_radius: 0.0,
                naps: 0,
                ap_table: None,
                l_groups: vec![],
                r_external: vec![],
                r_matrix_limited: Some(rml),
            }],
        }
    }

    fn log_grid(low: f64, high: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| low * (high / low).powf(i as f64 / (n - 1) as f64))
            .collect()
    }

    /// No levels: σ_t = (4π/k²) Σ g_J sin²φ_c, all of it elastic.
    #[test]
    fn hard_sphere_limit() {
        let radius = 7.0;
        let rml = RmlRange {
            pairs: vec![photon(), neutron()],
            incident_pair: 1,
            photon_pair: 0,
            spin_groups: vec![
                group(0.5, vec![channel(0, 0, 0.0), channel(1, 0, radius)], vec![]),
                group(1.5, vec![channel(0, 0, 0.0), channel(1, 1, radius)], vec![]),
            ],
        };
        let plan = RmlPlan::new(&rml);
        for e in [1e-3, 1.0, 1e3, 1e5] {
            let xs = plan.evaluate(e);
            let k = channel::wave_number(e, A);
            let phi0 = penetrability::phase_shift(0, k * radius);
            let phi1 = penetrability::phase_shift(1, k * radius);
            let expected = 4.0
                * channel::pi_over_k_squared_barns(e, A)
                * (1.0 * phi0.sin().powi(2) + 2.0 * phi1.sin().powi(2));
            assert!(
                (xs.total - expected).abs() <= 1e-13 * expected,
                "{e}: {xs:?}"
            );
            assert!(
                (xs.elastic - expected).abs() <= 1e-13 * expected,
                "{e}: {xs:?}"
            );
            assert_eq!((xs.capture, xs.reaction, xs.fission), (0.0, 0.0, 0.0));
        }
    }

    /// One level, one s-wave channel: capture is the Breit-Wigner form
    /// (π/k²) g Γ_n(E) Γ_γ / ((E − E₀)² + Γ²/4), with Γ_n(E) = Γ_n P(E)/P(E₀).
    #[test]
    fn single_level_capture_is_breit_wigner() {
        let (e0, gn, gg, radius) = (21.1, 0.04, 0.05, 7.64);
        let rml = RmlRange {
            pairs: vec![photon(), neutron()],
            incident_pair: 1,
            photon_pair: 0,
            spin_groups: vec![group(
                0.5,
                vec![channel(0, 0, 0.0), channel(1, 0, radius)],
                vec![RmlResonance {
                    energy: e0,
                    widths: vec![gg, gn],
                }],
            )],
        };
        let plan = RmlPlan::new(&rml);
        let p = |e: f64| penetrability::penetrability(0, channel::wave_number(e, A) * radius);
        for e in [0.5, 10.0, 20.9, 21.1, 21.3, 50.0, 200.0] {
            let xs = plan.evaluate(e);
            let gn_e = gn * p(e) / p(e0);
            let gamma = gn_e + gg;
            let expected = channel::pi_over_k_squared_barns(e, A) * gn_e * gg
                / ((e - e0).powi(2) + gamma * gamma / 4.0);
            assert!(
                (xs.capture - expected).abs() <= 1e-12 * expected,
                "{e}: {} vs {expected}",
                xs.capture
            );
        }
    }

    fn inelastic(q_value: f64) -> ParticlePair {
        ParticlePair {
            q_value,
            mt: 51,
            ..neutron()
        }
    }

    /// Two levels over an elastic and an inelastic channel: the parts sum to
    /// the total, and below the threshold the inelastic channel decouples.
    #[test]
    fn open_inelastic_channel_conserves_flux_and_closes_below_threshold() {
        let threshold = 30.0 * (A + 1.0) / A;
        let levels = vec![
            RmlResonance {
                energy: 25.0,
                widths: vec![0.04, 0.3, 0.2],
            },
            RmlResonance {
                energy: 60.0,
                widths: vec![0.05, -0.5, 0.4],
            },
        ];
        let two_channels = RmlRange {
            pairs: vec![photon(), neutron(), inelastic(-30.0)],
            incident_pair: 1,
            photon_pair: 0,
            spin_groups: vec![group(
                0.5,
                vec![channel(0, 0, 0.0), channel(1, 0, 7.0), channel(2, 0, 7.0)],
                levels.clone(),
            )],
        };
        let elastic_only = RmlRange {
            pairs: vec![photon(), neutron()],
            incident_pair: 1,
            photon_pair: 0,
            spin_groups: vec![group(
                0.5,
                vec![channel(0, 0, 0.0), channel(1, 0, 7.0)],
                levels
                    .iter()
                    .map(|r| RmlResonance {
                        energy: r.energy,
                        widths: r.widths[..2].to_vec(),
                    })
                    .collect(),
            )],
        };
        let (open, closed) = (RmlPlan::new(&two_channels), RmlPlan::new(&elastic_only));
        for e in [1.0, 20.0, 25.0, 30.0, threshold * (1.0 - 1e-9)] {
            let (a, b) = (open.evaluate(e), closed.evaluate(e));
            assert_eq!(a.reaction, 0.0, "{e}");
            assert!((a.total - b.total).abs() <= 1e-14 * b.total, "{e}");
            assert!((a.capture - b.capture).abs() <= 1e-14 * b.capture, "{e}");
        }
        for e in [threshold * (1.0 + 1e-9), 40.0, 59.9, 60.0, 61.0, 150.0] {
            let xs = open.evaluate(e);
            assert!(xs.reaction > 0.0, "{e}");
            let parts = xs.elastic + xs.capture + xs.reaction;
            assert!((xs.total - parts).abs() <= 1e-12 * xs.total, "{e}: {xs:?}");
        }
    }

    /// Two entrance channels (two channel spins) and a background term: the
    /// parts still sum to the total.
    #[test]
    fn two_entrance_channels_with_background_conserve_flux() {
        let mut target = neutron();
        target.spin_b = 0.5;
        let mut g = group(
            1.0,
            vec![
                channel(0, 0, 0.0),
                RmlChannel {
                    channel_spin: 0.0,
                    ..channel(1, 1, 7.6)
                },
                RmlChannel {
                    channel_spin: 1.0,
                    ..channel(1, 1, 7.6)
                },
            ],
            vec![RmlResonance {
                energy: 8.0,
                widths: vec![0.06, 0.003, -0.002],
            }],
        );
        g.background = vec![RmlBackground {
            channel: 2,
            e_down: 0.0,
            e_up: 1e4,
            r_con: 0.2,
            r_lin: 1e-5,
            r_quad: 0.0,
            s_con: 0.05,
            s_lin: 1e-6,
        }];
        let rml = RmlRange {
            pairs: vec![photon(), target],
            incident_pair: 1,
            photon_pair: 0,
            spin_groups: vec![g],
        };
        let plan = RmlPlan::new(&rml);
        for e in log_grid(0.5, 200.0, 400) {
            let xs = plan.evaluate(e);
            assert_eq!(xs.reaction, 0.0);
            let parts = xs.elastic + xs.capture;
            assert!((xs.total - parts).abs() <= 1e-12 * xs.total, "{e}: {xs:?}");
            assert!(xs.capture >= 0.0 && xs.elastic > 0.0, "{e}: {xs:?}");
        }
    }

    /// An l = 3 channel at 0.5 eV has P ≈ 1e-20; the result stays finite.
    #[test]
    fn tiny_penetrability_stays_finite() {
        let rml = RmlRange {
            pairs: vec![photon(), neutron()],
            incident_pair: 1,
            photon_pair: 0,
            spin_groups: vec![group(
                2.5,
                vec![channel(0, 0, 0.0), channel(1, 3, 6.0)],
                vec![RmlResonance {
                    energy: 3.0e4,
                    widths: vec![0.3, 2.0],
                }],
            )],
        };
        let xs = RmlPlan::new(&rml).evaluate(0.5);
        assert!(xs.total.is_finite() && xs.capture.is_finite() && xs.total > 0.0);
    }

    fn mf2_fixture(name: &str) -> ResonanceData {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/data/endf/mf2")
            .join(format!("{name}.endf"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        parse_endf_file2(&text).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    /// Total, elastic and capture of two evaluations agree within
    /// 1e-9 of the total plus 1e-13·4π/k². The LRF=3 path forms the total as
    /// 1 − Re U, whose round-off is a few ε times 4π/k².
    fn assert_same(name: &str, a: &ResonanceData, b: &ResonanceData, energies: &[f64], awr: f64) {
        let (xa, xb) = (
            cross_sections_on_grid(a, energies),
            cross_sections_on_grid(b, energies),
        );
        let mut worst = (0.0_f64, 0.0);
        for ((&e, p), q) in energies.iter().zip(&xa).zip(&xb) {
            let allowed = 1e-9 * p.total.max(q.total)
                + 1e-13 * 4.0 * channel::pi_over_k_squared_barns(e, awr);
            for (u, v) in [
                (p.total, q.total),
                (p.elastic, q.elastic),
                (p.capture, q.capture),
            ] {
                let ratio = (u - v).abs() / allowed;
                if ratio > worst.0 {
                    worst = (ratio, e);
                }
            }
        }
        assert!(
            worst.0 <= 1.0,
            "{name}: difference is {:.3} times the allowance at {} eV",
            worst.0,
            worst.1
        );
    }

    /// Grid of `n` log-spaced points over the range plus each level's E and E ± Γ/2.
    fn grid_with_levels(low: f64, high: f64, n: usize, levels: &[(f64, f64)]) -> Vec<f64> {
        let mut energies = log_grid(low, high, n);
        for &(e, width) in levels {
            for x in [e - width / 2.0, e, e + width / 2.0] {
                if x > low && x < high {
                    energies.push(x);
                }
            }
        }
        energies.sort_by(f64::total_cmp);
        energies
    }

    /// Fe-56 (ENDF/B-VIII.1, LRF=3) rewritten as LRF=7, one spin group per
    /// (l, J), evaluates as the LRF=3 path does.
    #[test]
    fn fe56_lrf3_and_its_lrf7_form_agree() {
        let lrf3 = mf2_fixture("Fe-56");
        let range = &lrf3.ranges[0];
        assert_eq!(range.formalism, ResonanceFormalism::ReichMoore);
        assert_eq!(range.target_spin, 0.0);
        // The LRF=3 path uses each L-group's AWRI, here 55.4545 for every group.
        let awr = range.l_groups[0].awr;
        assert!(range.l_groups.iter().all(|lg| lg.awr == awr));
        let mut pairs = vec![photon(), neutron()];
        pairs[0].mass_b = awr + 1.0;
        pairs[1].mass_b = awr;
        let mut spin_groups = Vec::new();
        let mut levels = Vec::new();
        for lg in &range.l_groups {
            let phase = if lg.apl > 0.0 {
                lg.apl
            } else {
                range.scattering_radius
            };
            let true_radius = if lg.apl > 0.0 {
                lg.apl
            } else if range.naps == 0 {
                channel::endf_channel_radius_fm(awr)
            } else {
                range.scattering_radius
            };
            let twice_l = 2 * lg.l as i64;
            for twice_j in ((twice_l - 1).abs()..=twice_l + 1).step_by(2) {
                let j = twice_j as f64 / 2.0;
                let resonances: Vec<RmlResonance> = lg
                    .resonances
                    .iter()
                    .filter(|r| (r.j.abs() - j).abs() < 1e-9)
                    .map(|r: &Resonance| {
                        levels.push((r.energy, r.gn.abs() + r.gg));
                        RmlResonance {
                            energy: r.energy,
                            widths: vec![r.gg, r.gn],
                        }
                    })
                    .collect();
                let neutron_channel = RmlChannel {
                    radius_phase: phase,
                    radius_true: true_radius,
                    ..channel(1, lg.l, 0.0)
                };
                spin_groups.push(group(
                    j,
                    vec![channel(0, 0, 0.0), neutron_channel],
                    resonances,
                ));
            }
        }
        let rml = RmlRange {
            pairs,
            incident_pair: 1,
            photon_pair: 0,
            spin_groups,
        };
        let mut lrf7 = data(rml, awr, range.energy_high);
        lrf7.ranges[0].energy_low = range.energy_low;
        let energies = grid_with_levels(1e-5, range.energy_high, 20_000, &levels);
        assert_same("Fe-56", &lrf3, &lrf7, &energies, awr);
    }

    /// W-182 (ENDF/B-VIII.1, LRF=7, one s- or p-wave neutron channel per group,
    /// APE = APT) rewritten as LRF=3 evaluates as the LRF=7 path does.
    #[test]
    fn w182_lrf7_and_its_lrf3_form_agree() {
        let lrf7 = mf2_fixture("W-182");
        let range = &lrf7.ranges[0];
        let rml = range.r_matrix_limited.as_ref().unwrap();
        let mut l_groups: Vec<LGroup> = Vec::new();
        let mut levels = Vec::new();
        for g in &rml.spin_groups {
            let photon = g
                .channels
                .iter()
                .position(|c| c.pair == rml.photon_pair)
                .unwrap();
            let neutron = g
                .channels
                .iter()
                .position(|c| c.pair == rml.incident_pair)
                .unwrap();
            let ch = &g.channels[neutron];
            assert_eq!(ch.radius_phase, ch.radius_true);
            let index = match l_groups.iter().position(|lg| lg.l == ch.l) {
                Some(i) => i,
                None => {
                    l_groups.push(LGroup {
                        l: ch.l,
                        awr: lrf7.awr,
                        apl: ch.radius_true,
                        qx: 0.0,
                        lrx: 0,
                        resonances: vec![],
                    });
                    l_groups.len() - 1
                }
            };
            assert_eq!(l_groups[index].apl, ch.radius_true);
            for r in &g.resonances {
                levels.push((r.energy, r.widths[photon] + r.widths[neutron].abs()));
                l_groups[index].resonances.push(Resonance {
                    energy: r.energy,
                    j: g.j,
                    gn: r.widths[neutron],
                    gg: r.widths[photon],
                    gfa: 0.0,
                    gfb: 0.0,
                });
            }
        }
        let mut lrf3 = lrf7.clone();
        lrf3.ranges.truncate(1);
        let lrf7_only = ResonanceData {
            ranges: vec![lrf7.ranges[0].clone()],
            ..lrf7.clone()
        };
        let r = &mut lrf3.ranges[0];
        r.formalism = ResonanceFormalism::ReichMoore;
        r.r_matrix_limited = None;
        r.scattering_radius = l_groups[0].apl;
        r.naps = 1;
        r.l_groups = l_groups;
        let energies = grid_with_levels(1e-5, r.energy_high, 20_000, &levels);
        assert_same("W-182", &lrf7_only, &lrf3, &energies, lrf7.awr);
    }
}
