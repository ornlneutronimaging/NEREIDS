//! Oak Ridge (ORR) resolution broadening.
//!
//! This implements the ORR pathways used by SAMMY ex007:
//! - water moderator (`kwatta=2`) with analytic weighting (`morr5`)
//! - tantalum target (`kwatta=1`) with numeric convolution using the same
//!   parameter transforms and component kernels (`morr3/morr4/morr6`)

use nereids_core::{EnergyGrid, PhysicsError, ResolutionFunction};

const SM2: f64 = 72.298_252_179_105_06; // SAMMY sqrt(m/2) constant, us*sqrt(eV)/m
const MIN_NORM: f64 = 1e-300;

/// Channel-width region: use `width_ns` for `E <= max_energy_ev`.
#[derive(Debug, Clone, Copy)]
pub struct OrrChannelWidth {
    pub max_energy_ev: f64,
    pub width_ns: f64,
}

/// ORR detector model.
#[derive(Debug, Clone)]
pub enum OrrDetector {
    /// Lithium-glass detector (`lithne=2` in SAMMY).
    LithiumGlass { d_ns: f64, f_inv_ns: f64, g: f64 },
    /// NE110 detector (`lithne=1` in SAMMY).
    ///
    /// `delta_mm` is the detector thickness-like parameter (PAR 9).
    /// `lambda_sigma_mm` is optional tabulated lambda*sigma vs energy (mm).
    /// If empty, `lambda_sigma_constant_mm` is used.
    Ne110 {
        delta_mm: f64,
        lambda_sigma_constant_mm: f64,
        lambda_sigma_mm: Vec<(f64, f64)>,
    },
}

/// ORR target/moderator model.
#[derive(Debug, Clone)]
pub enum OrrTarget {
    /// Water moderator model (`kwatta=2` in SAMMY).
    Water {
        lambda0_mm: f64,
        lambda1_mm: f64,
        lambda2_mm: f64,
        m: f64,
    },
    /// Tantalum target model (`kwatta=1` in SAMMY).
    ///
    /// These are the SAMMY ORR parameters 2..8 for tantalum.
    Tantalum {
        a_prime: f64,
        w_prime: f64,
        x1_prime: f64,
        x2_prime: f64,
        x3_prime: f64,
        x0_prime: f64,
        alpha: f64,
    },
}

/// ORR resolution parameters.
#[derive(Debug, Clone)]
pub struct OrrResolution {
    pub flight_path_m: f64,
    pub burst_width_ns: f64,
    pub target: OrrTarget,
    pub detector: OrrDetector,
    pub channel_widths: Vec<OrrChannelWidth>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DetectorMode {
    LithiumGlass,
    Ne110,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetMode {
    Water,
    Tantalum,
}

#[derive(Debug, Clone, Copy)]
struct EnergyParams {
    a_us_inv: f64,
    w_us_inv: f64,
    m: i32,
    x1_us: f64,
    x2_us: f64,
    x3_us: f64,
    x0_us: f64,
    alpha: f64,
    p_us: f64,
    c_us: f64,
    d_us: f64,
    f_us_inv: f64,
    g: f64,
    detector_mode: DetectorMode,
    target_mode: TargetMode,
    timej_us: f64,
    elow_ev: f64,
    eup_ev: f64,
}

#[derive(Debug, Clone)]
struct KernelPiecewise {
    numtim: usize,
    tsubm_us: [f64; 8],
    hhh: [[f64; 4]; 8], // [segment][term], term: 1..4 -> 0..3
}

fn qexp(x: f64) -> f64 {
    if -x < 100.0 {
        x.exp()
    } else {
        0.0
    }
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + a.abs() + b.abs())
}

fn time_from_energy_us(energy_ev: f64, flight_path_m: f64) -> f64 {
    SM2 * flight_path_m / energy_ev.sqrt()
}

fn compute_xcoef_weights(energies: &[f64]) -> Vec<f64> {
    let n = energies.len();
    if n < 4 {
        return vec![1.0; n];
    }

    let nn = n - 3;
    let nnp2 = n - 1;

    let mut x21 = vec![0.0; n];
    for i in 0..(n - 1) {
        x21[i] = energies[i + 1] - energies[i];
    }

    let mut x2 = vec![0.0; n];
    for i in 0..nnp2 {
        x2[i] = x21[i] * x21[i];
    }

    let mut x1 = vec![0.0; n];
    for i in 0..nnp2 {
        x1[i] = if x21[i] != 0.0 { 1.0 / x21[i] } else { 0.0 };
    }

    let mut wtmp = vec![0.0; n];
    for i in 0..nn {
        wtmp[i] = x2[i + 2] - x2[i];
    }
    for i in 0..nn {
        x2[i] = wtmp[i] * x1[i + 1];
    }
    for i in 0..nnp2 {
        x1[i] = 5.0 * x21[i];
    }

    let mut weights = vec![0.0; n];
    weights[0] = x1[0] + x21[1];
    weights[1] = x1[0] + x1[1] + x21[2] - x2[0];
    for k in 2..=(nn) {
        weights[k] = x21[k - 2] + x1[k - 1] + x1[k] + x21[k + 1] + x2[k - 2] - x2[k - 1];
    }
    weights[nn + 1] = x21[nn - 1] + x1[nn] + x1[nn + 1] + x2[nn - 1];
    weights[n - 1] = x21[nn] + x1[nn + 1];

    weights
}

fn map_m_value(raw_m: f64) -> i32 {
    let mut m = 10;
    if raw_m <= 9.5 {
        m = 9;
    }
    if raw_m <= 8.5 {
        m = 8;
    }
    if raw_m <= 7.5 {
        m = 7;
    }
    if raw_m <= 6.5 {
        m = 6;
    }
    if raw_m <= 5.5 {
        m = 5;
    }
    if raw_m <= 4.5 {
        m = 4;
    }
    if raw_m <= 3.5 {
        m = 3;
    }
    if raw_m <= 2.5 {
        m = 2;
    }
    if raw_m <= 1.5 {
        m = 1;
    }
    m
}

impl OrrResolution {
    fn interpolate_ne110_lambda_sigma_mm(
        &self,
        em_ev: f64,
        tab: &[(f64, f64)],
        fallback: f64,
    ) -> f64 {
        if tab.is_empty() {
            return fallback;
        }
        if tab.len() == 1 {
            return tab[0].1;
        }

        let mut idx = tab.partition_point(|(e, _)| *e < em_ev);
        if idx == 0 {
            idx = 1;
        }
        if idx >= tab.len() {
            idx = tab.len() - 1;
        }
        let (e0, y0) = tab[idx - 1];
        let (e1, y1) = tab[idx];
        if approx_eq(e0, e1) {
            y1
        } else {
            y0 + (em_ev - e0) * (y1 - y0) / (e1 - e0)
        }
    }

    fn channel_width_us(&self, energy_ev: f64) -> f64 {
        for c in &self.channel_widths {
            if energy_ev <= c.max_energy_ev {
                return c.width_ns / 1000.0;
            }
        }
        self.channel_widths
            .last()
            .map(|c| c.width_ns / 1000.0)
            .unwrap_or(0.0)
    }

    fn validate(&self) -> Result<(), PhysicsError> {
        if !self.flight_path_m.is_finite() || self.flight_path_m <= 0.0 {
            return Err(PhysicsError::InvalidParameter(
                "ORR flight_path_m must be finite and positive".to_string(),
            ));
        }
        for (name, value) in [("burst_width_ns", self.burst_width_ns)] {
            if !value.is_finite() {
                return Err(PhysicsError::InvalidParameter(format!(
                    "ORR {name} must be finite"
                )));
            }
            if value < 0.0 {
                return Err(PhysicsError::InvalidParameter(format!(
                    "ORR {name} must be non-negative"
                )));
            }
        }
        match &self.target {
            OrrTarget::Water {
                lambda0_mm,
                lambda1_mm,
                lambda2_mm,
                m,
            } => {
                for (name, value) in [
                    ("water_lambda0_mm", *lambda0_mm),
                    ("water_lambda1_mm", *lambda1_mm),
                    ("water_lambda2_mm", *lambda2_mm),
                    ("water_m", *m),
                ] {
                    if !value.is_finite() {
                        return Err(PhysicsError::InvalidParameter(format!(
                            "ORR {name} must be finite"
                        )));
                    }
                }
            }
            OrrTarget::Tantalum {
                a_prime,
                w_prime,
                x1_prime,
                x2_prime,
                x3_prime,
                x0_prime,
                alpha,
            } => {
                for (name, value) in [
                    ("tantalum_a_prime", *a_prime),
                    ("tantalum_w_prime", *w_prime),
                    ("tantalum_x1_prime", *x1_prime),
                    ("tantalum_x2_prime", *x2_prime),
                    ("tantalum_x3_prime", *x3_prime),
                    ("tantalum_x0_prime", *x0_prime),
                    ("tantalum_alpha", *alpha),
                ] {
                    if !value.is_finite() {
                        return Err(PhysicsError::InvalidParameter(format!(
                            "ORR {name} must be finite"
                        )));
                    }
                }
                if *a_prime < 0.0 || *w_prime < 0.0 {
                    return Err(PhysicsError::InvalidParameter(
                        "ORR tantalum a_prime and w_prime must be non-negative".to_string(),
                    ));
                }
            }
        }
        match &self.detector {
            OrrDetector::LithiumGlass { d_ns, f_inv_ns, g } => {
                if !d_ns.is_finite() || !f_inv_ns.is_finite() || !g.is_finite() {
                    return Err(PhysicsError::InvalidParameter(
                        "ORR lithium detector parameters must be finite".to_string(),
                    ));
                }
                if *d_ns < 0.0 {
                    return Err(PhysicsError::InvalidParameter(
                        "ORR lithium detector d_ns must be non-negative".to_string(),
                    ));
                }
                if *f_inv_ns <= 0.0 {
                    return Err(PhysicsError::InvalidParameter(
                        "ORR lithium detector f_inv_ns must be positive".to_string(),
                    ));
                }
            }
            OrrDetector::Ne110 {
                delta_mm,
                lambda_sigma_constant_mm,
                lambda_sigma_mm,
            } => {
                if !delta_mm.is_finite() || !lambda_sigma_constant_mm.is_finite() {
                    return Err(PhysicsError::InvalidParameter(
                        "ORR NE110 detector parameters must be finite".to_string(),
                    ));
                }
                if *delta_mm < 0.0 || *lambda_sigma_constant_mm < 0.0 {
                    return Err(PhysicsError::InvalidParameter(
                        "ORR NE110 detector parameters must be non-negative".to_string(),
                    ));
                }
                if lambda_sigma_mm
                    .iter()
                    .any(|(e, s)| !e.is_finite() || !s.is_finite() || *e <= 0.0 || *s < 0.0)
                {
                    return Err(PhysicsError::InvalidParameter(
                        "ORR NE110 table values must be finite, positive-energy, and non-negative"
                            .to_string(),
                    ));
                }
                if lambda_sigma_mm.windows(2).any(|w| w[1].0 < w[0].0) {
                    return Err(PhysicsError::InvalidParameter(
                        "ORR NE110 table energies must be sorted ascending".to_string(),
                    ));
                }
            }
        }
        if self.channel_widths.is_empty() {
            return Err(PhysicsError::InvalidParameter(
                "ORR channel_widths must not be empty".to_string(),
            ));
        }
        if self
            .channel_widths
            .iter()
            .any(|c| !c.max_energy_ev.is_finite() || !c.width_ns.is_finite() || c.width_ns < 0.0)
        {
            return Err(PhysicsError::InvalidParameter(
                "ORR channel widths must be finite and non-negative".to_string(),
            ));
        }
        if self
            .channel_widths
            .windows(2)
            .any(|w| w[1].max_energy_ev < w[0].max_energy_ev)
        {
            return Err(PhysicsError::InvalidParameter(
                "ORR channel widths must be sorted by max_energy_ev".to_string(),
            ));
        }
        Ok(())
    }

    fn t4b_mean_us(&self, d_us: f64, f_us_inv: f64, g: f64) -> f64 {
        if d_us == 0.0 {
            return 1.0 / f_us_inv;
        }
        if g == 0.0 {
            return 1.0 / f_us_inv + d_us;
        }
        let fd = f_us_inv * d_us;
        let gfd = g * fd;
        let h = f_us_inv * (1.0 + gfd);
        (0.5 * gfd * fd + 1.0 + fd) / h
    }

    fn t4a_mean_us(&self, d_us: f64, f_us_inv: f64) -> f64 {
        if f_us_inv == 0.0 {
            return 0.5 * d_us;
        }
        if d_us == 0.0 {
            return 0.0;
        }
        let exp_term = (-f_us_inv * d_us).exp();
        let h = 1.0 / (1.0 - exp_term);
        1.0 / f_us_inv - d_us * h * exp_term
    }

    fn simpson_integrate<F: Fn(f64) -> f64>(a: f64, b: f64, n_even: usize, f: F) -> f64 {
        if b <= a || n_even < 2 {
            return 0.0;
        }
        let n = if n_even.is_multiple_of(2) {
            n_even
        } else {
            n_even + 1
        };
        let h = (b - a) / n as f64;
        let mut sum = f(a) + f(b);
        for i in 1..n {
            let x = a + h * i as f64;
            if i.is_multiple_of(2) {
                sum += 2.0 * f(x);
            } else {
                sum += 4.0 * f(x);
            }
        }
        sum * h / 3.0
    }

    fn tantalum_target_value_raw(
        a_us_inv: f64,
        w_us_inv: f64,
        x1_us: f64,
        x2_us: f64,
        x3_us: f64,
        x0_us: f64,
        alpha: f64,
        t_us: f64,
    ) -> f64 {
        if t_us <= 0.0 || t_us <= x1_us {
            return 0.0;
        }
        let mut sum = 0.0;
        if t_us <= x3_us {
            if a_us_inv == 0.0 {
                sum += 1.0;
            } else {
                let at = a_us_inv * (t_us - x0_us);
                sum += (-(at * at)).exp();
            }
        }
        if alpha != 0.0 {
            let exp_term = (-(w_us_inv * (t_us - x2_us))).exp();
            if x2_us > 0.0 && t_us <= x2_us {
                sum += exp_term * alpha * t_us / x2_us;
            } else {
                sum += exp_term * alpha;
            }
        }
        sum
    }

    fn t2b_numerical(
        &self,
        a_us_inv: f64,
        w_us_inv: f64,
        x1_us: f64,
        x2_us: f64,
        x3_us: f64,
        x0_us: f64,
        alpha: f64,
    ) -> Result<f64, PhysicsError> {
        if w_us_inv == 0.0 && alpha != 0.0 {
            return Err(PhysicsError::InvalidParameter(
                "ORR tantalum cannot use w=0 with alpha>0".to_string(),
            ));
        }
        let mut upper = x1_us.max(x2_us).max(x3_us).max(x0_us);
        if w_us_inv > 0.0 {
            upper = upper.max(x2_us + 40.0 / w_us_inv);
        } else if a_us_inv > 0.0 {
            upper = upper.max(x0_us + 10.0 / a_us_inv);
        } else {
            upper = upper.max(1e-3);
        }
        if upper <= 0.0 || !upper.is_finite() {
            return Err(PhysicsError::InvalidParameter(
                "ORR tantalum produced invalid support bounds".to_string(),
            ));
        }
        let m0 = Self::simpson_integrate(0.0, upper, 256, |t| {
            Self::tantalum_target_value_raw(
                a_us_inv, w_us_inv, x1_us, x2_us, x3_us, x0_us, alpha, t,
            )
        });
        if !m0.is_finite() || m0.abs() <= MIN_NORM {
            return Err(PhysicsError::InvalidParameter(
                "ORR tantalum produced near-zero normalization".to_string(),
            ));
        }
        let m1 = Self::simpson_integrate(0.0, upper, 256, |t| {
            t * Self::tantalum_target_value_raw(
                a_us_inv, w_us_inv, x1_us, x2_us, x3_us, x0_us, alpha, t,
            )
        });
        Ok(m1 / m0)
    }

    fn gen_energy_params(&self, em_ev: f64) -> Result<EnergyParams, PhysicsError> {
        if em_ev <= 0.0 || !em_ev.is_finite() {
            return Err(PhysicsError::InvalidParameter(format!(
                "ORR center energy must be finite and positive, got {em_ev}"
            )));
        }

        let b = em_ev.sqrt() / SM2;
        let b1000 = b * 1000.0;

        let p_us = self.burst_width_ns / 1000.0;
        let (target_mode, a_us_inv, w_us_inv, m, x1_us, x2_us, x3_us, x0_us, alpha, x2_mean_us) =
            match &self.target {
                OrrTarget::Water {
                    lambda0_mm,
                    lambda1_mm,
                    lambda2_mm,
                    m,
                } => {
                    let ln_e = em_ev.ln();
                    let lambda_mm = *lambda0_mm + ln_e * (*lambda1_mm + *lambda2_mm * ln_e);
                    if lambda_mm <= 0.0 || !lambda_mm.is_finite() {
                        return Err(PhysicsError::InvalidParameter(format!(
                            "ORR water lambda must be finite and positive, got {lambda_mm}"
                        )));
                    }
                    let a_us_inv = b1000 / lambda_mm;
                    let m_i = map_m_value(*m);
                    (
                        TargetMode::Water,
                        a_us_inv,
                        0.0,
                        m_i,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        (m_i as f64 + 1.0) / a_us_inv,
                    )
                }
                OrrTarget::Tantalum {
                    a_prime,
                    w_prime,
                    x1_prime,
                    x2_prime,
                    x3_prime,
                    x0_prime,
                    alpha,
                } => {
                    let a_us_inv = *a_prime * b1000;
                    let w_us_inv = *w_prime * b1000;
                    let x1_us = *x1_prime / b1000;
                    let x2_us = *x2_prime / b1000;
                    let x3_us = *x3_prime / b1000;
                    let x0_us = *x0_prime / b1000;
                    let alpha_v = *alpha;
                    let x2_mean_us = self
                        .t2b_numerical(a_us_inv, w_us_inv, x1_us, x2_us, x3_us, x0_us, alpha_v)?;
                    (
                        TargetMode::Tantalum,
                        a_us_inv,
                        w_us_inv,
                        0,
                        x1_us,
                        x2_us,
                        x3_us,
                        x0_us,
                        alpha_v,
                        x2_mean_us,
                    )
                }
            };

        let c_us = self.channel_width_us(em_ev);
        let (d_us, f_us_inv, g, detector_mode) = match &self.detector {
            OrrDetector::LithiumGlass { d_ns, f_inv_ns, g } => (
                d_ns / 1000.0,
                f_inv_ns * 1000.0,
                *g,
                DetectorMode::LithiumGlass,
            ),
            OrrDetector::Ne110 {
                delta_mm,
                lambda_sigma_constant_mm,
                lambda_sigma_mm,
            } => {
                let lambda_sigma_mm = self.interpolate_ne110_lambda_sigma_mm(
                    em_ev,
                    lambda_sigma_mm,
                    *lambda_sigma_constant_mm,
                );
                (
                    delta_mm / b1000,
                    lambda_sigma_mm * b1000,
                    0.0,
                    DetectorMode::Ne110,
                )
            }
        };

        let time_us = self.flight_path_m / b;
        let x1 = 0.5 * p_us;
        let x2 = x2_mean_us;
        let x3 = 0.5 * c_us;
        let x4 = if detector_mode == DetectorMode::Ne110 {
            self.t4a_mean_us(d_us, f_us_inv)
        } else {
            self.t4b_mean_us(d_us, f_us_inv, g)
        };
        let timej_us = time_us + x1 + x2 + x3 + x4;

        let mut b_tail = p_us;
        if target_mode == TargetMode::Water {
            if a_us_inv != 0.0 {
                b_tail += 30.0 / a_us_inv;
            }
        } else if w_us_inv != 0.0 {
            b_tail += 20.0 / w_us_inv + x2_us;
        } else if a_us_inv != 0.0 {
            b_tail += 5.0 / a_us_inv + x0_us;
        }
        if detector_mode == DetectorMode::Ne110 {
            b_tail += d_us;
        } else {
            b_tail += 20.0 / f_us_inv + d_us;
        }
        b_tail += c_us;
        let tlow_us = timej_us - b_tail;
        if tlow_us <= 0.0 {
            return Err(PhysicsError::InvalidParameter(
                "ORR generated non-positive lower time bound".to_string(),
            ));
        }
        let tup_us = timej_us;
        let eup_ev = (SM2 * self.flight_path_m / tlow_us).powi(2);
        let elow_ev = (SM2 * self.flight_path_m / tup_us).powi(2);

        Ok(EnergyParams {
            a_us_inv,
            w_us_inv,
            m,
            x1_us,
            x2_us,
            x3_us,
            x0_us,
            alpha,
            p_us,
            c_us,
            d_us,
            f_us_inv,
            g,
            detector_mode,
            target_mode,
            timej_us,
            elow_ev,
            eup_ev,
        })
    }

    fn set_hhh_case_standard(&self, p: &EnergyParams) -> KernelPiecewise {
        let mut tsubm = [0.0; 8];
        tsubm[0] = 0.0;
        tsubm[1] = p.p_us;
        tsubm[2] = p.c_us;
        tsubm[3] = p.d_us;
        tsubm[4] = p.p_us + p.c_us;
        tsubm[5] = p.p_us + p.d_us;
        tsubm[6] = p.c_us + p.d_us;
        tsubm[7] = p.p_us + p.c_us + p.d_us;

        let mut hhh = [[0.0; 4]; 8];
        match p.detector_mode {
            DetectorMode::LithiumGlass => {
                if p.g != 0.0 {
                    let g2 = 0.5 * p.g;
                    hhh[0][1] = g2;
                    hhh[1][1] = -g2;
                    hhh[2][1] = -g2;
                    hhh[3][1] = -g2;
                    hhh[4][1] = g2;
                    hhh[5][1] = g2;
                    hhh[6][1] = g2;
                    hhh[7][1] = -g2;
                }
                hhh[3][2] = 1.0;
                hhh[5][2] = -1.0;
                hhh[6][2] = -1.0;
                hhh[7][2] = 1.0;
            }
            DetectorMode::Ne110 => {
                if p.f_us_inv != 0.0 {
                    let a = (-p.f_us_inv * p.d_us).exp();
                    hhh[0][2] = 1.0;
                    hhh[1][2] = -1.0;
                    hhh[2][2] = -1.0;
                    hhh[3][2] = -a;
                    hhh[4][2] = 1.0;
                    hhh[5][2] = a;
                    hhh[6][2] = a;
                    hhh[7][2] = -a;
                } else {
                    hhh[0][1] = 1.0;
                    hhh[1][1] = -1.0;
                    hhh[2][1] = -1.0;
                    hhh[3][1] = -1.0;
                    hhh[4][1] = 1.0;
                    hhh[5][1] = 1.0;
                    hhh[6][1] = 1.0;
                    hhh[7][1] = -1.0;
                }
            }
        }

        let mut iflip = false;
        for _ in 0..2 {
            let mut any = false;
            for i1 in [3usize, 4usize] {
                let i = i1 - 1;
                if tsubm[i - 1] <= tsubm[i] {
                    continue;
                }
                any = true;
                iflip = true;
                tsubm.swap(i - 1, i);
                hhh.swap(i - 1, i);

                let j1 = 9 - i1; // 1-based
                let j2 = 10 - i1;
                let a = j1 - 1;
                let b = j2 - 1;
                tsubm.swap(a, b);
                hhh.swap(a, b);
            }
            if !any {
                break;
            }
        }
        if iflip && tsubm[3] > tsubm[4] {
            tsubm.swap(3, 4);
            hhh.swap(3, 4);
        }

        if p.f_us_inv != 0.0 {
            for row in &mut hhh {
                row[0] = row[2];
                row[3] = -row[2];
            }
        }

        KernelPiecewise {
            numtim: 8,
            tsubm_us: tsubm,
            hhh,
        }
    }

    fn find_jjj(timeij_us: f64, piecewise: &KernelPiecewise) -> usize {
        if timeij_us <= 0.0 {
            return 0;
        }
        for j in 0..piecewise.numtim {
            if timeij_us < piecewise.tsubm_us[j] {
                return j + 1; // SAMMY 1-based equivalent
            }
        }
        piecewise.numtim + 1
    }

    fn bcd_value(&self, p: &EnergyParams, piecewise: &KernelPiecewise, t_us: f64) -> f64 {
        if t_us <= 0.0 {
            return 0.0;
        }
        let tmax = piecewise.tsubm_us[piecewise.numtim - 1];
        if t_us > tmax {
            return 0.0;
        }
        let idx = piecewise
            .tsubm_us
            .as_slice()
            .partition_point(|&x| x < t_us)
            .min(piecewise.numtim.saturating_sub(1));
        let row = piecewise.hhh[idx];
        row[0] * (-p.f_us_inv * t_us).exp() + row[1] * t_us * t_us + row[2] * t_us + row[3]
    }

    fn tantalum_target_value(&self, p: &EnergyParams, t_us: f64) -> f64 {
        Self::tantalum_target_value_raw(
            p.a_us_inv, p.w_us_inv, p.x1_us, p.x2_us, p.x3_us, p.x0_us, p.alpha, t_us,
        )
    }

    fn tantalum_sum(&self, p: &EnergyParams, piecewise: &KernelPiecewise, timeij_us: f64) -> f64 {
        if timeij_us <= 0.0 {
            return 0.0;
        }
        let u_max = timeij_us.min(piecewise.tsubm_us[piecewise.numtim - 1]);
        if u_max <= 0.0 {
            return 0.0;
        }
        Self::simpson_integrate(0.0, u_max, 80, |u| {
            let bcd = self.bcd_value(p, piecewise, u);
            if bcd == 0.0 {
                return 0.0;
            }
            let target = self.tantalum_target_value(p, timeij_us - u);
            bcd * target
        })
    }

    fn waterm_sum(
        &self,
        p: &EnergyParams,
        piecewise: &KernelPiecewise,
        timeij_us: f64,
        jjj: usize,
    ) -> f64 {
        let m = p.m;
        let a = p.a_us_inv;
        let f = p.f_us_inv;
        let c1 = (m as f64 + 1.0) * f / a;
        let c2 = (m as f64 + 2.0) * c1 * f / a;
        let c6 = if !approx_eq(a, f) {
            (a / (a - f)).powi(m + 1)
        } else {
            0.0
        };
        let small = 0.10_f64;

        let mut sum = 0.0_f64;
        let jjjm1 = jjj - 1;
        for i in 0..jjjm1 {
            let dt = timeij_us - piecewise.tsubm_us[i];
            let at = a * dt;
            let eat = qexp(-at);
            let ft = f * dt;
            let amft = (a - f) * dt;
            let abamft = amft.abs();

            let h1 = piecewise.hhh[i][0];
            let h2 = piecewise.hhh[i][1];
            let h3 = piecewise.hhh[i][2];
            let h4 = piecewise.hhh[i][3];

            if h2 != 0.0 {
                sum += h2 * (ft * ft - 2.0 * ft * c1 + c2);
            }
            if h3 != 0.0 {
                sum += h3 * (ft - c1);
            }
            if h4 != 0.0 {
                sum += h4;
            }

            let mut atk = 1.0_f64;
            if eat != 0.0 {
                let mut sumk = 1.0_f64;
                if m > 2 {
                    for k in 1..=(m - 2) {
                        atk *= at / (k as f64);
                        sumk += atk;
                    }
                }
                if h2 != 0.0 {
                    sum -= h2 * sumk * eat * ft * ft;
                }
                atk *= at / ((m - 1) as f64);
                sumk += atk;
                if h2 != 0.0 {
                    sum += h2 * sumk * eat * 2.0 * c1 * ft;
                }
                if h3 != 0.0 {
                    sum -= h3 * sumk * eat * ft;
                }
                atk *= at / (m as f64);
                sumk += atk;
                if h2 != 0.0 {
                    sum -= h2 * sumk * eat * c2;
                }
                if h3 != 0.0 {
                    sum += h3 * sumk * eat * c1;
                }
                if h4 != 0.0 {
                    sum -= h4 * sumk * eat;
                }
            }

            if h1 == 0.0 {
                continue;
            }
            if abamft <= small {
                if eat != 0.0 {
                    atk *= at / ((m + 1) as f64);
                    let mut sumj = 1.0_f64;
                    let mut atj = 1.0_f64;
                    if !approx_eq(a, f) {
                        for j in 1..=1_000_000_i32 {
                            atj *= amft / ((j + m + 1) as f64);
                            if (sumj + atj) == sumj {
                                break;
                            }
                            sumj += atj;
                        }
                    }
                    sum += atk * sumj * h1 * eat;
                }
            } else {
                let eft = qexp(-ft);
                if eft != 0.0 {
                    sum += c6 * h1 * eft;
                }
                if eat != 0.0 {
                    let aft = (a - f) * dt;
                    let mut sumk = 1.0_f64;
                    let mut aftk = 1.0_f64;
                    for k in 1..=m {
                        aftk *= aft / (k as f64);
                        sumk += aftk;
                    }
                    sum -= c6 * h1 * eat * sumk;
                }
            }
        }
        sum
    }
}

impl ResolutionFunction for OrrResolution {
    fn convolve(&self, energy: &EnergyGrid, spectrum: &[f64]) -> Result<Vec<f64>, PhysicsError> {
        self.validate()?;
        let energies = &energy.values;
        if energies.is_empty() {
            return Err(PhysicsError::EmptyEnergyGrid);
        }
        if spectrum.len() != energies.len() {
            return Err(PhysicsError::DimensionMismatch {
                expected: energies.len(),
                got: spectrum.len(),
            });
        }
        if energies.windows(2).any(|w| w[1] < w[0]) {
            return Err(PhysicsError::InvalidParameter(
                "energy grid must be sorted ascending".to_string(),
            ));
        }
        if energies.iter().any(|e| !e.is_finite() || *e <= 0.0) {
            return Err(PhysicsError::InvalidParameter(
                "energy grid must contain finite positive values".to_string(),
            ));
        }

        let xcoef = compute_xcoef_weights(energies);
        let mut out = vec![0.0; energies.len()];

        for (j, &em) in energies.iter().enumerate() {
            let p = self.gen_energy_params(em).map_err(|e| {
                PhysicsError::InvalidParameter(format!(
                    "ORR failed to generate parameters at index {j} for E={em} eV: {e:?}"
                ))
            })?;
            let piecewise = self.set_hhh_case_standard(&p);

            let ilow_raw = energies.partition_point(|&e| e < p.elow_ev);
            let ilow = ilow_raw.saturating_sub(2);
            let iup_raw = energies.partition_point(|&e| e < p.eup_ev);
            let iup = iup_raw
                .saturating_sub(1)
                .min(energies.len().saturating_sub(1));

            if ilow > iup {
                out[j] = spectrum[j];
                continue;
            }

            let mut weighted = 0.0_f64;
            let mut norm = 0.0_f64;
            for i in ilow..=iup {
                let ee = energies[i];
                let timeij = p.timej_us - time_from_energy_us(ee, self.flight_path_m);
                let jjj = Self::find_jjj(timeij, &piecewise);
                if jjj <= 1 {
                    continue;
                }
                let sum = if p.target_mode == TargetMode::Water {
                    self.waterm_sum(&p, &piecewise, timeij, jjj)
                } else {
                    self.tantalum_sum(&p, &piecewise, timeij)
                };
                let w = xcoef[i] * sum;
                if w == 0.0 || !w.is_finite() {
                    continue;
                }
                weighted += w * spectrum[i];
                norm += w;
            }

            out[j] = if norm.abs() > MIN_NORM {
                weighted / norm
            } else {
                spectrum[j]
            };
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_orr() -> OrrResolution {
        OrrResolution {
            flight_path_m: 201.578,
            burst_width_ns: 1.775,
            target: OrrTarget::Water {
                lambda0_mm: 0.200,
                lambda1_mm: 0.0,
                lambda2_mm: 0.0,
                m: 4.0,
            },
            detector: OrrDetector::LithiumGlass {
                d_ns: 5.0,
                f_inv_ns: 0.392_235,
                g: 1.009,
            },
            channel_widths: vec![OrrChannelWidth {
                max_energy_ev: 200_000.0,
                width_ns: 7.179,
            }],
        }
    }

    fn sample_orr_ne110() -> OrrResolution {
        OrrResolution {
            flight_path_m: 201.578,
            burst_width_ns: 2.20,
            target: OrrTarget::Water {
                lambda0_mm: 0.2015,
                lambda1_mm: 0.0,
                lambda2_mm: 0.0,
                m: 4.0,
            },
            detector: OrrDetector::Ne110 {
                delta_mm: 1.0,
                lambda_sigma_constant_mm: 27.0,
                lambda_sigma_mm: vec![(10.0, 27.2), (200_000.0, 14.8)],
            },
            channel_widths: vec![OrrChannelWidth {
                max_energy_ev: 200_000.0,
                width_ns: 8.54,
            }],
        }
    }

    #[test]
    fn test_orr_constant_signal_preserved() {
        let energy =
            EnergyGrid::new((0..200).map(|i| 180_000.0 + i as f64 * 10.0).collect()).unwrap();
        let spectrum = vec![0.42; energy.len()];
        let res = sample_orr();
        let out = res.convolve(&energy, &spectrum).unwrap();
        for y in out {
            assert!((y - 0.42).abs() < 1e-8);
        }
    }

    #[test]
    fn test_orr_dimension_mismatch_error() {
        let energy = EnergyGrid::new(vec![180_000.0, 180_100.0]).unwrap();
        let res = sample_orr();
        assert!(matches!(
            res.convolve(&energy, &[1.0]),
            Err(PhysicsError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_orr_ne110_constant_signal_preserved() {
        let energy =
            EnergyGrid::new((0..200).map(|i| 180_000.0 + i as f64 * 10.0).collect()).unwrap();
        let spectrum = vec![0.31; energy.len()];
        let res = sample_orr_ne110();
        let out = res.convolve(&energy, &spectrum).unwrap();
        for y in out {
            assert!((y - 0.31).abs() < 1e-8);
        }
    }

    #[test]
    fn test_orr_negative_burst_width_error() {
        let energy = EnergyGrid::new(vec![180_000.0, 180_100.0]).unwrap();
        let spectrum = vec![0.31, 0.32];
        let mut res = sample_orr();
        res.burst_width_ns = -1.0;
        assert!(matches!(
            res.convolve(&energy, &spectrum),
            Err(PhysicsError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_orr_negative_lithium_d_errors() {
        let energy = EnergyGrid::new(vec![180_000.0, 180_100.0]).unwrap();
        let spectrum = vec![0.31, 0.32];
        let mut res = sample_orr();
        res.detector = OrrDetector::LithiumGlass {
            d_ns: -1.0,
            f_inv_ns: 0.392_235,
            g: 1.009,
        };
        assert!(matches!(
            res.convolve(&energy, &spectrum),
            Err(PhysicsError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_orr_ne110_zero_f_tail_includes_detector_width() {
        let res = OrrResolution {
            flight_path_m: 201.578,
            burst_width_ns: 2.2,
            target: OrrTarget::Water {
                lambda0_mm: 1e-9,
                lambda1_mm: 0.0,
                lambda2_mm: 0.0,
                m: 4.0,
            },
            detector: OrrDetector::Ne110 {
                delta_mm: 1.0,
                lambda_sigma_constant_mm: 0.0,
                lambda_sigma_mm: vec![],
            },
            channel_widths: vec![OrrChannelWidth {
                max_energy_ev: 200_000.0,
                width_ns: 8.54,
            }],
        };
        let p = res.gen_energy_params(180_000.0).unwrap();
        let tlow_us = SM2 * res.flight_path_m / p.eup_ev.sqrt();
        let b_tail = p.timej_us - tlow_us;
        let expected_min = p.p_us + p.c_us + p.d_us;
        assert!(
            b_tail >= expected_min * 0.99,
            "expected b_tail to include detector width for NE110 with f=0: b_tail={b_tail}, expected_min={expected_min}"
        );
    }

    #[test]
    fn test_orr_param_generation_error_propagates() {
        let energy = EnergyGrid::new(vec![180_000.0, 180_100.0]).unwrap();
        let spectrum = vec![0.31, 0.32];
        let mut res = sample_orr();
        res.target = OrrTarget::Water {
            lambda0_mm: -0.1,
            lambda1_mm: 0.0,
            lambda2_mm: 0.0,
            m: 4.0,
        };
        assert!(matches!(
            res.convolve(&energy, &spectrum),
            Err(PhysicsError::InvalidParameter(_))
        ));
    }
}
