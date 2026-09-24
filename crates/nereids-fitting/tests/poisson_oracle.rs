use nereids_fitting::error::FittingError;
use nereids_fitting::lm::{FitModel, FlatMatrix};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::poisson::{PoissonConfig, PoissonResult, poisson_fit};
use serde::Deserialize;

#[derive(Deserialize)]
struct Oracle {
    models: Vec<Model>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(tag = "family", rename_all = "lowercase")]
enum Model {
    Decay {
        t: Vec<f64>,
    },
    Split {
        t: Vec<f64>,
    },
    Correlated {
        u: Vec<f64>,
        w: Vec<f64>,
    },
    Resonance {
        energy: Vec<f64>,
        flux: Vec<f64>,
        center: f64,
        peak: f64,
        width_300k: f64,
    },
    Saturated {
        energy: Vec<f64>,
        flux: Vec<f64>,
        center: f64,
        peak: f64,
        width_300k: f64,
    },
    Linear {
        x: Vec<Vec<f64>>,
        offset: Vec<f64>,
    },
}

#[derive(Deserialize)]
struct Case {
    name: String,
    model: usize,
    observed: Vec<f64>,
    start: Vec<f64>,
    start_valid: bool,
    lower: Vec<Option<f64>>,
    upper: Vec<Option<f64>>,
    reference: Vec<f64>,
    reference_deviance: f64,
    on_bound: Vec<bool>,
    sigma: Vec<Option<f64>>,
    covariance: Vec<Vec<Option<f64>>>,
    scale: Vec<Option<f64>>,
}

impl Model {
    fn columns(&self, p: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
        match self {
            Self::Decay { t } => {
                let e: Vec<f64> = t.iter().map(|&t| (-p[1] * t).exp()).collect();
                let mu = e.iter().map(|e| p[0] * e).collect();
                let db = t.iter().zip(&e).map(|(t, e)| -p[0] * t * e).collect();
                (mu, vec![e, db])
            }
            Self::Split { t } => {
                let e: Vec<f64> = t.iter().map(|&t| (-p[2] * t).exp()).collect();
                let mu = e.iter().map(|e| (p[0] + p[1]) * e).collect();
                let dc = t
                    .iter()
                    .zip(&e)
                    .map(|(t, e)| -(p[0] + p[1]) * t * e)
                    .collect();
                (mu, vec![e.clone(), e, dc])
            }
            Self::Correlated { u, w } => {
                let e: Vec<f64> = u
                    .iter()
                    .zip(w)
                    .map(|(&u, &w)| (-p[1] * u - p[2] * (u + w)).exp())
                    .collect();
                let mu = e.iter().map(|e| p[0] * e).collect();
                let da = u.iter().zip(&e).map(|(u, e)| -p[0] * u * e).collect();
                let db = u
                    .iter()
                    .zip(w)
                    .zip(&e)
                    .map(|((u, w), e)| -p[0] * (u + w) * e)
                    .collect();
                (mu, vec![e, da, db])
            }
            Self::Resonance {
                energy,
                flux,
                center,
                peak,
                width_300k,
            } => {
                let width = width_300k * (p[1] / 300.0).sqrt();
                let dwidth_dt = width_300k / (2.0 * (300.0 * p[1]).sqrt());
                let (mut mu, mut dn, mut dt) = (vec![], vec![], vec![]);
                for (&e, &f) in energy.iter().zip(flux) {
                    let delta = e - center;
                    let sigma = peak * (-(delta * delta) / (2.0 * width * width)).exp();
                    let trans = (-p[0] * sigma).exp();
                    mu.push(f * trans + p[2]);
                    dn.push(-f * trans * sigma);
                    dt.push(-f * trans * p[0] * sigma * delta * delta / width.powi(3) * dwidth_dt);
                }
                let db = vec![1.0; energy.len()];
                (mu, vec![dn, dt, db])
            }
            Self::Saturated {
                energy,
                flux,
                center,
                peak,
                width_300k,
            } => {
                let width = width_300k * (p[1] / 300.0).sqrt();
                let dwidth_dt = width_300k / (2.0 * (300.0 * p[1]).sqrt());
                let (mut mu, mut dn, mut dt) = (vec![], vec![], vec![]);
                for (&e, &f) in energy.iter().zip(flux) {
                    let delta = e - center;
                    let sigma = peak * (-(delta * delta) / (2.0 * width * width)).exp();
                    let counts = f * (-p[0] * sigma).exp();
                    mu.push(counts);
                    dn.push(-counts * sigma);
                    dt.push(-counts * p[0] * sigma * delta * delta / width.powi(3) * dwidth_dt);
                }
                (mu, vec![dn, dt])
            }
            Self::Linear { x, offset } => {
                let mu = x
                    .iter()
                    .zip(offset)
                    .map(|(row, o)| o + row.iter().zip(p).map(|(a, b)| a * b).sum::<f64>())
                    .collect();
                let cols = (0..p.len())
                    .map(|j| x.iter().map(|row| row[j]).collect())
                    .collect();
                (mu, cols)
            }
        }
    }
}

impl FitModel for Model {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        Ok(self.columns(params).0)
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        _y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let (mu, cols) = self.columns(params);
        let mut jacobian = FlatMatrix::zeros(mu.len(), free_param_indices.len());
        for (col, &index) in free_param_indices.iter().enumerate() {
            for (row, &value) in cols[index].iter().enumerate() {
                *jacobian.get_mut(row, col) = value;
            }
        }
        Some(jacobian)
    }
}

fn parameters(case: &Case, values: &[f64], fixed: bool) -> ParameterSet {
    ParameterSet::new(
        values
            .iter()
            .enumerate()
            .map(|(i, &value)| FitParameter {
                name: format!("p{i}").into(),
                value,
                lower: case.lower[i].unwrap_or(f64::NEG_INFINITY),
                upper: case.upper[i].unwrap_or(f64::INFINITY),
                fixed,
            })
            .collect(),
    )
}

fn fit(model: &Model, case: &Case, values: &[f64], fixed: bool) -> PoissonResult {
    let mut params = parameters(case, values, fixed);
    poisson_fit(
        model,
        &case.observed,
        &mut params,
        &PoissonConfig::default(),
    )
    .unwrap()
}

fn check(model: &Model, case: &Case) -> Vec<String> {
    let mut failures = vec![];
    let mut fail = |what: String| failures.push(format!("{}: {what}", case.name));

    let at_reference = fit(model, case, &case.reference, true);
    let parity = (at_reference.deviance - case.reference_deviance).abs();
    if parity > 1e-8 * case.reference_deviance.abs().max(1.0) {
        fail(format!(
            "deviance at the reference {} vs {}",
            at_reference.deviance, case.reference_deviance
        ));
    }

    let result = fit(model, case, &case.start, false);
    if !case.start_valid {
        if result.converged || result.iterations != 0 {
            fail(format!(
                "a start predicting zero where counts exist gave converged {} after {} steps",
                result.converged, result.iterations
            ));
        }
        return failures;
    }
    if !result.converged {
        fail(format!("not converged after {} steps", result.iterations));
    }
    if result.deviance > case.reference_deviance + 2e-4 {
        fail(format!(
            "deviance {} above the reference {}",
            result.deviance, case.reference_deviance
        ));
    }
    for i in 0..case.reference.len() {
        let unique = case.sigma[i].is_some() || case.on_bound[i];
        if let (true, Some(scale)) = (unique, case.sigma[i].or(case.scale[i])) {
            let miss = (result.params[i] - case.reference[i]) / scale;
            if miss.abs() > 0.02 {
                fail(format!(
                    "parameter {i} is {miss:.3} error bars from the reference"
                ));
            }
        }
    }

    let polished = fit(model, case, &case.reference, false);
    if !polished.converged || polished.iterations != 0 {
        fail(format!(
            "from the reference: converged {} after {} steps",
            polished.converged, polished.iterations
        ));
    }
    if polished.on_bound != case.on_bound {
        fail(format!(
            "on bound {:?} vs {:?}",
            polished.on_bound, case.on_bound
        ));
    }
    match &polished.covariance {
        None => fail("no covariance from the reference".into()),
        Some(covariance) => {
            for (i, row) in case.covariance.iter().enumerate() {
                for (j, want) in row.iter().enumerate() {
                    let got = covariance.get(i, j);
                    match (want, case.sigma[i].zip(case.sigma[j])) {
                        (Some(want), Some((si, sj))) if (got - want).abs() <= 1e-8 * si * sj => {}
                        (None, _) if got.is_nan() => {}
                        _ => fail(format!("covariance ({i}, {j}): {got} vs {want:?}")),
                    }
                }
            }
        }
    }
    match &polished.uncertainties {
        None => fail("no uncertainties from the reference".into()),
        Some(errors) => {
            for (i, (got, want)) in errors.iter().zip(&case.sigma).enumerate() {
                match (got, want) {
                    (Some(got), Some(want)) if ((got - want) / want).abs() <= 1e-8 => {}
                    (None, None) => {}
                    _ => fail(format!("error bar {i}: {got:?} vs {want:?}")),
                }
            }
        }
    }
    failures
}

#[test]
fn poisson_fit_matches_the_scipy_reference() {
    let oracle: Oracle = serde_json::from_str(include_str!("poisson_oracle/cases.json")).unwrap();
    let failures: Vec<String> = oracle
        .cases
        .iter()
        .flat_map(|case| check(&oracle.models[case.model], case))
        .collect();
    assert!(
        failures.is_empty(),
        "{} failures in {} cases:\n{}",
        failures.len(),
        oracle.cases.len(),
        failures.join("\n")
    );
}
