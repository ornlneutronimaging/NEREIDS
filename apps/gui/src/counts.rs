//! Counts-domain fitting from the GUI's loaded state.

use std::sync::Arc;

use ndarray::{Array2, Array3, Axis};
use nereids_endf::resonance::ResonanceData;
use nereids_endf::retrieval::EndfLibrary;
use nereids_pipeline::counts_fit::{
    self, Material, Measurement, NO_TIME_BACKGROUND, Region, Value,
};
use nereids_pipeline::counts_map::{self, MapMeasurement};
use nereids_pipeline::open_beam::{Calibration, Pulse};

/// The Ikeda-Carpenter pulse numbers used as the known calibration.
#[derive(Debug, Clone, Copy)]
pub struct PulseParams {
    pub alpha0: f64,
    pub alpha1: f64,
    pub beta0: f64,
    pub beta1: f64,
    pub r: f64,
    pub fwhm_us: f64,
}

impl Default for PulseParams {
    fn default() -> Self {
        Self {
            alpha0: 0.35,
            alpha1: 0.05,
            beta0: 0.25,
            beta1: 0.0,
            r: 0.15,
            fwhm_us: 0.0,
        }
    }
}

/// One isotope with its starting density.
#[derive(Clone)]
pub struct IsotopeInput {
    pub label: String,
    pub data: ResonanceData,
    pub density: f64,
}

/// The selection fitted by a single counts fit.
#[derive(Debug, Clone, Copy)]
pub enum RegionRequest {
    Pixel {
        y: usize,
        x: usize,
    },
    Roi {
        y0: usize,
        y1: usize,
        x0: usize,
        x1: usize,
    },
}

/// Everything one fit needs, owned so it can move to a worker thread.
#[derive(Clone)]
pub struct FitRequest {
    pub time_edges_us: Vec<f64>,
    pub charge_ratio: f64,
    pub normalization_fitted: bool,
    pub background_fitted: bool,
    pub isotopes: Vec<IsotopeInput>,
    pub temperature_k: f64,
    pub library: EndfLibrary,
    pub t0_us: f64,
    pub flight_path_m: f64,
    pub pulse: PulseParams,
    pub energy_span_ev: (f64, f64),
    pub sample_counts: Arc<Array3<f64>>,
    pub open_counts: Arc<Array3<f64>>,
    pub excluded: Arc<Array2<bool>>,
    pub region: RegionRequest,
    pub binning: usize,
}

/// A single counts fit's result.
#[derive(Debug, Clone)]
pub struct PixelFitSummary {
    pub labels: Vec<String>,
    pub densities: Vec<f64>,
    pub temperature_k: f64,
    pub deviance: f64,
    pub converged: bool,
    pub pixels: f64,
}

/// A counts-map fit's result.
#[derive(Debug, Clone)]
pub struct MapFitSummary {
    pub labels: Vec<String>,
    pub densities: Vec<Array2<f64>>,
    pub density_sd: Vec<Array2<f64>>,
    pub temperature_k: Array2<f64>,
    pub temperature_sd_k: Array2<f64>,
    pub trusted: Array2<bool>,
    pub converged: bool,
    pub deviance: f64,
    pub binning: usize,
}

/// The outcome of a background fit.
pub enum FitOutcome {
    Pixel(Result<PixelFitSummary, String>),
    Map(Result<Box<MapFitSummary>, String>),
}

fn bounds(region: RegionRequest) -> (usize, usize, usize, usize) {
    match region {
        RegionRequest::Pixel { y, x } => (y, y + 1, x, x + 1),
        RegionRequest::Roi { y0, y1, x0, x1 } => (y0, y1, x0, x1),
    }
}

fn selection_counts(
    counts: &Array3<f64>,
    region: RegionRequest,
) -> Result<(Vec<f64>, f64), String> {
    let shape = counts.shape();
    let (y0, y1, x0, x1) = bounds(region);
    if y0 >= y1 || x0 >= x1 {
        return Err("the selection is empty".into());
    }
    if y1 > shape[1] || x1 > shape[2] {
        return Err("the selection lies outside the image".into());
    }
    let block = counts.slice(ndarray::s![.., y0..y1, x0..x1]);
    let summed = block.sum_axis(Axis(2)).sum_axis(Axis(1));
    Ok((summed.to_vec(), ((y1 - y0) * (x1 - x0)) as f64))
}

fn assert_live(excluded: &Array2<bool>, region: RegionRequest) -> Result<(), String> {
    let (y0, y1, x0, x1) = bounds(region);
    for y in y0..y1 {
        for x in x0..x1 {
            if excluded[[y, x]] {
                return Err("the selection contains an excluded pixel".into());
            }
        }
    }
    Ok(())
}

fn calibration(request: &FitRequest) -> Calibration {
    let p = &request.pulse;
    Calibration {
        t0_us: Value::Known(request.t0_us),
        flight_path_m: Value::Known(request.flight_path_m),
        pulse: Pulse {
            alpha: [Value::Known(p.alpha0), Value::Known(p.alpha1)],
            beta: [Value::Known(p.beta0), Value::Known(p.beta1)],
            r: Value::Known(p.r),
            fwhm_squared_us2: Value::Known(p.fwhm_us * p.fwhm_us),
            r1: Value::Known(0.0),
            energy_span_ev: request.energy_span_ev,
            n_tau: 8,
            line_span_ev: None,
            prior: None,
        },
    }
}

fn material(request: &FitRequest) -> Material {
    Material {
        isotopes: request
            .isotopes
            .iter()
            .map(|i| (i.data.clone(), Value::Fitted(i.density)))
            .collect(),
        temperature_k: Value::Known(request.temperature_k),
        library: request.library,
        covariances: Vec::new(),
        thickness_variance: Value::Known(0.0),
    }
}

fn normalization(request: &FitRequest) -> Value {
    if request.normalization_fitted {
        Value::Fitted(1.0)
    } else {
        Value::Known(1.0)
    }
}

fn background(request: &FitRequest) -> [Value; 5] {
    let value = if request.background_fitted {
        Value::Fitted(0.0)
    } else {
        Value::Known(0.0)
    };
    [value, value, value, Value::Known(0.0), Value::Known(0.0)]
}

fn labels(request: &FitRequest) -> Vec<String> {
    request.isotopes.iter().map(|i| i.label.clone()).collect()
}

/// Fit the selected pixel or ROI with the counts engine.
pub fn run_pixel_fit(request: &FitRequest) -> Result<PixelFitSummary, String> {
    if request.isotopes.is_empty() {
        return Err("select at least one isotope with loaded ENDF data".into());
    }
    let (open_counts, pixels) = selection_counts(&request.open_counts, request.region)?;
    let (sample_counts, _) = selection_counts(&request.sample_counts, request.region)?;
    let shape = request.open_counts.shape();
    if request.excluded.dim() != (shape[1], shape[2]) {
        return Err("the excluded-pixel mask does not match the image".into());
    }
    assert_live(&request.excluded, request.region)?;
    let region = Region {
        open_counts,
        sample_counts,
        open_live: None,
        sample_live: None,
        background: background(request),
        pixels,
        gamma_transmission: Value::Known(1.0),
        capture_gammas: Value::Known(0.0),
        in_scatter: Value::Known(0.0),
        material: Some(material(request)),
    };
    let measurement = Measurement {
        time_edges_us: request.time_edges_us.clone(),
        charge_ratio: request.charge_ratio,
        normalization: normalization(request),
        time_background: NO_TIME_BACKGROUND,
        room_background: None,
        spread: None,
        regions: vec![region],
    };
    let fit =
        counts_fit::fit_counts(&measurement, &calibration(request)).map_err(|e| e.to_string())?;
    let region = fit
        .regions
        .first()
        .ok_or_else(|| "the fit returned no region".to_string())?;
    Ok(PixelFitSummary {
        labels: labels(request),
        densities: region.densities.clone(),
        temperature_k: region.temperature_k.unwrap_or(request.temperature_k),
        deviance: fit.deviance,
        converged: fit.converged,
        pixels,
    })
}

/// Fit every patch of the loaded cube with the counts map engine.
pub fn run_map_fit(request: &FitRequest) -> Result<MapFitSummary, String> {
    if request.isotopes.is_empty() {
        return Err("select at least one isotope with loaded ENDF data".into());
    }
    if request.binning == 0 {
        return Err("the patch side must be at least one pixel".into());
    }
    let excluded = request.excluded.as_ref();
    let (height, width) = excluded.dim();
    let sample = Array2::from_shape_fn((height, width), |(y, x)| !excluded[[y, x]]);
    let empty = Array2::from_elem((height, width), false);
    let map = MapMeasurement {
        time_edges_us: request.time_edges_us.clone(),
        charge_ratio: request.charge_ratio,
        normalization: normalization(request),
        time_background: NO_TIME_BACKGROUND,
        room_background: None,
        open_counts: request.open_counts.view(),
        sample_counts: request.sample_counts.view(),
        open_live: None,
        sample_live: None,
        excluded: excluded.view(),
        sample: sample.view(),
        empty: empty.view(),
        binning: request.binning,
        material: material(request),
        background: background(request),
        gamma_transmission: Value::Known(1.0),
        capture_gammas: Value::Known(0.0),
        in_scatter: Value::Known(0.0),
        empty_background: [Value::Known(0.0); 5],
    };
    let fit = counts_map::fit_map(&map, &calibration(request)).map_err(|e| e.to_string())?;
    Ok(MapFitSummary {
        labels: labels(request),
        densities: fit.densities.clone(),
        density_sd: fit.density_sd.clone(),
        temperature_k: fit.temperature_k.clone(),
        temperature_sd_k: fit.temperature_sd_k.clone(),
        trusted: fit.trusted.clone(),
        converged: fit.converged,
        deviance: fit.deviance,
        binning: request.binning,
    })
}
