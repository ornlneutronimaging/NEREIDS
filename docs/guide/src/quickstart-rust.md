# Quickstart: Rust

This example loads ENDF resonance data for U-238, computes a theoretical
transmission spectrum, and fits it to recover the areal density.

## Setup

```toml
# Cargo.toml
[dependencies]
nereids-endf = "0.1"
nereids-physics = "0.1"
nereids-fitting = "0.1"
ndarray = "0.16"
```

## Load ENDF Data

```rust,no_run
use nereids_endf::retrieval::{EndfLibrary, EndfRetriever};
use nereids_endf::parser::parse_endf_file2;

// Download and cache ENDF/B-VIII.1 data for U-238 (Z=92, A=238)
let retriever = EndfRetriever::new(EndfLibrary::Endf81);
let endf_text = retriever.fetch(92, 238)?;
let resonance_data = parse_endf_file2(&endf_text)?;

println!("U-238: {} resonances, AWR = {:.1}",
    resonance_data.total_resonances(),
    resonance_data.awr);
```

## Compute a Forward Model

```rust,no_run
use nereids_physics::transmission::{
    forward_model, InstrumentParams, SampleParams,
};

// Energy grid: 1 to 30 eV (covers the 6.67 eV and 20.9 eV resonances)
let energies: Vec<f64> = (0..2000)
    .map(|i| 1.0 + (i as f64) * 29.0 / 2000.0)
    .collect();

// Sample: U-238 at 0.001 atoms/barn, room temperature
let sample = SampleParams {
    isotopes: vec![(&resonance_data, 0.001)],
    temperature_k: 300.0,
};

// VENUS beamline parameters
let instrument = InstrumentParams {
    flight_path_m: Some(25.0),
    delta_t_us: Some(5.0),
    delta_l_m: Some(0.005),
    ..Default::default()
};

let transmission = forward_model(&energies, &sample, &instrument)?;
// transmission[i] is T(E_i) in [0, 1], with dips at resonance energies
```

## Fit a Measured Spectrum

```rust,no_run
use nereids_fitting::lm::{FitModel, LmConfig, lm_fit};
use nereids_fitting::transmission_model::TransmissionFitModel;
use nereids_fitting::parameters::{FitParameter, ParameterSet};

// Simulate measured data (in practice, load from TIFF/NeXus)
let measured_t = transmission.clone();
let sigma: Vec<f64> = vec![0.01; measured_t.len()];  // uniform uncertainty

// Set up the fit model
let model = TransmissionFitModel::new(
    &energies,
    &[&resonance_data],
    300.0,  // temperature_k
    &instrument,
)?;

// Initial guess: density = 0.0005 atoms/barn
let params = ParameterSet::new(vec![
    FitParameter::new(0.0005, 0.0, 1.0),  // (initial, lower, upper)
]);

let config = LmConfig::default();
let result = lm_fit(&model, &measured_t, &sigma, &params, &config)?;

println!("Fitted density: {:.6} atoms/barn", result.params[0]);
println!("Reduced chi-squared: {:.3}", result.reduced_chi_squared);
```

## Next Steps

- See the [API Reference](/api/nereids_pipeline/) for the full API
- Try [spatial mapping](../api/nereids_pipeline/spatial/) for per-pixel fitting
- Explore the [Python quickstart](./quickstart-python.md) for a NumPy-based workflow
