# Quickstart: Rust

This example loads ENDF resonance data for U-238, computes a theoretical
transmission spectrum, and fits it to recover the areal density.

## Setup

```toml
# Cargo.toml
[dependencies]
nereids-core = "0.1"
nereids-endf = "0.1"
nereids-physics = "0.1"
nereids-fitting = "0.1"
```

## Load ENDF Data

```rust,no_run
use nereids_core::types::Isotope;
use nereids_endf::retrieval::{EndfLibrary, EndfRetriever, mat_number};
use nereids_endf::parser::parse_endf_file2;

// Download and cache ENDF/B-VIII.1 data for U-238 (Z=92, A=238)
let isotope = Isotope::new(92, 238)?;
let retriever = EndfRetriever::new();
let mat = mat_number(&isotope).expect("U-238 has a known MAT number");
let (_path, endf_text) = retriever.get_endf_file(&isotope, EndfLibrary::EndfB8_1, mat)?;
let resonance_data = parse_endf_file2(&endf_text)?;

println!("U-238: {} resonances, AWR = {:.1}",
    resonance_data.total_resonances(),
    resonance_data.awr);
```

## Compute a Forward Model

```rust,no_run
use nereids_physics::transmission::{forward_model, SampleParams};

// Energy grid: 1 to 30 eV (covers the 6.67 eV and 20.9 eV resonances)
let energies: Vec<f64> = (0..2000)
    .map(|i| 1.0 + (i as f64) * 29.0 / 2000.0)
    .collect();

// Sample: U-238 at 0.001 atoms/barn, room temperature
let sample = SampleParams::new(300.0, vec![(resonance_data.clone(), 0.001)])?;

// No instrument resolution broadening for this example
let transmission = forward_model(&energies, &sample, None)?;
// transmission[i] is T(E_i) in [0, 1], with dips at resonance energies
```

## Fit a Measured Spectrum

```rust,no_run
use nereids_fitting::lm::{levenberg_marquardt, LmConfig};
use nereids_fitting::transmission_model::TransmissionFitModel;
use nereids_fitting::parameters::{FitParameter, ParameterSet};

// Simulate measured data (in practice, load from TIFF/NeXus)
let measured_t = transmission.clone();
let sigma: Vec<f64> = vec![0.01; measured_t.len()];

// Set up the fit model: one density parameter at index 0
let model = TransmissionFitModel::new(
    energies.clone(),
    vec![resonance_data],
    300.0,                   // temperature_k
    None,                    // no instrument resolution
    vec![0],                 // density_indices
    None,                    // no temperature fitting
    None,                    // no precomputed cross-sections
)?;

// Initial guess: density = 0.0005 atoms/barn (non-negative constraint)
let mut params = ParameterSet::new(vec![
    FitParameter::non_negative("U-238 density", 0.0005),
]);

let config = LmConfig::default();
let result = levenberg_marquardt(&model, &measured_t, &sigma, &mut params, &config)?;

println!("Fitted density: {:.6} atoms/barn", result.params[0]);
println!("Reduced chi-squared: {:.3}", result.reduced_chi_squared);
println!("Converged: {}", result.converged);
```

## Next Steps

- See the [API Reference](api/nereids_pipeline/) for the full API
- Explore the [Python quickstart](./quickstart-python.md) for a NumPy-based workflow
