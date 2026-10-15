# Quickstart: Rust

Add the crates you need:

```toml
[dependencies]
nereids-core = "0.3"
nereids-endf = "0.3"
nereids-physics = "0.3"
nereids-pipeline = "0.3"
```

The fitting entry point is `nereids_pipeline::counts_fit::fit_counts`. It takes

- a `Measurement`: raw open-beam and sample counts on the same time-of-flight
  bin edges, the sample run's proton-charge ratio, a normalization, a time
  background, and one or more regions, each with its background, pixel count,
  gamma transmission and material;
- a `Calibration`: `t0`, the flight path and the Ikeda-Carpenter pulse.

It returns a `CountsFit` carrying the fitted areal densities, temperatures,
backgrounds, timing, pulse numbers and covariance, plus the deviance and a
convergence flag. `nereids_pipeline::counts_map::fit_map` fits patches of
pixels as one counts fit.

Inputs come from `nereids_endf` (resonance data), `nereids_io` (raw counts)
and `nereids_pipeline::open_beam::fit_open_beam` (the incident beam fitted to
the open-beam counts). The `counts_fit` integration tests under
`crates/nereids-pipeline/tests/` are a worked end-to-end example.
