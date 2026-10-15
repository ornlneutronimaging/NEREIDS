# NEREIDS Test Data

The ENDF, SAMMY and synthetic fixtures in this directory support the Rust and
Python tests. Large PLEIADES fixtures are optional local research data, not a
submodule or a dependency of the test suite.

## Optional PLEIADES data

`tests/data/pleiades_data/` is ignored. Existing local checkouts and project
saves can remain there. A normal NEREIDS clone does not fetch these files.

The external PLEIADES test-data repository is
<https://code.ornl.gov/sns-hfir-scse/infrastructure/test-data/pleiades_data>.
Its `LANL-ORNL_example.tif` contains synthetic transmission images, which
are not raw open-beam/sample counts and cannot be supplied to the counts
likelihood. Its VENUS Hf sample and open-beam HDF5 files contain raw counts
in rustpix NeXus format. Their time-bin edges and run charge ratio must be
supplied to a counts fit.

The VENUS fixtures are a 256×256 crop of a private measurement cube, with
4367 time bins and a nominal flight path of 25 m. The HDF5 files do not
store proton charge; the original run's approximate sample/open charge ratio
was 5.98. This approximate value is provenance, not a measured calibration
for another run. Regeneration requires the maintainers' private source data
and extraction tooling.

## venus/aggregated_hf_120min.npz

The retained aggregated VENUS Hf spectrum has `energies_ev`, `sample_counts`,
`open_beam_counts` and `pc_ratio`. It comes from the same private measurement
cube. An energy-center array is not a substitute for the original time-bin
edges required by the counts API; this archive is research evidence, not an
end-to-end counts-fitting validation fixture.
