# Repository verification relevant to IC/UDR resolution

All commands ran from `/Users/chenzhang/github.com/NEREIDS/NEREIDS` on the
synchronized research branch.

## IC physics unit tests

```text
$ cargo test -p nereids-physics ikeda_carpenter::tests -- --nocapture
test result: ok. 21 passed; 0 failed; 0 ignored; 0 measured; 247 filtered out; finished in 0.61s
```

These cover pulse unit area/mean/nonnegativity, alpha≈beta stability, tail
orientation, optional-fold moments, mode anchoring, synthesis validity, and
direct/plan parity.

## Tabulated-resolution unit tests

```text
$ cargo test -p nereids-physics resolution::tests -- --nocapture
test result: ok. 73 passed; 0 failed; 0 ignored; 0 measured; 195 filtered out; finished in 0.04s
```

The expected `#[should_panic]` length-mismatch test printed its assertion and
then passed.

## Orientation, interpolation, and VENUS-plan integration

```text
$ cargo test -p nereids-physics --test kernel_orientation \
    --test kernel_width_interpolation --test venus_usr_resolution -- --nocapture
kernel_orientation:             1 passed; 0 failed; 1.15s
kernel_width_interpolation:     1 passed; 0 failed; 13.43s
venus_usr_resolution:           8 passed; 0 failed; 0.09s
3471-grid apply_r vs plan.apply observed max_hybrid_err = 6.661e-16
```

## Resolution calibrator unit filter

```text
$ cargo test -p nereids-fitting resolution_calib::tests -- --nocapture
test result: ok. 33 passed; 0 failed; 0 ignored; 0 measured; 156 filtered out; finished in 1329.99s
```

Six IC calibration cases exceeded Rust's 60-second long-test notice. The final
`ic_recovers_known_alpha` case accounted for most of the elapsed time.

## Generic SAMMY-reference physics suite

```text
$ cargo test -p nereids-physics --test samtry_validation -- --nocapture
test result: ok. 89 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 10.42s
```

This exercises generic Reich–Moore parsing/cross sections, Doppler broadening,
and transmission against permissive SAMMY fixtures. It is not a Ta/JENDL-5 or
VENUS response oracle: printed worst relative differences range from small to
several-fold in deliberately broad fixture tolerances.

## Infeasible fitted-PSR guard

```text
$ cargo test -p nereids-fitting rejects_infeasible_psr_start_width -- --nocapture
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 188 filtered out; finished in 0.00s
```

This confirms that the nominal `.05 us` lower PSR box can be rejected by the
tau-cap feasibility guard at the default beta/R start; it does not make the
declared rectangular bound feasible.

## What these tests do not verify

- The real VENUS UDR file is not in the repository; integration tests use a
  small symmetric synthetic kernel.
- IC loop closure generates truth with the same implementation and synthesis
  grid, so shared numerical/model-form errors cancel.
- No repository test covers the R threshold discontinuity, cap-boundary moment
  accuracy, dense/coarse convergence, non-unit-L physical scaling, out-of-range
  IC reuse, count-response formation, or mode/centroid convention against
  instrument data. The named worst-corner test uses `a0=.5`, not the declared
  `a0=5` maximum.
