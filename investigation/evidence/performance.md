# Measured IC path performance

Command:

```text
/usr/bin/time -p pixi run python investigation/performance_probe.py
```

Exit status 0; wall time 1.94 s. Each component value is the median of seven
release-build repetitions after warm-up on all 2,312 calibration-window bins.

```text
IC synthesis 48x400                 0.0075299 s
IC synthesis 64x500                 0.0142867 s
forward model, no resolution        0.0116955 s
forward model, reused IC table      0.1001856 s
fresh 48x400 IC + forward model     0.1091145 s
fresh-vs-reused output max abs      0.0
```

Thus kernel synthesis is only about 7% of a current composite objective call
at 48×400. Repeated nuclear/Doppler work plus the tabulated convolution is the
dominant measured cost; IC is not applied analytically after construction.

Independent archive timings:

```text
UDR spatial maps  =  99 s
IC spatial maps   = 602 s  (6.0808x)
IC calibrations   = 665–1490 s
```

The full debug calibration unit-test filter separately took 1329.99 s for
33 tests. The fixed IC table has 48 references; interpolated kernels at
8/10/20/45 eV contained 2,094/2,315/3,172/4,609 points and spanned about
38–42 µs, explaining why convolution/plan construction dominates synthesis.
