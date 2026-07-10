# Count-response formation probe

Command:

```text
/usr/bin/time -p pixi run python investigation/count_response_probe.py
```

Exit status 0; wall time 0.08 s. The probe constructs a five-bin response
matrix whose columns each sum to one, so each column is a measured-bin
probability conditioned on a true bin. It compares the current normalized
`R[T]` form with the physical count ratio `R[Phi*T]/R[Phi]`.

```text
response column sums = [1, 1, 1, 1, 1]
flat flux:       maximum |count ratio - R[T]| = 0.0
structured flux maximum |count ratio - R[T]| = 0.2106026786
structured center-bin values: R[T] = 0.4050000, count ratio = 0.2381562
```

The flat-flux equality control and structured-flux divergence validate the
mechanism independently of fit quality. This does not quantify the effect on
the archived VENUS data: the latent incident flux and a true-energy-conditioned
NEREIDS response matrix have not yet been jointly reconstructed.
