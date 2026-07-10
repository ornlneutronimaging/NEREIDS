# Independent numerical probes

These probes were compiled against the synchronized branch's debug
`nereids_physics` library. The durable sources are under
`investigation/probes/`.

Durable compile-and-run command:

```text
/usr/bin/time -p pixi run python investigation/run_code_probes.py
```

Exit status 0; wall time 3.04 s on the final replay. The runner first built
`nereids-physics`, selected
`target/debug/deps/libnereids_physics-8d202659f87bf9ad.rlib`, then compiled and
ran all six durable sources. Output:

```text
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.10s
nereids_physics_rlib=target/debug/deps/libnereids_physics-8d202659f87bf9ad.rlib
## grid
coarse_vs_dense_sample max_abs_diff=0.047622 at_E=15.100 fine_then_sample=0.952378 coarse_direct=1.000000
coarse_input_min=1.000000 fine_input_min=0.200000

## lscale
L_scale=1.005: sigma_current=0.049472940 sigma_physical=0.049227047 ratio=1.0049951 max_abs_output_diff=0.0001101
L_scale=1.020: sigma_current=0.049472940 sigma_physical=0.048503545 ratio=1.0199861 max_abs_output_diff=0.0004340

## tau
n_tau=8 points=8 dt=2.571429 variance=2.213234961 rel_err_vs_3=-26.226%
n_tau=16 points=16 dt=1.200000 variance=2.920245346 rel_err_vs_3=-2.658%
n_tau=64 points=64 dt=0.285714 variance=2.999036996 rel_err_vs_3=-0.032%
n_tau=600 points=600 dt=0.030050 variance=2.999297584 rel_err_vs_3=-0.023%

## r_threshold
R=0.0000e0 points=1004 dt=0.001387390 var=0.024854498 expected=0.024854536 rel=-0.00% span=1.391552
R=1.0000e-10 points=1004 dt=0.001387390 var=0.024854498 expected=0.024855036 rel=-0.00% span=1.391552
R=1.0000e-9 points=1004 dt=0.001387390 var=0.024854498 expected=0.024859536 rel=-0.02% span=1.391552
R=1.0001e-9 points=16 dt=0.097668172 var=0.023968606 expected=0.024859537 rel=-3.58% span=1.465023
R=1.0000e-8 points=16 dt=0.097668172 var=0.023968606 expected=0.024904536 rel=-3.76% span=1.465023
R=1.0000e-7 points=16 dt=0.097668172 var=0.023968607 expected=0.025354536 rel=-5.47% span=1.465023
R=1.0000e-6 points=16 dt=0.097668172 var=0.023968615 expected=0.029854534 rel=-19.72% span=1.465023

## resolution
flight_path=0.0: constructor=Ok broaden=[0.0, 1.0, 0.0] support=0.0
flight_path=-1.0: constructor=Ok broaden=[0.0, 1.0, 0.0] support=0.0
flight_path=NaN: constructor=Ok broaden=[NaN, NaN, NaN] support=NaN
flight_path=inf: constructor=Ok broaden=[NaN, NaN, NaN] support=NaN
all_zero_kernel: constructor=Ok broaden=[0.0, 1.0, 0.0]

## ic
ic_pulse(a=1e200,b=1.0,r=0.5,t=2e-200)=NaN
ic_pulse(a=1.0,b=1.0,r=1.0,t=2.5)=0.21376301724973645
ic_pulse(a=0.05,b=4.0,r=0.5,t=400.0)=2.072878694014749e-8
inverse_lambda_negative_tiny_denom_eval=999999999.9999999
negative_denom_model_constructor_ok=true
negative_tiny_kappa_eval=0.0 constructor_ok=true
true_box_corner=Err(... E = 24.178042795194287 eV ... α = 26.5856 µs⁻¹,
  β = 0.0200 µs⁻¹, R = 1.000 ... τ-step 0.0977 µs ... floor 0.0967 µs)
```

## Mechanisms established

- A narrow resonance absent from the coarse input grid cannot be recovered by
  IC/UDR convolution; the demonstrated output error is 0.047622.
- When `L_scale` is not one, current broadening scales physical timing
  widths by the same factor.
- The accepted cap floor of eight prompt samples understates Gamma(3) variance
  by 26.226%.
- Crossing `R=1e-9` changes the grid from 1004 to 16 retained points and
  introduces a discontinuous variance error.
- Rust tabulated-resolution constructors accept invalid flight paths and an
  all-zero kernel.
- The extreme-rate NaN is a direct-API robustness issue outside calibration
  bounds, not a production-domain explanation for the archive residuals.
- The actual declared calibration corner
  `a0=5,a1=2,beta=.02,R=1,PSR=.35 us` is not synthesizable under the tau cap.
  The repository test named `ic_box_worst_corner_synthesizes_within_tau_cap`
  uses `a0=.5`, so it does not cover that boundary.
