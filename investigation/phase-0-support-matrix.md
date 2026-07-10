# Phase 0 support matrix

Status vocabulary:

- **native** — objective matches the input contract;
- **approximation** — executable with a documented information/model loss;
- **legacy-invalid** — executable but the advertised statistical contract is
  contradicted by a mechanism test;
- **half-wired** — public representation exists but intended behavior is
  rejected or ambiguous;
- **rejected** — explicit unsupported combination;
- **calibration-only** / **research-only** — not a production fit route.
- **integration gap** — source-reachable but lacking a direct route test;
- **cross-surface divergence** — the same apparent request resolves to a
  different typed route or default on another public surface.

Test identifiers point to `investigation/evidence/phase-0/test-results.md`.

## Normalized compatibility-cell registry

This is the canonical R14 registry. A **support cell** means one `M` row in
this table; the explanatory roll-ups below do not create additional cells.
The registry is exhaustive over meaningful combinations, rather than the
literal Cartesian product of nonsensical combinations, by using these
normalized dimensions:

- backgrounds: **B0** none (for calibration, required profiled scale `a` but
  no additive term), **B1** SAMMY `Anorm+ABC`, **B2** optional SAMMY `D/F`
  extension to B1, **B3** multiplicative log-E baseline, **B4** combined
  SAMMY plus multiplicative stack (free `Anorm` forbidden; counts also forbid
  B2), **B5** detector-space `alpha1*Phi*T + alpha2*Bdet`, **B6** calibration
  profile `a*T+b0+b1*index`, and **B7** legacy fixed-flux Fisher background;
- resolutions: **R0** identity, **R1** Gaussian, **R2** tabulated/UDR,
  **R3** native Rust IC, and **R4** IC converted to tabulated form;
- routes: F1-F6, S1-S6, and C1-C3 retain the definitions in the path audit.

The base-route block covers B0 across every R family. F6/S6 cover B0-B5 in
one cell each because their input/solver rejection precedes either wrapper.
The background block covers B1-B5 for every otherwise executable F/S route
and B7 as the separate research helper. The calibration block covers the
only meaningful family pairs, C1/R1, C2/R2, and C3/R3, with B0 and B6.
Mismatched calibration families and production backgrounds on C1-C3 are not
API states. The M registry partitions **core semantic states** only. Surface
reachability is a separate, lossless P mapping below: every public surface and
route partition maps to its applicable M cells. Orthogonal wrapper defects use
X IDs and are requirements on a surface implementation, not second support
dispositions for the same route/background/resolution tuple. Thus an exact
public call has one M disposition plus zero or more X defects, never two M
dispositions. Reachable-but-ignored B5 configurations are retained as explicit
M cells rather than dismissed as nonsensical.

| ID | Route(s) | Background(s) | Resolution(s) | Effective behavior / status | Precise source | Evidence or explicit gap |
|---|---|---|---|---|---|---|
| M01 | F1 | B0 | R0-R2 | supplied-sigma LM/WLS; **native** | `crates/nereids-pipeline/src/pipeline.rs:1122-1249,1393-1529`; `crates/nereids-physics/src/resolution.rs:1152-1176` | E02, E08 |
| M02 | F1 | B0 | R3 | direct native-IC Rust fit; **native, integration gap** | `crates/nereids-pipeline/src/pipeline.rs:1393-1529`; `crates/nereids-physics/src/resolution.rs:1152-1176` | static-only test gap: no ordinary fit integration with `ResolutionFunction::IkedaCarpenter` |
| M03 | F1 | B0 | R4 | IC table uses tabulated production path; **native** | `bindings/python/src/lib.rs:764-848,2015-2047`; `crates/nereids-pipeline/src/pipeline.rs:1393-1529` | E11 |
| M04 | F2 | B0 | R0 | fractional-transmission Poisson NLL, supplied sigma ignored; **legacy-invalid** | `crates/nereids-pipeline/src/pipeline.rs:1251-1258,1551-1667`; `crates/nereids-fitting/src/poisson.rs:1-22,101-157` | E04, E08 |
| M05 | F2 | B0 | R1-R4 | same invalid statistic with broadening; **legacy-invalid** | `crates/nereids-pipeline/src/pipeline.rs:1551-1667`; `crates/nereids-physics/src/resolution.rs:1152-1176` | E04; static-only test gap: no F2 fit with a nontrivial resolution |
| M06 | F3 | B0 | R0-R1 | joint-Poisson profile deviance with explicit `c`; **native likelihood, incomplete response** | `crates/nereids-pipeline/src/pipeline.rs:1260-1284,1695-1913` | E02, E06, E07 |
| M07 | F3 | B0 | R2-R3 | core accepts either family but still evaluates same-bin `R[T]`; **native reachability, integration gap** | `crates/nereids-pipeline/src/pipeline.rs:1695-1913`; `crates/nereids-physics/src/resolution.rs:1152-1176` | static-only test gap: no ordinary counts fit integration for true UDR or native IC |
| M08 | F3 | B0 | R4 | joint-Poisson with IC-as-tabulated; **native likelihood, incomplete response** | `bindings/python/src/lib.rs:764-848,2015-2047`; `crates/nereids-pipeline/src/pipeline.rs:1695-1913` | E03, E11 |
| M09 | F4 | B0 | R0 | `S/O`, simplified sigma, then F1; `c` and OB variance ignored; **approximation with contract defect** | `crates/nereids-pipeline/src/pipeline.rs:1302-1320,1355-1389` | E04, E06, E07 |
| M10 | F4 | B0 | R1-R4 | same ratio fallback with broadening; **approximation with contract defect** | `crates/nereids-pipeline/src/pipeline.rs:1302-1320,1393-1529`; `crates/nereids-physics/src/resolution.rs:1152-1176` | E06; static-only test gap: no F4 nontrivial-resolution parity gate |
| M11 | F5 | B0 | R0 | caller `flux` is treated as paired Poisson O and only zero Bdet works; **half-wired** | `crates/nereids-pipeline/src/pipeline.rs:1286-1300,1695-1755` | E06, E10 |
| M12 | F5 | B0 | R1-R4 | same paired-zero-background path with broadening; **half-wired, integration gap** | `crates/nereids-pipeline/src/pipeline.rs:1286-1300,1695-1913`; `crates/nereids-physics/src/resolution.rs:1152-1176` | static-only test gap: no F5 nontrivial-resolution recovery gate |
| M13 | F6 | B0-B5 | R0-R4 | nuisance counts plus LM hard-rejected before wrappers; **rejected** | `crates/nereids-pipeline/src/pipeline.rs:1322-1327` | E06, E10 |
| M14 | S1 | B0 | R0-R2 | F1 per pixel with shared precompute; **native** | `crates/nereids-pipeline/src/spatial.rs:1227-1648,1818-1830`; `crates/nereids-physics/src/resolution.rs:1152-1176` | E08 |
| M15 | S1 | B0 | R3-R4 | spatial transmission IC is reachable; **native, integration gap** | `crates/nereids-pipeline/src/spatial.rs:1227-1648,1818-1830`; `crates/nereids-physics/src/resolution.rs:1152-1176` | static-only test gap: no spatial IC recovery test |
| M16 | S2 | B0 | R0 | F2 per pixel and negative T accepted; **legacy-invalid** | `crates/nereids-pipeline/src/spatial.rs:325-346,1818-1830`; `crates/nereids-pipeline/src/pipeline.rs:1551-1667` | E10 |
| M17 | S2 | B0 | R1-R4 | same invalid per-pixel statistic with broadening; **legacy-invalid** | `crates/nereids-pipeline/src/spatial.rs:325-346,1818-1830`; `crates/nereids-physics/src/resolution.rs:1152-1176` | E10; static-only test gap: no S2 nontrivial-resolution fit |
| M18 | S3 | B0 | R0 | joint-Poisson after replacing every pixel O by spatial mean O; **approximation** | `crates/nereids-pipeline/src/spatial.rs:1649-1705,1773-1805` | E05, E06, E12 |
| M19 | S3 | B0 | R1-R4 | same averaged-OB approximation with broadening; **approximation, integration gap** | `crates/nereids-pipeline/src/spatial.rs:1649-1705,1773-1805`; `crates/nereids-physics/src/resolution.rs:1152-1176` | E05, E12; static-only test gap: no S3 nontrivial-resolution recovery test |
| M20 | S4 | B0 | R0 | per-pixel F4 and ignored `c`; **approximation with contract defect** | `crates/nereids-pipeline/src/spatial.rs:1773-1805`; `crates/nereids-pipeline/src/pipeline.rs:1302-1320` | E06, E10 |
| M21 | S4 | B0 | R1-R4 | same per-pixel fallback with broadening; **approximation with contract defect** | `crates/nereids-pipeline/src/spatial.rs:1773-1805`; `crates/nereids-physics/src/resolution.rs:1152-1176` | E10; static-only test gap: no S4 nontrivial-resolution parity gate |
| M22 | S5 | B0 | R0 | paired per-pixel O, zero Bdet; **half-wired but mechanism-valid in tested scope** | `crates/nereids-pipeline/src/spatial.rs:1807-1816`; `crates/nereids-pipeline/src/pipeline.rs:1695-1755` | E05, E06 |
| M23 | S5 | B0 | R1-R4 | paired path with broadening; **half-wired, integration gap** | `crates/nereids-pipeline/src/spatial.rs:1807-1816`; `crates/nereids-physics/src/resolution.rs:1152-1176` | static-only test gap: no S5 nontrivial-resolution recovery test |
| M24 | S6 | B0-B5 | R0-R4 | nuisance counts plus LM rejected in preflight; **rejected** | `crates/nereids-pipeline/src/spatial.rs:916-934` | E06, E10 |
| M25 | F1, S1 | B1 | R0-R4 | SAMMY `Anorm+ABC` wrapper; **native** | `crates/nereids-pipeline/src/pipeline.rs:37-100,1412-1504`; `crates/nereids-pipeline/src/spatial.rs:936-945` | E08; static-only test gap: B1 plus spatial/native IC not exercised |
| M26 | F2, S2 | B1 | R0-R4 | wrapper executes around F2 statistic; **legacy-invalid** | `crates/nereids-pipeline/src/pipeline.rs:1551-1667`; `crates/nereids-pipeline/src/spatial.rs:325-346` | E08; static-only test gap: objective-specific B1/resolution recovery is absent |
| M27 | F3 | B1 | R0-R4 | `Anorm+ABC` supported with BackA interlock; **native likelihood, incomplete response** | `crates/nereids-pipeline/src/pipeline.rs:1695-1755,1796-1825` | E06; static-only test gap: all-resolution B1 coverage is absent |
| M28 | S3 | B1 | R0-R4 | B1 wraps the averaged-OB model; **approximation** | `crates/nereids-pipeline/src/spatial.rs:403-445,1649-1705`; `crates/nereids-pipeline/src/pipeline.rs:1695-1825` | E06, E12; static-only test gap: B1 plus nontrivial resolution not exercised |
| M29 | F4, S4 | B1 | R0-R4 | B1 inherited after ratio conversion; **approximation with contract defect** | `crates/nereids-pipeline/src/pipeline.rs:1302-1320,1412-1504`; `crates/nereids-pipeline/src/spatial.rs:1773-1805` | E06, E10; static-only test gap: B1-specific fallback recovery absent |
| M30 | F5, S5 | B1 | R0-R4 | B1 works only while Bdet stays zero; **half-wired** | `crates/nereids-pipeline/src/pipeline.rs:1695-1755,1796-1825`; `crates/nereids-pipeline/src/spatial.rs:403-445` | E06; static-only test gap: paired B1 recovery and nontrivial resolutions absent |
| M31 | F1, S1 | B2 | R0-R4 | paired BackD/F extension; **native, experimental** | `crates/nereids-pipeline/src/pipeline.rs:37-100,1476-1504,2291-2318`; `crates/nereids-pipeline/src/spatial.rs:936-970` | E08, E13; static-only test gap: no real or IC B2 gate |
| M32 | F2, S2 | B2 | R0-R4 | BackD/F can execute but inherits F2; **legacy-invalid** | `crates/nereids-pipeline/src/pipeline.rs:1551-1667,2291-2318`; `crates/nereids-pipeline/src/spatial.rs:936-970` | static-only test gap: no route-specific B2 test on F2/S2 |
| M33 | F3, S3, F5, S5 | B2 | R0-R4 | joint-Poisson rejects BackD/F; **rejected** | `crates/nereids-pipeline/src/pipeline.rs:1739-1746`; `crates/nereids-pipeline/src/spatial.rs:971-987` | E06, E13 |
| M34 | F4, S4 | B2 | R0-R4 | BackD/F inherited through LM conversion; **approximation with contract defect** | `crates/nereids-pipeline/src/pipeline.rs:1302-1320,1476-1504`; `crates/nereids-pipeline/src/spatial.rs:1773-1805` | static-only test gap: no B2 counts-to-LM recovery or rejection gate |
| M35 | F1, S1 | B3 | R0-R4 | outer multiplicative log-E baseline; **native** | `crates/nereids-pipeline/src/pipeline.rs:117-169,1422-1520,2321-2415`; `crates/nereids-pipeline/src/spatial.rs:1711-1759` | E06, E08, E13; static-only test gap: no IC baseline spatial gate |
| M36 | F2, S2 | B3 | R0-R4 | baseline wraps F2; **legacy-invalid** | `crates/nereids-pipeline/src/pipeline.rs:1551-1667,2321-2415`; `crates/nereids-pipeline/src/spatial.rs:1711-1759` | E08; static-only test gap: no F2/S2 baseline recovery test |
| M37 | F3 | B3 | R0-R4 | baseline changes T inside profile likelihood; **native likelihood, incomplete response** | `crates/nereids-pipeline/src/pipeline.rs:1804-1812,1823-1913` | E03, E06; static-only test gap: true UDR/native-IC baseline integration absent |
| M38 | S3 | B3 | R0-R4 | global/per-pixel baseline around averaged OB; **approximation** | `crates/nereids-pipeline/src/spatial.rs:1711-1759`; `crates/nereids-pipeline/src/pipeline.rs:1804-1812` | E06, E12; static-only test gap: nontrivial-resolution S3 baseline absent |
| M39 | F4, S4 | B3 | R0-R4 | baseline inherited after lossy ratio conversion; **approximation with contract defect** | `crates/nereids-pipeline/src/pipeline.rs:1302-1320,1422-1520`; `crates/nereids-pipeline/src/spatial.rs:1711-1759` | E06; static-only test gap: fallback-specific baseline oracle absent |
| M40 | F5, S5 | B3 | R0-R4 | baseline executable only for zero Bdet paired path; **half-wired** | `crates/nereids-pipeline/src/pipeline.rs:1695-1736,1804-1812`; `crates/nereids-pipeline/src/spatial.rs:1711-1759` | E06; static-only test gap: paired baseline recovery absent |
| M41 | F1, S1 | B4 | R0-R4 | combined stack valid only with fixed `Anorm`; **native under explicit rank constraint** | `crates/nereids-pipeline/src/pipeline.rs:1476-1520,2321-2415`; `crates/nereids-pipeline/src/spatial.rs:396-401` | E08, E13; static-only test gap: all-resolution combined-stack recovery absent |
| M42 | F2, S2 | B4 | R0-R4 | valid stack constraint around invalid F2 objective; **legacy-invalid** | `crates/nereids-pipeline/src/pipeline.rs:1551-1667,2321-2415`; `crates/nereids-pipeline/src/spatial.rs:396-401` | E13; static-only test gap: no successful F2/S2 combined-stack gate |
| M43 | F3 | B4 | R0-R4 | fixed-Anorm B1+B3 allowed and B2 forbidden; **native likelihood, incomplete response** | `crates/nereids-pipeline/src/pipeline.rs:1739-1755,1804-1812,2321-2415` | E06, E13; static-only test gap: full combined-stack resolution matrix absent |
| M44 | F4, S4 | B4 | R0-R4 | combined stack inherited after ratio conversion; **approximation with contract defect** | `crates/nereids-pipeline/src/pipeline.rs:1302-1320,2321-2415`; `crates/nereids-pipeline/src/spatial.rs:396-401` | E13; static-only test gap: fallback combined-stack recovery absent |
| M45 | F5, S5 | B4 | R0-R4 | fixed-Anorm combined stack only with zero Bdet and no B2; **half-wired** | `crates/nereids-pipeline/src/pipeline.rs:1695-1755,2321-2415`; `crates/nereids-pipeline/src/spatial.rs:403-445` | E13; static-only test gap: paired combined-stack recovery absent |
| M46 | F3, S3 | B5 alpha-fit config on ordinary counts | R0-R4 | fitted alpha flags are rejected before optimization; **half-wired/rejected feature** | `crates/nereids-pipeline/src/pipeline.rs:1704-1736`; `crates/nereids-pipeline/src/spatial.rs:403-433` | E10, E13; static-only test gap: direct spatial alpha-flag rejection is not preserved as its own command |
| M47 | F4, S4 | B5 alpha-fit config on ordinary counts | R0-R4 | LM never wires or counts the alpha flags, so they are silently ignored; **legacy-invalid ignored config** | `crates/nereids-pipeline/src/pipeline.rs:1302-1320,2600-2613`; `crates/nereids-pipeline/src/spatial.rs:380-445,1773-1805` | static-only test gap: no incompatible-config rejection exists |
| M48 | F5 | B5 detector/background input | R0-R4 | nonzero Bdet and either fitted alpha are rejected at the single-spectrum boundary; **half-wired/rejected feature** | `crates/nereids-pipeline/src/pipeline.rs:1704-1736` | E06, E10, E13 |
| M49 | F3-F5, S3-S5 (research analogue only) | B7 | R0-R4 | fixed-flux Jacobian/Fisher helper, no optimizer and not production-equivalent; **research-only** | `crates/nereids-pipeline/src/pipeline.rs:3335-3398` | E06 |
| M50 | C1 | B0 | R1 | Gaussian bounded outer calibration with profiled scale; **calibration-only** | `crates/nereids-fitting/src/resolution_calib.rs:296-409,577-590,799-870` | E09 |
| M51 | C1 | B6 | R1 | Gaussian calibration with profiled scale+index-linear nuisance; **calibration-only** | `crates/nereids-fitting/src/resolution_calib.rs:577-590,637-682,799-870` | E09 |
| M52 | C2 | B0 | R2 | UDR width scale/exponent calibration; **calibration-only** | `crates/nereids-fitting/src/resolution_calib.rs:296-409,591-597,799-870` | E09 |
| M53 | C2 | B6 | R2 | UDR correction plus discarded profiled nuisance; **calibration-only, coverage gap** | `crates/nereids-fitting/src/resolution_calib.rs:591-597,637-682,799-870` | static-only test gap: no C2+B6 family-specific recovery test |
| M54 | C3 | B0 | R3 | native IC synthesis inside bounded Nelder-Mead; **calibration-only, experimental** | `crates/nereids-fitting/src/resolution_calib.rs:598-634,799-870,1084-1116` | E09 |
| M55 | C3 | B6 | R3 | IC calibration plus discarded profiled nuisance; **calibration-only, experimental coverage gap** | `crates/nereids-fitting/src/resolution_calib.rs:598-682,1084-1116` | static-only test gap: no C3+B6 recovery/identifiability test |
| M56 | S5 | B5 detector/background input | R0-R4 | nonzero Bdet fails inside each pixel, but errors are swallowed into an accepted result with failed/nonconverged NaN pixels; **half-wired/unsafe result contract** | `crates/nereids-pipeline/src/spatial.rs:403-433,1807-1839,1954-2042`; `crates/nereids-pipeline/src/pipeline.rs:1704-1736` | E10 |
| M57 | F1 | B5 counts config attached through Rust `UnifiedFitConfig` | R0-R4 | transmission LM accepts but never consumes the counts background/alpha config; **legacy-invalid ignored config** | `crates/nereids-pipeline/src/pipeline.rs:369-395,739-742,1241-1249,2600-2613` | static-only test gap: no boundary rejection exists |
| M58 | F2 | B5 counts config attached through Rust `UnifiedFitConfig` | R0-R4 | transmission KL accepts and ignores the counts config in addition to using the invalid fractional-T statistic; **legacy-invalid ignored config** | `crates/nereids-pipeline/src/pipeline.rs:369-395,739-742,1251-1258,2600-2613` | static-only test gap: no boundary rejection exists |
| M59 | S1 | B5 counts config attached through Rust `UnifiedFitConfig` | R0-R4 | spatial transmission LM carries the config to each pixel, where it is ignored; **legacy-invalid ignored config** | `crates/nereids-pipeline/src/spatial.rs:1227-1648,1818-1839`; `crates/nereids-pipeline/src/pipeline.rs:369-395,1241-1249,2600-2613` | static-only test gap: no boundary rejection exists |
| M60 | S2 | B5 counts config attached through Rust `UnifiedFitConfig` | R0-R4 | spatial transmission KL carries and ignores the config while retaining F2's invalid statistic; **legacy-invalid ignored config** | `crates/nereids-pipeline/src/spatial.rs:325-346,1818-1839`; `crates/nereids-pipeline/src/pipeline.rs:369-395,1251-1258,2600-2613` | static-only test gap: no boundary rejection exists |
| M61 | S3 | B4 | R0-R4 | fixed-Anorm B1+B3 wraps the averaged-open-beam model and B2 is forbidden; **explicit spatial approximation** | `crates/nereids-pipeline/src/spatial.rs:971-987,1649-1759`; `crates/nereids-pipeline/src/pipeline.rs:1739-1755,1804-1812,2321-2415` | E06, E12-E13; static-only test gap: flat-beam combined-stack recovery is absent |

## Orthogonal cross-surface issue registry

X rows do not define additional support cells and receive no competing core
disposition. They identify wrapper/API obligations attached by the P mapping.

| ID | Route(s) | Background(s) | Resolution(s) | Effective behavior / status | Precise source | Evidence or explicit gap |
|---|---|---|---|---|---|---|
| X01 | F1-F6, S1-S6, C1-C3 (Rust typed API) | B0-B7 as applicable | R0-R3 | canonical typed dispatch; direct R3 exists; **native core with stale contract comments** | single `crates/nereids-pipeline/src/pipeline.rs:203-292,1122-1332`; spatial `crates/nereids-pipeline/src/spatial.rs:203-260,1227-2042`; calibration `crates/nereids-fitting/src/resolution_calib.rs:296-409,577-682,799-1228`; resolution `crates/nereids-physics/src/resolution.rs:1152-1176` | E06, E08, E09; static-only test gap: no serialized resolved-route provenance |
| X02 | F1-F5, S1-S5 (Python defaults) | B0-B4 | R0-R2, R4 | transmission defaults LM, counts/nuisance/spatial default Auto; **native but surface-specific** | `bindings/python/src/lib.rs:4751-4797,5203-5247,5961-5998` | E13 |
| X03 | F1-F6, S1-S6 (Python solver names) | B0-B5 | R0-R2, R4 | `poisson`/`joint_poisson` alias the domain-dependent KL enum, including F2 on transmission; **misleading legacy alias** | `bindings/python/src/lib.rs:4390-4441` | E10, E13 |
| X04 | C3 to F1-F5/S1-S5 (Python IC handoff) | B0-B6 as applicable | R3 to R4 | Python turns a C3 native-IC result into tabulated input in memory; C1 returns no table and C2 already carries R2; no public persistence artifact exists; **native IC handoff, persistence gap** | `bindings/python/src/lib.rs:764-848,1407-1425,2015-2047` | E09, E11; static-only test gap: no round-trip file/provenance export |
| X05 | F3-F6, S3-S6 (Python detector-background surface) | B5 | R0-R2, R4 | single F3 rejects alpha and F4 silently ignores it; spatial ordinary-count alpha rejects at the binding, while S5 nonzero Bdet returns an accepted all-failed/NaN result; **half-wired and surface-inconsistent** | `bindings/python/src/lib.rs:5041-5045,5203-5247,5455-5487`; `bindings/python/python/nereids/__init__.pyi:1798-1822`; `crates/nereids-pipeline/src/spatial.rs:403-433,1807-1839,1954-2042` | E10, E13 |
| X06 | F1-F4 (MCP single-count manifest) | B0-B2 | R0-R2, R4 | hidden default converts raw counts to transmission LM; explicit KL selects F3; **cross-surface divergence** | `bindings/python/python/nereids/mcp/server.py:663-717` | E13 |
| X07 | F1-F4 (MCP single-count manifest) | B0-B2 | R0-R2, R4 | any `fit_domain` string other than exact `counts` silently selects transmission, so KL typo reaches F2; **half-wired/unsafe fallback** | `bindings/python/python/nereids/mcp/server.py:684-717`; `bindings/python/python/nereids/mcp/server.py:1051-1157` | E13 |
| X08 | S3, S4 (MCP spatial counts) | B0-B2 | R0-R2, R4 | always constructs counts and Python Auto selects S3, unlike MCP single default; **cross-surface divergence** | `bindings/python/python/nereids/mcp/server.py:862-953`; `bindings/python/src/lib.rs:4751-4797` | E13; static-only test gap: FastMCP suite skipped and no live spatial manifest gate |
| X09 | F1-F5, S1-S5 (MCP config boundary) | B1-B5 | R1-R4 | allowlists silently drop valid-looking fields and validator omits solver/domain/schema checks; **half-wired** | `bindings/python/python/nereids/mcp/server.py:442-551,1051-1157` | E13; static-only test gap: no live FastMCP schema test |
| X10 | F1-F5, S1-S5 (MCP results) | B0-B5 | R0-R2, R4 | execution success is true even for nonconvergence and warnings/provenance are dropped; **half-wired result contract** | `bindings/python/python/nereids/mcp/server.py:592-629,1575-1598` | static-only test gap: no live MCP nonconvergence/result-contract test |
| X11 | F1-F4, S1-S4 (GUI route resolver) | B0-B4 | R0-R2 | default is KL and domain is inferred from solver plus OB presence, not explicitly selected; **cross-surface divergence** | `apps/gui/src/state.rs:1579-1625`; `apps/gui/src/guided/analyze.rs:175-214,2415-2445,2598-2635,2804-2828` | static-only test gap: no completed GUI route-selection test |
| X12 | F2, S2 reachable from GUI raw HDF5 without OB | B0-B4 | R0-R2 | raw sample counts are wrapped as T with sigma=1, then default KL remains reachable; **legacy-invalid data reinterpretation** | `apps/gui/src/guided/normalize.rs:525-632`; `apps/gui/src/state.rs:1613-1625` | static-only test gap: no GUI regression that blocks raw HDF5 fitting without OB |
| X13 | F3, S3 (GUI exposure handling) | B0-B4 | R0-R2 | normalization uses proton charges but count fits use independent `kl_c_ratio`; **half-wired duplicate source of truth** | `apps/gui/src/guided/normalize.rs:437-462`; `apps/gui/src/state.rs:1600-1603,1621-1625`; `apps/gui/src/guided/analyze.rs:2280-2330` | E02; static-only test gap: no GUI synchronization/persistence test |
| X14 | S3 (GUI displayed model) | B0-B4 | R0-R2 | fitter uses averaged OB while overlay scales by local pixel OB; **misleading approximation display** | `crates/nereids-pipeline/src/spatial.rs:1649-1705`; `apps/gui/src/guided/analyze.rs:1399-1472` | E05, E12; static-only test gap: no GUI effective-model overlay test |
| X15 | F1-F5, S1-S5, C1-C3 (GUI/MCP reachability) | B1-B6 | R1-R4 | GUI/MCP lack direct calibration/native-IC construction and durable calibrated-resolution provenance; **half-wired/absent workflow** | `apps/gui/src/state.rs:582-635`; `apps/gui/src/widgets/design.rs:572-699,852-877`; `bindings/python/python/nereids/mcp/server.py:442-467` | E09, E13; static-only test gap: no GUI/MCP calibration or resolution round trip exists |
| X16 | F1-F5, S1-S5 (cross-surface result semantics) | B0-B5 | R0-R4 | D/dof is mirrored into a chi-square field; some GUI/MCP labels, counters, and success semantics remain ambiguous; **half-wired result contract** | `crates/nereids-pipeline/src/pipeline.rs:1695-1702`; `apps/gui/src/guided/analyze.rs:1883-1906,2450-2475,2650-2666`; `apps/gui/src/guided/results.rs:47-76`; `bindings/python/python/nereids/mcp/server.py:592-629` | E13; static-only test gap: GUI labels and MCP result semantics lack end-to-end tests |
| X17 | F1-F5, S1-S5 (cross-surface defaults) | B0-B5 | R0-R4 | solver, temperature, library, and polish defaults are not one resolved contract; **cross-surface divergence** | `apps/gui/src/state.rs:1579-1625`; `bindings/python/src/lib.rs:4751-4797,5203-5247,5961-5998`; `crates/nereids-pipeline/src/pipeline.rs:397-403,1335-1352` | E13; static-only test gap: no cross-surface default-equivalence golden test |

## Lossless surface reachability mapping

P rows partition each public surface by route class. They map surface
reachability onto the core M partition and attach orthogonal X defects without
creating another disposition. Python C1, C2, and C3 are separate rows because
only C3 performs an R3-to-R4 IC conversion.

| ID | Public surface | Route partition | Resolution reachability | Applicable core cells | Orthogonal issues | Precise source/evidence |
|---|---|---|---|---|---|---|
| P01 | Rust typed single | F1-F6 | R0-R3 | M01-M13, M25-M48, M57-M58 as route/B applies | X01 | `crates/nereids-pipeline/src/pipeline.rs:203-292,1122-1332`; E06, E08, E10 |
| P02 | Rust typed spatial | S1-S6 | R0-R3 | M14-M24, M25-M48, M56, M59-M61 as route/B applies | X01 | `crates/nereids-pipeline/src/spatial.rs:203-260,1227-2042`; E05, E06, E08, E10, E12 |
| P03 | Rust calibration | C1-C3 | C1/R1, C2/R2, C3/R3 | M50-M55 | X01 | `crates/nereids-fitting/src/resolution_calib.rs:296-409,577-682,799-1228`; E09, E14 |
| P04 | Python typed single transmission | F1-F2 | R0-R2, R4 | M01, M03-M05, M25-M26, M31-M32, M35-M36, M41-M42 | X02, X03, X17; X04 only when R4 came from C3 | `bindings/python/src/lib.rs:4390-4441,5961-6235`; E04, E11, E13-E14 |
| P05 | Python typed single raw counts | F3-F4 | R0-R2, R4 | M06-M10, M27, M29, M33-M34, M37, M39, M43-M47 | X02, X03, X05, X17; X04 only when R4 came from C3 | `bindings/python/src/lib.rs:5203-5511`; E02-E04, E07, E10-E11, E13-E14 |
| P06 | Python typed single nuisance counts | F5-F6 | R0-R2, R4 | M11-M13, M30, M33, M40, M45, M48 | X02, X03, X05, X17; X04 only when R4 came from C3 | `bindings/python/src/lib.rs:5203-5511`; E10, E13 |
| P07 | Python typed spatial transmission | S1-S2 | R0-R2, R4 | M14-M17, M25-M26, M31-M32, M35-M36, M41-M42 | X02, X03, X17; X04 only when R4 came from C3 | `bindings/python/src/lib.rs:4680-5059`; E08, E10, E13-E14 |
| P08 | Python typed spatial raw counts | S3-S4 | R0-R2, R4 | M18-M21, M28-M29, M33-M34, M38-M39, M44, M61; M46-M47 are blocked at this binding | X02, X03, X05, X17; X04 only when R4 came from C3 | `bindings/python/src/lib.rs:4680-5059`; E05-E07, E10, E12-E14 |
| P09 | Python typed spatial nuisance counts | S5-S6 | R0-R2, R4 | M22-M24, M30, M33, M40, M45, M56 | X02, X03, X05, X17; X04 only when R4 came from C3 | `bindings/python/src/lib.rs:4680-5059`; E05, E10, E13 |
| P10 | Python calibration | C1 Gaussian | R1 | M50-M51 | none | `bindings/python/src/lib.rs:1490-1676`; E09, E14 |
| P11 | Python calibration | C2 UDR correction | R2 | M52-M53 | none | `bindings/python/src/lib.rs:1407-1425,1490-1676`; E09, E14 |
| P12 | Python calibration/handoff | C3 IC to production F/S | R3 to R4 | M54-M55 plus the selected production M cell | X04 | `bindings/python/src/lib.rs:1407-1425,1490-1676`; E09, E11 |
| P13 | MCP single spectrum | F1-F4 through Python | R0-R2, R4 | the applicable M01-M10/M25-M48 cell from P04/P05 | X06, X07, X09, X10, X15, X16, X17 | `bindings/python/python/nereids/mcp/server.py:442-717,1051-1157,1575-1598`; E13 |
| P14 | MCP spatial map | S1-S4 through Python | R0-R2, R4 | the applicable M14-M21/M25-M48/M61 cell from P07/P08 | X08, X09, X10, X15, X16, X17 | `bindings/python/python/nereids/mcp/server.py:862-953,1051-1157,1575-1598`; E13 |
| P15 | GUI single analysis | F1-F4 | R0-R2 | the applicable M01-M10/M25-M48 cell from P04/P05 | X11, X12, X13, X15, X16, X17 | `apps/gui/src/state.rs:1579-1625`; `apps/gui/src/guided/analyze.rs:175-214,2280-2475`; static-only test gap: no completed GUI route suite |
| P16 | GUI spatial analysis | S1-S4 | R0-R2 | the applicable M14-M21/M25-M48/M61 cell from P07/P08 | X11, X12, X13, X14, X15, X16, X17 | `apps/gui/src/guided/analyze.rs:1399-1472,2598-2828`; static-only test gap: no completed GUI route suite |
| P17 | Rust research helper | M49 fixed-flux Fisher/Jacobian | R0-R4 | M49 only | X01 | `crates/nereids-pipeline/src/pipeline.rs:3319-3611`; E06 |
| P18 | Python research helper | `compute_model_jacobian` to M49 | R0-R2, R4 | M49 only | none | `bindings/python/src/lib.rs:5656-5797`; `bindings/python/python/nereids/__init__.pyi:1874-1945`; E06, E13 |

## Explanatory roll-ups (non-registry)

The tables below preserve the original audit narrative. They are superseded
as support-cell definitions by the normalized registry above.

### Domain × solver × surface

| ID | Surface/input | Requested solver | Actual objective/data transformation | Status | Synthetic evidence | Real evidence | Preliminary disposition |
|---|---|---|---|---|---|---|---|
| F1 | single transmission | `auto`/`lm` | supplied-σ weighted LM on normalized T | native | transmission, background, baseline, energy-scale, fit-range, and Python IC-as-tabulated recovery pass | committed Hf+Gaussian anchor passes; χ²/ν≈219657 shows convergence is not adequacy | keep+improve |
| F2 | single transmission | `kl`, `poisson`, `joint_poisson` | single-arm Poisson NLL on fractional T; supplied σ absent from optimization | legacy-invalid | model-generated recovery tests pass, but uncertainty-reweight control is invariant and negative T is accepted | no route-specific real validity gate; Hf semantic probe confirms invariance | deprecate, then remove/replace |
| F3 | single raw counts | `auto`, KL aliases | profiled joint-Poisson conditional-binomial deviance with explicit `c` | native likelihood, incomplete response | matched stochastic ensemble: +5.0%, −0.30%, +0.36% bias at 25/250/2500 O counts; IC-as-tabulated recovery passes | Hf+Gaussian and Ta+IC/JENDL/baseline anchors pass | keep as primary counts route; improve response/background |
| F4 | single raw counts | `lm` | `T=S/O`, `σ≈√S/O`, then F1; ignores `c` and OB variance | approximation with contract defect | matched ensemble bias +52.8% at O=25, approaches F3 at high counts | real `c=5.982` Hf route hit density 0 and did not converge; manual OB prescale converged | deprecate; reject raw unequal-exposure use |
| F5 | single counts+nuisance | `auto`/KL | caller `flux` is treated as Poisson `O`; background must be zero; alpha fit rejected | half-wired | zero-background paired route works; rejection tests pass | no dedicated real gate | split/rename paired-open-beam route; complete or hide detector background |
| F6 | single counts+nuisance | `lm` | no objective; hard rejection | rejected | rejection test passes | n/a | keep unsupported; prevent misleading exposure |
| S1 | spatial transmission | `auto`/`lm` | F1 per pixel with shared precompute; optional global/per-pixel baseline | native | spatial transmission/Gaussian/tabulated smoke and baseline tests pass | no measured spatial validity gate | keep+improve |
| S2 | spatial transmission | KL aliases | F2 per pixel; fit range rejected | legacy-invalid | matched public smoke passes, while negative T is also accepted ([E10](evidence/phase-0/test-results.md#remaining-public-route-cells)) | no real route-validity gate | deprecate with F2 |
| S3 | spatial raw counts | `auto`/KL | F3 per pixel but every sample pixel uses spatially averaged open beam as `O` | explicit approximation | uniform-OB tests pass; 4:1 gradient control produced fully converged false density map | real VENUS OB totals have 4.55% nonzero-pixel CV vs 1.06% shot expectation and systematic half/quadrant differences | retain only as explicitly approximate mode; paired route should be scientific default |
| S4 | spatial raw counts | `lm` | F4 per pixel with per-pixel `S/O`; energy-scale fit rejected | approximation with contract defect | smoke tests pass; inherits F4 bias and ignored `c` | none | deprecate with F4 |
| S5 | spatial counts+nuisance | `auto`/KL | per-pixel supplied `flux` becomes Poisson `O`; zero background works, while nonzero-B pixel errors are swallowed | half-wired | paired gradient control recovered truth to 1.56e-10 relative; nonzero B returns `n_failed=1`, nonconvergence, and NaN on 1x1 input | none | keep/rename paired-open route; preflight-reject Bdet until complete |
| S6 | spatial counts+nuisance | `lm` | hard rejection | rejected | rejection/preflight tests pass | n/a | keep unsupported |
| C1 | resolution calibration: Gaussian | n/a | outer bounded Nelder–Mead; profiled transmission WSSR | calibration-only | matched Gaussian recovery/bounds tests | no real calibration fixture | keep+improve |
| C2 | resolution calibration: UDR correction | n/a | outer bounded Nelder–Mead over width scale/exponent; profiled transmission WSSR | calibration-only | matched base/raw UDR tests | real UDR file unavailable | keep+improve, with provenance gate |
| C3 | resolution calibration: IC | n/a | outer bounded Nelder–Mead over IC coordinates; profiled transmission WSSR | calibration-only | selected optimizer/convergence plus bounds/validation tests pass ([E09](evidence/phase-0/test-results.md#resolution-calibration-routes)); full recovery/PSR closed loops were not freshly rerun | archived calibration cannot be exactly replayed from supplied provenance | keep+improve; add count-response calibration before physical use |

Dispatch evidence: `crates/nereids-pipeline/src/pipeline.rs:1122-1332` and
`crates/nereids-pipeline/src/spatial.rs:215-228,1649-1830`. Calibration
evidence: `crates/nereids-fitting/src/resolution_calib.rs:637-682,799-1198`.

### Background/nuisance capability

| Model | F1/S1 | F2/S2 | F3/S3 | F4/S4 | F5/S5 | C1–C3 | Notes |
|---|---|---|---|---|---|---|---|
| none | yes | yes | yes | yes | only with zero supplied B | yes | baseline route for all comparisons |
| SAMMY Anorm+ABC | full | full but unvalidated | supported; A required when B/C free | inherited from LM | supported as transmission wrapper | no | post-physics apparent-transmission form |
| SAMMY BackD/F | full after validation fix | code supports, docs/tests inconsistent | rejected | inherited from LM | rejected | no | flags must be paired; single/spatial start validation differs |
| NEREIDS multiplicative log-E baseline | yes | yes but F2 deprecated | yes | yes | yes when otherwise executable | no | outermost; free b0 + free Anorm rejected |
| SAMMY + multiplicative baseline | only with fixed Anorm | same | same, without D/F | same | same, zero Bdet | no | structural normalization constraint |
| detector-space `α1 ΦT + α2 Bdet` | Rust counts config is silently ignored (M57/M59) | ignored plus invalid F2/S2 (M58/M60) | alpha flags reject; nonzero B redirects to F5/S5 | Rust/single-Python alpha is ignored, Python spatial rejects; detector input redirects to F6/S6 | F5 rejects; S5 returns accepted all-failed/NaN result | no | only the fixed-flux research helper implements the intended form; production behavior is split across M46-M48/M56-M60 |
| calibration profiled `aT+b0+b1·index` | no | no | no | no | no | optional | coefficients unbounded and discarded |
| legacy Fisher `b0+b1/√E`, alpha | no | no | research-only | research-only | research-only | no | not production joint-Poisson information geometry |

Definitions are at `pipeline.rs:37-148,294-348`, production counts gates at
`pipeline.rs:1704-1755`, and calibration nuisance at
`resolution_calib.rs:637-682`.

### Resolution family capability

| Resolution representation | Production fit reachability | Calibration reachability | Python reachability | Current measurement operator | Evidence/gap |
|---|---|---|---|---|---|
| none | all executable F/S routes | n/a | direct | identity | broad fit coverage |
| Gaussian | all executable F/S routes | C1 | Gaussian parameter triplet | `R[T]`; automatic auxiliary grid | real Hf F1/F3 plus synthetic spatial recovery |
| tabulated UDR | all executable F/S routes | base for C2 | `TabulatedResolution` | `R[T]`; caller-grid edge truncation/renormalization | synthetic kernel/plan tests; real UDR unavailable |
| native IC | Rust fits and C3 | C3 | fit APIs require `.as_tabulated()` | synthesized table then `R[T]`; caller-grid edge behavior | kernel/calibration tests, but no ordinary native-IC fit integration |
| IC converted to tabulated | all Python executable F/S routes | calibration result export in-memory | `.as_tabulated()` | same sampled kernel under tabulated capability/acceleration dispatch | real archived F3 and matched F1/F3 gates pass ([E03](evidence/phase-0/test-results.md#archived-real-ta-icjendl-counts-replay), [E11](evidence/phase-0/test-results.md#ordinary-fit-ic-handoff)); no spatial IC gate |

All current production routes form Beer–Lambert transmission and apply
resolution afterward (`nereids-fitting/src/transmission_model.rs:512-549`).
No route yet forms migrated `R[Φ]` and `R[ΦT]` count arms separately.

### Cross-surface reachability snapshot

| Surface | Transmission | Raw counts | Nuisance counts | Calibration | Important divergence |
|---|---|---|---|---|---|
| Rust typed API | F1/F2 | F3/F4 | F5/F6 | C1–C3 | direct native IC supported |
| Python typed API | F1/F2 | F3/F4 | F5/F6 through detector-background arg / constructors | C1–C3 | IC fit input must be tabulated; nuisance args overpromise |
| spatial Python | S1/S2 | S3/S4 | S5/S6 | none | raw counts average OB; global baseline default |
| MCP | transmission and counts-file routes | single counts default to hidden transmission-domain LM; spatial counts use Auto→joint-Poisson | selected but incomplete fields exposed | none found | typoed domain silently selects transmission; valid options are dropped |
| GUI | transmission LM or raw-count KL when both arms exist | F3 normally; raw HDF5 without OB is wrapped as transmission and default KL remains reachable | no working detector-bg workflow found | no direct calibrator workflow found | separate exposure ratios and overlay/fit model mismatch |
