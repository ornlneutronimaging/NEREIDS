# SAMMY ex003 Validation Specification for NEREIDS

**Date**: 2026-02-06
**Source**: SAMMY test case analysis
**Purpose**: Define validation strategy for 0K R-matrix cross section implementation

---

## 1. Test Case Overview

**Location**: `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY/samexm/samexm/ex003/`

**Physics**: Synthetic nuclide with 3 scattering channels (neutron + 2 fission)
- Tests all cross-section types: capture, elastic, fission, transmission, total
- 0K calculation (no Doppler broadening)
- Reich-Moore R-matrix formalism

**Variants**: 6 test cases (a, c, e, f, x, t)

| Variant | Data Type | Description | Chi-squared Tolerance |
|---|---|---|---|
| ex003c | Capture | σ_c (gamma capture) | χ²/N ≈ 1e-5 |
| ex003a | Absorption | σ_a = σ_c + σ_f | χ²/N ≈ 2e-9 |
| ex003e | Elastic | σ_el (scattering) | Similar to ex003c |
| ex003f | Fission | σ_f (total fission) | Similar to ex003c |
| ex003x | Transmission | T = exp(-nσt) | With sample thickness |
| ex003t | Total | σ_t = σ_el + σ_f + σ_c | χ²/N ≈ 7.6 (test mismatch) |

---

## 2. Input Files

### 2.1 Resonance Parameters

**File**: `ex003c.par` (shared across all variants)
**Location**: `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY/samexm/samexm/ex003/ex003c.par`

**Format**: Fixed-width, 5 columns of 11 characters each

| Column | Content | Units |
|---|---|---|
| 1 | Resonance energy E_λ | eV |
| 2 | Total width Γ_tot | milliEV |
| 3 | Neutron width Γ_n | milliEV |
| 4 | Fission width A (Γ_fa) | milliEV |
| 5 | Fission width B (Γ_fb) | milliEV |

**12 Resonances**:
```
Energy (eV):  0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0
All widths:   Γ_tot = 1.0, Γ_n = 0.5, Γ_fa = 0.5, Γ_fb = 0.5 milliEV
```

**Gamma width Γ_γ**: Derived from Γ_tot - Γ_n - Γ_fa - Γ_fb = 1.0 - 0.5 - 0.5 - 0.5 = -0.5 milliEV (!)
- **Note**: Negative gamma width is physically unusual but handled by SAMMY (sign preserved in amplitude)

### 2.2 Input Configuration

**File**: `ex003c.inp` (example for capture cross section)
**Location**: `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY/samexm/samexm/ex003/ex003c.inp`

**Key Parameters** (from LPT analysis):
- Element name: Synthetic
- Atomic weight: 10.000 amu
- Effective radius: 2.9080 fm
- Target spin I: Implied from spin group structure
- **1 spin group**: J = 0.5, 3 channels
  - Channel 1 (entrance): Neutron, L=0, J=0.5
  - Channel 2 (exit): Fission A
  - Channel 3 (exit): Fission B
- Data type: "CAPTUre cross section"
- Broadening: NOT wanted (0 K)
- Fitting: Do not solve Bayes equations (fixed parameters)

### 2.3 Experimental Data

**File**: `ex003c.dat` (synthetic data)
**Location**: `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY/samexm/samexm/ex003/ex003c.dat`

**Format**: "Twenty" format (20 characters per field, 3 fields per line)

| Column | Content |
|---|---|
| 1 | Energy (eV) |
| 2 | Cross section or transmission value |
| 3 | Uncertainty |

**Data range**:
- Emin: 9.999×10⁻⁶ eV (adjusted from 1×10⁻⁸ in input)
- Emax: 1200.0 eV
- **3112 data points**

---

## 3. Reference Outputs

### 3.1 Main Reference: LPT Files

**Location**: `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY/samexm/samexm/ex003/answers/`

**Files**:
- `ex003c.lpt` (capture)
- `ex003a.lpt` (absorption)
- `ex003e.lpt` (elastic)
- `ex003f.lpt` (fission)
- `ex003x.lpt` (transmission)
- `ex003t.lpt` (total)

**Content** (152 lines each):
1. SAMMY version and header
2. Input file names
3. Target element and atomic weight
4. Control flags parsed
5. Data type classification
6. Target thickness (0.0 or 5×10⁻⁵ atoms/barn)
7. **Spin group structure** (lines 68-77):
   - 1 spin group, J = 0.5
   - 3 particle channels
   - 12 resonances
8. **Resonance parameter table** (lines 93-113):
   - Energy (eV)
   - Gamma widths (milliEV): Γ_tot, Γ_n, Γ_fa, Γ_fb
9. **Energy range** (lines 122-135):
   - Emin: 9.90350×10⁻⁶ eV
   - Emax: 1200.0 eV
   - NDAT: 3112
10. **Chi-squared values** (lines 146-147):
    - Customary chi-squared
    - Chi-squared per data point

**Example Chi-Squared Values**:

| Variant | χ² (customary) | χ²/NDAT | Interpretation |
|---|---|---|---|
| ex003c | 3.287×10⁻² | 1.056×10⁻⁵ | Good fit (synthetic data) |
| ex003a | 6.389×10⁻⁶ | 2.053×10⁻⁹ | Excellent fit |
| ex003t | 23705.2 | 7.617 | Expected mismatch (wrong thickness) |

### 3.2 Parsing Strategy for LPT Files

**Step 1**: Read file line-by-line

**Step 2**: Extract resonance parameters (lines 93-113)
- Pattern: Lines starting with resonance energy values
- Parse 5 columns: E_λ, Γ_tot, Γ_n, Γ_fa, Γ_fb

**Step 3**: Extract chi-squared (lines 146-147)
- Pattern 1: Line containing "CHI SQUARED" followed by float
- Pattern 2: Next line contains χ²/NDAT value

**Step 4**: Extract data statistics
- Pattern: "Number of experimental data points" → extract integer (3112)

**Step 5**: Verify energy range
- Pattern: "Emin" and "Emax" followed by scientific notation floats

**Rust Example**:
```rust
fn parse_lpt_chi_squared(path: &Path) -> Result<(f64, f64), Error> {
    let contents = fs::read_to_string(path)?;
    let mut chi2 = None;
    let mut chi2_per_point = None;

    for (i, line) in contents.lines().enumerate() {
        if line.contains("CHI SQUARED") && chi2.is_none() {
            // Extract first float after "CHI SQUARED"
            if let Some(val) = extract_float(line) {
                chi2 = Some(val);
            }
        }
        if chi2.is_some() && chi2_per_point.is_none() {
            // Next line after chi2 contains per-point value
            if let Some(val) = extract_float(line) {
                chi2_per_point = Some(val);
                break;
            }
        }
    }

    Ok((chi2.unwrap(), chi2_per_point.unwrap()))
}
```

### 3.3 Alternative: ODF Files

**Files**: `ex003*.odf` (binary plotting data)
**Location**: Same `answers/` directory

**Structure** (from SAMMY documentation):
- Multi-section binary format
- Section 2: Cross section
- Section 4: Calculated cross section
- Sections 6/8/9: Transmission values

**Parsing**: Requires either:
1. SAMMY tool `samplt` to convert to ASCII `.lst`
2. Custom binary reader matching SAMMY's Fortran unformatted I/O

**Recommendation**: Start with LPT files (text), defer ODF until needed.

---

## 4. Validation Tolerance

**Source**: `/Users/chenzhang/github.com/NEREIDS/NEREIDS_claude/docs/adr/0002-sammy-reference-validation.md`

**Default**: 1.0e-4 relative error

**Formula**:
```rust
fn relative_error(a: f64, b: f64, cutoff: f64) -> f64 {
    let max_val = a.abs().max(b.abs());
    if max_val < cutoff {
        0.0  // Both below cutoff, ignore
    } else {
        (a - b).abs() / max_val
    }
}
```

**Cutoff**: 1e-30 (to avoid false positives on tiny values)

**Per-field overrides** (for future test cases):
- PAR files: May use 1e-3 (looser tolerance for fitted parameters)
- LST files: May use 1e-5 (tighter for high-precision cross sections)
- LPT chi-squared: Absolute difference tolerance (not relative)

---

## 5. Test Fixture Structure

### 5.1 Directory Layout

```
tests/fixtures/sammy_reference/ex003/
  input/
    ex003c.par
    ex003c.inp
    ex003c.dat
    ex003a.inp, ex003a.dat
    ex003e.inp, ex003e.dat
    ex003f.inp, ex003f.dat
    ex003x.inp, ex003x.dat
    ex003t.inp, ex003t.dat
  expected/
    ex003c.lpt
    ex003a.lpt
    ex003e.lpt
    ex003f.lpt
    ex003x.lpt
    ex003t.lpt
  config.toml
```

### 5.2 Configuration File

**File**: `config.toml`

```toml
[metadata]
name = "ex003"
description = "SAMMY R-matrix cross-section validation"
source = "samexm/samexm/ex003"

[physics]
formalism = "reich-moore"
temperature = 0.0  # 0 K, no Doppler broadening
element = "synthetic"
atomic_weight = 10.0
effective_radius_fm = 2.908

[[test_cases]]
id = "ex003c"
data_type = "capture"
input_par = "input/ex003c.par"
input_inp = "input/ex003c.inp"
input_dat = "input/ex003c.dat"
expected_lpt = "expected/ex003c.lpt"
tolerance = 1e-4
cutoff = 1e-30

[[test_cases]]
id = "ex003a"
data_type = "absorption"
input_par = "input/ex003c.par"
input_inp = "input/ex003a.inp"
input_dat = "input/ex003a.dat"
expected_lpt = "expected/ex003a.lpt"
tolerance = 1e-4
cutoff = 1e-30

# ... repeat for e, f, x, t variants

[validation]
compare_chi_squared = true
compare_resonance_params = false  # Fixed, not fitted
energy_range = [9.9e-6, 1200.0]
num_data_points = 3112
```

---

## 6. Rust Test Implementation Outline

```rust
#[test]
fn test_ex003_capture() -> Result<(), Box<dyn Error>> {
    // 1. Load input parameters
    let params = RMatrixParameters::from_par_file("tests/fixtures/sammy_reference/ex003/input/ex003c.par")?;
    let config = ForwardModelConfig {
        temperature_k: 0.0,
        normalization: 1.0,
        self_shielding: false,
    };

    // 2. Load experimental data
    let (energies, data, uncertainties) = load_dat_file("tests/fixtures/sammy_reference/ex003/input/ex003c.dat")?;
    let energy_grid = EnergyGrid::new(energies)?;

    // 3. Compute cross sections
    let forward_model = DefaultForwardModel::new(None);  // No resolution
    let sigma = forward_model.transmission(&energy_grid, &params, &config)?;

    // 4. Compute chi-squared
    let chi2 = compute_chi_squared(&sigma, &data, &uncertainties);
    let chi2_per_point = chi2 / (data.len() as f64);

    // 5. Load expected chi-squared from LPT
    let (expected_chi2, expected_chi2_per_point) = parse_lpt_chi_squared(
        "tests/fixtures/sammy_reference/ex003/expected/ex003c.lpt"
    )?;

    // 6. Compare with tolerance
    let rel_error = relative_error(chi2, expected_chi2, 1e-30);
    assert!(rel_error < 1e-4, "Chi-squared mismatch: {} vs {}", chi2, expected_chi2);

    Ok(())
}

// Repeat for ex003a, ex003e, ex003f, ex003x, ex003t
```

---

## 7. Comparison Strategy

### 7.1 Cross Section Values

**Option A**: Direct comparison at each energy point
- Load expected cross sections from ODF/LST files
- Compute relative error at each point
- Report max error and RMS error

**Option B**: Chi-squared comparison
- Compute χ² from NEREIDS output
- Compare with SAMMY's reported χ² value
- Tolerance: Absolute difference < 0.01 (1% of typical χ²)

**Recommendation**: Use **Option B** for ex003 (simpler, avoids parsing binary ODF)

### 7.2 Resonance Parameters

Not applicable for ex003 (fixed parameters, no fitting).

For future fitted cases:
- Compare final parameters from PAR file
- Tolerance: 1e-3 relative error (wider tolerance for nonlinear optimization)

---

## 8. Known Issues and Expected Behaviors

### 8.1 Negative Gamma Width

ex003 has Γ_γ = -0.5 milliEV (derived from total width balance).

**SAMMY handling**: Sign preserved in amplitude calculations (lines 102-106 in dopush1.f90)

**NEREIDS**: Must match this behavior:
```rust
let a_gamma = gamma_width.value.signum() * gamma_width.value.abs().sqrt();
```

### 8.2 ex003t Chi-Squared Mismatch

**Expected**: χ² ≈ 23705 (very large)

**Reason**: Test uses wrong sample thickness (5×10⁻⁵) to intentionally produce poor fit.

**Validation**: Confirm NEREIDS produces similar large χ² (within 10% relative error)

### 8.3 Zero Temperature

All ex003 variants have temperature = 0 K (no Doppler broadening).

**NEREIDS**: Ensure `ForwardModelConfig::temperature_k = 0.0` skips Doppler convolution.

---

## 9. Implementation Checklist

- [ ] Parse PAR file format (11-char fixed-width columns)
- [ ] Parse DAT file format (20-char "twenty" format)
- [ ] Load input configuration from INP file (or config.toml)
- [ ] Implement R-matrix cross-section calculation (see `reich_moore_physics.md`)
- [ ] Implement penetration/shift factors (see `penetration_shift_factors.md`)
- [ ] Compute chi-squared: χ² = Σ[(σ_calc - σ_obs)² / uncertainty²]
- [ ] Parse LPT file to extract expected chi-squared
- [ ] Create test fixture directory with all 6 variants
- [ ] Write Rust test functions for ex003a-f, ex003x, ex003t
- [ ] Validate all tests pass with 1e-4 relative error tolerance

---

**Next Steps**:
1. Copy SAMMY ex003 files to NEREIDS test fixtures
2. Implement file parsers (PAR, DAT, LPT)
3. Implement R-matrix physics modules
4. Run validation tests
5. Debug any discrepancies by comparing intermediate values (P_l, S_l, R-matrix elements)

---

**Source Files Referenced**:
- Input: `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY/samexm/samexm/ex003/*.{par,inp,dat}`
- Expected: `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY/samexm/samexm/ex003/answers/*.lpt`
- NEREIDS Docs: `docs/adr/0002-sammy-reference-validation.md`, `docs/sammy-validation-map.md`
