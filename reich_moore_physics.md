# Reich-Moore R-Matrix Physics for NEREIDS

**Date**: 2026-02-06
**Source**: SAMMY Fortran implementation analysis
**Purpose**: Physics reference for implementing 0K R-matrix cross sections in Rust

---

## 1. Core R-Matrix Equation

**Source**: `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY/sammy/src/clm/dopush1.f90` (lines 8-184)

### 1.1 Single Energy Point Evaluation

At neutron energy E, the R-matrix formalism constructs a **3×3 complex symmetric matrix**:

```
R-matrix = R(E) + iS(E)
```

where:
- **R(E)**: Real energy-dependent collision matrix
- **S(E)**: Imaginary shift matrix (from energy detuning)

**Three scattering channels**:
1. **Channel 1**: Neutron entrance/exit (n + target)
2. **Channel 2**: Fission channel A
3. **Channel 3**: Fission channel B

### 1.2 Matrix Element Construction

For each resonance λ with energy E_λ and widths Γ_n, Γ_γ, Γ_fa, Γ_fb:

**Energy-dependent factors** (lines 107-111):
```
Δ = E_λ - E                           (energy detuning)
D = Δ² + (Γ_γ/4)²                     (Breit-Wigner denominator)
R_factor = (Γ_γ/4) / D                (real resonance amplitude)
S_factor = Δ / (2D)                   (imaginary shift)
```

**Channel amplitudes** (lines 100-106):
```
A_n = √(Γ_n · P_e / P_r)              (neutron, penetration-corrected)
A_fa = ±√|Γ_fa|                       (fission A, sign preserved if negative)
A_fb = ±√|Γ_fb|                       (fission B, sign preserved if negative)
```

where:
- P_e = penetration factor at energy E
- P_r = penetration factor at resonance energy E_λ

**Matrix accumulation** (lines 114-126):
```
R(i,j) += R_factor · A_i · A_j        (real part)
S(i,j) -= S_factor · A_i · A_j        (shift correction)
```

Built as upper triangular, then symmetrized (lines 135-140).

### 1.3 S-Matrix Extraction

**Matrix inversion** (lines 142-143):
```
[R(E) + iS(E)]^(-1) = RI + iSI
```

Uses Frobenius-Schur method (subroutine `Frobns`, lines 189-222) for complex 3×3 inversion.

**Special case: No fission** (lines 151-159):
If both fission channels are closed (Γ_fa = Γ_fb = 0), reduce to 1D scalar:
```
RI(1,1) = R(1,1) / (R² + S²)
SI(1,1) = -S(1,1) / (R² + S²)
```

**Scattering matrix U** (lines 161-164):
```
U_11 = P₁(2·RI(1,1) - 1) + 2P₂·SI(1,1) + i[P₂(1 - 2·RI(1,1)) + 2P₁·SI(1,1)]
```

where P₁ = cos(2φ_L), P₂ = sin(2φ_L) from hard-sphere phase shift.

---

## 2. Cross Section Formulas

**Source**: Lines 164-175 in `dopush1.f90`

### 2.1 Partial Cross Sections

**Elastic scattering** (line 164):
```
σ_el = (2G_J/K²) · [(1 - Re{U_11})² + Im{U_11}²]
```

**Transmission** (line 165):
```
σ_trans = (2G_J/K²) · (1 - Re{U_11})
```

**Fission** (if channels open, lines 171-175):
```
σ_f = (4G_J/K²) · (T₁² + T₂² + T₃² + T₄²)
```
where T₁, T₂, T₃, T₄ are real and imaginary parts of off-diagonal inverse matrix elements RI(1,2), SI(1,2), RI(1,3), SI(1,3).

**Capture (gamma)** (line 176):
```
σ_c = σ_trans - σ_f - σ_el
```

**Total cross section** (line 179):
```
σ_t = σ_el + σ_f + σ_c
```

### 2.2 Potential Scattering Correction

For intermediate J values (not at boundaries, lines 166-169):
```
σ_el += (2G_J/K²) · (1 - P₁)
σ_trans += (2G_J/K²) · (1 - P₁)
```

This adds hard-sphere potential scattering background.

### 2.3 Normalization

Final normalization (lines 179-182):
```
σ → σ · π/K²
```

Wave number K (lines 44-47):
```
K = √(2m_r E) / ℏ = Zke · √|E|
```

where:
- Zke = Twomhb · A_rat ≈ 2.197×10⁻⁴ · (M_target / (M_target + M_neutron))
- Twomhb from constants module (line 15)
- Energy E in eV, distances in fm

---

## 3. Angular Momentum Summation

**Source**: Lines 73-176 in `dopush1.f90`

For target spin I and s-wave neutrons (L=0):
```
J_min = |I - 1/2|
J_max = I + 1/2
```

Each J contributes with statistical weight (line 76):
```
G_J = (2J + 1) / (2I + 1)
```

**Total cross section** = Σ_J [G_J · σ_J(E)]

---

## 4. Key Physical Constants

**Source**: `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY/sammy/src/mmas7.f90` (lines 7-108)

```
Neutron mass:     m_n ≈ 1.00866 amu
Twomhb:           2mℏ²[eV·fm²]^(-1) ≈ 2.197×10⁻⁴
Planck constant:  ℏc ≈ 197.327 MeV·fm
π/100:            For barn conversion (1 barn = 10⁻²⁴ cm²)
```

---

## 5. Numerical Algorithms

### 5.1 Complex Matrix Inversion: Frobenius-Schur

**Source**: Lines 189-222 in `dopush1.f90`

For 3×3 complex matrix C = R + iS:

1. Compute R^(-1) via Gaussian elimination (subroutine `Thrinv`)
2. If R is singular: Construct rank-reduction matrix D = B·R^(-1)·B
3. Form augmented matrix C' = C + D
4. Invert C' via standard complex inversion
5. Solve for C^(-1) using Frobenius identity:
   ```
   C^(-1) = R^(-1) - R^(-1)·C'^(-1)·R^(-1)·B·R^(-1)
   ```

**Advantages**:
- Handles near-singular R-matrices (resonance overlaps)
- Maintains numerical stability via rank augmentation

### 5.2 Symmetric Matrix Inversion

**Source**: Subroutine `Thrinv`, lines 249-294 in `dopush1.f90`

Specialized for 3×3 real symmetric matrices:
- Pivot-free method (diagonal-heavy matrices)
- Explicit formula (no loops): determinant + cofactor expansion
- Error flag if determinant < machine epsilon

---

## 6. Special Cases and Edge Handling

| Condition | Treatment | Source |
|---|---|---|
| **E < 0** | K = Zke·√\|E\| (keep K real) | Line 46 |
| **Negative Γ_f** | Sign preserved in A_fa, A_fb | Lines 102-106 |
| **No fission** | 1D scalar R-function | Lines 151-159 |
| **Singular R-matrix** | Frobenius-Schur rank reduction | Lines 205-220 |
| **Zero Γ_γ** | Threshold resonance (D = Δ²) | Line 109 |

---

## 7. Validation Reference

**Test case**: SAMMY `ex003` (synthetic nuclide)
- 12 resonances: 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 eV
- J = 1.0 (total angular momentum, column 2 in PAR file)
- All widths: Γ_n = 0.5, Γ_fa = 0.5, Γ_fb = 0.5 milliEV
- **PAR file format**: E_r (eV), J (dimensionless), Γ_n, Γ_fa, Γ_fb (milliEV)
- Expected output: capture, elastic, fission, transmission, total cross sections
- Tolerance: 1e-4 relative error

**Files**:
- Input: `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY/samexm/samexm/ex003/ex003c.par`
- Reference: `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY/samexm/samexm/ex003/answers/ex003*.lpt`

---

## 8. Implementation Checklist

- [ ] Wave number calculation (K from E, mass ratio)
- [ ] Penetration factors P_l (see separate document)
- [ ] Hard-sphere phase shifts φ_l (see separate document)
- [ ] Energy-dependent R-matrix construction (loop over resonances)
- [ ] Complex 3×3 matrix inversion (Frobenius-Schur)
- [ ] S-matrix element U_11 extraction
- [ ] Partial cross section formulas
- [ ] Angular momentum summation (loop over J)
- [ ] Potential scattering correction
- [ ] Final normalization π/K²

---

**Next**: See `penetration_shift_factors.md` for P_l, S_l, φ_l calculations.
