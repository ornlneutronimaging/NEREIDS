# Penetration and Shift Factors for R-Matrix Cross Sections

**Date**: 2026-02-06
**Source**: SAMMY Fortran implementation analysis
**Purpose**: Reference for implementing P_l, S_l, and φ_l calculations in Rust

---

## 1. Overview

Penetration factors **P_l** and shift factors **S_l** are energy- and angular-momentum-dependent functions that:
- Account for the **centrifugal barrier** in partial-wave scattering
- Modify resonance widths and phase shifts
- Depend on dimensionless wave number **ρ = ka** where:
  - k = neutron wave number
  - a = channel radius (scattering length)

**Physics**:
- Higher l → stronger centrifugal barrier → smaller P_l
- Shift factors S_l encode quantum phase corrections from potential well
- Hard-sphere phase shifts φ_l needed for potential scattering

---

## 2. Penetration Factors P_l

**Source**: `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY/sammy/src/xxx/mxxxb.f90`
**Subroutine**: `Facts_NML(L, Rho, Sf, Pf)` (lines 6-62)

### 2.1 Formulas

| l | P_l(ρ) | Implementation (line) |
|---|--------|-------------|
| 0 | ρ | `Pf = A` (24) |
| 1 | ρ³/(1+ρ²) | `Pf = A*A2/(1+A2)` (30) |
| 2 | ρ⁵/(9+3ρ²+ρ⁴) | `Pf = A*A4/D` where `D=9+A2*(3+A2)` (35-36) |
| 3 | ρ⁷/(225+45ρ²+6ρ⁴+ρ⁶) | `Pf = A*A6/D` where `D=225+A2*(45+A2*(6+A2))` (43-44) |
| 4 | ρ⁹/(11025+1575ρ²+135ρ⁴+10ρ⁶+ρ⁸) | `Pf = A*A8/D` where `D=11025+A2*(1575+A2*(135+A2*(10+A2)))` (52-53) |

**Variables**:
- `A = Rho` = ρ
- `A2 = A*A` = ρ²
- `A4 = A2*A2` = ρ⁴
- `A6 = A4*A2` = ρ⁶
- `A8 = A4*A4` = ρ⁸
- `D` = denominator polynomial

### 2.2 Numerical Method: Horner's Scheme

**Example (l=2, lines 32-36)**:
```fortran
A2 = A*A
A4 = A2*A2
D = 9.0d0 + A2*(3.0d0 + A2)    ! Horner nesting: 9 + ρ²(3 + ρ²)
Pf = A*A4/D
```

**Advantages**:
- Avoids computing large powers separately before summing
- Reduces floating-point error accumulation
- Minimizes operations: 3 multiplies + 2 adds (vs. 5+ if expanded)

**Pattern**:
```
P_l = ρ^(2l+1) / Denominator(ρ²)
```

Denominator coefficients derive from Coulomb wave function recursions (related to spherical Bessel functions).

---

## 3. Shift Factors S_l

**Source**: Same file, lines 6-62

### 3.1 Formulas

| l | S_l(ρ) | Implementation (line) |
|---|--------|-------------|
| 0 | 0 | `Sf = 0.0d0` (24) |
| 1 | -1/(1+ρ²) | `Sf = -1.0d0/(1.0d0+A2)` (29) |
| 2 | -(18+3ρ²)/(9+3ρ²+ρ⁴) | `Sf = -(18.0d0+3.0d0*A2)/D` (36) |
| 3 | -(675+90ρ²+6ρ⁴)/(225+45ρ²+6ρ⁴+ρ⁶) | `Sf = -(675.0d0+A2*(90.0d0+6.0d0*A2))/D` (44) |
| 4 | -(44100+4725ρ²+270ρ⁴+10ρ⁶)/(11025+1575ρ²+135ρ⁴+10ρ⁶+ρ⁸) | `Sf = -(44100.0d0+A2*(4725.0d0+A2*(270.0d0+10.0d0*A2)))/D` (53) |

**Properties**:
- Always **non-positive** (S_l ≤ 0)
- S_0 = 0 exactly (no shift for s-wave)
- Approach **-l** as ρ → 0 for l > 0 (S_1 → -1, S_2 → -2, S_3 → -3, etc.)

### 3.2 Physical Interpretation

S_l represents the **logarithmic derivative** of the Coulomb wave function at the channel radius:
```
S_l = a · (d/dr)[ln F_l(kr)]|_{r=a}
```

This modifies the resonance energy via:
```
E_eff = E_0 - (Γ_n / 2P_l) · S_l
```

---

## 4. Hard-Sphere Phase Shifts φ_l

**Source**: `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY/sammy/src/xxx/mxxxa.f90`
**Subroutine**: `Facphi_NML(Lspin, Rho, Cs, Si)` (lines 6-66)

### 4.1 Intermediate Variable B_l

For each l, compute auxiliary value B = B_l(ρ):

| l | B_l(ρ) | Implementation (line) |
|---|--------|-------------|
| 0 | 0 | `B = 0.0d0` (24) |
| 1 | ρ | `B = A` (28) |
| 2 | 3ρ/(3-ρ²) | `B = 3.0d0*A/(3.0d0-A2)` (32-34) |
| 3 | (15-ρ²)ρ/(15-6ρ²) | `B = (15.0d0-A2)*A/(15.0d0-6.0d0*A2)` (38-41) |
| 4 | (105-10ρ²)ρ/(105-45ρ²+ρ⁴) | `B = (105.0d0-10.0d0*A2)*A/(105.0d0-45.0d0*A2+A4)` (45-49) |

**⚠️ Singularities**:
- l=2: Denominator → 0 when ρ² = 3 (ρ ≈ 1.732)
- l=3: Denominator → 0 when ρ² = 15/6 = 2.5 (ρ ≈ 1.581)
- l=4: Denominator has complex roots (resonance region)

These correspond to **hard-sphere resonances** where the approximation breaks down.

### 4.2 Phase Shift Relation

**Physics** (line 55 comment):
```
φ_l = ρ - arctan(B_l)
```

But we need **cos(2φ_l)** and **sin(2φ_l)** for scattering amplitude, not φ itself.

### 4.3 Double-Angle Formulas

**Source**: Lines 56-63

```fortran
C = cos²(ρ)                          ! Line 57: C = C1*C1
D = sin(ρ)/cos(ρ) = tan(ρ)          ! Line 57: D = S1/C1
E = 2cos²(ρ)/(1+B²)                  ! Line 59: E = 2*C/(1.0d0+B*B)

cos(2φ) = E(1+BD)² - 1              ! Line 60: Cs = E*(1.0d0+B*D)**2 - 1.0d0
sin(2φ) = E(-B+D)(1+BD)             ! Line 62: Si = E*(-B+D)*(1.0d0+B*D)
```

**Derivation** (using arctan addition formulas):
```
2φ = 2ρ - 2arctan(B)
cos(2φ) = cos(2ρ)cos(2arctan B) + sin(2ρ)sin(2arctan B)
sin(2φ) = sin(2ρ)cos(2arctan B) - cos(2ρ)sin(2arctan B)
```

where:
```
cos(2arctan B) = (1-B²)/(1+B²)
sin(2arctan B) = 2B/(1+B²)
```

Substitute and simplify → formulas above.

---

## 5. Implementation Algorithm

### 5.1 Input

- `l`: Orbital angular momentum (0 ≤ l ≤ 4)
- `rho`: Dimensionless wave number ρ = ka

### 5.2 Output

- `P_l`: Penetration factor
- `S_l`: Shift factor
- `cos_2phi`: cos(2φ_l)
- `sin_2phi`: sin(2φ_l)

### 5.3 Rust Pseudocode

```rust
fn compute_factors(l: u32, rho: f64) -> (f64, f64, f64, f64) {
    let a = rho;
    let a2 = a * a;

    match l {
        0 => {
            let pf = a;
            let sf = 0.0;
            let b = 0.0;
            let (cos_2phi, sin_2phi) = double_angle(a, b);
            (pf, sf, cos_2phi, sin_2phi)
        },
        1 => {
            let pf = a * a2 / (1.0 + a2);
            let sf = -1.0 / (1.0 + a2);
            let b = a;
            let (cos_2phi, sin_2phi) = double_angle(a, b);
            (pf, sf, cos_2phi, sin_2phi)
        },
        2 => {
            let a4 = a2 * a2;
            let d = 9.0 + a2 * (3.0 + a2);  // Horner
            let pf = a * a4 / d;
            let sf = -(18.0 + 3.0 * a2) / d;
            let b = 3.0 * a / (3.0 - a2);    // ⚠️ Check denominator != 0
            let (cos_2phi, sin_2phi) = double_angle(a, b);
            (pf, sf, cos_2phi, sin_2phi)
        },
        // ... cases 3, 4 similar
        _ => panic!("Unsupported l > 4")
    }
}

fn double_angle(rho: f64, b: f64) -> (f64, f64) {
    let c1 = rho.cos();
    let s1 = rho.sin();
    let c = c1 * c1;
    let d = s1 / c1;  // tan(rho)
    let e = 2.0 * c / (1.0 + b * b);
    let cos_2phi = e * (1.0 + b * d).powi(2) - 1.0;
    let sin_2phi = e * (-b + d) * (1.0 + b * d);
    (cos_2phi, sin_2phi)
}
```

### 5.4 Edge Cases

| Issue | Condition | Mitigation |
|---|---|---|
| **Division by zero in B_l** | ρ² = 3 (l=2), ρ² = 2.5 (l=3) | Return error or use limiting value |
| **⚠️ CRITICAL: cos(ρ) = 0** | ρ = π/2, 3π/2, ... | **MUST CHECK** before tan(ρ) computation in double_angle |
| **Very small ρ** | ρ < 1e-10 | Use Taylor series: P_l → ρ^(2l+1), S_l → -l for l > 0 |
| **Very large ρ** | ρ > 100 | Asymptotic forms (beyond scope for resonance region) |

**Note on cos(ρ) singularity**: For high energies (E > 10 keV) or large channel radii (a > 10 fm), ρ can exceed π/2. The double-angle formula requires d = tan(ρ) = sin(ρ)/cos(ρ), which **diverges** when cos(ρ) = 0. This must be checked explicitly before computation.

---

## 6. Physical Validation

### 6.1 Limiting Behaviors

**ρ → 0**:
```
P_0 → ρ
P_l → ρ^(2l+1) for l > 0  (strong suppression)
S_0 = 0 (exactly)
S_l → -l for l > 0  (S_1 → -1, S_2 → -2, etc.)
```

**ρ → ∞** (outside typical resonance region):
```
P_l → constant (order unity)
S_l → 0
```

### 6.2 Test Values (from SAMMY validation)

Example: ρ = 1.0, l = 1
```
P_1 = 1³/(1+1) = 0.5
S_1 = -1/(1+1) = -0.5
```

Example: ρ = 2.0, l = 2
```
P_2 = ρ⁵/(9+3ρ²+ρ⁴) = 32/(9+12+16) = 32/37 ≈ 0.865
S_2 = -(18+3ρ²)/(9+3ρ²+ρ⁴) = -(18+12)/37 = -30/37 ≈ -0.811
```

### 6.3 Cross-Check Against SAMMY

Use SAMMY `ex003` test case:
- Effective radius a = 2.908 fm
- Energy range: 10 µeV to 1200 eV
- Compute ρ = k·a at each energy point
- Compare P_l, S_l values if SAMMY outputs them explicitly

---

## 7. Implementation Checklist

- [ ] Wave number k from energy: k = √(2m_r E)/ℏ
- [ ] Dimensionless ρ = ka computation
- [ ] P_l formulas for l=0,1,2,3,4 (Horner scheme)
- [ ] S_l formulas for l=0,1,2,3,4 (Horner scheme)
- [ ] B_l intermediate values
- [ ] Double-angle formulas for cos(2φ), sin(2φ)
- [ ] Edge case handling (division by zero, very small/large ρ)
- [ ] Unit tests with known values
- [ ] Validation against SAMMY ex003

---

**Next**: See `ex003_validation_spec.md` for test case details and comparison strategy.
